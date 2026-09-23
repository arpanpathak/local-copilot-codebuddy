// A thin C interface over TensorRT-LLM's C++ Executor, so Rust can call it.
//
// The Executor is thread-safe: requests can be started, awaited and cancelled
// from different threads. Every function catches C++ exceptions and reports
// them through the `error` buffer instead of letting them cross into Rust.

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

struct EdgeEngine
{
    tle::Executor executor;
};

// Generation settings for one request.
struct EdgeSampling
{
    int32_t max_new_tokens;
    float temperature; // 0 means greedy
    float top_p;
    int32_t top_k;            // 0 means no top-k limit
    float repetition_penalty; // 1 means none
    uint64_t random_seed;
    int32_t end_token;
};

// Called once for each generated token.
using EdgeTokenCallback = void (*)(void* context, int32_t token);

static void write_error(char* error, size_t error_len, char const* message)
{
    if (error_len == 0)
    {
        return;
    }
    std::strncpy(error, message, error_len - 1);
    error[error_len - 1] = '\0';
}

// The weights now live in GPU memory: give the file cache of the engines in
// `engine_dir` back to the rest of the system instead of letting it linger.
static void drop_engine_file_cache(char const* engine_dir)
{
    std::error_code ignored;
    for (auto const& entry : std::filesystem::directory_iterator(engine_dir, ignored))
    {
        if (entry.path().extension() != ".engine")
        {
            continue;
        }
        int const fd = open(entry.path().c_str(), O_RDONLY | O_CLOEXEC);
        if (fd >= 0)
        {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
        }
    }
}

extern "C"
{

// Loads the engine in `engine_dir` (the output of trtllm-build).
// `paged_context` must match how the engine was built (--use_paged_context_fmha);
// it enables chunked prefill and reusing the KV cache of earlier turns.
// Returns null on failure, with the reason in `error`.
EdgeEngine* edge_engine_open(
    char const* engine_dir, int32_t kv_cache_tokens, bool paged_context, char* error, size_t error_len)
{
    try
    {
        // TensorRT-LLM logs to the terminal, which would draw over the chat UI.
        // Errors still reach Rust through `error`.
        tlc::Logger::getLogger()->setLevel(tlc::Logger::ERROR);
        initTrtLlmPlugins();
        // Jetson memory is shared with the CPU: cap the KV cache explicitly.
        // Block reuse keeps the cache of the conversation so far, so a new turn
        // only processes the new message instead of the whole history.
        auto const kv_cache = tle::KvCacheConfig(/*enableBlockReuse=*/paged_context, kv_cache_tokens);
        auto const config = tle::ExecutorConfig(
            /*maxBeamWidth=*/1, tle::SchedulerConfig(), kv_cache, /*enableChunkedContext=*/paged_context);
        // Memory-map the engine instead of reading it into a heap buffer. Read
        // into the heap, the 5+ GB file sits in RAM twice while its weights are
        // copied to the GPU (which shares that RAM on Jetson), pushing the rest
        // of the system into swap. Mapped pages are file cache the kernel can
        // drop as soon as they are copied.
        auto engine = new EdgeEngine{
            tle::Executor(engine_dir, tle::ModelType::kDECODER_ONLY, config, /*useMMap=*/true)};
        drop_engine_file_cache(engine_dir);
        return engine;
    }
    catch (std::exception const& exception)
    {
        write_error(error, error_len, exception.what());
        return nullptr;
    }
}

void edge_engine_close(EdgeEngine* engine)
{
    delete engine;
}

// Starts generating a reply to `prompt` (token ids), streamed token by token.
// Returns the request id, or 0 on failure with the reason in `error`.
uint64_t edge_engine_start(EdgeEngine* engine, int32_t const* prompt, size_t prompt_len, EdgeSampling sampling,
    char* error, size_t error_len)
{
    try
    {
        auto sampling_config = tle::SamplingConfig(/*beamWidth=*/1);
        if (sampling.temperature > 0.0f)
        {
            sampling_config.setTemperature(sampling.temperature);
            sampling_config.setTopP(sampling.top_p);
            if (sampling.top_k > 0)
            {
                sampling_config.setTopK(sampling.top_k);
            }
            sampling_config.setRandomSeed(sampling.random_seed);
        }
        else
        {
            sampling_config.setTopK(1); // greedy
        }
        sampling_config.setRepetitionPenalty(sampling.repetition_penalty);

        auto const output_config = tle::OutputConfig(false, false, false, /*excludeInputFromOutput=*/true);
        auto request = tle::Request(tle::VecTokens(prompt, prompt + prompt_len), sampling.max_new_tokens,
            /*streaming=*/true, sampling_config, output_config, sampling.end_token);
        return engine->executor.enqueueRequest(request);
    }
    catch (std::exception const& exception)
    {
        write_error(error, error_len, exception.what());
        return 0;
    }
}

// Waits until request `request_id` produces tokens, passing each one to
// `on_token`. Sets `*finished` once the reply is complete (or cancelled).
// Returns false on failure, with the reason in `error`.
bool edge_engine_wait(EdgeEngine* engine, uint64_t request_id, EdgeTokenCallback on_token, void* context,
    bool* finished, char* error, size_t error_len)
{
    try
    {
        *finished = false;
        for (auto const& response : engine->executor.awaitResponses(request_id))
        {
            if (response.hasError())
            {
                write_error(error, error_len, response.getErrorMsg().c_str());
                return false;
            }
            auto const result = response.getResult();
            for (auto const token : result.outputTokenIds.at(0))
            {
                on_token(context, token);
            }
            *finished = *finished || result.isFinal;
        }
        return true;
    }
    catch (std::exception const& exception)
    {
        write_error(error, error_len, exception.what());
        return false;
    }
}

// Stops request `request_id`; a pending edge_engine_wait then reports it finished.
void edge_engine_cancel(EdgeEngine* engine, uint64_t request_id)
{
    try
    {
        engine->executor.cancelRequest(request_id);
    }
    catch (std::exception const&)
    {
        // Already finished: nothing to cancel.
    }
}

} // extern "C"
