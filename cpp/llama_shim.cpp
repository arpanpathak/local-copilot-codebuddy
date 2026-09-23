// A thin C interface over llama.cpp, the second engine. It runs GGUF models,
// including architectures TensorRT-LLM 0.12 cannot run (e.g. Qwen3.5).
//
// One engine serves one conversation at a time. It remembers the tokens its
// cache holds, so a new turn only processes what changed since the last one.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

struct EdgeLlama
{
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    llama_vocab const* vocab = nullptr;
    // The tokens whose keys and values the context's memory holds, in order.
    std::vector<llama_token> cached;
};

// Generation settings for one request.
struct EdgeLlamaSampling
{
    int32_t max_new_tokens;
    float temperature; // 0 means greedy
    float top_p;
    int32_t top_k;            // 0 means no top-k limit
    float repetition_penalty; // 1 means none
    uint32_t random_seed;
};

// Called with each generated token, and with -1 while the prompt is being
// read. Returning false stops generation.
using EdgeLlamaTokenCallback = bool (*)(void* context, int32_t token);

static void write_error(char* error, size_t error_len, char const* message)
{
    if (error_len == 0)
    {
        return;
    }
    std::strncpy(error, message, error_len - 1);
    error[error_len - 1] = '\0';
}

// llama.cpp logs to the terminal, which would draw over the chat UI.
static void discard_log(ggml_log_level, char const*, void*) {}

extern "C"
{

// Loads the model in `path` with all layers on the GPU. Returns null on
// failure, with the reason in `error`.
EdgeLlama* edge_llama_open(char const* path, char* error, size_t error_len)
{
    llama_log_set(discard_log, nullptr);
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = -1;
    llama_model* model = llama_model_load_from_file(path, params);
    if (model == nullptr)
    {
        write_error(error, error_len, "llama.cpp could not load the model");
        return nullptr;
    }
    auto* engine = new EdgeLlama;
    engine->model = model;
    engine->vocab = llama_model_get_vocab(model);
    return engine;
}

// (Re)creates the context with room for `n_ctx` tokens. Returns false on failure.
bool edge_llama_set_context(EdgeLlama* engine, uint32_t n_ctx, char* error, size_t error_len)
{
    if (engine->context != nullptr)
    {
        llama_free(engine->context);
        engine->context = nullptr;
    }
    engine->cached.clear();
    auto params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = 2048;
    params.n_ubatch = 512;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    engine->context = llama_init_from_model(engine->model, params);
    if (engine->context == nullptr)
    {
        write_error(error, error_len, "llama.cpp could not create a context");
        return false;
    }
    return true;
}

void edge_llama_close(EdgeLlama* engine)
{
    if (engine->context != nullptr)
    {
        llama_free(engine->context);
    }
    llama_model_free(engine->model);
    delete engine;
}

// The context length the model was trained for.
int32_t edge_llama_n_ctx_train(EdgeLlama const* engine)
{
    return llama_model_n_ctx_train(engine->model);
}

// The model's chat template (Jinja), or null.
char const* edge_llama_chat_template(EdgeLlama const* engine)
{
    return llama_model_chat_template(engine->model, nullptr);
}

// Tokenizes `text`, reading special tokens such as <|im_start|> as tokens.
// Returns the number of tokens, or minus the number needed if `max` is too small.
int32_t edge_llama_tokenize(EdgeLlama const* engine, char const* text, int32_t text_len, int32_t* tokens, int32_t max)
{
    return llama_tokenize(engine->vocab, text, text_len, tokens, max, /*add_special=*/false, /*parse_special=*/true);
}

// Writes the text of `token` to `buf`. Returns its length, or minus the length
// needed if `buf` is too small.
int32_t edge_llama_token_piece(EdgeLlama const* engine, int32_t token, char* buf, int32_t len)
{
    return llama_token_to_piece(engine->vocab, token, buf, len, /*lstrip=*/0, /*special=*/false);
}

// Generates a reply to `prompt`, passing each token to `on_token` until the
// model ends its turn, `max_new_tokens` is reached or `on_token` returns false.
// Returns false on failure, with the reason in `error`.
bool edge_llama_generate(EdgeLlama* engine, int32_t const* prompt, size_t prompt_len, EdgeLlamaSampling sampling,
    EdgeLlamaTokenCallback on_token, void* context, char* error, size_t error_len)
{
    llama_context* ctx = engine->context;
    llama_memory_t memory = llama_get_memory(ctx);

    // Keep the cache of the conversation so far; only process what changed.
    size_t reused = 0;
    auto const& cached = engine->cached;
    while (reused < cached.size() && reused < prompt_len && cached[reused] == prompt[reused])
    {
        ++reused;
    }
    reused = std::min(reused, prompt_len - 1); // the last prompt token must be processed for its logits
    if (reused < cached.size() && !llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(reused), -1))
    {
        // Some models (with recurrent layers) cannot drop only the end: start over.
        llama_memory_clear(memory, true);
        reused = 0;
    }
    engine->cached.resize(reused);

    // Read the rest of the prompt in chunks.
    size_t const batch = llama_n_batch(ctx);
    for (size_t start = reused; start < prompt_len; start += batch)
    {
        if (!on_token(context, -1))
        {
            return true;
        }
        auto const count = static_cast<int32_t>(std::min(batch, prompt_len - start));
        std::vector<llama_token> chunk(prompt + start, prompt + start + count);
        if (llama_decode(ctx, llama_batch_get_one(chunk.data(), count)) != 0)
        {
            llama_memory_clear(memory, true);
            engine->cached.clear();
            write_error(error, error_len, "llama.cpp could not process the prompt");
            return false;
        }
        engine->cached.insert(engine->cached.end(), chunk.begin(), chunk.end());
    }

    auto* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler,
        llama_sampler_init_penalties(llama_vocab_n_tokens(engine->vocab), 64, sampling.repetition_penalty, 0.0f, 0.0f));
    if (sampling.temperature > 0.0f)
    {
        if (sampling.top_k > 0)
        {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(sampling.top_k));
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(sampling.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(sampling.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(sampling.random_seed));
    }
    else
    {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }

    bool succeeded = true;
    for (int32_t generated = 0; generated < sampling.max_new_tokens; ++generated)
    {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(engine->vocab, token) || !on_token(context, token))
        {
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0)
        {
            write_error(error, error_len, "llama.cpp could not continue generating");
            succeeded = false;
            break;
        }
        engine->cached.push_back(token);
    }
    llama_sampler_free(sampler);
    return succeeded;
}

} // extern "C"
