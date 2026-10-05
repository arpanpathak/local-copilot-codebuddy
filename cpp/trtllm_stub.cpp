// Stands in for trtllm_shim.cpp in a build without the TensorRT-LLM runtime,
// such as a llama.cpp-only build on a device with no TensorRT-LLM 0.12. It
// exports the same C functions; opening an engine fails with a message that
// says why, so GGUF models still work and TensorRT-LLM models fail cleanly.

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

struct EdgeEngine;

struct EdgeSampling {
    int32_t max_new_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    float repetition_penalty;
    uint64_t random_seed;
    int32_t end_token;
};

typedef void (*EdgeTokenCallback)(void* context, int32_t token);

static void write_error(char* error, size_t error_len) {
    static char const message[] =
        "this build has no TensorRT-LLM runtime: run engine/install-runtime.sh and reinstall, or pick a .gguf model";
    if (error != nullptr && error_len > 0) {
        std::strncpy(error, message, error_len - 1);
        error[error_len - 1] = '\0';
    }
}

EdgeEngine* edge_engine_open(char const*, int32_t, bool, char* error, size_t error_len) {
    write_error(error, error_len);
    return nullptr;
}

void edge_engine_close(EdgeEngine*) {}

uint64_t edge_engine_start(EdgeEngine*, int32_t const*, size_t, EdgeSampling, char* error, size_t error_len) {
    write_error(error, error_len);
    return 0;
}

bool edge_engine_wait(EdgeEngine*, uint64_t, EdgeTokenCallback, void*, bool* finished, char* error, size_t error_len) {
    if (finished != nullptr) {
        *finished = true;
    }
    write_error(error, error_len);
    return false;
}

void edge_engine_cancel(EdgeEngine*, uint64_t) {}

}
