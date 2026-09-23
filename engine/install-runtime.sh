#!/usr/bin/env bash
# Copies TensorRT-LLM's C++ runtime out of the container image, so local-copilot-codebuddy
# can link against it and run natively (no container, no Python at runtime).
#
# Installs into $TRTLLM_ROOT (default ~/.local/lib/local-copilot-codebuddy-trtllm):
#   include/   TensorRT-LLM executor headers and the matching TensorRT headers
#   lib/       libtensorrt_llm.so and friends, plus TensorRT 10.4 (the engine
#              is tied to the TensorRT version that built it; JetPack 6.2 ships 10.7)
set -euo pipefail

TRTLLM_ROOT="${TRTLLM_ROOT:-$HOME/.local/lib/local-copilot-codebuddy-trtllm}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE=local-copilot-codebuddy-trtllm

docker build -q -t "$IMAGE" "$SCRIPT_DIR" >/dev/null
mkdir -p "$TRTLLM_ROOT/lib" "$TRTLLM_ROOT/include"

docker run --rm --user "$(id -u):$(id -g)" -v "$TRTLLM_ROOT:/out" "$IMAGE" bash -c '
    set -e
    cp -a /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/. /out/lib/
    cp -a /usr/lib/aarch64-linux-gnu/libnvinfer.so.10* /out/lib/
    cp -a /opt/TensorRT-LLM/cpp/include/. /out/include/
    cp -a /usr/include/aarch64-linux-gnu/NvInfer*.h /out/include/
'
# The plugin library's soname carries a version suffix its file name lacks.
ln -sf libnvinfer_plugin_tensorrt_llm.so "$TRTLLM_ROOT/lib/libnvinfer_plugin_tensorrt_llm.so.10"
echo "TensorRT-LLM runtime installed in $TRTLLM_ROOT"
