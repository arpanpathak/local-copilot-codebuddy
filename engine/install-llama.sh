#!/usr/bin/env bash
# Builds llama.cpp with CUDA for this Jetson and installs the parts
# local-copilot-codebuddy links against, so it can run GGUF models.
#
# Installs into $LLAMA_ROOT (default ~/.local/lib/local-copilot-codebuddy-llama):
#   include/   llama.h and the ggml headers
#   lib/       libllama.so and the ggml libraries (CUDA backend included)
#
# The source is kept in $LLAMA_SRC (default ~/.local/src/llama.cpp). Building
# takes about 20 minutes on a Jetson Orin NX; an existing build is only updated.
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:-$HOME/.local/lib/local-copilot-codebuddy-llama}"
LLAMA_SRC="${LLAMA_SRC:-$HOME/.local/src/llama.cpp}"
# The GPU architecture: 87 is Jetson Orin.
CUDA_ARCH="${CUDA_ARCH:-87}"

if [[ ! -d "$LLAMA_SRC" ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
fi

echo "==> building llama.cpp (about 20 minutes the first time)"
# $ORIGIN: the libraries find each other in whatever folder they are installed to.
cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
# Four jobs: CUDA compiles are memory-hungry, and more would push the Jetson into swap.
cmake --build "$LLAMA_SRC/build" -j4 --target llama

mkdir -p "$LLAMA_ROOT/include" "$LLAMA_ROOT/lib"
cp "$LLAMA_SRC/include/llama.h" "$LLAMA_SRC"/ggml/include/*.h "$LLAMA_ROOT/include/"
cp -a "$LLAMA_SRC"/build/bin/libllama.so* "$LLAMA_SRC"/build/bin/libggml*.so* "$LLAMA_ROOT/lib/"
echo "llama.cpp installed in $LLAMA_ROOT"
