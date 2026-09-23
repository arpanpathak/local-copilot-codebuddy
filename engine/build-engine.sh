#!/usr/bin/env bash
# Builds a TensorRT-LLM engine from a Hugging Face GPTQ-Int4 Qwen2/Qwen2.5 checkpoint.
#
# Usage: engine/build-engine.sh [HF_MODEL_ID]
#   default model: Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4
#
# Everything lands in $MODELS_DIR (default ~/models/trt):
#   <name>/          the downloaded Hugging Face weights and tokenizer
#   <name>-ckpt/     the TensorRT-LLM checkpoint (intermediate)
#   <name>-engine/   the TensorRT engine that local-copilot-codebuddy loads
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4}"
MODELS_DIR="${MODELS_DIR:-$HOME/models/trt}"
# Qwen2.5 handles 32K tokens natively. The KV cache for 32K tokens of a 7B model
# is about 1.75 GB, which fits next to the ~5.3 GB of weights on a 16 GB Jetson.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"      # conversation + reply
MAX_INPUT_LEN="${MAX_INPUT_LEN:-30720}"  # conversation, leaving at least 2K tokens for the reply
# Tokens processed per step. Longer prompts are split into chunks of this size
# (chunked prefill), which keeps scratch memory small. Chunking and KV-cache
# reuse between turns both need paged context attention.
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-2048}"

NAME="$(basename "$MODEL_ID")"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$MODELS_DIR"

if [[ ! -f "$MODELS_DIR/$NAME/config.json" ]]; then
    echo "==> downloading $MODEL_ID"
    hf download "$MODEL_ID" --local-dir "$MODELS_DIR/$NAME"
fi

echo "==> building the container image"
docker build -q -t local-copilot-codebuddy-trtllm "$SCRIPT_DIR" >/dev/null

run_in_container() {
    # Run as the calling user so the outputs are not owned by root.
    docker run --rm --runtime nvidia \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -w /tmp \
        -v "$MODELS_DIR:/models" \
        -v "$SCRIPT_DIR:/engine:ro" \
        local-copilot-codebuddy-trtllm "$@"
}

if [[ ! -f "$MODELS_DIR/$NAME-ckpt/config.json" ]]; then
    echo "==> converting GPTQ weights to a TensorRT-LLM checkpoint"
    run_in_container python3 /engine/convert_gptq.py \
        --model_dir "/models/$NAME" \
        --output_dir "/models/$NAME-ckpt"
fi

echo "==> building the TensorRT engine"
# Build next to the current engine and swap only on success, so a failed build
# never leaves you without a working engine.
rm -rf "$MODELS_DIR/$NAME-engine.new"
run_in_container trtllm-build \
    --checkpoint_dir "/models/$NAME-ckpt" \
    --output_dir "/models/$NAME-engine.new" \
    --gemm_plugin float16 \
    --use_paged_context_fmha enable \
    --max_batch_size 1 \
    --max_input_len "$MAX_INPUT_LEN" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --max_num_tokens "$MAX_NUM_TOKENS"

rm -rf "$MODELS_DIR/$NAME-engine.old"
if [[ -d "$MODELS_DIR/$NAME-engine" ]]; then
    mv "$MODELS_DIR/$NAME-engine" "$MODELS_DIR/$NAME-engine.old"
fi
mv "$MODELS_DIR/$NAME-engine.new" "$MODELS_DIR/$NAME-engine"
echo "==> done: $MODELS_DIR/$NAME-engine (previous engine kept in $NAME-engine.old)"
