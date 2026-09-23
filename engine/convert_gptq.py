"""Converts a Hugging Face GPTQ-Int4 Qwen2 / Qwen2.5 model to a TensorRT-LLM checkpoint.

TensorRT-LLM 0.12's own examples/qwen/convert_checkpoint.py first loads the
model through transformers, which requires the auto-gptq package, and
auto-gptq has no build for Jetson. The GPTQ weight loader behind it only calls
`state_dict()` on that model, so this script reads the tensors straight from
the .safetensors files and hands them over.

Usage (inside the container):
    python3 convert_gptq.py --model_dir /models/<hf-model> --output_dir /models/<name>-ckpt
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.qwen.config import QWenConfig
from tensorrt_llm.models.qwen.convert import load_weights_from_hf_gptq_model
from tensorrt_llm.quantization import QuantAlgo


class SafetensorsWeights:
    """Stands in for a transformers model: the GPTQ loader only calls `state_dict()`."""

    def __init__(self, model_dir: Path):
        self.tensors = {}
        for shard in sorted(model_dir.glob("*.safetensors")):
            self.tensors.update(load_file(shard))

    def state_dict(self):
        return self.tensors


def gptq_quant_config(model_dir: Path) -> QuantConfig:
    """Reads the GPTQ settings (4-bit, group size) from the model's config.json."""
    hf_config = json.loads((model_dir / "config.json").read_text())
    gptq = hf_config["quantization_config"]
    if gptq.get("bits") != 4 or gptq.get("quant_method") != "gptq":
        raise SystemExit(f"expected a 4-bit GPTQ model, got {gptq}")

    return QuantConfig(
        quant_algo=QuantAlgo.W4A16_GPTQ,
        group_size=gptq["group_size"],
        has_zero_point=True,
        pre_quant_scale=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_dir", type=Path, required=True, help="Hugging Face GPTQ model directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="where to write the TensorRT-LLM checkpoint")
    args = parser.parse_args()

    config = QWenConfig.from_hugging_face(
        str(args.model_dir), dtype="float16", quant_config=gptq_quant_config(args.model_dir)
    )
    weights = load_weights_from_hf_gptq_model(SafetensorsWeights(args.model_dir), config)

    model = QWenForCausalLM(config)
    model.load(weights)
    model.save_checkpoint(str(args.output_dir), save_config=True)
    print(f"checkpoint written to {args.output_dir}")


if __name__ == "__main__":
    main()
