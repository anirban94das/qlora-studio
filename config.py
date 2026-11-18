"""
Central configuration for QLoRA Studio.

Same pattern as Claude_Cowork/gemma-web-agent/config.py: every tunable lives
here, and anything machine-specific can be overridden with an environment
variable so nobody has to edit code.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Where trained adapters and logs go.
OUTPUT_DIR = Path(os.environ.get("QLORA_OUTPUT_DIR", ROOT / "outputs"))
SAMPLE_DATA = ROOT / "data" / "pirate_sample.jsonl"

# Where logs go (see logs.py for what each file contains).
LOG_DIR = Path(os.environ.get("QLORA_LOG_DIR", ROOT / "logs"))

# Optional Hugging Face token for gated models (Llama, Gemma, ...).
# Get one at https://huggingface.co/settings/tokens and set HF_TOKEN.
HF_TOKEN = os.environ.get("HF_TOKEN") or None

PORT = int(os.environ.get("QLORA_PORT", "7861"))

# Models offered in the dropdown. You can also type any Hugging Face repo id.
# Comment shows roughly what QLoRA training needs (4-bit base + adapters).
MODEL_CHOICES = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",  # tiny, trains on almost anything
    "Qwen/Qwen2.5-0.5B-Instruct",           # ~2 GB GPU
    "Qwen/Qwen2.5-1.5B-Instruct",           # ~4 GB GPU  (good default for 6 GB cards)
    "Qwen/Qwen3-1.7B",                      # ~4-5 GB GPU
    "meta-llama/Llama-3.2-1B-Instruct",     # gated: needs HF_TOKEN + accepted licence
    "microsoft/Phi-3.5-mini-instruct",      # ~6-7 GB GPU
    "mistralai/Mistral-7B-Instruct-v0.3",   # explore only on small GPUs (~8 GB+ to train)
]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
