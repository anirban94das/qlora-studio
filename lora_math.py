"""
The arithmetic behind QLoRA, kept separate so the UI can explain every number.

LoRA in one line: instead of updating a big weight W (out x in), freeze it and
learn a low-rank update  dW = B @ A * (alpha / r)  with A: (r x in), B: (out x r).
So each adapted Linear adds r * (in + out) trainable numbers.

QLoRA = LoRA where the frozen W is stored in 4-bit NF4 (about 0.5 bytes per
weight instead of 2), which is what lets a 1.5B model train on a 6 GB laptop GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from inspect_model import ModelInfo

GB = 1024**3


@dataclass
class Plan:
    lora_params: int
    lora_pct: float
    per_module: dict[str, int]         # short name -> trainable params across all layers
    mem_base: float                    # GB, each part of the estimate
    mem_lora: float
    mem_activations: float
    mem_overhead: float

    @property
    def mem_total(self) -> float:
        return self.mem_base + self.mem_lora + self.mem_activations + self.mem_overhead


def plan(info: ModelInfo, targets: list[str], r: int, batch: int, seq_len: int) -> Plan:
    per_module: dict[str, int] = {}
    for short in targets:
        n = sum(r * (l.in_features + l.out_features) for l in info.linear_by_short(short)
                if l.in_features and l.out_features)
        per_module[short] = n * info.num_layers
    lora = sum(per_module.values())

    # Frozen base: block Linears in NF4 (~0.5 B/param + ~6% quant constants),
    # everything else (embeddings, lm_head, norms) stays 16-bit.
    q = info.quantizable_params
    rest = info.total_params - q
    mem_base = (q * 0.5 * 1.06 + rest * 2) / GB

    # Adapters in fp32: weight 4 + grad 4 + paged 8-bit Adam states 2 = ~10 B/param.
    mem_lora = lora * 10 / GB

    # Activations with gradient checkpointing: one saved hidden state per layer,
    # plus the working set of one layer being recomputed, plus fp32 logits and
    # their gradient (big for 150k-word vocabularies!).
    tokens = batch * seq_len
    act = tokens * info.hidden * 2 * info.num_layers                     # checkpoints
    act += tokens * max(info.intermediate or 4 * info.hidden, info.hidden) * 2 * 6  # one live layer
    act += tokens * info.vocab * 4 * 2                                    # logits + grad
    mem_act = act / GB

    overhead = 0.6  # CUDA context, cuBLAS workspaces, allocator fragmentation
    return Plan(lora, 100 * lora / max(info.total_params, 1), per_module,
                mem_base, mem_lora, mem_act, overhead)


def peft_snippet(repo: str, targets: list[str], r: int, alpha: int, dropout: float) -> str:
    t = ", ".join(f'"{x}"' for x in targets)
    return f'''import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,                 # the "Q" in QLoRA
    bnb_4bit_quant_type="nf4",         # 4-bit NormalFloat
    bnb_4bit_use_double_quant=True,    # also quantise the quantisation constants
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained("{repo}", quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)   # enables gradient checkpointing

lora = LoraConfig(r={r}, lora_alpha={alpha}, lora_dropout={dropout},
                  target_modules=[{t}], task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()
'''
