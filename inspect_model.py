"""
Look inside a real open-source model WITHOUT downloading its weights.

The trick (same spirit as visualize_NNs_VC_/visualize_gemma_model.py, which
built untrained modules from a config): fetch only the model's config.json
(a few KB), then build the real `transformers` model on PyTorch's "meta"
device. Meta tensors have shapes and dtypes but no storage, so a 7B model
"loads" in a second using ~0 RAM -- and every module, layer and weight shape
you see is the genuine architecture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from torch import nn

from config import HF_TOKEN

log = logging.getLogger("qlora.inspect")


@dataclass
class Leaf:
    """One weight-carrying module inside a decoder block (e.g. self_attn.q_proj)."""

    name: str            # path inside the block, e.g. "self_attn.q_proj"
    group: str           # first path component, e.g. "self_attn"
    kind: str            # "linear" | "norm" | "other"
    params: int
    in_features: int | None = None
    out_features: int | None = None
    shape: tuple = ()

    @property
    def short(self) -> str:
        return self.name.split(".")[-1]


@dataclass
class ModelInfo:
    repo: str
    model_type: str
    arch_class: str
    total_params: int
    num_layers: int
    hidden: int
    vocab: int
    heads: int | None
    kv_heads: int | None
    intermediate: int | None
    tied_embeddings: bool
    embed_params: int
    head_params: int
    block_params: int                     # params in ONE decoder block
    other_params: int                     # final norm etc.
    leaves: list[Leaf] = field(default_factory=list)   # contents of block 0
    layer_path: str = ""                 # e.g. "model.layers"
    tree_text: str = ""

    @property
    def linear_names(self) -> list[str]:
        """Unique short names of Linear layers in a block -> LoRA target candidates."""
        seen: list[str] = []
        for leaf in self.leaves:
            if leaf.kind == "linear" and leaf.short not in seen:
                seen.append(leaf.short)
        return seen

    def linear_by_short(self, short: str) -> list[Leaf]:
        return [l for l in self.leaves if l.kind == "linear" and l.short == short]

    @property
    def quantizable_params(self) -> int:
        """Params bitsandbytes turns into 4-bit: Linear weights inside the blocks."""
        return sum(l.params for l in self.leaves if l.kind == "linear") * self.num_layers


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _find_layer_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The decoder stack = the biggest ModuleList whose children share one class."""
    best_name, best = "", None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if len({type(c) for c in mod}) == 1 and (best is None or len(mod) > len(best)):
                best_name, best = name, mod
    if best is None:
        raise ValueError("Could not find a stack of repeated decoder layers in this model.")
    return best_name, best


def _leaf_kind(mod: nn.Module) -> str:
    if isinstance(mod, nn.Linear) or type(mod).__name__ in ("Linear4bit", "Linear8bitLt", "Conv1D"):
        return "linear"
    if "norm" in type(mod).__name__.lower():
        return "norm"
    return "other"


def _block_leaves(block: nn.Module) -> list[Leaf]:
    leaves = []
    for name, mod in block.named_modules():
        if not name or any(True for _ in mod.children()):
            continue
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        if n == 0:
            continue
        kind = _leaf_kind(mod)
        leaf = Leaf(name=name, group=name.split(".")[0], kind=kind, params=n)
        w = getattr(mod, "weight", None)
        if w is not None:
            leaf.shape = tuple(w.shape)
        if kind == "linear":
            if isinstance(mod, nn.Linear):
                leaf.in_features, leaf.out_features = mod.in_features, mod.out_features
            elif leaf.shape:  # GPT-2 style Conv1D stores (in, out)
                leaf.in_features, leaf.out_features = leaf.shape[0], leaf.shape[-1]
        leaves.append(leaf)
    return leaves


def _tree(module: nn.Module, name: str = "model", depth: int = 0, max_depth: int = 4) -> list[str]:
    """Indented module tree; repeated layers are collapsed to 'layer 0 (x N)'."""
    pad = "  " * depth
    n = _count(module)
    lines = [f"{pad}{name}: {type(module).__name__}  [{n:,} params]"]
    if depth >= max_depth:
        return lines
    children = list(module.named_children())
    if isinstance(module, nn.ModuleList) and len(children) > 1 and len({type(c) for _, c in children}) == 1:
        lines += _tree(children[0][1], f"0 .. {len(children) - 1}  (x{len(children)}, identical)", depth + 1, max_depth)
        return lines
    for cname, child in children:
        lines += _tree(child, cname, depth + 1, max_depth)
    return lines


@lru_cache(maxsize=16)
def inspect(repo: str) -> ModelInfo:
    """Build `repo` on the meta device and summarise its real architecture."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    repo = repo.strip()
    t = time.perf_counter()
    log.info("inspecting %s (config.json only, meta device)", repo)
    cfg = AutoConfig.from_pretrained(repo, token=HF_TOKEN)
    log.debug("config for %s: %s", repo, cfg.to_json_string(use_diff=False))
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    try:
        model.tie_weights()  # from_config doesn't always tie embedding <-> lm_head
    except Exception:
        pass

    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    layer_path, layers = _find_layer_list(model)
    block = layers[0]

    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = head is not None and emb is not None and head.weight is emb.weight
    embed_params = _count(emb) if emb is not None else 0
    head_params = 0 if (head is None or tied) else _count(head)
    block_params = _count(block)
    total = _count(model)  # parameters() de-duplicates tied weights
    other = total - embed_params - head_params - block_params * len(layers)

    info = ModelInfo(
        repo=repo,
        model_type=getattr(cfg, "model_type", "?"),
        arch_class=type(model).__name__,
        total_params=total,
        num_layers=len(layers),
        hidden=getattr(text_cfg, "hidden_size", None) or getattr(text_cfg, "n_embd", 0),
        vocab=getattr(text_cfg, "vocab_size", 0),
        heads=getattr(text_cfg, "num_attention_heads", None),
        kv_heads=getattr(text_cfg, "num_key_value_heads", None),
        intermediate=getattr(text_cfg, "intermediate_size", None),
        tied_embeddings=tied,
        embed_params=embed_params,
        head_params=head_params,
        block_params=block_params,
        other_params=max(other, 0),
        leaves=_block_leaves(block),
        layer_path=layer_path,
        tree_text="\n".join(_tree(model)),
    )
    log.info("%s: %s, %s params, %d x %s at %s, hidden %s, vocab %s, tied=%s, linears=%s (%.2fs)",
             repo, info.arch_class, f"{info.total_params:,}", info.num_layers, type(block).__name__,
             layer_path, info.hidden, info.vocab, tied, info.linear_names, time.perf_counter() - t)
    for leaf in info.leaves:
        log.debug("  block0 leaf %-40s %-6s shape=%s params=%s", leaf.name, leaf.kind, leaf.shape,
                  f"{leaf.params:,}")
    return info


def human(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def gpu_info() -> tuple[str, float]:
    """(description, total GB). (\"No CUDA GPU\", 0) on CPU-only machines."""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return p.name, p.total_memory / 1024**3
    return "No CUDA GPU detected", 0.0
