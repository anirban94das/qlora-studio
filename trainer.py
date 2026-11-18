"""
Real QLoRA training, run in a background thread so the UI stays live.

Pipeline: load base model in 4-bit (bitsandbytes NF4) -> attach LoRA adapters
(peft) -> train with the Hugging Face Trainer -> save just the adapter
(a few MB) -> keep the model in memory so you can chat with base vs tuned.

Everything is logged (see studio_log.py): the main log, plus per run
run.log / run_config.json / metrics.jsonl / summary.json in the run folder.
"""

from __future__ import annotations

import logging
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import HF_TOKEN, OUTPUT_DIR, SAMPLE_DATA
from studio_log import (COMPARE_LOG, FMT, DATEFMT, RUNS_INDEX, append_jsonl, format_exc,
                        gpu_mem, write_json)

log = logging.getLogger("qlora.trainer")


@dataclass
class RunState:
    status: str = "idle"          # idle | loading | training | done | stopped | error
    message: str = ""
    step: int = 0
    max_steps: int = 0
    history: list[dict] = field(default_factory=list)   # {step, loss, lr}
    adapter_dir: str = ""
    run_dir: str = ""
    started: float = 0.0
    stop_requested: bool = False
    model: object = None
    tokenizer: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = RunState()


def _set(status: str | None = None, message: str | None = None) -> None:
    """Update the run state AND log the change, so the log mirrors the UI."""
    if status and status != STATE.status:
        log.info("status: %s -> %s", STATE.status, status)
        STATE.status = status
    if message and message != STATE.message:
        log.info("%s", message)
        STATE.message = message


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def _to_messages(row: dict) -> list[dict] | None:
    """Accept the common dataset shapes and turn each row into chat messages."""
    if "messages" in row and isinstance(row["messages"], list):
        return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    if "conversations" in row and isinstance(row["conversations"], list):  # ShareGPT
        role = {"human": "user", "gpt": "assistant", "system": "system"}
        return [{"role": role.get(m.get("from"), "user"), "content": m.get("value", "")}
                for m in row["conversations"]]
    q = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    a = row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
    if q and a:
        if row.get("instruction") and row.get("input"):
            q = f"{row['instruction']}\n\n{row['input']}"
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return None


def load_rows(source: str, limit: int) -> list[str] | list[list[dict]]:
    from datasets import load_dataset

    source = (source or "").strip()
    if not source or source == "sample":
        log.info("dataset: built-in sample %s", SAMPLE_DATA)
        ds = load_dataset("json", data_files=str(SAMPLE_DATA), split="train")
    elif Path(source).exists():
        log.info("dataset: local file %s", source)
        ds = load_dataset("json", data_files=source, split="train")
    else:
        log.info("dataset: Hugging Face hub '%s'", source)
        ds = load_dataset(source, split="train", token=HF_TOKEN)
    log.info("dataset: %d rows available, columns=%s, using up to %d", len(ds), ds.column_names, limit)
    ds = ds.select(range(min(limit, len(ds))))
    out, skipped = [], 0
    for row in ds:
        if "text" in row and isinstance(row["text"], str):
            out.append(row["text"])
        else:
            msgs = _to_messages(row)
            if msgs:
                out.append(msgs)
            else:
                skipped += 1
    if skipped:
        log.warning("dataset: skipped %d rows with no usable columns", skipped)
    if not out:
        raise ValueError("No usable rows. Expected columns like text / messages / "
                         "instruction+output / prompt+response.")
    return out


def render_text(tok, item) -> str:
    if isinstance(item, str):
        return item
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(item, tokenize=False)
    return "\n\n".join(f"### {m['role'].title()}:\n{m['content']}" for m in item) + tok.eos_token


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def start(repo: str, targets: list[str], r: int, alpha: int, dropout: float, lr: float,
          steps: int, batch: int, seq_len: int, dataset: str, n_rows: int) -> str:
    with STATE.lock:
        if STATE.status in ("loading", "training"):
            log.warning("start refused: a run is already %s", STATE.status)
            return "A run is already going — stop it first."
        STATE.status, STATE.message, STATE.step, STATE.max_steps = "loading", "Starting…", 0, steps
        STATE.history, STATE.stop_requested, STATE.started = [], False, time.time()
        STATE.run_dir = STATE.adapter_dir = ""
        if STATE.model is not None:
            log.info("releasing previously trained model from GPU memory")
        STATE.model = STATE.tokenizer = None
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args = (repo, targets, r, alpha, dropout, lr, steps, batch, seq_len, dataset, n_rows)
    threading.Thread(target=_run, args=args, daemon=True, name="train").start()
    return "Started."


def stop() -> str:
    log.info("stop requested by user at step %d/%d", STATE.step, STATE.max_steps)
    STATE.stop_requested = True
    return "Stop requested — finishing the current step…"


def _run(repo, targets, r, alpha, dropout, lr, steps, batch, seq_len, dataset, n_rows):
    run_dir = OUTPUT_DIR / f"{re.sub(r'[^A-Za-z0-9._-]', '_', repo)}-r{r}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    STATE.run_dir = str(run_dir)

    # Per-run log file: receives every log record while this run is going.
    run_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(logging.Formatter(FMT, DATEFMT))
    logging.getLogger().addHandler(run_handler)

    config = dict(repo=repo, targets=targets, r=r, alpha=alpha, dropout=dropout, learning_rate=lr,
                  max_steps=steps, batch_size=batch, max_length=seq_len, dataset=dataset,
                  max_examples=n_rows, run_dir=str(run_dir))
    summary = dict(config=config, status="error", started=time.strftime(DATEFMT))
    metrics_path = run_dir / "metrics.jsonl"
    t0 = time.time()
    log.info("=== training run %s ===", run_dir.name)
    for k, v in config.items():
        log.info("config | %-13s %s", k, v)
    write_json(run_dir / "run_config.json", config)

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  DataCollatorForLanguageModeling, Trainer, TrainerCallback,
                                  TrainingArguments)
        from datasets import Dataset

        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA needs an NVIDIA GPU with CUDA (bitsandbytes 4-bit).")
        bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
        log.info("compute dtype: %s", dtype)

        _set(message=f"Downloading / loading {repo} in 4-bit… (first time can take a few minutes)")
        t = time.time()
        tok = AutoTokenizer.from_pretrained(repo, token=HF_TOKEN)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            log.info("tokenizer had no pad token; using eos (%r)", tok.eos_token)
        log.info("tokenizer: %s, vocab %d, chat template: %s", type(tok).__name__, len(tok),
                 bool(getattr(tok, "chat_template", None)))
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=bnb, device_map={"": 0},
                                                     dtype=dtype, token=HF_TOKEN)
        log.info("model loaded in 4-bit in %.1fs | footprint %.2f GB | %s", time.time() - t,
                 model.get_memory_footprint() / 1024**3, gpu_mem())
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
        trainable, total = model.get_nb_trainable_parameters()
        n_adapted = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A)
        log.info("LoRA attached to %d Linear layers | trainable %s / %s (%.4f%%) | %s",
                 n_adapted, f"{trainable:,}", f"{total:,}", 100 * trainable / total, gpu_mem())
        summary.update(trainable_params=trainable, total_params=total, adapted_layers=n_adapted)

        _set(message="Preparing data…")
        rows = load_rows(dataset, n_rows)
        texts = [render_text(tok, x) for x in rows]
        log.debug("example 0 as the model will see it:\n%s", texts[0])
        ds = Dataset.from_dict({"text": texts}).map(
            lambda b: tok(b["text"], truncation=True, max_length=seq_len),
            batched=True, remove_columns=["text"])
        lens = [len(x) for x in ds["input_ids"]]
        truncated = sum(1 for n in lens if n >= seq_len)
        log.info("tokenised %d examples | tokens min %d / median %d / max %d | %d truncated at %d",
                 len(lens), min(lens), int(statistics.median(lens)), max(lens), truncated, seq_len)
        if truncated:
            log.warning("%d examples were cut at max length %d — consider raising it", truncated, seq_len)
        summary.update(examples=len(lens), tokens_median=int(statistics.median(lens)), truncated=truncated)

        class Live(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                log.info("training begins | %d steps | %s", state.max_steps, gpu_mem())

            def on_log(self, args, state, control, logs=None, **kw):
                logs = logs or {}
                rec = {"step": state.global_step, "epoch": round(state.epoch or 0, 4),
                       "elapsed_s": round(time.time() - t0, 2), **logs, **gpu_mem()}
                append_jsonl(metrics_path, rec)
                if "loss" in logs:
                    STATE.history.append({"step": state.global_step, "loss": float(logs["loss"]),
                                          "lr": float(logs.get("learning_rate", 0.0))})
                    log.info("step %d/%d | loss %.4f | lr %.2e | grad_norm %s | epoch %.2f | GPU %.2f GB (peak %.2f)",
                             state.global_step, state.max_steps, float(logs["loss"]),
                             float(logs.get("learning_rate", 0)), logs.get("grad_norm"),
                             state.epoch or 0, rec.get("gpu_alloc_gb", 0), rec.get("gpu_peak_gb", 0))
                else:
                    log.info("trainer: %s", logs)

            def on_step_end(self, args, state, control, **kw):
                STATE.step = state.global_step
                if STATE.stop_requested:
                    log.info("stopping at step %d (user request)", state.global_step)
                    control.should_training_stop = True

            def on_epoch_end(self, args, state, control, **kw):
                log.info("epoch %.0f finished at step %d", state.epoch or 0, state.global_step)

            def on_train_end(self, args, state, control, **kw):
                log.info("training ended at step %d | %s", state.global_step, gpu_mem())

        targs = TrainingArguments(
            output_dir=str(run_dir), max_steps=steps, per_device_train_batch_size=batch,
            gradient_accumulation_steps=1, learning_rate=lr, lr_scheduler_type="cosine",
            warmup_steps=max(1, steps // 20), logging_steps=1, save_strategy="no",
            optim="paged_adamw_8bit", bf16=bf16, fp16=not bf16, report_to="none",
            gradient_checkpointing=True, dataloader_pin_memory=False,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=ds,
                          data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
                          callbacks=[Live()])
        _set("training", f"Training {trainable:,} of {total:,} params "
                         f"({100 * trainable / total:.3f}%) on {len(texts)} examples.")
        result = trainer.train()
        log.info("trainer result: %s", result.metrics)

        model.save_pretrained(run_dir)
        tok.save_pretrained(run_dir)
        size = sum(p.stat().st_size for p in run_dir.glob("adapter_model*")) / 1024**2
        log.info("adapter saved to %s (%.1f MB)", run_dir, size)
        model.eval()
        STATE.model, STATE.tokenizer, STATE.adapter_dir = model, tok, str(run_dir)
        losses = [h["loss"] for h in STATE.history]
        summary.update(status="stopped" if STATE.stop_requested else "done", steps_done=STATE.step,
                       first_loss=losses[0] if losses else None, final_loss=losses[-1] if losses else None,
                       min_loss=min(losses) if losses else None, adapter_mb=round(size, 2),
                       trainer_metrics=result.metrics)
        _set(summary["status"], f"Finished. Adapter saved to {run_dir}")
    except Exception as e:  # surface every failure in the UI and the logs
        log.exception("training run failed")
        hint = ""
        if "out of memory" in str(e).lower():
            hint = " → Try a smaller model, batch size 1, or a shorter max length."
        summary.update(status="error", error=f"{type(e).__name__}: {e}", traceback=format_exc())
        _set("error", f"{type(e).__name__}: {e}{hint}  (full details in the Logs tab)")
    finally:
        summary.update(finished=time.strftime(DATEFMT), runtime_s=round(time.time() - t0, 1),
                       peak_gpu=gpu_mem())
        write_json(run_dir / "summary.json", summary)
        append_jsonl(RUNS_INDEX, {k: v for k, v in summary.items() if k not in ("traceback", "trainer_metrics")})
        log.info("=== run %s finished: %s in %.1fs ===", run_dir.name, summary["status"], summary["runtime_s"])
        logging.getLogger().removeHandler(run_handler)
        run_handler.close()


# ----------------------------------------------------------------------------
# after training
# ----------------------------------------------------------------------------
def adapter_norms() -> dict[tuple[int, str], float]:
    """||B @ A|| * scaling for every adapted Linear, keyed by (layer index, name)."""
    import torch

    model = STATE.model
    if model is None:
        return {}
    out = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                m = re.search(r"\.(\d+)\.", name)
                if not m:
                    continue
                A = mod.lora_A["default"].weight.float()
                B = mod.lora_B["default"].weight.float()
                scale = mod.scaling["default"]
                out[(int(m.group(1)), name.split(".")[-1])] = float(torch.linalg.norm(B @ A) * scale)
    if out:
        top = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:5]
        log.info("adapter norms: %d modules | biggest changes: %s", len(out),
                 ", ".join(f"L{l}.{n}={v:.4f}" for (l, n), v in top))
        if STATE.run_dir:
            write_json(Path(STATE.run_dir) / "adapter_norms.json",
                       {f"layer{l}.{n}": v for (l, n), v in sorted(out.items())})
    return out


def generate(prompt: str, max_new_tokens: int = 160) -> tuple[str, str]:
    """(base model answer, fine-tuned answer) for the same prompt."""
    import torch

    model, tok = STATE.model, STATE.tokenizer
    if model is None:
        log.info("compare requested but no trained model in memory")
        return "Train a model first.", ""
    msgs = [{"role": "user", "content": prompt}]
    text = (tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if getattr(tok, "chat_template", None) else f"### User:\n{prompt}\n\n### Assistant:\n")
    ids = tok(text, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)

    def run(label):
        t = time.time()
        with torch.no_grad():
            out = model.generate(**ids, **kw)
        n_new = out.shape[1] - ids["input_ids"].shape[1]
        ans = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        dt = time.time() - t
        log.info("generate[%s]: %d tokens in %.2fs (%.1f tok/s)", label, n_new, dt, n_new / max(dt, 1e-6))
        return ans, round(dt, 2), int(n_new)

    model.config.use_cache = True
    with model.disable_adapter():
        base, base_t, base_n = run("base")
    tuned, tuned_t, tuned_n = run("tuned")
    log.info("compare | prompt: %r\n  base : %r\n  tuned: %r", prompt, base, tuned)
    append_jsonl(COMPARE_LOG, dict(adapter=STATE.adapter_dir, prompt=prompt, base=base, tuned=tuned,
                                   base_seconds=base_t, tuned_seconds=tuned_t,
                                   base_tokens=base_n, tuned_tokens=tuned_n))
    return base, tuned
