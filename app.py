"""
QLoRA Studio — see inside real open-source LLMs and fine-tune them with QLoRA.

Run:  python app.py      (or double-click run.bat)
Then the browser opens at http://localhost:7861

Tabs:
  1. Explore  — the genuine architecture of any Hugging Face model (no weights downloaded)
  2. Plan     — pick LoRA settings, see exactly where adapters go + memory estimate
  3. Train    — real QLoRA training with a live loss curve
  4. Compare  — chat with base vs fine-tuned, and see which layers changed most

UI layout/launch pattern borrowed from Claude_Cowork/gemma-web-agent/app.py.
"""

from __future__ import annotations

# Logging first, so every later import (torch, transformers, gradio) is captured.
import studio_log
studio_log.setup()

import logging
import time

import gradio as gr
import pandas as pd

import diagram
import lora_math
import trainer
from config import DEFAULT_MODEL, MODEL_CHOICES, PORT
from inspect_model import gpu_info, human, inspect
from studio_log import LOG_DIR, MAIN_LOG, list_log_files, logged, tail

log = logging.getLogger("qlora.app")
NL = chr(10)
GPU_NAME, GPU_GB = gpu_info()

INTRO = f"""
# 🔬 QLoRA Studio
**See inside real open-source language models — then teach one a new trick on your own GPU.**

Your GPU: **{GPU_NAME}**{f" · {GPU_GB:.1f} GB" if GPU_GB else ""}  ·  Start at tab ①, then go left to right.
"""

EXPLAIN_LORA = """
<details><summary><b>What are LoRA and QLoRA? (30-second version)</b></summary>

* A model is mostly big grids of numbers (**weight matrices**) inside `Linear` layers — the boxes in the diagram.
* **Full fine-tuning** changes all of them: billions of numbers, needs a huge GPU.
* **LoRA** freezes them and adds a tiny *add-on* to chosen layers: two thin matrices **A** and **B**.
  The layer's new behaviour is `W + B·A`. Only A and B are trained — often **< 1 %** of the model.
  * **r (rank)** = how thin A and B are. Bigger r → more capacity, more memory.
  * **alpha** = how loudly the add-on speaks (its output is scaled by alpha / r).
* **QLoRA** also squeezes the frozen weights into **4-bit** numbers, so a 1.5B model fits in ~2 GB.
</details>
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(repo: str):
    try:
        return inspect(repo), None
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            msg = ("This model is **gated**. Accept its licence on huggingface.co, create a token at "
                   "https://huggingface.co/settings/tokens, set `HF_TOKEN`, and restart.")
        logging.getLogger("qlora.inspect").exception("could not inspect %r", repo)
        return None, f"⚠️ Could not load `{repo}`: {msg}"


@logged
def explore(repo: str):
    info, err = _load(repo)
    if err:
        return err, "", "", "", gr.update(choices=[], value=[])
    gqa = ""
    if info.heads and info.kv_heads and info.kv_heads != info.heads:
        gqa = f" ({info.kv_heads} key/value heads — *grouped-query attention*)"
    summary = f"""
### {info.repo}
| | |
|---|---|
| **Total parameters** | **{human(info.total_params)}** ({info.total_params:,}) |
| Architecture | `{info.arch_class}` (`{info.model_type}`) |
| Decoder blocks | {info.num_layers} identical blocks stacked on top of each other |
| Hidden size | {info.hidden} numbers describe each token at every step |
| Attention heads | {info.heads}{gqa} |
| MLP width | {info.intermediate} |
| Vocabulary | {info.vocab:,} tokens |
| Weights in 16-bit / 4-bit | ~{info.total_params * 2 / 1024**3:.1f} GB / ~{(info.quantizable_params * 0.53 + (info.total_params - info.quantizable_params) * 2) / 1024**3:.1f} GB |

*Built from the model's `config.json` only — no weights were downloaded. Every box is the real module.*
"""
    default_targets = info.linear_names  # all Linear layers: strongest effect, still < 1-2 % of params
    return (summary, diagram.model_stack(info), diagram.block_detail(info), info.tree_text,
            gr.update(choices=info.linear_names, value=default_targets))


@logged
def plan(repo, targets, r, alpha, dropout, batch, seq_len):
    info, err = _load(repo)
    if err:
        return err, "", "", ""
    targets = targets or []
    p = lora_math.plan(info, targets, int(r), int(batch), int(seq_len))
    fits = ""
    if GPU_GB:
        verdict = ("✅ should fit" if p.mem_total < GPU_GB * 0.9 else
                   "⚠️ tight — may run out of memory" if p.mem_total < GPU_GB * 1.05 else
                   "❌ probably won't fit — try a smaller model, batch 1, or shorter length")
        fits = f"on your {GPU_GB:.1f} GB GPU: **{verdict}**"
    bar_w = lambda gb: max(2, int(300 * gb / max(p.mem_total, GPU_GB or p.mem_total)))
    bar = lambda label, gb, color: (
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<div style="width:150px;font-size:12px">{label}</div>'
        f'<div style="width:{bar_w(gb)}px;height:14px;background:{color};border-radius:3px"></div>'
        f'<div style="font-size:12px">{gb:.2f} GB</div></div>')
    mem = (bar("Frozen model (4-bit)", p.mem_base, "#3b82f6") + bar("LoRA + optimizer", p.mem_lora, "#f97316")
           + bar("Activations", p.mem_activations, "#10b981") + bar("CUDA overhead", p.mem_overhead, "#94a3b8"))
    per = "\n".join(f"| `{k}` | {v:,} |" for k, v in p.per_module.items()) or "| — | pick some modules |"
    text = f"""
### Trainable: **{p.lora_params:,}** params = **{p.lora_pct:.3f}%** of the model
| adapted module (all {info.num_layers} layers) | trainable params |
|---|---|
{per}

**Estimated training memory: {p.mem_total:.1f} GB** {fits}
<br><sub>Rough estimate; real usage varies ±30 %.</sub>
"""
    code = lora_math.peft_snippet(info.repo, targets, int(r), int(alpha), float(dropout))
    return text, mem, diagram.block_detail(info, targets, int(r)), code


@logged
def run_training(repo, targets, r, alpha, dropout, lr, steps, batch, seq_len, dataset, n_rows):
    if not targets:
        yield "⚠️ Pick at least one module to adapt in tab ②.", None
        return
    msg = trainer.start(repo, list(targets), int(r), int(alpha), float(dropout), float(lr),
                        int(steps), int(batch), int(seq_len), dataset, int(n_rows))
    if msg != "Started.":
        yield msg, None
        return
    while True:
        s = trainer.STATE
        elapsed = time.time() - s.started
        pct = 100 * s.step / max(s.max_steps, 1)
        head = {"loading": "⏳ Loading", "training": "🏋️ Training", "done": "✅ Done",
                "stopped": "⏹️ Stopped", "error": "❌ Error"}.get(s.status, s.status)
        status = f"**{head}** — step {s.step}/{s.max_steps} ({pct:.0f}%) · {elapsed:.0f}s\n\n{s.message}"
        df = pd.DataFrame(s.history, columns=["step", "loss", "lr"]).astype(float)
        yield status, df
        if s.status in ("done", "stopped", "error"):
            log.info("training stream ended: %s | %s", s.status, s.message)
            return
        time.sleep(1.0)


@logged
def compare(prompt):
    base, tuned = trainer.generate(prompt)
    return base, tuned


@logged
def show_changes():
    norms = trainer.adapter_norms()
    if not norms:
        return "<p>Train a model in tab ③ first.</p>"
    return diagram.delta_heatmap(norms)


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
with gr.Blocks(title="QLoRA Studio") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        repo = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, allow_custom_value=True, scale=4,
                           label="Model (pick one, or paste any Hugging Face repo id)")
        load_btn = gr.Button("Look inside 🔍", variant="primary", scale=1)

    with gr.Tabs():
        # ---------------- 1 ----------------
        with gr.Tab("① Explore the network"):
            summary = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### The whole model, top to bottom")
                    stack_svg = gr.HTML()
                with gr.Column(scale=3):
                    gr.Markdown("#### Zoom into one block (hover boxes for shapes)")
                    block_svg = gr.HTML()
            with gr.Accordion("Raw PyTorch module tree", open=False):
                tree = gr.Code(language=None, lines=24)

        # ---------------- 2 ----------------
        with gr.Tab("② Plan QLoRA"):
            gr.Markdown(EXPLAIN_LORA)
            with gr.Row():
                with gr.Column(scale=1):
                    targets = gr.CheckboxGroup(label="Which layers get a LoRA add-on?",
                                               info="all = strongest (default) · only q_proj + v_proj = classic & cheapest")
                    r = gr.Slider(1, 128, value=16, step=1, label="r (rank)")
                    alpha = gr.Slider(1, 256, value=32, step=1, label="alpha (usually 2 × r)")
                    dropout = gr.Slider(0, 0.3, value=0.05, step=0.01, label="dropout")
                    batch = gr.Slider(1, 8, value=2, step=1, label="batch size")
                    seq_len = gr.Slider(64, 2048, value=256, step=64, label="max text length (tokens)")
                with gr.Column(scale=2):
                    plan_text = gr.Markdown()
                    mem_html = gr.HTML()
            plan_svg = gr.HTML()
            with gr.Accordion("Equivalent Python code", open=False):
                code = gr.Code(language="python")

        # ---------------- 3 ----------------
        with gr.Tab("③ Train"):
            gr.Markdown("Uses the settings from tab ②. The built-in **sample** dataset teaches the model to "
                        "talk like a pirate 🏴‍☠️ — easy to see whether training worked. You can also give a "
                        "Hugging Face dataset id (e.g. `yahma/alpaca-cleaned`) or a path to a `.jsonl` file.")
            with gr.Row():
                dataset = gr.Textbox(value="sample", label="Dataset", scale=3)
                n_rows = gr.Number(value=200, precision=0, label="max examples", scale=1)
                steps = gr.Number(value=60, precision=0, label="training steps", scale=1)
                lr = gr.Number(value=3e-4, label="learning rate", scale=1)
            with gr.Row():
                train_btn = gr.Button("Start training 🚀", variant="primary")
                stop_btn = gr.Button("Stop")
            train_status = gr.Markdown("_Not started._")
            loss_plot = gr.LinePlot(x="step", y="loss", title="Training loss (lower = model fits your data better)",
                                    height=320)

        # ---------------- 4 ----------------
        with gr.Tab("④ Compare & inspect"):
            gr.Markdown("Same prompt, same model — once **without** the adapter and once **with** it.")
            with gr.Row():
                prompt = gr.Textbox(value="What is the tallest mountain in the world?", label="Prompt", scale=4)
                ask_btn = gr.Button("Ask both", variant="primary", scale=1)
            with gr.Row():
                base_out = gr.Textbox(label="Base model", lines=8)
                tuned_out = gr.Textbox(label="Fine-tuned (base + LoRA)", lines=8)
            gr.Markdown("#### Where did learning happen?")
            heat_btn = gr.Button("Show adapter heat-map")
            heat = gr.HTML()

        # ---------------- 5 ----------------
        with gr.Tab("⑤ Logs"):
            gr.Markdown(
                f"Everything the app does is written to log files. **Main folder:** `{LOG_DIR}`" + NL + NL +
                "* `qlora-studio.log` — everything: your clicks, model loading, every training step, "
                "library messages, warnings and full error tracebacks" + NL +
                "* `console.log` — copy of the console window · `runs.jsonl` — one line per training run · "
                "`compare.jsonl` — every base-vs-tuned question and answer" + NL +
                "* Each training run folder in `outputs/` also has `run.log`, `run_config.json`, "
                "`metrics.jsonl` (every step: loss, learning rate, GPU memory) and `summary.json`.")
            with gr.Row():
                log_file = gr.Dropdown(list_log_files(), value=str(MAIN_LOG), label="Log file",
                                       allow_custom_value=True, scale=4)
                log_lines = gr.Slider(50, 5000, value=400, step=50, label="last N lines", scale=2)
                newest = gr.Checkbox(value=True, label="newest first", scale=1)
                auto = gr.Checkbox(value=True, label="auto-refresh (3 s)", scale=1)
                refresh_logs = gr.Button("Refresh", scale=1)
            log_view = gr.Textbox(lines=28, max_lines=28, autoscroll=False, show_label=False)
            log_download = gr.File(label="Download this file")
            timer = gr.Timer(3.0)

    # ---------------- wiring ----------------
    plan_inputs = [repo, targets, r, alpha, dropout, batch, seq_len]
    plan_outputs = [plan_text, mem_html, plan_svg, code]
    for ev in (load_btn.click, demo.load):
        ev(explore, repo, [summary, stack_svg, block_svg, tree, targets]).then(plan, plan_inputs, plan_outputs)
    for comp in (targets, r, alpha, dropout, batch, seq_len):
        comp.change(plan, plan_inputs, plan_outputs)
    train_btn.click(run_training, [repo, targets, r, alpha, dropout, lr, steps, batch, seq_len, dataset, n_rows],
                    [train_status, loss_plot])
    stop_btn.click(trainer.stop, None, train_status)
    ask_btn.click(compare, prompt, [base_out, tuned_out])
    heat_btn.click(show_changes, None, heat)

    def _read_log(path, n, newest_first):
        files = list_log_files()
        path = path if path in files else str(MAIN_LOG)
        return gr.update(choices=files, value=path), tail(path, n, newest_first), path

    def _auto(path, n, newest_first, on):
        return tail(path, n, newest_first) if on and path else gr.update()

    log_outputs = [log_file, log_view, log_download]
    for ev in (refresh_logs.click, log_file.change, log_lines.release, newest.change, demo.load):
        ev(_read_log, [log_file, log_lines, newest], log_outputs, show_progress="hidden")
    timer.tick(_auto, [log_file, log_lines, newest, auto], log_view, show_progress="hidden")


if __name__ == "__main__":
    log.info("launching UI on http://127.0.0.1:%d", PORT)
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT, inbrowser=True,
                        theme=gr.themes.Soft(primary_hue="orange"))
