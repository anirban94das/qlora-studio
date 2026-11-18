# CLAUDE.md

Guidance for Claude Code when working in `qlora-studio/`.

## What this is

A beginner-friendly Gradio app with five tabs. It shows the **real architecture of any Hugging Face causal LM** (Explore), lets you plan a **QLoRA** run (Plan), trains it (Train), compares base vs fine-tuned output (Compare), and shows every log file live (Logs).

It carries over the core idea of `../visualize_NNs_VC_`: draw the *actual* network. The UI layout and launch pattern come from `../Claude_Cowork/gemma-web-agent` (`app.py`, `config.py` env-override style, `setup.bat`).

## Commands

```bat
setup.bat                      :: one-time: creates .venv, installs CUDA torch (cu128) + requirements.txt
run.bat                        :: launches app.py -> http://localhost:7861 (opens browser)
.venv\Scripts\python.exe app.py
```

There is no test suite. To verify a change, launch the app and drive it. Headless Playwright works: wait for `text=Total parameters` after load, and for `text=✅ Done` after training. For a fast end-to-end GPU check, call `trainer.start(...)` directly with `HuggingFaceTB/SmolLM2-360M-Instruct` (~40 steps, ~40 s, <1 GB VRAM).

Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence the Windows HF cache warning.

## Architecture

| File | Responsibility |
|---|---|
| `config.py` | Model dropdown list, `DEFAULT_MODEL`, `PORT` (7861), `OUTPUT_DIR`, `HF_TOKEN`. All values can be overridden with env vars. |
| `inspect_model.py` | `inspect(repo)` downloads **only `config.json`** and builds the model under `accelerate.init_empty_weights()` (meta device). It returns `ModelInfo`: param counts, the leaf modules of block 0 (`Leaf`) and the module tree. The result is cached with `lru_cache`. |
| `diagram.py` | Builds SVG as plain strings, with no plotting library. `model_stack()` draws the whole model, `block_detail()` draws one decoder block with LoRA badges, and `delta_heatmap()` draws the after-training heat-map. Text uses `currentColor` and fills are translucent, so light and dark themes both work. |
| `lora_math.py` | `plan()` returns the trainable-param count (exact) and a memory estimate (heuristic). `peft_snippet()` generates the code shown in the Plan tab. |
| `trainer.py` | One global `STATE: RunState`. `start()` launches `_run()` on a daemon thread: 4-bit NF4 load, `prepare_model_for_kbit_training`, `get_peft_model`, then HF `Trainer` (paged_adamw_8bit, cosine schedule, grad checkpointing). A `TrainerCallback` pushes loss into `STATE.history` and honours `STATE.stop_requested`. The adapter is saved to `outputs/<repo>-r<r>-<timestamp>/`. The trained model stays in `STATE.model` for `generate()` and `adapter_norms()`. |
| `studio_log.py` | `setup()` runs first thing in `app.py`. It tees stdout/stderr into `logs/console.log` (tqdm carriage-return progress bars collapsed), sets a root `RotatingFileHandler` on `logs/qlora-studio.log` at DEBUG plus console output at INFO, captures warnings, routes transformers/datasets loggers into root, and installs `sys.excepthook` and `threading.excepthook`. Also provides `@logged` (logs a UI handler's args, duration and exceptions; generator-aware), `gpu_mem()`, `append_jsonl`/`write_json` (for `runs.jsonl` and `compare.jsonl`), and `list_log_files`/`tail` for the Logs tab. |
| `app.py` | Gradio Blocks UI. `run_training` is a **generator** that polls `STATE` once a second and yields the status markdown and a loss DataFrame. |
| `data/pirate_sample.jsonl` | 30 instruction/output pairs written in pirate-speak. They make it obvious whether training worked. |

### Logging

Every module uses `logging.getLogger("qlora.<area>")`. Never use `print`. `trainer._run` attaches a temporary `FileHandler` for `<run_dir>/run.log`, writes `run_config.json` up front, appends to `metrics.jsonl` on every Trainer `on_log`, and always writes `summary.json` and a `logs/runs.jsonl` line in `finally`, even on errors. State changes go through `trainer._set()` so the log mirrors what the UI shows. Never log the `HF_TOKEN` value. The logging module is called `studio_log.py`, not `logs.py`, so it doesn't clash with the `logs/` folder. Env vars: `QLORA_LOG_DIR`, `QLORA_LOG_LEVEL`.

### Non-obvious details

- **Finding the decoder stack generically:** `_find_layer_list` picks the largest `nn.ModuleList` whose children all share one class. Embedding and LM head come from `get_input_embeddings()` and `get_output_embeddings()`.
- **Tied embeddings:** `from_config` doesn't always tie the LM head to the embedding, so `inspect()` calls `model.tie_weights()`. Without it, Qwen2.5-1.5B reports 1.78B params instead of 1.54B.
- **Block flow order is inferred from module names** (`_group_rank` in `diagram.py`). PyTorch registration order puts the norms last, which doesn't match the real data flow. Extend `_group_rank` if a new architecture's norms end up in the wrong place.
- **LoRA targets come from the model itself:** they are the unique short names of the `Linear` leaves in block 0 (`ModelInfo.linear_names`). Phi-3.5, for example, gives `qkv_proj` and `gate_up_proj`. The default target set is *all* of them. With only `q_proj`/`v_proj` and few steps, the pirate effect barely shows.
- **Dataset loading** (`trainer.load_rows`) accepts any of these columns: `text`, `messages`, ShareGPT `conversations`, or instruction/prompt/question + output/response/answer. Rows are rendered with the tokenizer's chat template when it has one.
- **Base vs tuned:** both answers come from the same model. The base answer is generated inside `with model.disable_adapter():`.
- **Concurrency:** only one training run at a time, guarded in `start()`. Gradio's queue runs `run_training` as a long-lived generator.
- **Loss chart:** the DataFrame must be numeric (`.astype(float)`). An object-dtype column makes `gr.LinePlot` treat loss as categorical, which scrambles the y-axis.

## Verified behaviour (RTX 3060 Laptop, 6 GB, torch 2.11+cu128, transformers 5.23, bitsandbytes 0.50, Gradio 6)

- Explore works for Qwen2.5, Qwen3, SmolLM2, Mistral-7B and Phi-3.5 (configs only, ~1 s each).
- Qwen2.5-1.5B-Instruct QLoRA run (all linears, r=16, batch 2, 256 tokens, 60 steps): ~50 s, loss 4 → ~0.1, the pirate style is clearly learned.
- The memory estimate in `lora_math.plan` is deliberately conservative. SmolLM2 peaked at 0.6 GB actual against a 1.1 GB estimate.

## Gotchas

- QLoRA training needs an NVIDIA GPU (bitsandbytes 4-bit). The Explore and Plan tabs work on CPU.
- Gated models (Llama, Gemma) need `HF_TOKEN` and an accepted licence on huggingface.co.
- Local GGUF files (LM Studio / Ollama) **cannot** be used. This app needs HF safetensors repos.
- `outputs/` and `.venv/` are gitignored.
