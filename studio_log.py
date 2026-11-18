"""
Logging for QLoRA Studio: everything that happens ends up in logs/.

    logs/
      qlora-studio.log   main log (DEBUG): UI actions, model inspection, training,
                         library messages (transformers, peft, datasets, ...),
                         warnings and every exception with full traceback.
                         Rotates at 10 MB, keeps 5 old files.
      console.log        raw copy of everything printed to the console window
                         (progress bars collapsed to their final state).
      runs.jsonl         one line per training run: settings, result, timings.
      compare.jsonl      every base-vs-tuned comparison: prompt + both answers.

    outputs/<run>/       per training run, next to the saved adapter:
      run.log            this run's log lines only
      run_config.json    every setting used
      metrics.jsonl      every step: loss, lr, grad norm, GPU memory, time
      summary.json       final status, runtime, losses, peak memory, error

Call setup() once, as early as possible (app.py does it first).
HF_TOKEN's value is never written anywhere -- only whether it is set.
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

from config import LOG_DIR

MAIN_LOG = LOG_DIR / "qlora-studio.log"
CONSOLE_LOG = LOG_DIR / "console.log"
RUNS_INDEX = LOG_DIR / "runs.jsonl"
COMPARE_LOG = LOG_DIR / "compare.jsonl"

FMT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(threadName)-12s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger("qlora")
_done = False


class _Tee:
    """Write to the real console AND console.log. Carriage-return progress bars
    (tqdm) are collapsed so the file holds only each bar's last state per line."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.buf = ""
        self.lock = threading.Lock()

    def write(self, text):
        try:
            self.stream.write(text)
        except Exception:
            pass
        with self.lock:
            self.buf += text
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                line = line.rsplit("\r", 1)[-1]
                if line.strip():
                    self.file.write(f"{time.strftime(DATEFMT)} | {line}\n")
            if len(self.buf) > 20000:      # endless \r bar without newline
                self.buf = self.buf.rsplit("\r", 1)[-1]
        return len(text)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def setup(level: str | None = None) -> None:
    global _done
    if _done:
        return
    _done = True
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. console mirror (before anything prints)
    if sys.stdout is not None:
        sys.stdout = _Tee(sys.stdout, CONSOLE_LOG)
    if sys.stderr is not None:
        sys.stderr = _Tee(sys.stderr, CONSOLE_LOG)

    # 2. main log file (DEBUG) + console (INFO)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.handlers.RotatingFileHandler(MAIN_LOG, maxBytes=10 * 1024**2, backupCount=5,
                                              encoding="utf-8")
    fh.setLevel(getattr(logging, (level or os.environ.get("QLORA_LOG_LEVEL", "DEBUG")).upper(),
                        logging.DEBUG))
    fh.setFormatter(logging.Formatter(FMT, DATEFMT))
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(fh)
    root.addHandler(ch)

    # Very chatty libraries: keep their INFO, drop their DEBUG noise.
    for name in ("urllib3", "httpx", "httpcore", "asyncio", "PIL", "matplotlib", "filelock",
                 "fsspec", "multipart", "python_multipart", "uvicorn.access", "websockets"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 3. Python warnings -> log
    logging.captureWarnings(True)

    # 4. Hugging Face libraries use their own loggers; route them to ours.
    try:
        import transformers
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_propagation()
    except Exception:
        log.debug("transformers logging hookup skipped", exc_info=True)
    try:
        import datasets
        datasets.utils.logging.set_verbosity_info()
        # API differs across datasets versions: strip its own handlers, then propagate to ours.
        dl = logging.getLogger("datasets")
        for h in list(dl.handlers):
            dl.removeHandler(h)
        dl.propagate = True
    except Exception:
        log.debug("datasets logging hookup skipped", exc_info=True)

    # 5. Uncaught exceptions anywhere (main thread or background threads)
    def _hook(exc_type, exc, tb):
        logging.getLogger("qlora.crash").critical("Uncaught exception",
                                                  exc_info=(exc_type, exc, tb))
    sys.excepthook = _hook
    threading.excepthook = lambda a: logging.getLogger("qlora.crash").critical(
        f"Uncaught exception in thread {a.thread.name if a.thread else '?'}",
        exc_info=(a.exc_type, a.exc_value, a.exc_traceback))

    log.info("=" * 70)
    log.info("QLoRA Studio starting — logs in %s", LOG_DIR)
    log_system_info()


def log_system_info() -> None:
    info = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "cwd": os.getcwd(), "pid": os.getpid(),
        "HF_TOKEN set": bool(os.environ.get("HF_TOKEN")),
    }
    for env in ("QLORA_PORT", "QLORA_OUTPUT_DIR", "QLORA_LOG_DIR", "QLORA_LOG_LEVEL",
                "HF_HOME", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(env):
            info[env] = os.environ[env]
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets",
                "gradio", "pandas"):
        try:
            from importlib.metadata import version
            info[pkg] = version(pkg)
        except Exception:
            info[pkg] = "not installed"
    try:
        import torch
        info["cuda available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = f"{p.name}, {p.total_memory / 1024**3:.1f} GB, cc {p.major}.{p.minor}"
            info["torch cuda"] = torch.version.cuda
            info["bf16"] = torch.cuda.is_bf16_supported()
    except Exception as e:
        info["torch error"] = repr(e)
    for k, v in info.items():
        log.info("system | %-16s %s", k, v)


def gpu_mem() -> dict:
    """Current / peak GPU memory in GB (empty dict without CUDA)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"gpu_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "gpu_used_total_gb": round((total - free) / 1024**3, 3)}
    except Exception:
        return {}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime(DATEFMT), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _short(v, n=200):
    s = repr(v)
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n} chars)"


def logged(fn):
    """Decorator for UI handlers: log call, arguments, duration, and any error.
    Works for plain functions and generators (e.g. the training stream)."""
    lg = logging.getLogger(f"qlora.ui.{fn.__name__}")

    def _start(args, kwargs):
        lg.info("called  args=%s%s", ", ".join(_short(a, 120) for a in args),
                f" kwargs={_short(kwargs)}" if kwargs else "")
        return time.perf_counter()

    import inspect as _inspect
    if _inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def gen(*args, **kwargs):
            t = _start(args, kwargs)
            n = 0
            try:
                for item in fn(*args, **kwargs):
                    n += 1
                    yield item
                lg.info("finished after %.2fs (%d updates)", time.perf_counter() - t, n)
            except Exception:
                lg.exception("FAILED after %.2fs", time.perf_counter() - t)
                raise
        return gen

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t = _start(args, kwargs)
        try:
            out = fn(*args, **kwargs)
            lg.info("done in %.2fs", time.perf_counter() - t)
            return out
        except Exception:
            lg.exception("FAILED after %.2fs", time.perf_counter() - t)
            raise
    return wrapper


def list_log_files() -> list[str]:
    """Every log file, newest first: main logs plus each run's files."""
    from config import OUTPUT_DIR
    files = [p for p in LOG_DIR.glob("*") if p.is_file()]
    files += [p for p in OUTPUT_DIR.glob("*/*") if p.suffix in (".log", ".jsonl", ".json")
              and p.name not in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json",
                                 "special_tokens_map.json", "added_tokens.json", "vocab.json",
                                 "generation_config.json", "chat_template.json")]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def tail(path: str, lines: int = 300, newest_first: bool = False) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()[-int(lines):]
        if newest_first:
            # reverse whole entries, keeping multi-line ones (tracebacks, configs) intact
            import re
            entries: list[list[str]] = []
            for line in data:
                if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", line) or not entries:
                    entries.append([line])
                else:
                    entries[-1].append(line)
            data = [l for e in reversed(entries) for l in e]
        return "".join(data) or "(empty)"
    except FileNotFoundError:
        return "(file not found yet)"
    except Exception as e:
        return f"(could not read: {e})"


def format_exc() -> str:
    return traceback.format_exc()
