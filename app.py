"""
FastAPI web server: the thing you deploy to Railway.

Endpoints:
    GET  /              -> the browser UI (Chat + Train tabs)
    GET  /health        -> health check (used by Railway)
    POST /generate      -> {prompt, system?, max_new_tokens?} -> {response}
    POST /train         -> start a fine-tune in the background
    GET  /train/status  -> progress/log of the running (or last) training

Run locally:
    uvicorn app:app --reload
Then open http://localhost:8000
"""
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
import inference
import train as train_module

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="My LLM")

_train_state = {"running": False, "done": False, "error": None, "log": []}


class GenRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    max_new_tokens: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_model": config.BASE_MODEL_ID,
        "base_downloaded": config.has_base_model(),
        "fine_tuned": config.has_adapter(),
    }


@app.post("/generate")
def generate(req: GenRequest):
    text = inference.generate(
        req.prompt, system=req.system, max_new_tokens=req.max_new_tokens
    )
    return {"response": text}


def _run_training():
    _train_state.update(running=True, done=False, error=None, log=[])
    try:
        train_module.train(log_cb=lambda m: _train_state["log"].append(m))
        inference.reset()  # reload so /generate uses the new adapter
        _train_state["done"] = True
    except Exception as exc:  # noqa: BLE001 - surface any error to the UI
        _train_state["error"] = str(exc)
        _train_state["log"].append(f"ERROR: {exc}")
    finally:
        _train_state["running"] = False


@app.post("/train")
def start_training():
    """Kick off fine-tuning in a background thread.

    WARNING: training is heavy. This is meant for local / Colab use.
    Railway's CPU containers will be very slow and may hit memory limits.
    """
    if _train_state["running"]:
        return {"status": "already_running"}
    threading.Thread(target=_run_training, daemon=True).start()
    return {"status": "started"}


@app.get("/train/status")
def training_status():
    return _train_state
