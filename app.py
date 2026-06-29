"""
FastAPI web server: the thing you deploy to Railway.

Now backed by a database (Postgres on Railway, SQLite locally):
  - logs every chat to chat_logs
  - stores training examples you can manage in the UI
  - simple user accounts (register / login)

Endpoints:
    GET  /                 -> the browser UI
    GET  /health           -> health check (used by Railway)
    POST /auth/register    -> {username, password}
    POST /auth/login       -> {username, password} -> {token}
    POST /auth/logout      -> (Bearer token)
    GET  /me               -> who am I (Bearer token)
    POST /generate         -> {prompt, system?, max_new_tokens?} -> {response}  (logged)
    GET  /history          -> recent chats (yours if logged in)
    GET  /examples         -> list training examples
    POST /examples         -> add a training example
    DELETE /examples/{id}  -> remove a training example
    POST /train?from_db=.. -> start a fine-tune (from DB or the jsonl file)
    GET  /train/status     -> training progress/log

Run locally:
    uvicorn app:app --reload
"""
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
import inference
import database
import auth
import train as train_module

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="My LLM")

_train_state = {"running": False, "done": False, "error": None, "log": []}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class GenRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    max_new_tokens: Optional[int] = None


class AuthRequest(BaseModel):
    username: str
    password: str


class ExampleRequest(BaseModel):
    instruction: str
    input: str = ""
    output: str


# ---------------------------------------------------------------------------
# Auth helper: resolve the optional "Authorization: Bearer <token>" header
# ---------------------------------------------------------------------------
def current_user(authorization: Optional[str] = Header(default=None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    return auth.user_for_token(token)


@app.on_event("startup")
def _startup():
    database.init_db()


# ---------------------------------------------------------------------------
# Pages / health
# ---------------------------------------------------------------------------
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
        "database": "postgres" if not database.DB_URL.startswith("sqlite") else "sqlite",
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/auth/register")
def register(req: AuthRequest):
    if not req.username.strip() or not req.password:
        raise HTTPException(400, "username and password required")
    uid = auth.create_user(req.username, req.password)
    if uid is None:
        raise HTTPException(409, "username already taken")
    # auto-login after register
    return auth.login(req.username, req.password)


@app.post("/auth/login")
def do_login(req: AuthRequest):
    result = auth.login(req.username, req.password)
    if not result:
        raise HTTPException(401, "invalid username or password")
    return result


@app.post("/auth/logout")
def do_logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        auth.logout(authorization.split(" ", 1)[1].strip())
    return {"ok": True}


@app.get("/me")
def me(user=Depends(current_user)):
    return {"user": user}


# ---------------------------------------------------------------------------
# Chat (logged to the database)
# ---------------------------------------------------------------------------
@app.post("/generate")
def generate(req: GenRequest, user=Depends(current_user)):
    text = inference.generate(
        req.prompt, system=req.system, max_new_tokens=req.max_new_tokens
    )
    with database.SessionLocal() as db:
        db.add(
            database.ChatLog(
                user_id=user["id"] if user else None,
                system=req.system,
                prompt=req.prompt,
                response=text,
                model=config.BASE_MODEL_ID,
            )
        )
        db.commit()
    return {"response": text}


@app.get("/history")
def history(user=Depends(current_user), limit: int = 50):
    with database.SessionLocal() as db:
        q = db.query(database.ChatLog).order_by(database.ChatLog.id.desc())
        if user:
            q = q.filter(database.ChatLog.user_id == user["id"])
        rows = q.limit(min(limit, 200)).all()
        return [
            {
                "id": r.id,
                "prompt": r.prompt,
                "response": r.response,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Training examples (your fine-tuning data, stored in the DB)
# ---------------------------------------------------------------------------
@app.get("/examples")
def list_examples():
    with database.SessionLocal() as db:
        rows = db.query(database.TrainingExample).order_by(
            database.TrainingExample.id.desc()
        ).all()
        return [
            {"id": r.id, "instruction": r.instruction, "input": r.input, "output": r.output}
            for r in rows
        ]


@app.post("/examples")
def add_example(ex: ExampleRequest, user=Depends(current_user)):
    if not ex.instruction.strip() or not ex.output.strip():
        raise HTTPException(400, "instruction and output are required")
    with database.SessionLocal() as db:
        row = database.TrainingExample(
            user_id=user["id"] if user else None,
            instruction=ex.instruction,
            input=ex.input or "",
            output=ex.output,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return {"id": row.id}


@app.delete("/examples/{ex_id}")
def delete_example(ex_id: int):
    with database.SessionLocal() as db:
        row = db.get(database.TrainingExample, ex_id)
        if row:
            db.delete(row)
            db.commit()
    return {"deleted": ex_id}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_training(from_db: bool):
    _train_state.update(running=True, done=False, error=None, log=[])
    try:
        train_module.train(
            log_cb=lambda m: _train_state["log"].append(m), from_db=from_db
        )
        inference.reset()  # reload so /generate uses the new adapter
        _train_state["done"] = True
    except Exception as exc:  # noqa: BLE001 - surface any error to the UI
        _train_state["error"] = str(exc)
        _train_state["log"].append(f"ERROR: {exc}")
    finally:
        _train_state["running"] = False


@app.post("/train")
def start_training(from_db: bool = False):
    """Kick off fine-tuning in a background thread.

    from_db=True trains on the examples stored in the database;
    otherwise it trains on data/train.jsonl.

    WARNING: training is heavy — meant for local / Colab use, not Railway CPU.
    """
    if _train_state["running"]:
        return {"status": "already_running"}
    threading.Thread(target=_run_training, args=(from_db,), daemon=True).start()
    return {"status": "started", "from_db": from_db}


@app.get("/train/status")
def training_status():
    return _train_state
