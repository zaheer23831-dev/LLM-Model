"""
Database layer (SQLAlchemy).

- On Railway: uses Postgres automatically via the DATABASE_URL env var.
- Locally: falls back to a SQLite file (llm_app.db) — zero setup.

Tables:
  users              - accounts (username + hashed password)
  sessions           - login tokens
  chat_logs          - every prompt + the model's reply
  training_examples  - your fine-tuning data, manageable from the app
"""
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, String, Text, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return "sqlite:///llm_app.db"  # local default — a single file
    # Some providers hand out the old "postgres://" scheme; SQLAlchemy wants "postgresql://"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


DB_URL = _db_url()
# SQLite + a multi-threaded web server needs this flag
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SessionToken(Base):
    __tablename__ = "sessions"
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatLog(Base):
    __tablename__ = "chat_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    system: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    model: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TrainingExample(Base):
    __tablename__ = "training_examples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    instruction: Mapped[str] = mapped_column(Text)
    input: Mapped[str] = mapped_column(Text, default="")
    output: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def init_db():
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(engine)


def get_training_examples():
    """Return all training examples as plain dicts (used by train.py)."""
    with SessionLocal() as db:
        rows = db.query(TrainingExample).order_by(TrainingExample.id).all()
        return [
            {"instruction": r.instruction, "input": r.input or "", "output": r.output}
            for r in rows
        ]
