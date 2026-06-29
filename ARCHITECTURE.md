# 🏗️ Architecture — My LLM

A self-hosted, fine-tunable LLM chat application. You **download an open-source
model, fine-tune it on your own data, chat with it, and deploy it** — with user
accounts, chat history, and a managed database. Nothing is sent to a third-party
AI: the model runs on your own infrastructure.

- **Live app:** https://llm-model-production-5149.up.railway.app
- **Base model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache-2.0)
- **Fine-tuning:** LoRA adapters (PEFT)
- **Serving:** FastAPI + token streaming
- **Storage:** Postgres (Railway) / SQLite (local)
- **Deploy:** Docker → Railway

---

## 1. System overview

```mermaid
flowchart TB
    subgraph Client["🌐 Browser"]
        UI["Web UI (web/index.html)<br/>Chat · Data · History · Train tabs"]
    end

    subgraph Railway["☁️ Railway — your cloud"]
        subgraph App["App Service (Docker container)"]
            API["FastAPI — app.py<br/>routes + auth + logging"]
            AUTH["auth.py<br/>PBKDF2 + bearer tokens"]
            INF["inference.py<br/>load + generate + stream"]
            TRAIN["train.py<br/>LoRA fine-tuning"]
            MODEL["🧠 Qwen2.5-0.5B<br/>+ your LoRA adapter"]
        end
        DB[("🗄️ Postgres<br/>users · sessions · chats · examples")]
    end

    HF["🤗 Hugging Face Hub"]

    UI -->|HTTP + streamed tokens| API
    API --> AUTH
    API --> INF
    API <-->|read / write| DB
    API --> TRAIN
    INF --> MODEL
    TRAIN -->|writes adapter| MODEL
    INF -. downloads weights once .-> HF
```

**Key idea:** the browser talks only to *your* FastAPI app. The app runs the
model itself (CPU inference) and persists everything to your own database. The
only outbound call is a **one-time download** of the model weights from Hugging
Face when the container first boots.

---

## 2. Components

| File | Responsibility |
|------|----------------|
| [config.py](config.py) | Single source of truth: model id, paths, hyperparameters (env-overridable) |
| [download_model.py](download_model.py) | Fetches the base model into `models/base/` |
| [inference.py](inference.py) | Loads base + adapter; `generate()` and streaming `generate_stream()`; uses all CPU cores |
| [train.py](train.py) | LoRA fine-tuning — CPU/GPU aware; trains from `train.jsonl` **or** the database |
| [app.py](app.py) | FastAPI server — all HTTP endpoints, chat logging, background training |
| [auth.py](auth.py) | Password hashing (stdlib PBKDF2) + session-token auth |
| [database.py](database.py) | SQLAlchemy models + engine (Postgres on Railway, SQLite locally) |
| [web/index.html](web/index.html) | Single-page UI: Chat, Data, History, Train + login |
| [data/train.jsonl](data/train.jsonl) | Default training examples (replace with your own) |
| [Dockerfile](Dockerfile) / [railway.json](railway.json) | Deployment config |
| [train_on_colab.ipynb](train_on_colab.ipynb) | One-click free-GPU training notebook |

### Tech stack
- **Python 3.10**, **FastAPI** + **Uvicorn**
- **PyTorch** (CPU build), **Transformers**, **PEFT** (LoRA)
- **SQLAlchemy 2.0** + **Postgres / SQLite**
- **Docker** on **Railway**

---

## 3. Request flows

### 3a. Chat (streaming)

```mermaid
sequenceDiagram
    participant U as 🌐 Browser
    participant A as FastAPI (app.py)
    participant I as inference.py
    participant M as 🧠 Model
    participant D as 🗄️ Postgres

    U->>A: POST /generate/stream { prompt, system }
    A->>I: generate_stream(prompt)
    Note over I,M: first call only — download + load model
    I->>M: model.generate(..., streamer)
    loop for each token
        M-->>I: next token
        I-->>A: yield token
        A-->>U: stream token (types out live)
    end
    A->>D: INSERT ChatLog(prompt, full_reply, user_id)
    Note over U: reply appears word-by-word
```

### 3b. Authentication

```mermaid
sequenceDiagram
    participant U as 🌐 Browser
    participant A as FastAPI
    participant Au as auth.py
    participant D as 🗄️ Postgres

    U->>A: POST /auth/register { username, password }
    A->>Au: hash_password (PBKDF2)
    Au->>D: INSERT user
    A-->>U: { token }
    Note over U: token saved in localStorage

    U->>A: POST /generate/stream (Authorization: Bearer token)
    A->>Au: user_for_token(token)
    Au->>D: lookup session → user
    Au-->>A: user
    Note over A: chat is logged against this user
```

### 3c. Fine-tuning

```mermaid
flowchart LR
    subgraph Source["Training data"]
        F["data/train.jsonl"]
        DBX["DB examples<br/>(Data tab)"]
    end
    Source --> L["train.py<br/>load_rows()"]
    L --> P["Tokenize + mask prompt<br/>(loss only on the answer)"]
    P --> Q["Attach LoRA<br/>(train ~1.75% of params)"]
    Q --> R{"Hardware?"}
    R -->|GPU: Colab| S1["fp16 — minutes"]
    R -->|CPU: laptop/Railway| S2["fp32 — slow"]
    S1 --> W["💾 models/adapter/"]
    S2 --> W
    W --> INF["inference.py loads<br/>base + adapter"]
    INF --> OUT["✅ Customized replies"]
```

> **Where to train:** GPU (Colab/Kaggle) for real runs, CPU only for tiny tests.
> The result — a ~35 MB adapter — is what gets deployed, not the whole model.

---

## 4. Data model

```mermaid
erDiagram
    users ||--o{ sessions : "has"
    users ||--o{ chat_logs : "writes"
    users ||--o{ training_examples : "owns"

    users {
        int id PK
        string username UK
        string password_hash
        datetime created_at
    }
    sessions {
        string token PK
        int user_id FK
        datetime created_at
    }
    chat_logs {
        int id PK
        int user_id FK "nullable (anonymous ok)"
        text prompt
        text response
        string model
        datetime created_at
    }
    training_examples {
        int id PK
        int user_id FK "nullable"
        text instruction
        text input
        text output
        datetime created_at
    }
```

---

## 5. API reference

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/` | — | Web UI |
| GET | `/health` | — | Status + which DB is in use |
| POST | `/auth/register` | — | Create account (auto-login) |
| POST | `/auth/login` | — | Get a session token |
| POST | `/auth/logout` | Bearer | Invalidate token |
| GET | `/me` | Bearer | Current user |
| POST | `/generate` | optional | Full reply (non-streaming) |
| POST | `/generate/stream` | optional | **Streamed** reply (used by UI) |
| GET | `/history` | optional | Recent chats (yours if logged in) |
| GET | `/examples` | — | List training examples |
| POST | `/examples` | optional | Add a training example |
| DELETE | `/examples/{id}` | — | Remove an example |
| POST | `/train?from_db=` | — | Start a fine-tune (background) |
| GET | `/train/status` | — | Training progress/log |

---

## 6. Deployment topology

```mermaid
flowchart TB
    DEV["💻 Laptop<br/>code · local SQLite"]
    GH["🐙 GitHub<br/>zaheer23831-dev/LLM-Model"]
    COLAB["📓 Colab — free GPU<br/>train → adapter.zip"]

    subgraph RW["☁️ Railway Project: LLM-Model"]
        APP["App Service<br/>(Dockerfile, $PORT, DATABASE_URL)"]
        PG[("Postgres Service")]
        APP <-->|private network| PG
    end

    USERS["🌍 Users"]

    DEV -->|git push| GH
    DEV -->|railway up| APP
    GH -. open in Colab .-> COLAB
    COLAB -. download adapter .-> DEV
    APP -->|public HTTPS URL| USERS
```

### What lives where (and what persists)

| Item | Location | Survives redeploy? |
|------|----------|--------------------|
| App code | Docker image | ✅ (rebuilt from source) |
| Base model (~1 GB) | downloaded into container at runtime | ❌ re-downloads each boot* |
| Fine-tuned adapter (~35 MB) | baked into the Docker image | ✅ |
| Users / chats / examples | Postgres | ✅ |

\* *Add a Railway **Volume** mounted at `/app/models` to cache the base model and skip the re-download.*

---

## 7. The data flywheel

The database turns the app into a self-improving system:

```mermaid
flowchart LR
    A["👥 Users chat"] --> B["📝 Logged to Postgres"]
    B --> C["👀 Review good replies"]
    C --> D["➕ Add to training examples"]
    D --> E["🎓 Re-train (from_db)"]
    E --> F["✨ Better model"]
    F --> A
```

---

## 8. Running it

### Local
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python download_model.py      # one-time: fetch the base model
uvicorn app:app --reload      # http://localhost:8000  (uses local SQLite)
```

### Deploy to Railway
```bash
railway up --detach           # build + deploy the app
railway add --database postgres
# set on app service: DATABASE_URL=${{Postgres.DATABASE_URL}} and PORT=8000
```

### Train on a free GPU
Open [`train_on_colab.ipynb`](train_on_colab.ipynb) in Colab → set GPU runtime →
Run all → download `adapter.zip` → drop into `models/adapter/` → `git push` →
`railway up`.

---

## 9. Design decisions

- **LoRA over full fine-tuning** — trains ~1.75% of parameters; runs on modest
  hardware and produces a tiny, portable adapter.
- **Self-hosted model over an API** — full ownership and privacy; prompts never
  leave your infrastructure. Trade-off: a small CPU model is slower/less capable
  than a giant hosted model.
- **Adapter in the image, base model at runtime** — keeps the git repo and image
  small while still shipping your fine-tune.
- **DB abstraction (Postgres/SQLite)** — zero-setup local dev, managed DB in prod,
  same code path via `DATABASE_URL`.
- **Streaming responses** — CPU generation is slow, so stream tokens to keep the
  UI responsive while the answer is produced.

---

## 10. Roadmap / ideas

- Replace sample data with a **real `train.jsonl`** for your use case
- **GGUF quantization** (llama.cpp) for ~4–8× faster CPU inference
- **Railway Volume** to persist the base model
- Per-user conversation threads; RAG for knowledge-heavy use cases
