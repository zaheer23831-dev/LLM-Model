# My LLM 🧠

A complete starter kit to **download an open-source LLM, fine-tune it on your own
data, chat with it, and deploy it to Railway** — with a simple web UI.

- **Base model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache-2.0, small enough for a CPU)
- **Fine-tuning:** LoRA (trains a tiny adapter — works on modest hardware)
- **Serving:** FastAPI + a browser UI (Chat + Train tabs)
- **Deploy:** Docker → Railway

> ⚠️ **Reality check.** Railway and most laptops have **no GPU**.
> - *Inference* (chatting) on a CPU works fine for this small model — just a bit slow.
> - *Training* on a CPU works for tiny data but is **slow**. For anything real, run
>   `train.py` on a **free GPU** (Google Colab / Kaggle), then download the
>   `models/adapter` folder back into this project.

---

## 1. Setup (local)

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the base model into ./models/base  (one time, ~1 GB)
python download_model.py
```

## 2. Chat with the model

**Option A — web UI (recommended):**
```bash
uvicorn app:app --reload
```
Open http://localhost:8000 → **Chat** tab. (The very first reply is slow because
it loads the model into memory.)

**Option B — terminal:**
```bash
python inference.py
```

## 3. Fine-tune it on your own data

1. Edit **`data/train.jsonl`**. One JSON object per line:
   ```json
   {"instruction": "your question/task", "input": "optional extra context", "output": "the answer you want"}
   ```
   Quality matters more than quantity — even 50–500 good examples help.
2. Run training:
   ```bash
   python train.py
   ```
   …or use the **Train** tab in the web UI.
3. The result is saved to `models/adapter/` and is **automatically used** the next
   time you chat. To go back to the plain base model, delete that folder.

> Tip: to train on a free GPU, upload `train.py`, `config.py`, and `data/train.jsonl`
> to a Google Colab notebook (set runtime → GPU), run it, then download the
> generated `models/adapter` folder into this project.

## 4. Deploy to Railway

Railway builds the included `Dockerfile` (CPU-only). The model is **not** committed
to git — the app downloads it on first request.

1. Push this folder to a GitHub repo.
2. On [railway.app](https://railway.app): **New Project → Deploy from GitHub repo**.
3. Railway auto-detects the `Dockerfile` and `railway.json`. No extra config needed.
4. (Optional) Add environment variables from `.env.example` under the service's
   **Variables** tab.
5. Open the generated URL → you get the same web UI.

**Notes for Railway:**
- First request after deploy is slow (it downloads the model). The `/health`
  endpoint stays green meanwhile.
- The filesystem is ephemeral — the model re-downloads on every redeploy. To avoid
  that (and to keep a trained adapter), attach a **Volume** mounted at `/app/models`.
- Don't try to *train* on Railway's CPU — it'll be painfully slow and may hit RAM
  limits. Train locally/Colab and deploy the adapter via a Volume or by baking it
  into the image.
- Watch usage: a service kept "always on" with the model loaded consumes
  RAM-hours continuously.

---

## File guide

| File | What it does |
|------|--------------|
| `config.py` | All settings in one place (model id, paths, hyperparameters) |
| `download_model.py` | Downloads the base model into `models/base` |
| `train.py` | LoRA fine-tuning (CLI + importable) |
| `inference.py` | Loads base + adapter and generates text (CLI + importable) |
| `app.py` | FastAPI server: web UI + `/generate` + `/train` endpoints |
| `web/index.html` | Browser UI (Chat + Train tabs) |
| `data/train.jsonl` | **Your training examples — edit this** |
| `Dockerfile` / `railway.json` | Deployment config for Railway |

## API endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | status + whether fine-tuned |
| POST | `/generate` | `{"prompt": "...", "system": "...", "max_new_tokens": 256}` | `{"response": "..."}` |
| POST | `/train` | — | starts background fine-tune |
| GET | `/train/status` | — | training progress/log |

Example:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Who are you?\"}"
```
