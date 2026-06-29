"""
Central configuration for the whole project.

Everything (download, training, inference, the web app) reads from here,
so you only change settings in ONE place. You can also override any value
with an environment variable (handy on Railway).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "base"      # the downloaded open-source model
ADAPTER_DIR = MODELS_DIR / "adapter"      # your fine-tuned LoRA weights
DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"     # your training examples

# ---------------------------------------------------------------------------
# Base model (open source, Apache-2.0, small enough to run on a CPU)
# Want something different? Set BASE_MODEL_ID env var, e.g.
#   Qwen/Qwen2.5-1.5B-Instruct  (bigger, smarter, slower)
#   meta-llama/Llama-3.2-1B-Instruct  (needs HF access approval)
# ---------------------------------------------------------------------------
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))


# ---------------------------------------------------------------------------
# Training hyper-parameters (kept small so it can run on a CPU laptop)
# ---------------------------------------------------------------------------
class TrainCfg:
    epochs = float(os.environ.get("EPOCHS", "3"))
    learning_rate = float(os.environ.get("LR", "2e-4"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "512"))
    # LoRA settings (the small "adapter" we actually train)
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def has_base_model() -> bool:
    """True if the base model has already been downloaded locally."""
    return (BASE_MODEL_DIR / "config.json").exists()


def has_adapter() -> bool:
    """True if you've trained an adapter (fine-tune) at least once."""
    return (ADAPTER_DIR / "adapter_config.json").exists()


def model_source() -> str:
    """Where to load the base model from: local folder if present, else HF hub."""
    return str(BASE_MODEL_DIR) if has_base_model() else BASE_MODEL_ID


def ensure_base_model() -> str:
    """Download the base model into models/base if it isn't there yet."""
    if has_base_model():
        return str(BASE_MODEL_DIR)
    from huggingface_hub import snapshot_download

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[config] Downloading {BASE_MODEL_ID} -> {BASE_MODEL_DIR} (first time only)...")
    snapshot_download(
        repo_id=BASE_MODEL_ID,
        local_dir=str(BASE_MODEL_DIR),
        # skip formats we don't use to save space/time
        ignore_patterns=["*.gguf", "*.pth", "original/*", "*.onnx"],
    )
    print("[config] Download complete.")
    return str(BASE_MODEL_DIR)
