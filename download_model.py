"""
Download the open-source base model into ./models/base

Usage:
    python download_model.py

This is a one-time step. After it finishes, the model lives in your codebase
under models/base and everything else (training, inference) uses it from there.
"""
import config

if __name__ == "__main__":
    path = config.ensure_base_model()
    print(f"\nModel ready at: {path}")
    print(f"Model id: {config.BASE_MODEL_ID}")
