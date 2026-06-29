"""
Run the model: load the base model (plus your fine-tuned adapter if it exists)
and generate text.

Use it as a CLI for a quick chat:
    python inference.py

Or import generate() from the web app.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

_tokenizer = None
_model = None


def reset():
    """Forget the loaded model so it reloads next time (e.g. after training)."""
    global _tokenizer, _model
    _tokenizer = None
    _model = None


def load():
    """Lazily load the model into memory (only once)."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    config.ensure_base_model()
    src = config.model_source()
    print(f"[inference] Loading model from {src} ...")
    _tokenizer = AutoTokenizer.from_pretrained(src)
    _model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)

    # If you've fine-tuned, layer your adapter on top of the base model.
    if config.has_adapter():
        from peft import PeftModel
        print(f"[inference] Applying fine-tuned adapter from {config.ADAPTER_DIR}")
        _model = PeftModel.from_pretrained(_model, str(config.ADAPTER_DIR))

    _model.eval()
    print("[inference] Model ready.")
    return _tokenizer, _model


def generate(prompt, system=None, max_new_tokens=None):
    tokenizer, model = load()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("Chat with your model. Type 'quit' to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        print("Bot:", generate(user), "\n")
