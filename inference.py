"""
Run the model: load the base model (plus your fine-tuned adapter if it exists)
and generate text.

Use it as a CLI for a quick chat:
    python inference.py

Or import generate() from the web app.
"""
import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import config

# Use all available CPU cores — meaningfully faster generation on CPU-only hosts.
try:
    torch.set_num_threads(os.cpu_count() or 1)
except Exception:
    pass

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


def _build_inputs(tokenizer, prompt, system):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")


def _gen_kwargs(tokenizer, inputs, max_new_tokens):
    return dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
        do_sample=True,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )


def generate(prompt, system=None, max_new_tokens=None):
    """Generate the full reply, then return it."""
    tokenizer, model = load()
    inputs = _build_inputs(tokenizer, prompt, system)
    with torch.no_grad():
        output = model.generate(**_gen_kwargs(tokenizer, inputs, max_new_tokens))
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_stream(prompt, system=None, max_new_tokens=None):
    """Yield the reply token-by-token as it's produced (for live streaming)."""
    tokenizer, model = load()
    inputs = _build_inputs(tokenizer, prompt, system)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = _gen_kwargs(tokenizer, inputs, max_new_tokens)
    kwargs["streamer"] = streamer
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
    thread.join()


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
