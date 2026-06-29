"""
Fine-tune the base model on your own data using LoRA.

LoRA = we freeze the big model and train a tiny "adapter" on top. This is the
only realistic way to fine-tune on modest hardware. The result is saved to
models/adapter and is automatically picked up by inference.py / the web app.

Run it directly:
    python train.py

Or import and call train() from elsewhere (the web app does this).

NOTE: Training needs real compute. On a CPU laptop this works for the small
0.5B model + a small dataset, but it is SLOW. For anything serious, run this
same file on a free GPU (Google Colab / Kaggle).
"""
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

import config


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(from_db=False):
    """Get training rows either from the database or the jsonl file."""
    if from_db:
        import database
        rows = database.get_training_examples()
        print(f"Loaded {len(rows)} examples from the database")
        return rows
    return _read_jsonl(config.TRAIN_FILE)


def build_dataset(tokenizer, rows, max_len):
    """
    Turn each {instruction, input, output} row into tokenized training data.
    We mask the prompt tokens (-100) so the model only learns to produce the
    *answer*, not to repeat the question.
    """
    examples = []
    for r in rows:
        instruction = (r.get("instruction") or "").strip()
        extra_input = (r.get("input") or "").strip()
        output = (r.get("output") or "").strip()
        if not instruction or not output:
            continue

        user_msg = instruction if not extra_input else f"{instruction}\n\n{extra_input}"

        prompt_messages = [{"role": "user", "content": user_msg}]
        full_messages = prompt_messages + [{"role": "assistant", "content": output}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100  # ignore prompt tokens in the loss

        examples.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    if not examples:
        raise ValueError("No valid training examples found")
    return examples


class _Collator:
    """Pads a batch of variable-length examples to the same length."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention, labels = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad)
            attention.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "labels": torch.tensor(labels),
        }


class _LogCallback(TrainerCallback):
    def __init__(self, cb):
        self.cb = cb

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.cb and logs:
            self.cb(f"step {state.global_step}: {logs}")


def train(log_cb=None, from_db=False):
    """Run the fine-tune. log_cb (optional) receives progress strings.

    from_db=True trains on examples stored in the database; otherwise it uses
    data/train.jsonl. Can also be forced with the TRAIN_FROM_DB=1 env var.
    """
    import os
    if os.environ.get("TRAIN_FROM_DB", "").lower() in ("1", "true", "yes"):
        from_db = True

    def log(msg):
        print(msg)
        if log_cb:
            log_cb(msg)

    on_gpu = torch.cuda.is_available()
    log(f"Device: {torch.cuda.get_device_name(0) if on_gpu else 'CPU'} "
        f"({'GPU — fast' if on_gpu else 'CPU — slow; use Colab/Kaggle for real runs'})")

    config.ensure_base_model()
    src = config.model_source()

    log(f"Loading base model: {src}")
    tokenizer = AutoTokenizer.from_pretrained(src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    model.config.use_cache = False  # required when training

    log("Attaching LoRA adapter...")
    lora = LoraConfig(
        r=config.TrainCfg.lora_r,
        lora_alpha=config.TrainCfg.lora_alpha,
        lora_dropout=config.TrainCfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    log(f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)")

    source = "database" if from_db else str(config.TRAIN_FILE)
    log(f"Loading data from {source}")
    rows = load_rows(from_db=from_db)
    dataset = build_dataset(tokenizer, rows, config.TrainCfg.max_seq_len)
    log(f"Loaded {len(dataset)} examples")

    args = TrainingArguments(
        output_dir=str(config.ADAPTER_DIR / "_checkpoints"),
        num_train_epochs=config.TrainCfg.epochs,
        per_device_train_batch_size=config.TrainCfg.batch_size,
        gradient_accumulation_steps=config.TrainCfg.grad_accum,
        learning_rate=config.TrainCfg.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=on_gpu,  # mixed-precision speedup on GPU; ignored on CPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=_Collator(tokenizer.pad_token_id),
        callbacks=[_LogCallback(log_cb)] if log_cb else None,
    )

    log("Starting training... (this can take a while on CPU)")
    trainer.train()

    config.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.ADAPTER_DIR))
    log(f"Done. Adapter saved to {config.ADAPTER_DIR}")
    return str(config.ADAPTER_DIR)


if __name__ == "__main__":
    train(log_cb=None)
