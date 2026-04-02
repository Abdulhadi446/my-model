import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def install_dependencies(python_bin: str):
    packages = [
        "torch>=2.3.0",
        "transformers>=4.44.0",
        "datasets>=2.20.0",
        "peft>=0.11.1",
        "accelerate>=0.33.0",
        "trl>=0.9.6",
        "bitsandbytes>=0.43.1",
        "safetensors>=0.4.3",
    ]
    cmd = [python_bin, "-m", "pip", "install", "-U"] + packages
    print("Installing dependencies...")
    subprocess.run(cmd, check=True)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
            if "instruction" not in row or "response" not in row:
                raise ValueError(
                    f"Line {i} must contain 'instruction' and 'response' keys"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def prompt_from_row(row):
    text = f"### Instruction:\n{row['instruction']}\n\n"
    if row.get("input"):
        text += f"### Input:\n{row['input']}\n\n"
    text += "### Response:\n"
    return text


def train_text_from_row(row):
    return prompt_from_row(row) + row["response"]


def token_f1(prediction: str, reference: str):
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    for t in pred_tokens:
        c = ref_counts.get(t, 0)
        if c > 0:
            overlap += 1
            ref_counts[t] = c - 1

    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def benchmark_model(model, tokenizer, eval_rows, max_new_tokens=128):
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    f1_scores = []
    total_gen_tokens = 0
    total_seconds = 0.0

    for row in eval_rows:
        prompt = prompt_from_row(row)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        total_gen_tokens += int(new_tokens.shape[0])
        total_seconds += elapsed
        f1_scores.append(token_f1(prediction, row["response"]))

    avg_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    toks_per_sec = total_gen_tokens / total_seconds if total_seconds > 0 else 0.0

    return {
        "samples": len(eval_rows),
        "avg_token_f1": avg_f1,
        "gen_tokens": total_gen_tokens,
        "gen_seconds": total_seconds,
        "tokens_per_sec": toks_per_sec,
    }


def split_rows(rows, eval_ratio, seed):
    random.seed(seed)
    rows = rows[:]
    random.shuffle(rows)

    eval_size = max(1, int(len(rows) * eval_ratio))
    if len(rows) <= 2:
        eval_size = 1
    if eval_size >= len(rows):
        eval_size = max(1, len(rows) - 1)

    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]
    if not train_rows:
        raise ValueError("Not enough rows to create a train split.")
    return train_rows, eval_rows


def load_base_model(model_name, load_in_4bit, max_memory=None):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    quant_config = None
    if torch.cuda.is_available() and load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        max_memory=max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install, prepare dataset, train Qwen 7B LoRA, and benchmark"
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--chats_dir", default="../chats")
    parser.add_argument("--dataset_path", default="data/train_from_chats.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen7b-lora")
    parser.add_argument("--eval_ratio", type=float, default=0.25)
    parser.add_argument("--benchmark_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    if not args.skip_install:
        install_dependencies(sys.executable)

    # Reuse the chat dataset converter so chats/*.txt becomes training JSONL.
    converter_script = root / "prepare_chats_dataset.py"
    subprocess.run(
        [
            sys.executable,
            str(converter_script),
            "--chats_dir",
            args.chats_dir,
            "--output",
            args.dataset_path,
        ],
        check=True,
        cwd=str(root),
    )

    dataset_path = (root / args.dataset_path).resolve()
    rows = load_jsonl(dataset_path)
    train_rows, eval_rows = split_rows(rows, args.eval_ratio, args.seed)
    eval_rows = eval_rows[: args.benchmark_samples]

    print(f"Dataset ready: {len(train_rows)} train / {len(eval_rows)} eval")

    load_in_4bit = not args.no_4bit
    if not torch.cuda.is_available():
        print("Warning: No CUDA GPU detected. Qwen 7B training will be very slow or fail.")

    print("Loading base model for pre-finetune benchmark...")
    base_model, tokenizer = load_base_model(args.model_name, load_in_4bit=load_in_4bit)

    base_metrics = benchmark_model(
        base_model,
        tokenizer,
        eval_rows,
        max_new_tokens=args.max_new_tokens,
    )
    print("Base benchmark:")
    print(json.dumps(base_metrics, indent=2))

    train_dataset = Dataset.from_dict({"text": [train_text_from_row(r) for r in train_rows]})

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    print("Starting LoRA fine-tuning...")
    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        args=training_args,
        max_seq_length=args.max_seq_length,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    del trainer
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading base+adapter for post-finetune benchmark...")
    post_base, post_tokenizer = load_base_model(
        args.model_name,
        load_in_4bit=load_in_4bit,
    )
    tuned_model = PeftModel.from_pretrained(post_base, args.output_dir)

    tuned_metrics = benchmark_model(
        tuned_model,
        post_tokenizer,
        eval_rows,
        max_new_tokens=args.max_new_tokens,
    )
    print("Fine-tuned benchmark:")
    print(json.dumps(tuned_metrics, indent=2))

    summary = {
        "model_name": args.model_name,
        "dataset_path": str(dataset_path),
        "adapter_path": str((root / args.output_dir).resolve()),
        "base": base_metrics,
        "fine_tuned": tuned_metrics,
        "delta_avg_token_f1": tuned_metrics["avg_token_f1"] - base_metrics["avg_token_f1"],
        "delta_tokens_per_sec": tuned_metrics["tokens_per_sec"] - base_metrics["tokens_per_sec"],
    }

    summary_path = (root / args.output_dir / "benchmark_summary.json").resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Benchmark summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
