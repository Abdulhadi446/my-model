import argparse
import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
            if "instruction" not in row or "response" not in row:
                raise ValueError(
                    f"Line {i} must include 'instruction' and 'response' keys"
                )
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty. Add at least one JSONL row.")
    return rows


def format_example(row):
    inp = row.get("input", "")
    if inp:
        return (
            "### Instruction:\n"
            f"{row['instruction']}\n\n"
            "### Input:\n"
            f"{inp}\n\n"
            "### Response:\n"
            f"{row['response']}"
        )
    return (
        "### Instruction:\n"
        f"{row['instruction']}\n\n"
        "### Response:\n"
        f"{row['response']}"
    )


@dataclass
class Config:
    model_name: str
    dataset_path: str
    output_dir: str
    target_modules: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_seq_length: int
    logging_steps: int
    save_steps: int


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Coder LoRA SFT fine-tuning script")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset_path", default="data/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen-coder-lora")
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of modules for LoRA",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    args = parser.parse_args()
    return Config(**vars(args))


def main():
    cfg = parse_args()

    rows = read_jsonl(cfg.dataset_path)
    dataset = Dataset.from_dict({"text": [format_example(r) for r in rows]})

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    target_modules = [m.strip() for m in cfg.target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("At least one target module is required.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        args=args,
        max_seq_length=cfg.max_seq_length,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Training complete. Adapter saved to: {cfg.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
