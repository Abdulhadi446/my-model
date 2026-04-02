import argparse
import json
import os

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


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
    user_text = f"Instruction:\n{row['instruction']}"
    if row.get("input"):
        user_text += f"\n\nInput:\n{row['input']}"

    # Chat template keeps tokenization consistent with Qwen Instruct models.
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": row["response"]},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth + Qwen Coder LoRA fine-tuning")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset_path", default="data/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen-coder-unsloth")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--full_finetune", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=args.full_finetune,
    )

    if not args.full_finetune:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    rows = read_jsonl(args.dataset_path)
    formatted = []
    for r in rows:
        messages = format_example(r)
        formatted.append(tokenizer.apply_chat_template(messages, tokenize=False))
    dataset = Dataset.from_dict({"text": formatted})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            bf16=torch.cuda.is_available(),
            optim="adamw_8bit",
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Saved to: {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
