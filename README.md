# Qwen Coder Fine-Tuning Starter

This repository contains practical scripts to fine-tune coding models (especially Qwen Coder) using LoRA.

It includes:
- dataset preparation from chat logs
- standard Transformers + PEFT training
- Unsloth training path
- an end-to-end Qwen 7B pipeline with base vs fine-tuned benchmarking

## Repository layout

- `chats/`: your raw chat logs
- `finetune-python/`: all training scripts, requirements files, and datasets

Main scripts in `finetune-python/`:
- `prepare_chats_dataset.py`: convert `chats/*.txt` into JSONL training data
- `train_lora_sft.py`: generic LoRA SFT starter (TinyLlama by default)
- `train_lora_qwen_coder.py`: LoRA SFT for Qwen Coder
- `train_unsloth_qwen_coder.py`: Qwen Coder fine-tuning with Unsloth
- `train_qwen7b_end_to_end.py`: install deps, prepare dataset, train Qwen 7B, and benchmark

## Quick start (recommended)

1. Move into project folder:

```bash
cd finetune-python
```

2. Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Run the full Qwen 7B pipeline:

```bash
python train_qwen7b_end_to_end.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --chats_dir ../chats \
  --dataset_path data/train_from_chats.jsonl \
  --output_dir outputs/qwen7b-lora \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```

This single command will:
- install dependencies
- build dataset from your chat logs
- fine-tune with LoRA
- benchmark base vs fine-tuned model
- save report to `outputs/qwen7b-lora/benchmark_summary.json`

## Colab / GPU recommendation

For Qwen 7B training, use a CUDA GPU environment such as:
- Google Colab
- Kaggle
- RunPod

Without GPU, 7B training will be very slow or may fail due to memory limits.

## Alternative training paths

- Lightweight local test:

```bash
python train_lora_qwen_coder.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset_path data/train_from_chats.jsonl \
  --output_dir outputs/qwen-coder-lora
```

- Unsloth path:

```bash
pip install -r requirements-unsloth.txt
python train_unsloth_qwen_coder.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset_path data/train_from_chats.jsonl \
  --output_dir outputs/qwen-coder-unsloth \
  --load_in_4bit
```

## Dataset format

JSONL records use:
- `instruction` (required)
- `response` (required)
- `input` (optional)

Example:

```json
{"instruction":"Explain recursion simply.","response":"Recursion is when a function calls itself until a base case stops it."}
```

## Notes

- The detailed script-level guide is in `finetune-python/README.md`.
- Start with a small model and short run to validate your pipeline before long training jobs.
