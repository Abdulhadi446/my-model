# Python AI Fine-Tuning Starter (LoRA)

This project gives you a simple way to fine-tune a small language model with your own instruction dataset.

## What this does
- Loads a base model (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- Applies LoRA adapters (parameter-efficient fine-tuning)
- Trains on `data/train.jsonl`
- Saves your fine-tuned adapter to `outputs/tinyllama-lora`

## 1) Install dependencies

```bash
cd /workspaces/codespaces-jupyter/finetune-python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Prepare your dataset

Use JSONL with one record per line.
Required keys:
- `instruction`
- `response`
Optional key:
- `input`

Example is already in `data/train.jsonl`.

## 3) Start fine-tuning

```bash
python train_lora_sft.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset_path data/train.jsonl \
  --output_dir outputs/tinyllama-lora \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 512
```

## 3b) Qwen Coder version

Use this command to fine-tune Qwen Coder directly:

```bash
python train_lora_qwen_coder.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset_path data/train.jsonl \
  --output_dir outputs/qwen-coder-lora \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 768
```

Default target modules for Qwen Coder are:
- `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

If your model variant uses different module names, override with:

```bash
python train_lora_qwen_coder.py --target_modules q_proj,k_proj,v_proj,o_proj
```

## 3c) Qwen Coder with Unsloth (faster)

Install Unsloth stack:

```bash
pip install -r requirements-unsloth.txt
```

Run Unsloth fine-tuning:

```bash
python train_unsloth_qwen_coder.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dataset_path data/train.jsonl \
  --output_dir outputs/qwen-coder-unsloth \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 2048 \
  --load_in_4bit
```

Tips:
- Use `--load_in_4bit` if VRAM is tight.
- Remove `--load_in_4bit` for higher precision if you have enough VRAM.
- Use `--full_finetune` only if you have large GPU memory.

## 4) Where output goes
- Adapter + tokenizer are saved in `outputs/tinyllama-lora`

## Notes
- GPU is strongly recommended for speed.
- Start with a small model and small dataset to validate the pipeline.
- For better quality, use a larger clean dataset and run multiple epochs.
