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

### Build dataset from your chat logs

If you want to train on files in `/workspaces/codespaces-jupyter/chats`:

```bash
cd /workspaces/codespaces-jupyter/finetune-python
python prepare_chats_dataset.py \
  --chats_dir ../chats \
  --output data/train_from_chats.jsonl
```

Then train with `--dataset_path data/train_from_chats.jsonl`.

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

## 3d) One-script Qwen 7B pipeline (install + dataset + train + benchmark)

This runs everything in one command:
- installs dependencies
- builds dataset from `../chats`
- fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` with LoRA
- benchmarks base model vs fine-tuned model

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

Outputs:
- adapter: `outputs/qwen7b-lora`
- benchmark report: `outputs/qwen7b-lora/benchmark_summary.json`

Notes:
- Use a CUDA GPU environment (Colab/Kaggle/RunPod) for 7B training.
- Add `--skip_install` if your environment is already prepared.

## 4) Where output goes
- Adapter + tokenizer are saved in `outputs/tinyllama-lora`

## 5) Chat with the model

Chat with base model:

```bash
python chat_with_model.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --load_in_4bit
```

Chat with your fine-tuned adapter:

```bash
python chat_with_model.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter_path outputs/qwen7b-lora \
  --load_in_4bit
```

Inside chat:
- Type your prompt and press Enter
- Type `/exit` to quit

## Notes
- GPU is strongly recommended for speed.
- Start with a small model and small dataset to validate the pipeline.
- For better quality, use a larger clean dataset and run multiple epochs.
