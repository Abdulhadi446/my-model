import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_name: str, adapter_path: str | None, load_in_4bit: bool):
    quant_config = None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if torch.cuda.is_available() and load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.9

    return model, tokenizer


def build_messages(user_text: str, system_prompt: str | None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def generate_response(
    model,
    tokenizer,
    user_text: str,
    system_prompt: str | None,
    max_new_tokens: int,
):
    messages = build_messages(user_text, system_prompt)

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"User: {user_text}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with base or fine-tuned model")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Path to LoRA adapter directory. Omit to chat with base model.",
    )
    parser.add_argument("--system_prompt", default="You are a helpful coding assistant.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )

    print("Interactive chat started.")
    print("Type /exit to quit.")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            print("Bye.")
            break

        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_text=user_text,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"\nModel: {answer}")


if __name__ == "__main__":
    main()
