#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-Coder-Next"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # float16 for CUDA, float32 elsewhere for compatibility
    if device == "cuda":
        return torch.float16
    return torch.float32


def main() -> None:
    device = pick_device()
    dtype = pick_dtype(device)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Loading model on {device} with dtype={dtype} ...")
    # IMPORTANT:
    # - No load_in_4bit / load_in_8bit
    # - No quantization_config
    # - No device_map="auto"
    # These avoid meta-tensor + bitsandbytes hook issues.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.to(device)
    model.eval()

    prompt = "Explain in a few lines what a transformer model is."
    messages = [{"role": "user", "content": prompt}]

    # Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,  # practical default
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n=== Model output ===\n")
    print(output_text)


if __name__ == "__main__":
    main()
