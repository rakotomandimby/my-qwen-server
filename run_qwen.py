#!/usr/bin/env python3

import os
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_NAME = "Qwen/Qwen3.5-2B"

def ensure_vocab_size(config, tokenizer):
    """Ensures config.vocab_size using text_config.vocab_size, tokenizer.vocab_size, or len(tokenizer)."""
    if getattr(config, "vocab_size", None) is not None:
        return

    text_config = getattr(config, "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)
        except TypeError as exc:
            raise ValueError("Unable to determine vocab_size from config or tokenizer.") from exc
    if vocab_size is None:
        raise ValueError("Unable to determine vocab_size from config or tokenizer.")
    config.vocab_size = vocab_size

def get_device_and_dtype():
    """Identifies the best available hardware accelerator and compatible dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype

def pick_max_memory():
    """Dynamically calculates max memory bounds to ensure stable CPU/Disk offloading."""
    if not torch.cuda.is_available():
        return None
    
    # Leave 1 GB VRAM headroom for CUDA context & overhead
    total_vram_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    gpu_budget_gib = max(1, total_vram_gib - 1)
    
    # Calculate System RAM
    try:
        total_ram_gib = int((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3))
    except (AttributeError, ValueError, OSError):
        total_ram_gib = 32
        
    # Leave 4 GB System RAM headroom for OS tasks
    cpu_budget_gib = max(4, total_ram_gib - 4)
    
    return {
        0: f"{gpu_budget_gib}GiB",
        "cpu": f"{cpu_budget_gib}GiB",
    }

def main() -> None:
    device, dtype = get_device_and_dtype()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ensure_vocab_size(config, tokenizer)
    
    model_kwargs = {
        "torch_dtype": dtype,
        "config": config,
        "trust_remote_code": True,
    }

    # --- GPU VRAM BUDGETING, QUANTIZATION & OFFLOADING ---
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        
        # Explicit folder for layers that completely overflow RAM to Disk
        os.makedirs("offload", exist_ok=True)
        model_kwargs["offload_folder"] = "offload"
        
        # Restrict memory bounds to guarantee stability
        mem_map = pick_max_memory()
        if mem_map:
            model_kwargs["max_memory"] = mem_map

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected CUDA device with {total_vram_gb:.1f} GiB VRAM. Max mapped VRAM: {mem_map[0]}")
        
        if total_vram_gb < 20.0:
            print("Using 4-bit NF4 quantization. Allowing fp32 CPU offloading for layers that don't fit...")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,  
            )
    else:
        model_kwargs["low_cpu_mem_usage"] = True

    print(f"Loading model on {device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception as e:
        print(f"\nFailed to load model! Error details: {e}\n")
        raise

    if device != "cuda":
        model.to(device)
        
    model.eval()

    # --- PROMPTING ---
    prompt = "Tell me about OCaml"
    messages = [{"role": "user", "content": prompt}]

    print("Formatting prompt via chat template...")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # --- OUTPUT PARSING ---
    input_length = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_length:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n=== Model output ===\n")
    print(output_text.strip())


if __name__ == "__main__":
    main()
