#!/usr/bin/env python3

import os
import transformers
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_NAME = "Qwen/Qwen3.5-2B"
TEXT_CONFIG_FIELDS = (
    "attention_dropout",
    "attention_bias",
    "bos_token_id",
    "eos_token_id",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "max_position_embeddings",
    "max_window_layers",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "rope_parameters",
    "rope_scaling",
    "rope_theta",
    "sliding_window",
    "tie_word_embeddings",
    "use_sliding_window",
    "use_cache",
)

def get_config_value(config, key):
    """Gets a config value from either a config object or a plain dict."""
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)

def ensure_vocab_size(config, tokenizer):
    """Ensures config.vocab_size using text_config.vocab_size, tokenizer.vocab_size, or len(tokenizer)."""
    if get_config_value(config, "vocab_size") is not None:
        return

    text_config = getattr(config, "text_config", None)
    vocab_size = get_config_value(text_config, "vocab_size")
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

def ensure_text_config_fields(config):
    """Copies missing ``text_config`` fields to the top level when needed.

    Some Qwen 3.5 checkpoints keep core text-model settings such as
    ``num_hidden_layers`` under ``config.text_config`` while the installed
    Transformers model loader still reads them from the top-level config.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return

    for key in TEXT_CONFIG_FIELDS:
        value = get_config_value(text_config, key)
        if get_config_value(config, key) is None and value is not None:
            setattr(config, key, value)

def ensure_layer_types(config):
    """Ensures ``config.layer_types`` exists for Qwen 3.5 text-model loading.

    Older Transformers releases may instantiate the top-level ``Qwen3_5Config``
    without copying ``text_config.layer_types`` to the root config, even though
    the model loader expects it there. When the checkpoint also omits an
    explicit ``layer_types`` list, mirror the library default pattern.
    """
    if get_config_value(config, "layer_types") is not None:
        return

    num_hidden_layers = get_config_value(config, "num_hidden_layers")
    if num_hidden_layers is None:
        return

    text_config = getattr(config, "text_config", None)
    interval_pattern = get_config_value(text_config, "full_attention_interval")
    if interval_pattern is None:
        interval_pattern = get_config_value(config, "full_attention_interval")
    if interval_pattern is None:
        # Mirrors the Qwen 3.5 Transformers config default when the checkpoint
        # does not provide an explicit full_attention_interval value.
        interval_pattern = 4
    elif not isinstance(interval_pattern, int) or interval_pattern <= 0:
        raise ValueError("full_attention_interval must be a positive integer.")

    config.layer_types = [
        "full_attention" if (layer_idx + 1) % interval_pattern == 0 else "linear_attention"
        for layer_idx in range(num_hidden_layers)
    ]

def ensure_pad_token_id(config, tokenizer):
    """Ensure ``config.pad_token_id`` exists before model initialization.

    Fallback order:
    1. ``config.text_config.pad_token_id``
    2. ``tokenizer.pad_token_id``
    3. ``config.eos_token_id``
    4. ``tokenizer.eos_token_id``
    """
    if get_config_value(config, "pad_token_id") is not None:
        return

    text_config = getattr(config, "text_config", None)
    pad_token_id = get_config_value(text_config, "pad_token_id")
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = get_config_value(config, "eos_token_id")
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is not None:
        config.pad_token_id = pad_token_id

def get_model_loader(config):
    """Selects the safest loader class for the checkpoint-backed config."""
    conditional_generation_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    architectures = get_config_value(config, "architectures") or ()
    text_config = getattr(config, "text_config", None)
    model_type = get_config_value(config, "model_type")
    text_model_type = get_config_value(text_config, "model_type")

    if conditional_generation_cls is not None:
        if "Qwen3_5ForConditionalGeneration" in architectures:
            return conditional_generation_cls

        # Some checkpoints expose the composite Qwen 3.5 config without an
        # architectures hint, but the paired weights still live under the
        # conditional-generation wrapper's ``model.language_model.*`` prefix.
        if text_config is not None and (model_type == "qwen3_5" or text_model_type == "qwen3_5"):
            return conditional_generation_cls

        # Keep supporting composite configs that do not expose a ``model_type``
        # hint but still bundle text and vision sub-configs under the wrapper.
        if text_config is not None and getattr(config, "vision_config", None) is not None:
            return conditional_generation_cls

    return AutoModelForCausalLM

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
    ensure_text_config_fields(config)
    ensure_layer_types(config)
    ensure_pad_token_id(config, tokenizer)
    
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

    model_loader = get_model_loader(config)
    print(f"Loading model on {device} with {model_loader.__name__}...")
    try:
        model = model_loader.from_pretrained(MODEL_NAME, **model_kwargs)
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
