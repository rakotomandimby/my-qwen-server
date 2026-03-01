import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

original_new = bnb.nn.Params4bit.__new__

def patched_new(cls, *args, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return original_new(cls, *args, **kwargs)

bnb.nn.Params4bit.__new__ = staticmethod(patched_new)

model_name = "Qwen/Qwen3-Coder-Next"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, 
    llm_int8_enable_fp32_cpu_offload=True  
)

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype="auto",
  device_map="auto",
  quantization_config=quantization_config,
  offload_folder="model_offload"           
)

# prepare the model input
prompt = "Write a quick sort algorithm."
messages =[
  {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=65536
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
