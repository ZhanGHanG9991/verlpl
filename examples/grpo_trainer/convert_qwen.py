import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 使用 Shell 传入的路径
model_path = "/workspace/opt/models/Qwen3-0.6B"
output_dir = "/workspace/opt/models/Qwen3-0.6B-BF16-Fixed"

print(f"Loading model from {model_path}...")

# 关键步骤：
# 1. torch_dtype=torch.float16: 强制将权重转为 bf16，解决 float32 报错
# 2. attn_implementation="flash_attention_2": 显式开启 FA2
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# 确保 config 文件中也记录了正确的设置
model.config.attn_implementation = "flash_attention_2"
model.config.dtype = "float16"

print(f"Saving fixed model to {output_dir}...")
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(output_dir)
print("Model conversion and fix complete!")
