set -x
export WANDB_API_KEY='xxx'

# 定义源模型路径和转换后的目标路径
ORIGINAL_MODEL_PATH="/workspace/opt/models/Qwen2.5-0.5B-Instruct"
FIXED_MODEL_PATH="/workspace/opt/models/Qwen2.5-0.5B-Instruct-BF16-Fixed"

# -----------------------------------------------------------------------------
# 步骤 1: 创建并执行 Python 脚本，将模型转换为 bfloat16 并修复配置
# -----------------------------------------------------------------------------
echo "Generating model conversion script..."

cat > convert_qwen.py << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 使用 Shell 传入的路径
model_path = "${ORIGINAL_MODEL_PATH}"
output_dir = "${FIXED_MODEL_PATH}"

print(f"Loading model from {model_path}...")

# 关键步骤：
# 1. torch_dtype=torch.float16: 强制将权重转为 bf16，解决 float32 报错
# 2. attn_implementation="flash_attention_2": 显式开启 FA2
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
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
EOF

# 执行转换脚本
python3 convert_qwen.py

# -----------------------------------------------------------------------------
# 步骤 2: 启动训练 (使用修复后的模型路径 FIXED_MODEL_PATH)
# -----------------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/opt/datasets/verl/gsm8k/train.parquet \
    data.val_files=/workspace/opt/datasets/verl/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${FIXED_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.model.save_dtype=float16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=float16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_0.5b_grpo_tp2' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@