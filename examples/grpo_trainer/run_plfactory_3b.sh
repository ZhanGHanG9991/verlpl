set -x
export WANDB_API_KEY='975b4b9d4759ebc4c5ee764c39fe0e9034ae64f5'

# 定义源模型路径和转换后的目标路径
ORIGINAL_MODEL_PATH="/workspace/opt/models/plfactory-3b-sft"
FIXED_MODEL_PATH="/workspace/opt/models/plfactory-3b-sft-bf16"

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
# 1. torch_dtype=torch.bfloat16: 强制将权重转为 bf16，解决 float32 报错
# 2. attn_implementation="flash_attention_2": 显式开启 FA2
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# 确保 config 文件中也记录了正确的设置
model.config.attn_implementation = "flash_attention_2"
model.config.dtype = "bfloat16"

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
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/opt/projects/verlpl/examples/results/plfactory_rl_train.parquet \
    data.val_files=/workspace/opt/projects/verlpl/examples/results/plfactory_rl_test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${FIXED_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl-plfactory' \
    trainer.experiment_name='plfactory-3b-sft-new' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=77 \
    trainer.test_freq=6 \
    trainer.total_epochs=2 $@