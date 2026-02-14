#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

# GPU Allocation Strategy:
# - GPU 0-1: Training (ActorRolloutRef with FSDP + vLLM time-multiplexed)
# - GPU 2: Reward model (VLLM offline inference)
# Total: 3 GPUs required
DEVICES="4,5,2,3"

RAY_CPUS=${RAY_CPUS:-8} CUDA_VISIBLE_DEVICES=$DEVICES python3 -m verl.trainer.main \
    config=../examples/config.yaml \
    data.train_files=SecCoderX/SecCoderX_Qwen2.5_Coder_7B_GRPO_dataset \
    data.val_files=SecCoderX/SecCoderX_Qwen2.5_Coder_7B_GRPO_dataset \
    worker.actor.model.model_path=Qwen/Qwen2.5-Coder-7B-Instruct\
    data.max_prompt_length=2048 \
    data.max_response_length=3072 \
    data.format_prompt=../examples/format_prompt/plain.jinja \
    trainer.experiment_name=experiment_name \
    trainer.n_gpus_per_node=4 \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.rollout.tensor_parallel_size=2 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=10 \
    worker.rollout.temperature=0.8 \
    trainer.total_epochs=5 \
    trainer.save_checkpoint_path=path/to/your/checkpoint/dir \
    trainer.find_last_checkpoint=true \
    worker.reward.use_vllm_reward_model=true \
    worker.reward.vllm_model_path=seccoderx_reward_model \
    worker.reward.vllm_tensor_parallel_size=1 \
    worker.reward.vllm_num_engines=2 \
    worker.reward.vllm_gpu_memory_utilization=0.9 \
    worker.reward.vllm_sampling_params.temperature=0.0 \
    worker.reward.vllm_sampling_params.max_tokens=4096 \
    worker.reward.vllm_sampling_params.skip_special_tokens=false \
    worker.reward.vllm_sampling_params.include_stop_str_in_output=true \
    worker.reward.reward_function=../examples/reward_function/seccoderx_align.py:compute_score
