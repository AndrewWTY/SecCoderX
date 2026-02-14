#!/bin/bash

set -x

# conda init
# conda activate easyr1
# export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98  # Set the number of CPUs for Ray
DEVICES="0,1,2,3"


MODEL_PATH=/path/to/your/model  # replace it with your local file path
# export RAY_memory_monitor_refresh_ms=0

RAY_CPUS=${RAY_CPUS:-8} CUDA_VISIBLE_DEVICES=$DEVICES python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=SecCoderX/SecCoderX_Reward_Model_GRPO_dataset \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.format_prompt=examples/format_prompt/plain.jinja \
    trainer.experiment_name=experiment_name \
    trainer.n_gpus_per_node=4\
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.actor.fsdp.torch_dtype=bf16\
    worker.actor.optim.strategy=adamw_bf16\
    worker.reward.reward_function=examples/reward_function/seccoderx_rm.py:compute_score_without_cwe\
    worker.rollout.n=10 \
    worker.rollout.temperature=0.8 \
    worker.rollout.max_num_batched_tokens=16384 \
    trainer.save_checkpoint_path=path/to/your/checkpoint/dir \
    trainer.total_epochs=1