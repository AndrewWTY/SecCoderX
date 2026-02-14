#!/bin/bash

set -x

# conda init
# conda activate easyr1
# export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98  # Set the number of CPUs for Ray
DEVICES="1,2,3,4"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
# export RAY_memory_monitor_refresh_ms=0

RAY_CPUS=${RAY_CPUS:-8} CUDA_VISIBLE_DEVICES=$DEVICES python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=4\
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.actor.fsdp.torch_dtype=bf16\
    worker.actor.optim.strategy=adamw_bf16