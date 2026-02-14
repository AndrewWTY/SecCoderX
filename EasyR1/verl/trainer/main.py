# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"

        # Compute GPU allocation for reward model FIRST
        num_reward_gpus = 0
        if config.worker.reward.use_vllm_reward_model:
            # Total GPUs = TP size per engine × number of engines
            num_reward_gpus = (config.worker.reward.vllm_tensor_parallel_size *
                             config.worker.reward.vllm_num_engines)
            config.worker.reward.num_gpus = num_reward_gpus
            print(f"[DEBUG] Allocating {num_reward_gpus} GPU(s) for VLLM reward model")
            print(f"[DEBUG]   Num engines: {config.worker.reward.vllm_num_engines}")
            print(f"[DEBUG]   TP size per engine: {config.worker.reward.vllm_tensor_parallel_size}")
            print(f"[DEBUG]   Reward model path: {config.worker.reward.vllm_model_path}")
        else:
            print(f"[DEBUG] No VLLM reward model - num_reward_gpus = 0")

        # Adjust GPU count for resource pool by reserving GPUs for reward model
        num_reward_gpus_reserved = num_reward_gpus
        training_gpus = config.trainer.n_gpus_per_node - num_reward_gpus_reserved

        print(f"[DEBUG] Total GPUs: {config.trainer.n_gpus_per_node}, Reward GPUs reserved: {num_reward_gpus_reserved}, Training GPUs: {training_gpus}")

        resource_pool_spec = {
            global_pool_id: [training_gpus] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        print(f"[DEBUG] Creating resource pool with spec: {resource_pool_spec}")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        # Create reward manager Ray actors with GPU resources
        print(f"[DEBUG] Creating RewardManager with num_cpus={config.worker.reward.num_cpus}, num_gpus={num_reward_gpus}")
        RemoteRewardManager = ray.remote(RewardManager).options(
            num_cpus=config.worker.reward.num_cpus,
            num_gpus=num_reward_gpus
        )

        # Create one shared reward manager for both training and validation
        # (avoids consuming 2 GPUs when using VLLM reward model)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = reward_fn  # Share the same instance
        print(f"[DEBUG] Using shared reward manager for both training and validation")

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        }
        
        # Allow setting Ray CPU count via environment variable
        import os
        ray_init_kwargs = {"runtime_env": runtime_env}
        if "RAY_CPUS" in os.environ:
            ray_init_kwargs["num_cpus"] = int(os.environ["RAY_CPUS"])
            
        ray.init(**ray_init_kwargs)

    # Debug: Print Ray's detected GPU resources
    available_resources = ray.available_resources()
    gpu_count = available_resources.get('GPU', 0)
    print(f"[DEBUG] Ray detected {gpu_count} GPU(s)")
    print(f"[DEBUG] Ray available resources: {available_resources}")

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
