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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig

# Conditional VLLM imports

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
VLLM_AVAILABLE = True


class RewardInput(TypedDict):
    prompt: str
    response: str
    response_length: int
    ground_truth: str
    metadata: dict  # All additional fields from the dataset


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            
            # Decode prompt
            valid_prompt_ids = prompt_ids[i]
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            # Decode response
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            # Collect all metadata from non_tensor_batch (excluding ground_truth which we pass explicitly)
            metadata = {}
            for key, value in data.non_tensor_batch.items():
                if key != "ground_truth":
                    metadata[key] = value[i]
            
            score = self.reward_fn(
                {
                    "prompt": prompt_str,
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "metadata": metadata,
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        # First, load the reward function module (parent class behavior)
        # We need to do this manually instead of calling super().__init__()
        # because we want to modify reward_function_kwargs before creating the partial
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        # Initialize VLLM reward model(s) if configured
        self.vllm_engines = None
        self.vllm_tokenizer = None
        self.vllm_sampling_params = None

        if config.use_vllm_reward_model:
            if config.vllm_model_path is None:
                raise ValueError(
                    "vllm_model_path must be specified when use_vllm_reward_model=True"
                )

            import ray

            # Get GPU IDs assigned to this actor by Ray
            gpu_ids = ray.get_gpu_ids()
            original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

            num_engines = config.vllm_num_engines
            print(f"[RewardManager] Initializing {num_engines} VLLM reward engine(s): {config.vllm_model_path}")
            print(f"[RewardManager]   Tensor parallel size per engine: {config.vllm_tensor_parallel_size}")
            print(f"[RewardManager]   GPU memory utilization: {config.vllm_gpu_memory_utilization}")
            print(f"[RewardManager]   Ray GPU IDs: {gpu_ids}")
            print(f"[RewardManager]   Original CUDA_VISIBLE_DEVICES: {original_cuda_visible_devices}")
            print(f"[RewardManager]   torch.cuda.device_count(): {torch.cuda.device_count()}")

            if len(gpu_ids) < num_engines:
                raise ValueError(
                    f"Not enough GPUs allocated: {len(gpu_ids)} GPUs assigned but {num_engines} engines requested"
                )

            # Initialize multiple VLLM engines (one per GPU for data parallelism)
            self.vllm_engines = []
            for i in range(num_engines):
                # Get the GPU ID for this engine
                engine_gpu_id = gpu_ids[i]

                print(f"[RewardManager] Creating engine {i} on physical GPU {engine_gpu_id}")

                # Set CUDA_VISIBLE_DEVICES to limit the GPUs visible to this engine
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(engine_gpu_id))

                # Set the default CUDA device
                torch.cuda.set_device(0)  # Since only one GPU is visible, it's cuda:0

                print(f"[RewardManager]   CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                print(f"[RewardManager]   torch.cuda.device_count()={torch.cuda.device_count()}")

                # Initialize the LLM model
                engine = LLM(
                    model=config.vllm_model_path,
                    tensor_parallel_size=config.vllm_tensor_parallel_size,
                    gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                    trust_remote_code=config.vllm_trust_remote_code,
                    max_model_len=config.vllm_max_model_len,
                    enforce_eager=False,
                    device="cuda:0"
                )
                self.vllm_engines.append(engine)
                print(f"[RewardManager] Engine {i} created successfully")

            # Restore original CUDA_VISIBLE_DEVICES
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
            print(f"[RewardManager] All {num_engines} engines created. Restored CUDA_VISIBLE_DEVICES")

            # Load tokenizer once (shared across all engines)
            self.vllm_tokenizer = AutoTokenizer.from_pretrained(
                config.vllm_model_path,
                trust_remote_code=config.vllm_trust_remote_code
            )
            stop_tokens = []
            if self.vllm_tokenizer and self.vllm_tokenizer.eos_token:
                stop_tokens.append(self.vllm_tokenizer.eos_token)
            stop_tokens.append("<|im_end|>")
            config.vllm_sampling_params['stop'] = stop_tokens if stop_tokens else None

            # Create sampling params by unpacking config dictionary
            self.vllm_sampling_params = SamplingParams(**config.vllm_sampling_params)

            print(f"VLLM sampling params: {config.vllm_sampling_params}")
            print(f"Successfully initialized {num_engines} VLLM reward engine(s)")

            # Inject VLLM objects into reward function kwargs
            # Pass list of engines for data parallelism
            config.reward_function_kwargs['vllm_engines'] = self.vllm_engines
            config.reward_function_kwargs['vllm_tokenizer'] = self.vllm_tokenizer
            config.reward_function_kwargs['vllm_sampling_params'] = self.vllm_sampling_params

        # Create reward function with kwargs (including VLLM objects if applicable)
        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        # print(f"data.batch: {data.batch}")
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            # Collect all metadata from non_tensor_batch (excluding ground_truth which we pass explicitly)
            metadata = {}
            for key, value in data.non_tensor_batch.items():
                if key != "ground_truth":
                    metadata[key] = value[i]
            
            reward_inputs.append(
                {
                    # "prompt": prompt_str,
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "metadata": metadata,
                }
            )
            # print(f"prompt_str: {prompt_str}")
            # print(f"metadata: {metadata}")

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
