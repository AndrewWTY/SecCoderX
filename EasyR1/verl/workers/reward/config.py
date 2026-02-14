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
"""
Reward config
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RewardConfig:
    reward_type: str = "batch"
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1

    # VLLM reward model configuration
    use_vllm_reward_model: bool = False
    vllm_model_path: Optional[str] = None
    vllm_tensor_parallel_size: int = 1  # TP size per engine
    vllm_num_engines: int = 1  # Number of engines (data parallelism)
    vllm_gpu_memory_utilization: float = 0.9
    vllm_trust_remote_code: bool = True
    vllm_max_model_len: Optional[int] = None
    num_gpus: int = 0  # Auto: vllm_tensor_parallel_size * vllm_num_engines

    # VLLM sampling parameters (unpacked into vllm.SamplingParams)
    vllm_sampling_params: dict = field(default_factory=lambda: {
        "temperature": 0.0,
        "max_tokens": 4096,
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
    })

    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.reward_function):  # ray job uses absolute path
                self.reward_function = os.path.abspath(self.reward_function)
            else:
                print(f"Reward function {self.reward_function} not found.")
                self.reward_function = None
