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

import re
from typing import Any, List


def extract_answer(response: str) -> str:
    """Extract the answer from between <answer> and </answer> tags."""
    matches = re.findall(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def extract_think_content(response: str) -> str:
    """Extract the content from between <think> and </think> tags."""
    matches = re.findall(r'<think>(.*?)</think>', response, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def check_cwe_mentioned(response: str, cwe_types: List[str]) -> bool:
    """Check if any of the CWE types are mentioned in the <think> section."""
    if not cwe_types:
        return False
    
    think_content = extract_think_content(response)
    if not think_content:
        return False
    
    # Convert think content to lowercase for case-insensitive matching
    think_lower = think_content.lower()
    
    # Check if any CWE type is mentioned in the thinking section
    for cwe in cwe_types:
        if cwe.lower() in think_lower:
            return True
    
    return False


def format_reward(response: str) -> float:
    """Check if response follows the expected format with <think> and <answer> tags."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 0.2 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the extracted answer matches the ground truth for vulnerability detection."""
    answer = extract_answer(response)
    if not answer:
        return 0.0
    
    # Parse combined ground truth format: "Vulnerable;[CWE-89, CWE-72]" or "Not Vulnerable;[]"
    if ';' in ground_truth:
        vulnerability_status = ground_truth.split(';')[0].strip()
    else:
        vulnerability_status = ground_truth.strip()
    
    # Case-insensitive comparison for vulnerability detection
    return 1.0 if answer.lower() == vulnerability_status.lower() else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for r2vul reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        cwe_type_str = reward_input.get("cwe_type", "")
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, ground_truth)
        
        # Parse CWE types from ground_truth if it contains semicolon
        if ';' in ground_truth:
            parts = ground_truth.split(';', 1)
            if len(parts) > 1:
                cwe_part = parts[1].strip()
                # Extract CWE types from format like "[CWE-89, CWE-72]"
                if cwe_part.startswith('[') and cwe_part.endswith(']'):
                    cwe_part = cwe_part[1:-1]
                    cwe_types = [cwe.strip() for cwe in cwe_part.split(',') if cwe.strip()]
                else:
                    cwe_types = [cwe.strip() for cwe in cwe_part.split(',') if cwe.strip()]
            else:
                cwe_types = []
        else:
            # Fallback to legacy cwe_type field
            cwe_types = [cwe.strip() for cwe in cwe_type_str.split(",") if cwe.strip()] if cwe_type_str else []
        
        # Calculate CWE bonus based on the new scoring system
        cwe_bonus_score = 0.0
        # # import pdb; pdb.set_trace()
        # if cwe_types:
        #     cwe_mentioned = check_cwe_mentioned(response, cwe_types)
            
        #     if cwe_mentioned and accuracy_score > 0:
        #         # Mentioned correct CWE + accurate prediction
        #         cwe_bonus_score = 1.0
        #     elif cwe_mentioned and accuracy_score == 0:
        #         # Mentioned correct CWE + wrong prediction
        #         cwe_bonus_score = 0.3
        #     elif not cwe_mentioned and accuracy_score > 0:
        #         # Failed to mention correct CWE + accurate prediction
        #         cwe_bonus_score = -0.2
        #     else:
        #         # Failed to mention correct CWE + wrong prediction
        #         cwe_bonus_score = 0.0
        
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score + cwe_bonus_score
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
            "cwe_bonus": cwe_bonus_score,
        })

    return scores


def compute_score_without_cwe(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for r2vul reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, ground_truth)
        
        scores.append({
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        })

    return scores