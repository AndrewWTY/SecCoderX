

from ast import Sub
import re
import traceback
from typing import Any, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from codebleu import calc_codebleu

# Load prompt template at module level
TEMPLATE_PATH = "/path/to/project/vulnerability_eval_pipeline/prompt_templates/think_templates_withcwe_new.txt"

def load_prompt_template() -> str:
    """Load the prompt template from file."""
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return f.read().strip()

# Cache the template
PROMPT_TEMPLATE = load_prompt_template()


def parse_reward_model_answer(vllm_response: str) -> str:
    """Extract the answer from between <answer> and </answer> tags.

    Args:
        vllm_response: The full response from VLLM model

    Returns:
        Extracted answer string (e.g., "vulnerable" or "not vulnerable")
    """
    matches = re.findall(r'<answer>(.*?)</answer>', vllm_response, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip().lower()
    return ""


def extract_code_from_response(response: str) -> tuple[str, bool]:
    """Extract the code from between triple backticks.

    Args:
        response: The model's response containing code

    Returns:
        Tuple of (extracted_code, success_flag)
    """
    # Match triple backtick blocks that may optionally have a language label
    matches = re.findall(r'```[\w]*\n(.*?)```', response, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[0].strip(), True
    return response.strip(), False


def format_prompt_for_vllm(
    code: str,
    cwe_id: List[str],
    cwe_names: List[str],
    cwe_descriptions: List[str],
    language_suffix: str
) -> str:
    """Format the vulnerability detection prompt using the template.

    Args:
        code: The code to analyze
        cwe_id: List of CWE IDs
        cwe_names: List of CWE names
        cwe_descriptions: List of CWE descriptions
        language_suffix: Language file suffix (e.g., "c", "py")

    Returns:
        Formatted prompt string
    """
    # Format specific CWE description
    specific_cwe_template = """\
Type: {cwe_id}: {cwe_name}
Description: {cwe_desc}
"""

    specific_cwe_desc = ""
    for id, name, desc in zip(cwe_id, cwe_names, cwe_descriptions):
        specific_cwe_desc += specific_cwe_template.format(
            cwe_id=id, cwe_name=name, cwe_desc=desc
        ) + "\n"

    # Format the prompt using the template
    prompt = PROMPT_TEMPLATE.format(
        code=code,
        specific_cwe_desc=specific_cwe_desc,
        language_suffix=language_suffix
    )

    return prompt


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Apply the model's chat template to the prompt.

    Args:
        tokenizer: The VLLM tokenizer
        prompt: The formatted text prompt

    Returns:
        Chat-templated prompt string
    """
    messages = [{"role": "user", "content": prompt}]

    # Apply chat template
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return chat_prompt


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    vllm_engines=None,  # NOW A LIST OF ENGINES
    vllm_tokenizer=None,
    vllm_sampling_params=None,
) -> List[Dict[str, float]]:
    """Compute vulnerability detection reward scores using data-parallel VLLM inference.

    This function receives multiple VLLM engines from RewardManager and distributes
    inference work across them for higher throughput.

    Args:
        reward_inputs: List of dictionaries containing:
            - "response": The generated code to evaluate
            - "ground_truth": Ground truth label (not used for scoring)
            - "metadata": Dict with "cwe_id", "cwe_name", "cwe_description",
                         "language_suffix", reference code, etc.
        vllm_engines: LIST of VLLM LLM engines (data parallelism)
        vllm_tokenizer: Tokenizer for chat template (passed from RewardManager)
        vllm_sampling_params: Sampling parameters (passed from RewardManager)

    Returns:
        List of score dictionaries with "overall", "functionality", "format",
        "vulnerability", "length" keys
    """
    # Validate inputs - support both single engine (backward compat) and multiple engines
    if vllm_engines is None:
        raise ValueError(
            "This reward function requires VLLM engine(s). "
            "Set use_vllm_reward_model=True in config."
        )

    # Handle backward compatibility: convert single engine to list
    if not isinstance(vllm_engines, list):
        vllm_engines = [vllm_engines]

    if vllm_tokenizer is None:
        raise ValueError("vllm_tokenizer is required")

    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for VLLM reward function.")

    num_engines = len(vllm_engines)
    print(f"Processing {len(reward_inputs)} samples with {num_engines} VLLM engine(s) (data-parallel)")

    # Phase 1: Extract codes and compute format/length rewards
    extracted_data = []
    for idx, reward_input in enumerate(reward_inputs):
        response = reward_input["response"]
        metadata = reward_input.get("metadata", {})

        # Extract code from response
        code, format_success = extract_code_from_response(response)
        format_reward = 1.0 if format_success else 0.0

        # Get reference code for functionality and length comparison
        reference_code = metadata.get("generated_code", "")
        codebleu_lang = metadata.get("codebleu_lang", "")

        # Compute length reward
        if reference_code and code:
            ref_word_count = len(reference_code.splitlines())
            gen_word_count = len(code.splitlines())

            if ref_word_count > 0:
                length_diff_ratio = (gen_word_count - ref_word_count) / ref_word_count
                if -0.3 < length_diff_ratio <= 0.5:
                    length_reward = 1.0
                elif -0.5 <= length_diff_ratio <= -0.3:
                    length_reward = -0.5
                elif length_diff_ratio < -0.5:
                    length_reward = -2.0
                else:
                    length_reward = 0.0
            else:
                length_reward = 0.0
        else:
            length_reward = 0.0

        try:
            result = calc_codebleu(
                [reference_code],
                [code],
                lang=codebleu_lang,
                weights=(0.1, 0.1, 0.7, 0.1),
                tokenizer=None
            )
            functionality_reward = result['codebleu']
        except Exception as e:
            print(f"CodeBLEU error for sample {idx}:")
            # traceback.print_exc()
            functionality_reward = 0.0
            # breakpoint()
            # import pdb; pdb.set_trace()
        # else:
        #     functionality_reward = 0.0

        extracted_data.append({
            'idx': idx,
            'code': code,
            'metadata': metadata,
            'format_reward': format_reward,
            'length_reward': length_reward,
            'functionality_reward': functionality_reward,
        })

    # Phase 2: Build prompts for vulnerability detection
    vllm_prompts = []
    valid_indices = []

    for data in extracted_data:
        metadata = data['metadata']
        code = data['code']

        # Extract CWE information (note: singular "cwe_name" and "cwe_description")
        cwe_id = metadata.get("cwe_id", [])
        cwe_names = metadata.get("cwe_name", [])
        cwe_descriptions = metadata.get("cwe_description", [])
        language_suffix = metadata.get("language_suffix", "")

        # Handle string vs list types
        if type(cwe_id) == str:
            cwe_id = [cwe_id]
        if type(cwe_names) == str:
            cwe_names = [cwe_names]
        if type(cwe_descriptions) == str:
            cwe_descriptions = [cwe_descriptions]

        # Handle None values
        if cwe_id is None:
            cwe_id = []
        if cwe_names is None:
            cwe_names = []
        if cwe_descriptions is None:
            cwe_descriptions = []

        # Format prompt using text template
        text_prompt = format_prompt_for_vllm(
            code=code,
            cwe_id=cwe_id,
            cwe_names=cwe_names,
            cwe_descriptions=cwe_descriptions,
            language_suffix=language_suffix
        )

        # Apply chat template
        chat_prompt = apply_chat_template(vllm_tokenizer, text_prompt)

        vllm_prompts.append(chat_prompt)
        valid_indices.append(data['idx'])

    # print(vllm_prompts[0])
    # breakpoint()
    # import pdb; pdb.set_trace()
    # Phase 3: Distribute prompts across engines (round-robin for load balancing)
    engine_prompts = [[] for _ in range(num_engines)]
    engine_indices = [[] for _ in range(num_engines)]

    for idx, (prompt, orig_idx) in enumerate(zip(vllm_prompts, valid_indices)):
        engine_idx = idx % num_engines
        engine_prompts[engine_idx].append(prompt)
        engine_indices[engine_idx].append(orig_idx)

    # Print distribution stats
    print(f"Distributing {len(vllm_prompts)} prompts across {num_engines} engine(s):")
    for i, prompts in enumerate(engine_prompts):
        print(f"  Engine {i}: {len(prompts)} prompts")

    # Phase 4: Batch inference on each engine in parallel (data parallelism)
    all_outputs = [None] * len(vllm_prompts)

    def run_engine_inference(engine_idx, engine, prompts, indices):
        """Run inference on a single engine (executed in parallel thread)."""
        if not prompts:
            return engine_idx, []

        print(f"Engine {engine_idx}: Starting VLLM batch inference on {len(prompts)} prompts...")
        outputs = engine.generate(prompts, vllm_sampling_params)
        print(f"Engine {engine_idx}: Completed VLLM batch inference")
        return engine_idx, outputs

    # Run all engines in parallel using ThreadPoolExecutor
    print(f"Running {num_engines} engines in parallel...")
    with ThreadPoolExecutor(max_workers=num_engines) as executor:
        # Submit all engine inference tasks
        future_to_engine = {
            executor.submit(run_engine_inference, engine_idx, engine, prompts, indices): engine_idx
            for engine_idx, (engine, prompts, indices) in enumerate(
                zip(vllm_engines, engine_prompts, engine_indices)
            )
        }

        # Collect results as they complete
        for future in as_completed(future_to_engine):
            engine_idx, outputs = future.result()

            # Map outputs back to original indices
            for local_idx, output in enumerate(outputs):
                original_position = engine_idx + local_idx * num_engines
                all_outputs[original_position] = output

    print(f"All {num_engines} engines completed")

    # Debug: Print 3 pairs of (prompt, reward model response)
    print("\n" + "="*80)
    print("DEBUG: Sample prompt-response pairs from reward model")
    print("="*80)
    num_samples_to_print = min(3, len(all_outputs))
    for i in range(num_samples_to_print):
        if all_outputs[i] is not None:
            print(f"\n--- Sample {i+1} ---")
            print(f"PROMPT (truncated to 500 chars):\n{vllm_prompts[i]}...")
            print(f"\nREWARD MODEL RESPONSE:\n{all_outputs[i].outputs[0].text}")
            print(f"Original code: {reward_inputs[i]['metadata']['generated_code']}")
            print(f"Generated code: {reward_inputs[i]['response']}")
            print("-" * 40)
    print("="*80 + "\n")

    # Phase 5: Parse VLLM responses and compute vulnerability rewards
    vulnerability_results = {}
    for output, idx in zip(all_outputs, valid_indices):
        if output is None:
            print(f"Warning: No output for index {idx}")
            vulnerability_results[idx] = {
                'prediction': 'error',
                'vulnerability_reward': 0.0
            }
            continue

        generated_text = output.outputs[0].text
        prediction = parse_reward_model_answer(generated_text)

        # 1.0 if not vulnerable, 0.0 if vulnerable
        is_not_vulnerable = prediction in ["not vulnerable", "non-vulnerable"]
        vulnerability_reward = 2.0 if is_not_vulnerable else 0.0

        vulnerability_results[idx] = {
            'prediction': prediction,
            'vulnerability_reward': vulnerability_reward
        }

    # Phase 6: Combine all rewards
    final_scores = []
    for data in extracted_data:
        idx = data['idx']
        format_reward = data['format_reward']
        length_reward = data['length_reward']
        functionality_reward = data['functionality_reward']

        vuln_data = vulnerability_results.get(idx, {'vulnerability_reward': 0.0})
        vulnerability_reward = vuln_data['vulnerability_reward']
        # Total reward formula (matching original implementation)
        total_reward = (
            format_reward +
            vulnerability_reward +
            length_reward +
            length_reward * vulnerability_reward +
            length_reward * vulnerability_reward * functionality_reward
        )
        final_scores.append({
            "overall": total_reward,
            "length*vulnerability": length_reward*vulnerability_reward,
            "functionality": functionality_reward,
            "length*vulnerability*functionality": length_reward*vulnerability_reward*functionality_reward,
            "format": format_reward,
            "vulnerability": vulnerability_reward,
            "length": length_reward,
        })

    avg_reward = sum(s['overall'] for s in final_scores) / len(final_scores) if final_scores else 0.0
    print(f"Data-parallel VLLM inference complete. Average reward: {avg_reward:.3f}")

    return final_scores
