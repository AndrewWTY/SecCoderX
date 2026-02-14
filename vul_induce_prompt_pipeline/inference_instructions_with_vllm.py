#!/usr/bin/env python3
"""
Run VLLM offline inference on flattened instructions dataset.
Uses the instruction_text field as the prompt for code generation.
"""

import argparse
import json
import re
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Default template path - can be overridden via argument
DEFAULT_TEMPLATE_PATH = "/path/to/project/EasyR1/examples/format_prompt/r2vul.jinja"


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_prompt_template(template_path: str) -> Template:
    """Load the Jinja2 prompt template."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return Template(f.read())


def extract_code_from_response(response: str) -> str:
    """Extract the code from between <code> and </code> tags.

    Args:
        response: The model's response containing code

    Returns:
        Extracted code string, or empty string if no code tags found
    """
    matches = re.findall(r'<code>(.*?)</code>', response, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        # Match triple backtick blocks that may optionally have a language label (like ```python\n)
        matches = re.findall(r'```[\w]*\n(.*?)```', response, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip()
    return ""  # Return empty string if no code tags found


def run_inference_batch(
    dataset: List[Dict[str, Any]],
    model_name: str,
    template_path: str,
    use_template: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    append_additional_instruction: bool = False,
    max_prompt_tokens: int = None,
    disable_thinking: bool = False
) -> tuple:
    """Run offline batch inference using VLLM.

    Args:
        dataset: List of dataset samples with 'instruction_text' field
        model_name: Model name/path for VLLM
        template_path: Path to Jinja2 template
        use_template: Whether to use Jinja2 template or use instruction_text directly
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization fraction
        append_additional_instruction: Append additional instruction to the prompt
        max_prompt_tokens: Maximum prompt tokens allowed (entries exceeding this are skipped)
        disable_thinking: Disable thinking mode for Qwen3 models

    Returns:
        Tuple of (responses, valid_indices, skipped_indices)
    """
    print(f"Loading model: {model_name}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")

    # Load tokenizer for chat template
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Initialize VLLM model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True
    )

    # Get max_model_len from the loaded model
    model_max_len = llm.llm_engine.model_config.max_model_len
    print(f"Model max context length: {model_max_len}")

    # Default max_prompt_tokens to max_model_len if not specified
    if max_prompt_tokens is None:
        max_prompt_tokens = model_max_len
        print(f"Using model's max_model_len as max_prompt_tokens: {max_prompt_tokens}")

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )

    # Format prompts for all samples
    print("Formatting prompts...")
    all_prompts = []

    if use_template:
        template = load_prompt_template(template_path)
        for sample in tqdm(dataset, desc="Formatting"):
            instruction_text = sample.get('instruction_text', '')
            # First render the content with jinja template
            content = template.render(content=instruction_text)

            # Add additional instruction if requested
            if append_additional_instruction:
                content += "\n\nOutput the code strictly in the following format: ```[your code]```"

            # Create messages and apply chat template
            messages = [{"role": "user", "content": content}]
            chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if disable_thinking:
                chat_template_kwargs["enable_thinking"] = False
            prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            all_prompts.append(prompt)
    else:
        # Use instruction_text directly without template
        for sample in tqdm(dataset, desc="Formatting"):
            instruction_text = sample.get('instruction_text', '')

            # Add additional instruction if requested
            if append_additional_instruction:
                instruction_text += "\n\nOutput the code strictly in the following format: ```[your code]```"

            # Apply chat template even without jinja template
            messages = [{"role": "user", "content": instruction_text}]
            chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if disable_thinking:
                chat_template_kwargs["enable_thinking"] = False
            prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            all_prompts.append(prompt)

    # Filter prompts by token length (max_prompt_tokens defaults to model's max_model_len)
    valid_indices = []
    skipped_indices = []
    prompts = []

    print(f"\nFiltering prompts exceeding {max_prompt_tokens} tokens...")
    for idx, prompt in enumerate(tqdm(all_prompts, desc="Checking token lengths")):
        token_count = len(tokenizer.encode(prompt))
        if token_count <= max_prompt_tokens:
            valid_indices.append(idx)
            prompts.append(prompt)
        else:
            skipped_indices.append(idx)

    print(f"\n{'=' * 60}")
    print(f"PROMPT FILTERING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Valid prompts: {len(valid_indices)}")
    print(f"Skipped prompts (exceeding {max_prompt_tokens} tokens): {len(skipped_indices)}")
    print(f"{'=' * 60}\n")

    if not prompts:
        print("No valid prompts to process after filtering!")
        return [], valid_indices, skipped_indices

    # Run batch inference
    print("\nSample prompts:")
    print("=" * 80)
    print(prompts[0])
    print("=" * 80)
    if len(prompts) > 1:
        print(prompts[1])
        print("=" * 80)

    print(f"\nRunning inference on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Extract generated texts
    responses = [output.outputs[0].text for output in outputs]

    return responses, valid_indices, skipped_indices


def main():
    parser = argparse.ArgumentParser(
        description='Run VLLM offline inference on flattened instructions dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (from flatten_instructions.py)'
    )

    # Model
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2.5-Coder-7B-Instruct',
        help='Model name/path for VLLM'
    )
    parser.add_argument(
        '--template-path',
        type=str,
        default=DEFAULT_TEMPLATE_PATH,
        help='Path to Jinja2 prompt template'
    )
    parser.add_argument(
        '--no-template',
        action='store_true',
        help='Use instruction_text directly without Jinja2 template'
    )

    # Generation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--append-additional-instruction',
        action='store_true',
        default=False,
        help='Append additional instruction to the prompt'
    )

    # Hardware
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism'
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization fraction (0.0-1.0)'
    )

    # Output
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path to save results as dataset (optional)'
    )
    parser.add_argument(
        '--output-jsonl',
        type=str,
        default=None,
        help='Output path to save results as JSONL (optional)'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push results to HuggingFace Hub'
    )
    parser.add_argument(
        '--output-repo',
        type=str,
        default=None,
        help='Output HuggingFace repo name (required if --push-to-hub)'
    )

    # Prompt filtering
    parser.add_argument(
        '--max-prompt-tokens',
        type=int,
        default=None,
        help='Maximum prompt tokens allowed. Entries exceeding this are skipped and reported.'
    )

    # Qwen3 thinking mode
    parser.add_argument(
        '--disable-thinking',
        action='store_true',
        default=False,
        help='Disable thinking mode for Qwen3 models'
    )

    # Testing
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )

    args = parser.parse_args()

    # Load dataset into pandas DataFrame
    print(f"Loading dataset from: {args.input}")
    df = pd.read_json(args.input, lines=True)

    # Limit samples if specified
    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples for testing")
        df = df.head(args.max_samples)

    # Convert to list of dicts for processing
    dataset_list = df.to_dict('records')

    print(f"\nProcessing {len(dataset_list)} samples")
    print(f"Model: {args.model_name}")
    print(f"Template: {'Direct instruction' if args.no_template else args.template_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")

    # Verify instruction_text field exists
    if 'instruction_text' not in dataset_list[0]:
        raise ValueError("Dataset must have 'instruction_text' field. Did you run flatten_instructions.py?")

    # Run offline batch inference
    responses, valid_indices, skipped_indices = run_inference_batch(
        dataset_list,
        args.model_name,
        args.template_path,
        use_template=not args.no_template,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        append_additional_instruction=args.append_additional_instruction,
        max_prompt_tokens=args.max_prompt_tokens,
        disable_thinking=args.disable_thinking
    )

    # Filter DataFrame to only valid entries
    df_valid = df.iloc[valid_indices].copy()

    # Extract code from responses
    print("\nExtracting code from responses...")
    extracted_codes = []
    for response in tqdm(responses, desc="Extracting code"):
        code = extract_code_from_response(response)
        extracted_codes.append(code)

    # Create column name from model name (simplify the name)
    column_name = args.model_name.replace('/', '_').replace('-', '_').replace('.', '_')

    # Add extracted codes to DataFrame
    print(f"\nAdding results to DataFrame")
    print(f"  - Generated code column: generated_code_{column_name}")
    print(f"  - Raw response column: response_{column_name}")

    df_valid[f"generated_code_{column_name}"] = extracted_codes
    df_valid[f"response_{column_name}"] = responses

    # Save results
    if args.output_jsonl:
        print(f"Saving to JSONL: {args.output_jsonl}")
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        df_valid.to_json(args.output_jsonl, orient='records', lines=True, force_ascii=False)

    if args.output_path:
        print(f"Saving DataFrame to pickle: {args.output_path}")
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        df_valid.to_pickle(args.output_path)

    if args.push_to_hub:
        print("Warning: --push-to-hub not supported with pandas backend. Use --output-jsonl instead.")

    print("\n=== Done! ===")
    print(f"Total samples in input: {len(dataset_list)}")
    print(f"Samples processed: {len(responses)}")
    print(f"Samples skipped (prompt too long): {len(skipped_indices)}")
    if responses:
        print(f"\nSample instruction:")
        print(dataset_list[valid_indices[0]]['instruction_text'])
        print("\nSample extracted code:")
        print(extracted_codes[0])


if __name__ == "__main__":
    main()
