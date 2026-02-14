#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate reference answers using Gemini Batch API or vLLM server.

Usage:
    # Option 1: Gemini Batch API
    # Step 1: Prepare the batch request file
    python -m CybersecurityBenchmarks.scripts.generate_reference_answers prepare \
        --prompt-path /path/to/project/PurpleLlama/CybersecurityBenchmarks/datasets/instruct/instruct-v2.json \
        --batch-file /path/to/project/PurpleLlama/CybersecurityBenchmarks/datasets/instruct/batch_requests-v2.jsonl \
        --limit 10  # Optional

    # Step 2: Submit batch, wait for completion, download results, and process
    python -m CybersecurityBenchmarks.scripts.generate_reference_answers submit \
        --prompt-path datasets/instruct/instruct.json \
        --batch-file datasets/instruct/batch_requests.jsonl \
        --output-path datasets/instruct/instruct_with_reference.json \
        --model gemini-2.5-pro \
        --api-key YOUR_API_KEY

    # Option 2: vLLM Server (OpenAI-compatible API)
    python -m CybersecurityBenchmarks.scripts.generate_reference_answers vllm \
        --input datasets/instruct/instruct.json \
        --output datasets/instruct/instruct_with_reference.json \
        --model-name "codellama-7b-instruct" \
        --base-url "http://localhost:8000/v1" \
        --parallel 4
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

LOG = logging.getLogger(__name__)

REFERENCE_CODE_PROMPT = """{test_case_prompt}"""


def extract_code_from_response(response: str) -> str:
    """Extract code from markdown code blocks if present."""
    if not response:
        return ""
    code_blocks = re.findall(r"```(?:\w+)?\n?(.*?)```", response, re.DOTALL)
    return code_blocks[0].strip() if code_blocks else response.strip()


def sanitize_model_name(name: str) -> str:
    """Sanitize model name for use as field suffix.

    Examples:
        "codellama/CodeLlama-7b-Instruct-hf" -> "codellama-7b-instruct-hf"
        "Qwen/Qwen2.5-Coder-7B-Instruct" -> "qwen2.5-coder-7b-instruct"
    """
    # Take the last part after /
    name = name.split("/")[-1]
    # Convert to lowercase and replace underscores with hyphens
    return name.lower().replace("_", "-")


def cmd_vllm(args):
    """Generate reference answers using vLLM server (OpenAI-compatible API)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from tqdm import tqdm

    # Load prompts
    input_path = Path(args.input)
    prompts = json.loads(input_path.read_text())
    LOG.info(f"Loaded {len(prompts)} prompts from {input_path}")

    # Apply limit if specified
    if args.limit is not None:
        prompts = prompts[:args.limit]
        LOG.info(f"Limited to {len(prompts)} samples")

    # Determine field name
    field_name = f"reference_code_{sanitize_model_name(args.model_name)}"
    LOG.info(f"Will save responses to field: {field_name}")

    # Create OpenAI client for vLLM server
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    # Query function
    def query_single(prompt_data):
        test_case_prompt = prompt_data.get("test_case_prompt", "")
        prompt_id = str(prompt_data.get("prompt_id", ""))
        # print(f'test_case_prompt: {test_case_prompt}')
        # breakpoint()
        if not test_case_prompt:
            return prompt_id, ""

        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": test_case_prompt}],
                max_tokens=args.max_tokens,
                temperature=0.0
            )
            text = response.choices[0].message.content
            return prompt_id, extract_code_from_response(text)
        except Exception as e:
            LOG.warning(f"Failed to query prompt {prompt_id}: {e}")
            return prompt_id, ""

    # Generate responses with parallelism
    responses = {}
    print(f"\n{'='*60}")
    print(f"Generating reference answers:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Server: {args.base_url}")
    print(f"  - Parallel requests: {args.parallel}")
    print(f"  - Max tokens: {args.max_tokens}")
    print(f"  - Number of prompts: {len(prompts)}")
    print(f"  - Output field: {field_name}")
    print(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(query_single, p): p for p in prompts}
        for future in tqdm(as_completed(futures), total=len(prompts), desc="Generating"):
            prompt_id, code = future.result()
            responses[prompt_id] = code

    LOG.info(f"Generated {len(responses)} responses")

    # Load existing output file if it exists (to preserve existing reference codes)
    output_path = Path(args.output)
    if output_path.exists():
        existing_prompts = json.loads(output_path.read_text())
        # Create a map from prompt_id to existing data
        existing_map = {str(p.get("prompt_id", "")): p for p in existing_prompts}
        # Merge: update existing prompts with new field
        for prompt_data in prompts:
            prompt_id = str(prompt_data.get("prompt_id", ""))
            if prompt_id in existing_map:
                # Update existing entry
                existing_map[prompt_id][field_name] = responses.get(prompt_id, "")
            else:
                # Add new entry with the field
                prompt_data[field_name] = responses.get(prompt_id, "")
                existing_map[prompt_id] = prompt_data
        # Convert back to list and sort
        prompts = list(existing_map.values())
    else:
        # No existing file, add field to prompts
        for prompt_data in prompts:
            prompt_id = str(prompt_data.get("prompt_id", ""))
            prompt_data[field_name] = responses.get(prompt_id, "")

    # Sort by prompt_id
    prompts.sort(key=lambda x: x.get("prompt_id", 0))

    # Save output
    output_path.write_text(json.dumps(prompts, indent=2))
    LOG.info(f"Saved {len(prompts)} prompts to {output_path}")

    # Print stats
    num_with_reference = sum(1 for p in prompts if p.get(field_name))
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  - Generated: {num_with_reference}/{len(prompts)} reference answers")
    print(f"  - Output file: {output_path}")
    print(f"  - Field name: {field_name}")
    print(f"{'='*60}\n")


def cmd_prepare(args):
    """Prepare batch request JSONL file."""
    # Load prompts
    prompt_path = Path(args.prompt_path)
    prompts = json.loads(prompt_path.read_text())
    LOG.info(f"Loaded {len(prompts)} prompts from {prompt_path}")

    # Apply limit if specified
    if args.limit is not None:
        prompts = prompts[:args.limit]
        LOG.info(f"Limited to {len(prompts)} samples")

    # Create batch requests in JSONL format
    batch_file = Path(args.batch_file)
    count = 0
    with open(batch_file, 'w') as f:
        for prompt_data in prompts:
            test_case_prompt = prompt_data.get("test_case_prompt", "")
            if not test_case_prompt:
                continue

            reference_prompt = REFERENCE_CODE_PROMPT.format(test_case_prompt=test_case_prompt)
            request = {
                "key": str(prompt_data.get("prompt_id", "")),
                "request": {
                    "contents": [{
                        "parts": [{"text": reference_prompt}],
                        "role": "user"
                    }]
                }
            }
            f.write(json.dumps(request) + '\n')
            count += 1

    LOG.info(f"Wrote {count} batch requests to {batch_file}")

    # Show sample
    print(f"\n{'='*60}")
    print(f"Batch request file prepared:")
    print(f"  - File: {batch_file}")
    print(f"  - Requests: {count}")
    print(f"{'='*60}")
    print(f"\nSample request (first line):")
    with open(batch_file, 'r') as f:
        first_line = f.readline()
        print(json.dumps(json.loads(first_line), indent=2)[:500] + "...")
    print(f"\n{'='*60}")
    print("Next step:")
    print("  Run 'submit' command to upload, submit, and process results")
    print(f"{'='*60}")


def cmd_submit(args):
    """Submit batch job, poll for completion, download and process results."""
    from google import genai
    from google.genai import types

    # Load original prompts
    prompt_path = Path(args.prompt_path)
    prompts = json.loads(prompt_path.read_text())
    LOG.info(f"Loaded {len(prompts)} prompts from {prompt_path}")

    # Apply limit if specified
    if args.limit is not None:
        prompts = prompts[:args.limit]
        LOG.info(f"Limited to {len(prompts)} samples")

    # Create client
    client = genai.Client(api_key=args.api_key)
    LOG.info(f"Using model: {args.model}")

    # Check batch file exists
    batch_file = Path(args.batch_file)
    if not batch_file.exists():
        LOG.error(f"Batch file not found: {batch_file}")
        LOG.error("Run 'prepare' command first to create the batch file")
        return

    # Count requests
    with open(batch_file, 'r') as f:
        num_requests = sum(1 for line in f if line.strip())

    # Prompt user for confirmation
    print(f"\n{'='*60}")
    print(f"Ready to submit batch job:")
    print(f"  - Model: {args.model}")
    print(f"  - Batch file: {batch_file}")
    print(f"  - Number of requests: {num_requests}")
    print(f"  - Output path: {args.output_path}")
    print(f"{'='*60}")
    confirm = input("Proceed with batch submission? [y/N]: ").strip().lower()
    if confirm != 'y':
        LOG.info("Batch submission cancelled by user.")
        return

    # Upload file
    LOG.info(f"Uploading batch file: {batch_file}")
    uploaded_file = client.files.upload(
        file=str(batch_file),
        config=types.UploadFileConfig(display_name='batch_requests', mime_type='application/jsonl')
    )
    LOG.info(f"Uploaded file: {uploaded_file.name}")

    # Submit batch job
    LOG.info("Submitting batch job...")
    batch_job = client.batches.create(
        model=f"models/{args.model}",
        src=uploaded_file.name,
        config={"display_name": "reference-answers-generation"},
    )
    LOG.info(f"Batch job created: {batch_job.name}")

    # Poll for completion
    LOG.info("Polling for batch job completion...")
    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:
        batch_job = client.batches.get(name=batch_job.name)
        state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
        LOG.info(f"Batch job state: {state}")

        if state in completed_states:
            break

        LOG.info(f"Waiting {args.poll_interval}s before next poll...")
        time.sleep(args.poll_interval)

    if state != "JOB_STATE_SUCCEEDED":
        LOG.error(f"Batch job failed with state: {state}")
        return

    # Extract results
    LOG.info("Extracting results...")
    response_map = {}

    # Check for inlined responses
    if batch_job.dest and hasattr(batch_job.dest, 'inlined_responses') and batch_job.dest.inlined_responses:
        LOG.info("Processing inlined responses...")
        for resp in batch_job.dest.inlined_responses:
            key = resp.key if hasattr(resp, 'key') else None
            if key and hasattr(resp, 'response'):
                try:
                    text = resp.response.candidates[0].content.parts[0].text
                    response_map[key] = extract_code_from_response(text)
                except (IndexError, AttributeError) as e:
                    LOG.warning(f"Failed to extract response for key {key}: {e}")
                    response_map[key] = ""

    # Check for file-based responses
    elif batch_job.dest and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
        LOG.info(f"Downloading results from file: {batch_job.dest.file_name}")
        file_content = client.files.download(file=batch_job.dest.file_name)

        # Save response file for debugging
        response_file = Path(args.output_path).parent / "batch_responses.jsonl"
        response_file.write_bytes(file_content)
        LOG.info(f"Saved raw responses to: {response_file}")

        # Parse JSONL responses
        for line in file_content.decode('utf-8').strip().split('\n'):
            if not line:
                continue
            resp_data = json.loads(line)
            key = resp_data.get("key")
            if key:
                try:
                    text = resp_data["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    response_map[key] = extract_code_from_response(text)
                except (KeyError, IndexError) as e:
                    LOG.warning(f"Failed to extract response for key {key}: {e}")
                    response_map[key] = ""

    LOG.info(f"Extracted {len(response_map)} responses")

    # Map responses back to prompts
    for prompt_data in prompts:
        prompt_id = str(prompt_data.get("prompt_id", ""))
        prompt_data["reference_code"] = response_map.get(prompt_id, "")

    # Sort by prompt_id to maintain order
    prompts.sort(key=lambda x: x.get("prompt_id", 0))

    # Save output
    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(prompts, indent=2))
    LOG.info(f"Saved {len(prompts)} prompts with reference answers to {output_path}")

    # Print stats
    num_with_reference = sum(1 for p in prompts if p.get("reference_code"))
    LOG.info(f"Successfully generated {num_with_reference}/{len(prompts)} reference answers")


def main():
    parser = argparse.ArgumentParser(description="Generate reference answers using Gemini Batch API")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare batch request JSONL file")
    prepare_parser.add_argument("--prompt-path", required=True, help="Path to instruct.json")
    prepare_parser.add_argument("--batch-file", required=True, help="Output path for batch request JSONL")
    prepare_parser.add_argument("--limit", type=int, default=None, help="Limit to N samples")

    # Submit command (upload, submit, poll, download, process)
    submit_parser = subparsers.add_parser("submit", help="Submit batch, wait, download and process results")
    submit_parser.add_argument("--prompt-path", required=True, help="Path to original instruct.json")
    submit_parser.add_argument("--batch-file", required=True, help="Path to batch request JSONL")
    submit_parser.add_argument("--output-path", required=True, help="Output path for merged JSON")
    submit_parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model to use")
    submit_parser.add_argument("--api-key", required=True, help="Google API key")
    submit_parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status polls")
    submit_parser.add_argument("--limit", type=int, default=None, help="Limit to N samples (must match prepare)")

    # vLLM command (generate using vLLM server)
    vllm_parser = subparsers.add_parser("vllm", help="Generate references using vLLM server (OpenAI-compatible API)")
    vllm_parser.add_argument("--input", required=True, help="Input JSON file (e.g., instruct.json)")
    vllm_parser.add_argument("--output", required=True, help="Output JSON file")
    vllm_parser.add_argument("--model-name", required=True, help="Model name for API calls and field naming")
    vllm_parser.add_argument("--base-url", required=True, help="vLLM server URL (e.g., http://localhost:8000/v1)")
    vllm_parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for generation")
    vllm_parser.add_argument("--limit", type=int, default=None, help="Limit to N samples")
    vllm_parser.add_argument("--parallel", type=int, default=1, help="Number of parallel requests")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "vllm":
        cmd_vllm(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
