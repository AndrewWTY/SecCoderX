#!/usr/bin/env python3
"""
Generate Task Instructions from Scenarios using Gemini Batch API
Reads JSONL with scenarios and generates natural coding task instructions using batch processing.
"""

import argparse
import json
import logging
import os
import random
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
from google import genai
from google.genai import types

# Available programming languages for instruction generation
PROGRAMMING_LANGUAGES = ['c', 'cpp', 'py', 'java', 'js']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str) -> dict:
    """Load prompt templates from YAML file."""
    with open(prompts_path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> Iterator[dict]:
    """Load JSONL file line by line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_history_filename(temp_dir: Path) -> str:
    """
    Generate history filename inside temp directory.

    Args:
        temp_dir: Temp directory for this dataset

    Returns:
        Path to history file

    Example:
        Input: data/temp/with_scenarios/
        Output: data/temp/with_scenarios/sampling_history.jsonl
    """
    history_path = temp_dir / 'sampling_history.jsonl'
    return str(history_path)


def get_temp_dir_for_input(input_file: str, custom_folder_name: Optional[str] = None) -> Path:
    """
    Get temp directory for organizing batch files by input dataset.

    Args:
        input_file: Path to input file
        custom_folder_name: Optional custom folder name under data/temp/
                           If not provided, uses input file stem

    Returns:
        Path to temp directory

    Examples:
        Input: data/output/with_scenarios.jsonl, custom_folder_name=None
        Output: data/temp/with_scenarios/

        Input: data/output/with_scenarios.jsonl, custom_folder_name="experiment_1"
        Output: data/temp/experiment_1/
    """
    if custom_folder_name:
        temp_dir = Path("data/temp") / custom_folder_name
    else:
        input_path = Path(input_file)
        temp_dir = Path("data/temp") / input_path.stem

    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_next_batch_number(temp_dir: Path) -> int:
    """
    Get the next batch number based on existing files in temp directory.

    Args:
        temp_dir: Temp directory for this dataset

    Returns:
        Next batch number (1, 2, 3, ...)
    """
    # Check both requests and results files to find max batch number
    existing_files = list(temp_dir.glob("batch_requests_*.jsonl")) + list(temp_dir.glob("batch_results_*.jsonl"))
    if not existing_files:
        return 1

    batch_numbers = []
    for file in existing_files:
        try:
            num = int(file.stem.split('_')[-1])
            batch_numbers.append(num)
        except ValueError:
            continue

    return max(batch_numbers) + 1 if batch_numbers else 1


def cleanup_batch_files(temp_dir: Path, batch_number: int):
    """
    Clean up batch request and metadata files.

    Args:
        temp_dir: Temp directory for this dataset
        batch_number: Batch number to clean up
    """
    files_to_delete = [
        temp_dir / f"batch_requests_{batch_number}.jsonl",
        temp_dir / f"batch_metadata_{batch_number}.json",
    ]

    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted: {file_path.name}")
        else:
            logger.debug(f"File not found (already deleted): {file_path.name}")


def load_cwe_config(config_file: str) -> Dict[str, int]:
    """
    Load CWE-specific generation counts from a JSON file.

    Args:
        config_file: Path to JSON file with format {"CWE-79": 10, "CWE-89": 20}
                    or {"79": 10, "89": 20}

    Returns:
        Dict mapping normalized CWE IDs to counts
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)

    # Normalize CWE IDs (handle both "CWE-79" and "79" formats)
    normalized_config = {}
    for cwe_id, count in raw_config.items():
        # Remove "CWE-" prefix if present for normalization
        normalized_id = str(cwe_id).upper().replace('CWE-', '').strip()
        normalized_config[normalized_id] = int(count)

    logger.info(f"Loaded CWE config with {len(normalized_config)} CWE types")
    for cwe_id, count in sorted(normalized_config.items()):
        logger.info(f"  CWE-{cwe_id}: {count} samples requested")

    return normalized_config


def load_sampling_history(history_file: str) -> set:
    """
    Load sampling history from file.

    Returns:
        Set of scenario identifiers that have been sampled before
    """
    if not history_file or not Path(history_file).exists():
        logger.info("No sampling history found - starting fresh")
        return set()

    history = set()
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Create unique identifier: record_idx + scenario_id
                identifier = f"{entry['record_idx']}_{entry['scenario_id']}"
                history.add(identifier)

    logger.info(f"Loaded sampling history: {len(history)} scenarios previously sampled")
    return history


def save_sampling_history(history_file: str, sampled_scenarios: List[dict], mode: str = 'append'):
    """
    Save sampled scenarios to history file.

    Args:
        history_file: Path to history file
        sampled_scenarios: List of scenario info dicts
        mode: 'append' to add to existing history, 'write' to overwrite
    """
    if not history_file:
        return

    # Ensure directory exists
    Path(history_file).parent.mkdir(parents=True, exist_ok=True)

    open_mode = 'a' if mode == 'append' else 'w'
    with open(history_file, open_mode, encoding='utf-8') as f:
        for scenario_info in sampled_scenarios:
            f.write(json.dumps(scenario_info, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(sampled_scenarios)} sampled scenarios to history: {history_file}")


def create_batch_request(
    scenario: dict,
    record_metadata: dict,
    prompt_template: str,
    request_key: str
) -> dict:
    """
    Create a single batch request for Gemini Batch API.

    Args:
        scenario: Scenario dict
        record_metadata: Metadata from original record
        prompt_template: Prompt template
        request_key: Unique key for this request

    Returns:
        Batch request dict in Gemini format
    """
    scenario_text = scenario.get('scenario_text', '')
    cwe_id = record_metadata.get('cwe', 'UNKNOWN')
    cwe_name = record_metadata.get('cwe_name', '')
    cwe_description = record_metadata.get('cwe_description', '')
    cwe_extended = record_metadata.get('cwe_extended_description', '')
    complete_cwe_description = cwe_description+' '+cwe_extended
    # Randomly sample a programming language
    sampled_language = random.choice(PROGRAMMING_LANGUAGES)

    # Format prompt with sampled language
    prompt = prompt_template.format(
        scenario=scenario_text,
        cwe_id=cwe_id,
        cwe_name=cwe_name,
        cwe_description=complete_cwe_description,
        language=sampled_language
    )

    # Create batch request in Gemini format
    batch_request = {
        "key": request_key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generation_config": {
                "response_mime_type": "application/json"
            }
        }
    }

    return batch_request, sampled_language


def prepare_batch_requests(
    input_file: str,
    prompt_template: str,
    temp_dir: Path,
    batch_number: int,
    limit: Optional[int] = None,
    history_file: Optional[str] = None,
    cwe_config: Optional[Dict[str, int]] = None
) -> tuple[List[dict], Dict[str, dict], str, List[dict], List[dict]]:
    """
    Prepare all batch requests from input file.

    Args:
        input_file: Input JSONL path (with scenarios)
        prompt_template: Prompt template
        limit: Limit number of instructions to generate (even sampling across CWEs)
        history_file: Path to sampling history file (to avoid duplicates)
        cwe_config: Dict mapping CWE IDs to counts, e.g., {"79": 10, "89": 20}
                   Takes precedence over limit if provided

    Returns:
        Tuple of (batch_requests, metadata_map, batch_file_path, all_records, sampled_scenario_info)
    """
    logger.info(f"Preparing batch requests from {input_file}")

    # Load sampling history
    sampling_history = load_sampling_history(history_file)

    # Load all records
    all_records = list(load_jsonl(input_file))
    logger.info(f"Loaded {len(all_records)} records")

    # Flatten all scenarios with their record metadata
    scenarios_with_metadata = []
    for record_idx, record in enumerate(all_records):
        scenarios = record.get('scenarios', [])

        # Extract and prepare record metadata
        cwe_description = record.get('cwe_description', '')
        cwe_extended = record.get('cwe_extended_description', '')
        full_cwe_description = cwe_description or ''
        if cwe_extended:
            full_cwe_description += ' ' + cwe_extended

        record_metadata = {
            'cwe': record.get('cwe', 'UNKNOWN'),
            'cwe_name': record.get('cwe_name', ''),
            'cwe_description': full_cwe_description.strip(),
            'cve': record.get('cve', ''),
        }

        for scenario in scenarios:
            scenario_id = scenario.get('scenario_id', 0)
            identifier = f"{record_idx}_{scenario_id}"

            # Skip if already sampled
            if identifier in sampling_history:
                continue

            scenarios_with_metadata.append({
                'scenario': scenario,
                'record_idx': record_idx,
                'record_metadata': record_metadata,
                'cwe': record.get('cwe', 'UNKNOWN'),
                'identifier': identifier
            })

    total_scenarios = len(scenarios_with_metadata)
    logger.info(f"Total scenarios available (excluding history): {total_scenarios}")
    if sampling_history:
        logger.info(f"Excluded {len(sampling_history)} previously sampled scenarios")

    # Group scenarios by CWE (needed for both cwe_config and limit modes)
    scenarios_by_cwe = defaultdict(list)
    for item in scenarios_with_metadata:
        # Normalize CWE ID for matching
        cwe_normalized = str(item['cwe']).upper().replace('CWE-', '').strip()
        item['cwe_normalized'] = cwe_normalized
        scenarios_by_cwe[cwe_normalized].append(item)

    num_cwes = len(scenarios_by_cwe)
    logger.info(f"Scenarios span {num_cwes} different CWE types")

    # Determine sampling strategy
    if cwe_config:
        # CWE-specific sampling (takes precedence over limit)
        logger.info("Using CWE-specific config for sampling...")
        scenarios_to_process = []

        for cwe_id, target_count in sorted(cwe_config.items()):
            cwe_scenarios = scenarios_by_cwe.get(cwe_id, [])
            available = len(cwe_scenarios)

            if available == 0:
                logger.warning(f"  CWE-{cwe_id}: No scenarios available (requested {target_count})")
                continue

            # Sample up to target_count, or take all if not enough
            sample_size = min(target_count, available)
            if sample_size < target_count:
                logger.info(f"  CWE-{cwe_id}: Only {available} available, taking all (requested {target_count})")
            else:
                logger.info(f"  CWE-{cwe_id}: Sampling {sample_size}/{available} scenarios")

            sampled = random.sample(cwe_scenarios, sample_size)
            scenarios_to_process.extend(sampled)

        logger.info(f"Total scenarios selected from CWE config: {len(scenarios_to_process)}")

    elif limit:
        # Even sampling across all CWE types
        logger.info(f"Limit provided: {limit} instructions")
        logger.info("Sampling scenarios evenly across CWE types...")

        # Calculate samples per CWE for even distribution
        samples_per_cwe = max(1, limit // num_cwes)
        logger.info(f"Target samples per CWE: {samples_per_cwe}")

        # Sample evenly from each CWE
        scenarios_to_process = []
        for cwe, cwe_scenarios in sorted(scenarios_by_cwe.items()):
            sample_size = min(samples_per_cwe, len(cwe_scenarios))
            sampled = random.sample(cwe_scenarios, sample_size)
            scenarios_to_process.extend(sampled)
            logger.info(f"  CWE-{cwe}: sampled {sample_size}/{len(cwe_scenarios)} scenarios")

        # If we haven't reached the limit, add more samples
        if len(scenarios_to_process) < limit:
            remaining = limit - len(scenarios_to_process)
            logger.info(f"Adding {remaining} more samples to reach limit...")

            # Optimize: use set of IDs for O(1) lookup instead of O(N) list lookup
            selected_ids = {(s['record_idx'], s['scenario']['scenario_id']) for s in scenarios_to_process}
            unsampled = [s for s in scenarios_with_metadata
                        if (s['record_idx'], s['scenario']['scenario_id']) not in selected_ids]

            if unsampled:
                additional = random.sample(unsampled, min(remaining, len(unsampled)))
                scenarios_to_process.extend(additional)

        logger.info(f"Total scenarios selected: {len(scenarios_to_process)}")

    else:
        # No limit, process all scenarios
        scenarios_to_process = scenarios_with_metadata
        logger.info(f"No limit - processing all {total_scenarios} scenarios")

    # Create batch requests
    batch_requests = []
    metadata_map = {}  # Maps request_key to metadata

    for idx, scenario_item in enumerate(scenarios_to_process):
        request_key = f"request_{idx}"
        scenario = scenario_item['scenario']
        record_idx = scenario_item['record_idx']
        record_metadata = scenario_item['record_metadata']

        batch_request, sampled_language = create_batch_request(
            scenario, record_metadata, prompt_template, request_key
        )
        batch_requests.append(batch_request)

        # Store metadata for later mapping
        metadata_map[request_key] = {
            'record_idx': record_idx,
            'scenario': scenario,
            'record_metadata': record_metadata,
            'language': sampled_language
        }

    logger.info(f"Created {len(batch_requests)} batch requests")

    # Build sampled scenario info for history tracking
    sampled_scenario_info = []
    for scenario_item in scenarios_to_process:
        scenario = scenario_item['scenario']
        record_idx = scenario_item['record_idx']
        scenario_id = scenario.get('scenario_id', 0)
        cwe = scenario_item['cwe']

        sampled_scenario_info.append({
            'record_idx': record_idx,
            'scenario_id': scenario_id,
            'cwe': cwe,
            'scenario_text': scenario.get('scenario_text', '')[:100] + '...'  # Truncate for readability
        })

    # Write batch requests to JSONL file (numbered by batch)
    batch_file_path = temp_dir / f"batch_requests_{batch_number}.jsonl"

    with open(batch_file_path, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    logger.info(f"Batch requests saved to: {batch_file_path}")

    # Save metadata_map for later aggregation
    metadata_file_path = temp_dir / f"batch_metadata_{batch_number}.json"
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_map, f, ensure_ascii=False, indent=2)

    logger.info(f"Batch metadata saved to: {metadata_file_path}")

    # Log sampling summary by CWE
    if sampled_scenario_info:
        logger.info("=" * 60)
        logger.info("📊 Sampling Summary:")
        cwe_counts = defaultdict(int)
        for info in sampled_scenario_info:
            cwe_counts[info['cwe']] += 1
        for cwe in sorted(cwe_counts.keys()):
            logger.info(f"  CWE-{cwe}: {cwe_counts[cwe]} scenarios")
        logger.info("=" * 60)

    return batch_requests, metadata_map, batch_file_path, all_records, sampled_scenario_info


def submit_batch_job(
    client: genai.Client,
    batch_file_path: str,
    model: str,
    display_name: str = "instruction_generation"
) -> types.BatchJob:
    """
    Upload batch file and submit batch job to Gemini.

    Args:
        client: Gemini client
        batch_file_path: Path to batch requests JSONL file
        model: Model name
        display_name: Display name for the batch job

    Returns:
        BatchJob object
    """
    logger.info(f"Uploading batch file: {batch_file_path}")

    # Upload file to Gemini File API
    uploaded_file = client.files.upload(
        file=batch_file_path,
        config=types.UploadFileConfig(mime_type='application/jsonl')
    )
    logger.info(f"File uploaded: {uploaded_file.name}")
    logger.info(f"File URI: {uploaded_file.uri}")

    # Create batch job
    logger.info(f"Creating batch job with model: {model}")
    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name,  # FIXED: Use .name instead of .uri
        config=types.CreateBatchJobConfig(
            display_name=display_name
        )
    )

    logger.info(f"Batch job created: {batch_job.name}")
    logger.info(f"Initial state: {batch_job.state.name}")

    return batch_job


def poll_batch_job(
    client: genai.Client,
    job_name: str,
    poll_interval: int = 30
) -> types.BatchJob:
    """
    Poll batch job until completion.

    Args:
        client: Gemini client
        job_name: Batch job name
        poll_interval: Seconds between polls

    Returns:
        Completed BatchJob object
    """
    logger.info(f"Polling batch job: {job_name}")
    logger.info(f"Poll interval: {poll_interval} seconds")

    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name

        logger.info(f"Job state: {state}")

        if state == "JOB_STATE_SUCCEEDED":
            logger.info("✅ Batch job completed successfully!")
            return batch_job
        elif state == "JOB_STATE_FAILED":
            logger.error("❌ Batch job failed!")
            raise RuntimeError(f"Batch job failed: {batch_job}")
        elif state == "JOB_STATE_CANCELLED":
            logger.error("❌ Batch job was cancelled!")
            raise RuntimeError("Batch job was cancelled")
        elif state == "JOB_STATE_EXPIRED":
            logger.error("❌ Batch job expired!")
            raise RuntimeError("Batch job expired (exceeded 48-hour window)")
        elif state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            logger.info(f"Waiting {poll_interval} seconds before next poll...")
            time.sleep(poll_interval)
        else:
            logger.warning(f"Unknown job state: {state}")
            time.sleep(poll_interval)


def download_batch_results(
    client: genai.Client,
    batch_job: types.BatchJob,
    temp_dir: Path,
    batch_number: int
) -> List[dict]:
    """
    Download and parse batch results.

    Args:
        client: Gemini client
        batch_job: Completed BatchJob object
        temp_dir: Temp directory for this dataset
        batch_number: Batch number

    Returns:
        List of result dicts
    """
    logger.info("Downloading batch results...")

    # Generate numbered result file path
    output_path = temp_dir / f"batch_results_{batch_number}.jsonl"

    # Download results file
    result_file_name = batch_job.dest.file_name
    logger.info(f"Result file: {result_file_name}")

    # Download file content using proper API method
    file_content = client.files.download(file=result_file_name)

    # Save raw results
    with open(output_path, 'wb') as f:
        f.write(file_content)

    logger.info(f"Results saved to: {output_path}")

    # Parse JSONL results
    results = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    logger.info(f"Parsed {len(results)} results")

    return results


def parse_batch_results(
    results: List[dict],
    metadata_map: Dict[str, dict]
) -> Dict[int, tuple[List[dict], List[dict]]]:
    """
    Parse batch results and map back to scenarios/instructions.

    Args:
        results: List of result dicts from batch API
        metadata_map: Maps request_key to metadata

    Returns:
        Dict mapping record_idx to (scenarios, instructions)
    """
    logger.info("Parsing batch results...")

    scenarios_by_record = defaultdict(list)
    instructions_by_record = defaultdict(list)

    success_count = 0
    failure_count = 0

    for result in results:
        request_key = result.get('key')

        if request_key not in metadata_map:
            logger.warning(f"Unknown request key: {request_key}")
            continue

        metadata = metadata_map[request_key]
        record_idx = metadata['record_idx']
        scenario = metadata['scenario']
        record_metadata = metadata['record_metadata']
        sampled_language = metadata['language']

        # Add scenario
        scenarios_by_record[record_idx].append(scenario)

        # Check for errors in result
        if 'error' in result:
            error_info = result.get('error', {})
            logger.warning(f"Error in request {request_key}: {error_info}")
            failure_count += 1
            continue

        # Parse response
        response = result.get('response')

        if not response:
            logger.warning(f"No response for request: {request_key}")
            failure_count += 1
            continue

        # Extract text from response
        try:
            candidates = response.get('candidates', [])
            if not candidates:
                logger.warning(f"No candidates in response for request: {request_key}")
                failure_count += 1
                continue

            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            if not parts:
                logger.warning(f"No parts in response for request: {request_key}")
                failure_count += 1
                continue

            text = parts[0].get('text', '')

            # Parse JSON response
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for request: {request_key}")
                failure_count += 1
                continue

            # Extract instruction text
            if isinstance(parsed, dict):
                instruction_text = parsed.get('coding_task_prompt', '')
                design_plan = parsed.get('design_plan', '')
                implementation_language = parsed.get('implementation_language', '')
            else:
                logger.warning(f"Unexpected response format for request: {request_key}")
                failure_count += 1
                continue

            if not instruction_text:
                logger.warning(f"Empty instruction for request: {request_key}")
                failure_count += 1
                continue

            # Create instruction dict
            scenario_id = scenario.get('scenario_id', 0)
            instruction = {
                'scenario_id': scenario_id,
                'scenario_text': scenario.get('scenario_text', ''),
                'instruction_text': instruction_text,
                'design_plan': design_plan,
                'sampled_language': sampled_language,
                'implementation_language': implementation_language,
                'cwe': record_metadata.get('cwe', 'UNKNOWN')
            }

            instructions_by_record[record_idx].append(instruction)
            success_count += 1

        except Exception as e:
            logger.warning(f"Error parsing response for request {request_key}: {e}")
            failure_count += 1
            continue

    logger.info(f"✅ Successfully parsed {success_count} instructions")
    logger.info(f"❌ Failed to parse {failure_count} instructions")

    return scenarios_by_record, instructions_by_record


def aggregate_all_batch_results(temp_dir: Path, all_records: List[dict]) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]]]:
    """
    Aggregate instructions from all previous batch results.

    Args:
        temp_dir: Temp directory containing batch result files
        all_records: All original records

    Returns:
        Tuple of (scenarios_by_record, instructions_by_record) from ALL batches
    """
    logger.info(f"Aggregating results from all batches in: {temp_dir}")

    # Find all batch result files (sort numerically, not lexicographically)
    result_files = sorted(
        temp_dir.glob("batch_results_*.jsonl"),
        key=lambda f: int(f.stem.split('_')[-1])
    )
    if not result_files:
        logger.warning("No previous batch results found")
        return {}, {}

    logger.info(f"Found {len(result_files)} batch result files")

    # Aggregate instructions from all batches
    all_instructions_by_record = defaultdict(list)
    all_scenarios_by_record = defaultdict(list)

    for result_file in result_files:
        logger.info(f"Loading results from: {result_file.name}")

        # Load metadata_map for this batch
        batch_num = int(result_file.stem.split('_')[-1])
        metadata_file = temp_dir / f"batch_metadata_{batch_num}.json"

        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file.name}, skipping batch {batch_num}")
            continue

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_map = json.load(f)

        # Load results
        with open(result_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f if line.strip()]

        # Parse results using metadata
        for result in results:
            request_key = result.get('key')
            if request_key not in metadata_map:
                continue

            metadata = metadata_map[request_key]
            record_idx = metadata['record_idx']
            scenario = metadata['scenario']

            # Check for errors
            if 'error' in result:
                continue

            response = result.get('response')
            if not response:
                continue

            # Extract instruction
            try:
                candidates = response.get('candidates', [])
                if not candidates:
                    continue
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if not parts:
                    continue
                text = parts[0].get('text', '')

                # Parse JSON
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    instruction_text = parsed.get('coding_task_prompt', '')
                    design_plan = parsed.get('design_plan', '')
                    implementation_language = parsed.get('implementation_language', '')
                else:
                    instruction_text = str(parsed)
                    design_plan = ''
                    implementation_language = ''
                if not instruction_text:
                    continue

                # Create instruction
                instruction = {
                    'scenario_id': scenario.get('scenario_id', 0),
                    'scenario_text': scenario.get('scenario_text', ''),
                    'instruction_text': instruction_text,
                    'design_plan': design_plan,
                    'sampled_language': metadata.get('language', ''),
                    'implementation_language': implementation_language,
                    'cwe': metadata['record_metadata'].get('cwe', 'UNKNOWN')
                }

                all_instructions_by_record[record_idx].append(instruction)
                all_scenarios_by_record[record_idx].append(scenario)

            except Exception as e:
                logger.warning(f"Error parsing result for key {request_key}: {e}")
                continue

    logger.info(f"Aggregated {sum(len(v) for v in all_instructions_by_record.values())} total instructions from {len(result_files)} batches")

    return dict(all_scenarios_by_record), dict(all_instructions_by_record)


def write_output(
    output_file: str,
    all_records: List[dict],
    scenarios_by_record: Dict[int, List[dict]],
    instructions_by_record: Dict[int, List[dict]]
):
    """Write final output JSONL with instructions."""
    logger.info(f"Writing output to: {output_file}")

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total_instructions = 0
    processed_records = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record_idx in sorted(scenarios_by_record.keys()):
            record = all_records[record_idx].copy()
            record['scenarios'] = scenarios_by_record[record_idx]
            record['instructions'] = instructions_by_record[record_idx]

            # Update metadata
            if '_metadata' not in record:
                record['_metadata'] = {}
            record['_metadata'].update({
                'num_instructions_requested': len(scenarios_by_record[record_idx]),
                'num_instructions_generated': len(instructions_by_record[record_idx])
            })

            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            out_f.flush()

            processed_records += 1
            total_instructions += len(instructions_by_record[record_idx])

    logger.info(f"✅ Completed! Processed {processed_records} records")
    logger.info(f"Generated {total_instructions} total instructions")
    logger.info(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate task instructions from scenarios using Gemini Batch API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file with scenarios'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='data/output/with_instructions_batch.jsonl',
        help='Path to output JSONL file'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--prompts',
        type=str,
        default='config/prompts.yaml',
        help='Path to prompts file'
    )

    # Batch settings
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-pro',
        help='Gemini model to use'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=30,
        help='Seconds between batch job status polls'
    )

    parser.add_argument(
        '--display-name',
        type=str,
        default='instruction_generation',
        help='Display name for batch job'
    )

    parser.add_argument(
        '--skip-confirmation',
        action='store_true',
        help='Skip confirmation prompt and submit batch job immediately'
    )

    parser.add_argument(
        '--parse-only',
        action='store_true',
        help='Skip batch submission, only aggregate existing results and write output'
    )

    parser.add_argument(
        '--history-file',
        type=str,
        default=None,
        help='Path to sampling history file (default: auto-generated in temp folder)'
    )

    parser.add_argument(
        '--temp-folder-name',
        type=str,
        default=None,
        help='Custom folder name under data/temp/ for batch files (default: input file stem)'
    )

    # Sampling options
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of instructions to generate (sampled evenly across CWEs)'
    )

    parser.add_argument(
        '--cwe-config',
        type=str,
        default=None,
        help='JSON file with CWE-specific counts, e.g., {"CWE-79": 10, "CWE-89": 20}. Takes precedence over --limit'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    # Get temp directory for batch files
    temp_dir = get_temp_dir_for_input(args.input_file, args.temp_folder_name)
    logger.info(f"Using temp directory: {temp_dir}")

    # Auto-generate history file path if not provided (inside temp directory)
    if args.history_file is None:
        args.history_file = get_history_filename(temp_dir)
        logger.info(f"Auto-generated history file: {args.history_file}")
    else:
        logger.info(f"Using specified history file: {args.history_file}")

    # Get next batch number
    batch_number = get_next_batch_number(temp_dir)
    logger.info(f"Batch number: {batch_number}")

    # Load configuration
    try:
        config = load_config(args.config)
        prompts = load_prompts(args.prompts)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        logger.error("Please ensure config.yaml and prompts.yaml exist")
        return 1

    # Load CWE config if provided
    cwe_config = None
    if args.cwe_config:
        try:
            cwe_config = load_cwe_config(args.cwe_config)
        except FileNotFoundError:
            logger.error(f"CWE config file not found: {args.cwe_config}")
            return 1
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in CWE config file: {e}")
            return 1

    # Handle --parse-only mode: skip to aggregation
    if args.parse_only:
        logger.info("=" * 80)
        logger.info("🔄 Parse-only mode: skipping batch submission")
        logger.info("=" * 80)

        # Load all records
        all_records = list(load_jsonl(args.input_file))
        logger.info(f"Loaded {len(all_records)} records from input file")

        # Aggregate ALL results from all batches
        logger.info("Aggregating results from all existing batches...")
        all_scenarios_by_record, all_instructions_by_record = aggregate_all_batch_results(temp_dir, all_records)

        if not all_instructions_by_record:
            logger.warning("No batch results found to aggregate!")
            return 1

        # Write output with ALL aggregated instructions
        write_output(
            output_file=args.output,
            all_records=all_records,
            scenarios_by_record=all_scenarios_by_record,
            instructions_by_record=all_instructions_by_record
        )

        logger.info("🎉 Parse-only complete!")
        return 0

    # Load Gemini API key
    api_key = None
    key_file = "config/gemini_api_key.txt"
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
        logger.info(f"Loaded API key from {key_file}")
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info("Loaded API key from environment variable")

    if not api_key:
        logger.error("API key not found!")
        logger.error(f"Please either:")
        logger.error(f"  1. Create {key_file} with your Gemini API key")
        logger.error(f"  2. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return 1

    # Initialize Gemini client
    try:
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return 1

    # Prepare batch requests
    batch_requests, metadata_map, batch_file_path, all_records, sampled_scenario_info = prepare_batch_requests(
        input_file=args.input_file,
        prompt_template=prompts['instruction_synthesis_prompt'],
        temp_dir=temp_dir,
        batch_number=batch_number,
        limit=args.limit,
        history_file=args.history_file,
        cwe_config=cwe_config
    )

    # Ask user for confirmation before submitting (unless --skip-confirmation)
    logger.info("=" * 80)
    logger.info("📄 Batch request file created!")
    logger.info(f"📁 Location: {batch_file_path}")
    logger.info(f"📊 Total requests: {len(batch_requests)}")
    if args.history_file:
        logger.info(f"📋 History file: {args.history_file}")
    logger.info("=" * 80)

    # Show 3 example formatted prompts
    logger.info("\n" + "=" * 80)
    logger.info("📝 Example Prompts (showing 3 samples):")
    logger.info("=" * 80)
    num_examples = min(3, len(batch_requests))
    for i in range(num_examples):
        request = batch_requests[i]
        key = request.get('key', 'unknown')
        prompt_text = request.get('request', {}).get('contents', [{}])[0].get('parts', [{}])[0].get('text', '')

        logger.info(f"\n{'─' * 80}")
        logger.info(f"[Example {i+1}] Request Key: {key}")
        logger.info(f"{'─' * 80}")
        logger.info(f"\n{prompt_text}\n")
        logger.info("─" * 80)
    logger.info("=" * 80 + "\n")

    if args.skip_confirmation:
        logger.info("⚡ Skipping confirmation (--skip-confirmation flag set)")
        logger.info("✅ Proceeding with batch submission...")
    else:
        logger.info("\n⚠️  Please review the batch file before submission.")
        logger.info("You can inspect the file to verify the requests are correct.\n")

        while True:
            user_input = input("Do you want to submit this batch job? (yes/no): ").strip().lower()
            if user_input in ['yes', 'y']:
                logger.info("✅ User confirmed. Proceeding with batch submission...")
                break
            elif user_input in ['no', 'n']:
                logger.info("❌ User cancelled. Cleaning up batch files...")
                cleanup_batch_files(temp_dir, batch_number)
                logger.info("Batch files deleted. Exiting without submission.")
                # Don't save to history if user cancelled
                return 0
            else:
                logger.warning("Invalid input. Please enter 'yes' or 'no'.")

    # Submit batch job and process (with cleanup on failure)
    try:
        # Submit batch job (don't save history yet - wait for success)
        batch_job = submit_batch_job(
            client=client,
            batch_file_path=batch_file_path,
            model=args.model,
            display_name=args.display_name
        )

        # Poll until completion
        completed_job = poll_batch_job(
            client=client,
            job_name=batch_job.name,
            poll_interval=args.poll_interval
        )

        # Download results
        results = download_batch_results(
            client=client,
            batch_job=completed_job,
            temp_dir=temp_dir,
            batch_number=batch_number
        )

        # Parse results from current batch
        scenarios_by_record, instructions_by_record = parse_batch_results(
            results=results,
            metadata_map=metadata_map
        )

        logger.info(f"Current batch generated {sum(len(v) for v in instructions_by_record.values())} instructions")

        # Save sampled scenarios to history (only after successful completion)
        if sampled_scenario_info and args.history_file:
            logger.info(f"💾 Saving {len(sampled_scenario_info)} sampled scenarios to history...")
            save_sampling_history(args.history_file, sampled_scenario_info, mode='append')
            logger.info("✅ History updated successfully")

        # Aggregate ALL results from all batches
        logger.info("=" * 80)
        logger.info("Aggregating results from all batches...")
        all_scenarios_by_record, all_instructions_by_record = aggregate_all_batch_results(temp_dir, all_records)

        # Write output with ALL aggregated instructions
        write_output(
            output_file=args.output,
            all_records=all_records,
            scenarios_by_record=all_scenarios_by_record,
            instructions_by_record=all_instructions_by_record
        )

        logger.info("🎉 All done!")
        return 0

    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        logger.error("Cleaning up batch files...")
        cleanup_batch_files(temp_dir, batch_number)
        logger.error("Batch files deleted due to failure.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
