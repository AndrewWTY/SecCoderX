#!/usr/bin/env python3
"""
Generate Scenarios from Vulnerable Code using Gemini Batch API
Reads JSONL with vulnerable code and generates realistic application scenarios using batch processing.
"""

import argparse
import json
import logging
import os
import random
import time
import yaml
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
from google import genai
from google.genai import types

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
    """
    history_path = temp_dir / 'processing_history.jsonl'
    return str(history_path)


def get_temp_dir_for_input(input_file: str, custom_folder_name: Optional[str] = None) -> Path:
    """
    Get temp directory for organizing batch files by input dataset.

    Args:
        input_file: Path to input file
        custom_folder_name: Optional custom folder name under data/temp_scenarios/
                           If not provided, uses input file stem

    Returns:
        Path to temp directory
    """
    if custom_folder_name:
        temp_dir = Path("data/temp_scenarios") / custom_folder_name
    else:
        input_path = Path(input_file)
        temp_dir = Path("data/temp_scenarios") / input_path.stem

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


def load_processing_history(history_file: str) -> set:
    """
    Load processing history from file.

    Returns:
        Set of record indices that have been processed before
    """
    if not history_file or not Path(history_file).exists():
        logger.info("No processing history found - starting fresh")
        return set()

    history = set()
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                history.add(entry['record_idx'])

    logger.info(f"Loaded processing history: {len(history)} records previously processed")
    return history


def save_processing_history(history_file: str, processed_records: List[dict], mode: str = 'append'):
    """
    Save processed records to history file.

    Args:
        history_file: Path to history file
        processed_records: List of record info dicts
        mode: 'append' to add to existing history, 'write' to overwrite
    """
    if not history_file:
        return

    # Ensure directory exists
    Path(history_file).parent.mkdir(parents=True, exist_ok=True)

    open_mode = 'a' if mode == 'append' else 'w'
    with open(history_file, open_mode, encoding='utf-8') as f:
        for record_info in processed_records:
            f.write(json.dumps(record_info, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(processed_records)} processed records to history: {history_file}")


def calculate_cwe_counts(records: List[dict]) -> Dict[str, int]:
    """
    Calculate N_CWE for each CWE in the dataset.

    Args:
        records: List of all records

    Returns:
        Dict mapping CWE ID to count
    """
    logger.info("Calculating CWE counts...")
    cwe_counts = Counter()

    for record in records:
        cwe = record.get('cwe', 'UNKNOWN')
        cwe_counts[cwe] += 1

    logger.info(f"Found {len(cwe_counts)} unique CWEs")
    logger.info(f"Top 5 CWEs: {cwe_counts.most_common(5)}")

    return dict(cwe_counts)


def determine_scenario_count(
    cwe: str,
    cwe_counts: Dict[str, int],
    threshold: int,
    scenarios_above: int,
    scenarios_below: int,
    no_threshold: bool
) -> int:
    """
    Determine how many scenarios to generate based on CWE frequency.

    Args:
        cwe: CWE identifier
        cwe_counts: Dict of CWE counts
        threshold: N_CWE threshold
        scenarios_above: Count when N_CWE >= threshold
        scenarios_below: Count when N_CWE < threshold
        no_threshold: If True, always return scenarios_below

    Returns:
        Number of scenarios to generate
    """
    if no_threshold:
        return scenarios_below

    n_cwe = cwe_counts.get(cwe, 0)

    if n_cwe >= threshold:
        return scenarios_above
    else:
        return scenarios_below


def create_batch_request(
    record: dict,
    record_idx: int,
    prompt_template: str,
    num_scenarios: int,
    request_key: str
) -> dict:
    """
    Create a single batch request for Gemini Batch API.

    Args:
        record: Vulnerability record
        record_idx: Record index
        prompt_template: Prompt template
        num_scenarios: Number of scenarios to generate
        request_key: Unique key for this request

    Returns:
        Batch request dict in Gemini format
    """
    code = record.get('code', '')
    language = record.get('language_suffix', 'unknown')

    # Format prompt
    prompt = prompt_template.format(
        num_scenarios=num_scenarios,
        code_snippet=code,
        language=language
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

    return batch_request


def prepare_batch_requests(
    input_file: str,
    prompt_template: str,
    temp_dir: Path,
    batch_number: int,
    cwe_counts: Dict[str, int],
    threshold: int,
    scenarios_above: int,
    scenarios_below: int,
    no_threshold: bool,
    limit: Optional[int] = None,
    history_file: Optional[str] = None
) -> Tuple[List[dict], Dict[str, dict], Path, List[dict], List[dict]]:
    """
    Prepare all batch requests from input file.

    Args:
        input_file: Input JSONL path (vulnerable code)
        prompt_template: Prompt template
        temp_dir: Temp directory for batch files
        batch_number: Current batch number
        cwe_counts: CWE frequency counts
        threshold: N_CWE threshold
        scenarios_above: Scenarios when above threshold
        scenarios_below: Scenarios when below threshold
        no_threshold: Disable threshold
        limit: Limit number of records to process
        history_file: Path to processing history file

    Returns:
        Tuple of (batch_requests, metadata_map, batch_file_path, all_records, processed_record_info)
    """
    logger.info(f"Preparing batch requests from {input_file}")

    # Load processing history
    processing_history = load_processing_history(history_file)

    # Load all records
    all_records = list(load_jsonl(input_file))
    logger.info(f"Loaded {len(all_records)} records")

    # Filter out already processed records
    records_to_process = []
    for record_idx, record in enumerate(all_records):
        if record_idx not in processing_history:
            records_to_process.append((record_idx, record))

    logger.info(f"Records available (excluding history): {len(records_to_process)}")
    if processing_history:
        logger.info(f"Excluded {len(processing_history)} previously processed records")

    # Apply limit if provided
    if limit and len(records_to_process) > limit:
        records_to_process = records_to_process[:limit]
        logger.info(f"Limited to {limit} records")

    # Create batch requests
    batch_requests = []
    metadata_map = {}
    processed_record_info = []

    for record_idx, record in records_to_process:
        request_key = f"request_{record_idx}"
        cwe = record.get('cwe', 'UNKNOWN')
        n_cwe = cwe_counts.get(cwe, 0)

        # Determine scenario count
        num_scenarios = determine_scenario_count(
            cwe, cwe_counts, threshold, scenarios_above, scenarios_below, no_threshold
        )

        batch_request = create_batch_request(
            record, record_idx, prompt_template, num_scenarios, request_key
        )
        batch_requests.append(batch_request)

        # Store metadata for later mapping
        metadata_map[request_key] = {
            'record_idx': record_idx,
            'cwe': cwe,
            'n_cwe': n_cwe,
            'num_scenarios_requested': num_scenarios
        }

        # Track processed record info for history
        processed_record_info.append({
            'record_idx': record_idx,
            'cwe': cwe,
            'num_scenarios_requested': num_scenarios
        })

    logger.info(f"Created {len(batch_requests)} batch requests")

    # Write batch requests to JSONL file
    batch_file_path = temp_dir / f"batch_requests_{batch_number}.jsonl"

    logger.info(f"Writing {len(batch_requests)} batch requests to file...")
    with open(batch_file_path, 'w', encoding='utf-8') as f:
        content = '\n'.join(json.dumps(request, ensure_ascii=False) for request in batch_requests) + '\n'
        f.write(content)

    logger.info(f"Batch requests saved to: {batch_file_path}")

    # Save metadata_map for later aggregation
    metadata_file_path = temp_dir / f"batch_metadata_{batch_number}.json"
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_map, f, ensure_ascii=False, indent=2)

    logger.info(f"Batch metadata saved to: {metadata_file_path}")

    # Log summary by CWE
    if processed_record_info:
        logger.info("=" * 60)
        logger.info("Processing Summary:")
        cwe_request_counts = defaultdict(int)
        for info in processed_record_info:
            cwe_request_counts[info['cwe']] += 1
        for cwe in sorted(cwe_request_counts.keys()):
            logger.info(f"  CWE-{cwe}: {cwe_request_counts[cwe]} records")
        logger.info("=" * 60)

    return batch_requests, metadata_map, batch_file_path, all_records, processed_record_info


def submit_batch_job(
    client: genai.Client,
    batch_file_path: str,
    model: str,
    display_name: str = "scenario_generation"
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
        src=uploaded_file.name,
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
            logger.info("Batch job completed successfully!")
            return batch_job
        elif state == "JOB_STATE_FAILED":
            logger.error("Batch job failed!")
            raise RuntimeError(f"Batch job failed: {batch_job}")
        elif state == "JOB_STATE_CANCELLED":
            logger.error("Batch job was cancelled!")
            raise RuntimeError("Batch job was cancelled")
        elif state == "JOB_STATE_EXPIRED":
            logger.error("Batch job expired!")
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

    # Download file content
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
) -> Dict[int, List[dict]]:
    """
    Parse batch results and extract scenarios.

    Args:
        results: List of result dicts from batch API
        metadata_map: Maps request_key to metadata

    Returns:
        Dict mapping record_idx to list of scenarios
    """
    logger.info("Parsing batch results...")

    scenarios_by_record = defaultdict(list)

    success_count = 0
    failure_count = 0

    for result in results:
        request_key = result.get('key')

        if request_key not in metadata_map:
            logger.warning(f"Unknown request key: {request_key}")
            continue

        metadata = metadata_map[request_key]
        record_idx = metadata['record_idx']
        n_cwe = metadata['n_cwe']

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

            # Extract scenarios
            if isinstance(parsed, list):
                scenario_list = parsed
            elif isinstance(parsed, dict) and 'scenarios' in parsed:
                scenario_list = parsed['scenarios']
            elif isinstance(parsed, dict):
                scenario_list = [parsed]
            else:
                logger.warning(f"Unexpected response format for request: {request_key}")
                failure_count += 1
                continue

            # Process each scenario
            for item in scenario_list:
                if isinstance(item, dict):
                    scenario_id = item.get('scenario_id', len(scenarios_by_record[record_idx]) + 1)
                    scenario_text = item.get('scenario', item.get('scenario_text', ''))

                    if scenario_text:
                        scenarios_by_record[record_idx].append({
                            'scenario_id': scenario_id,
                            'scenario_text': scenario_text,
                            'n_cwe_count': n_cwe
                        })
                        success_count += 1

        except Exception as e:
            logger.warning(f"Error parsing response for request {request_key}: {e}")
            failure_count += 1
            continue

    logger.info(f"Successfully parsed {success_count} scenarios")
    logger.info(f"Failed to parse {failure_count} requests")

    return dict(scenarios_by_record)


def aggregate_all_batch_results(temp_dir: Path, all_records: List[dict]) -> Dict[int, List[dict]]:
    """
    Aggregate scenarios from all previous batch results.

    Args:
        temp_dir: Temp directory containing batch result files
        all_records: All original records

    Returns:
        Dict mapping record_idx to list of scenarios from ALL batches
    """
    logger.info(f"Aggregating results from all batches in: {temp_dir}")

    # Find all batch result files (sort numerically, not lexicographically)
    result_files = sorted(
        temp_dir.glob("batch_results_*.jsonl"),
        key=lambda f: int(f.stem.split('_')[-1])
    )
    if not result_files:
        logger.warning("No previous batch results found")
        return {}

    logger.info(f"Found {len(result_files)} batch result files")

    # Aggregate scenarios from all batches
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
            n_cwe = metadata['n_cwe']

            # Check for errors
            if 'error' in result:
                continue

            response = result.get('response')
            if not response:
                continue

            # Extract scenarios
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

                if isinstance(parsed, list):
                    scenario_list = parsed
                elif isinstance(parsed, dict) and 'scenarios' in parsed:
                    scenario_list = parsed['scenarios']
                elif isinstance(parsed, dict):
                    scenario_list = [parsed]
                else:
                    continue

                for item in scenario_list:
                    if isinstance(item, dict):
                        scenario_id = item.get('scenario_id', len(all_scenarios_by_record[record_idx]) + 1)
                        scenario_text = item.get('scenario', item.get('scenario_text', ''))

                        if scenario_text:
                            all_scenarios_by_record[record_idx].append({
                                'scenario_id': scenario_id,
                                'scenario_text': scenario_text,
                                'n_cwe_count': n_cwe
                            })

            except Exception as e:
                logger.warning(f"Error parsing result for key {request_key}: {e}")
                continue

    total_scenarios = sum(len(v) for v in all_scenarios_by_record.values())
    logger.info(f"Aggregated {total_scenarios} total scenarios from {len(result_files)} batches")

    return dict(all_scenarios_by_record)


def write_output(
    output_file: str,
    all_records: List[dict],
    scenarios_by_record: Dict[int, List[dict]]
):
    """Write final output JSONL with scenarios."""
    logger.info(f"Writing output to: {output_file}")

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total_scenarios = 0
    processed_records = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record_idx in sorted(scenarios_by_record.keys()):
            record = all_records[record_idx].copy()
            record['scenarios'] = scenarios_by_record[record_idx]

            # Update metadata
            if '_metadata' not in record:
                record['_metadata'] = {}
            record['_metadata'].update({
                'record_index': record_idx,
                'num_scenarios_generated': len(scenarios_by_record[record_idx])
            })

            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            out_f.flush()

            processed_records += 1
            total_scenarios += len(scenarios_by_record[record_idx])

    logger.info(f"Completed! Processed {processed_records} records")
    logger.info(f"Generated {total_scenarios} total scenarios")
    logger.info(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate scenarios from vulnerable code using Gemini Batch API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file with vulnerable code'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='data/output/with_scenarios_batch.jsonl',
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

    # Threshold settings
    parser.add_argument(
        '--threshold',
        type=int,
        default=None,
        help='CWE count threshold (overrides config)'
    )

    parser.add_argument(
        '--scenarios-above',
        type=int,
        default=None,
        help='Number of scenarios when N_CWE >= threshold'
    )

    parser.add_argument(
        '--scenarios-below',
        type=int,
        default=None,
        help='Number of scenarios when N_CWE < threshold'
    )

    parser.add_argument(
        '--no-threshold',
        action='store_true',
        help='Disable threshold and always generate scenarios-below count'
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
        default='scenario_generation',
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
        help='Path to processing history file (default: auto-generated in temp folder)'
    )

    parser.add_argument(
        '--temp-folder-name',
        type=str,
        default=None,
        help='Custom folder name under data/temp_scenarios/ for batch files (default: input file stem)'
    )

    # Sampling options
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of records to process'
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

    # Override config with command-line arguments
    threshold = args.threshold if args.threshold is not None else config['scenarios']['threshold']
    scenarios_above = args.scenarios_above if args.scenarios_above is not None else config['scenarios']['count_above_threshold']
    scenarios_below = args.scenarios_below if args.scenarios_below is not None else config['scenarios']['count_below_threshold']

    # Handle --parse-only mode: skip to aggregation
    if args.parse_only:
        logger.info("=" * 80)
        logger.info("Parse-only mode: skipping batch submission")
        logger.info("=" * 80)

        # Load all records
        all_records = list(load_jsonl(args.input_file))
        logger.info(f"Loaded {len(all_records)} records from input file")

        # Aggregate ALL results from all batches
        logger.info("Aggregating results from all existing batches...")
        all_scenarios_by_record = aggregate_all_batch_results(temp_dir, all_records)

        if not all_scenarios_by_record:
            logger.warning("No batch results found to aggregate!")
            return 1

        # Write output with ALL aggregated scenarios
        write_output(
            output_file=args.output,
            all_records=all_records,
            scenarios_by_record=all_scenarios_by_record
        )

        logger.info("Parse-only complete!")
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

    # Load all records and calculate CWE counts
    all_records = list(load_jsonl(args.input_file))
    cwe_counts = calculate_cwe_counts(all_records)

    # Prepare batch requests
    batch_requests, metadata_map, batch_file_path, all_records, processed_record_info = prepare_batch_requests(
        input_file=args.input_file,
        prompt_template=prompts['scenario_generation_prompt'],
        temp_dir=temp_dir,
        batch_number=batch_number,
        cwe_counts=cwe_counts,
        threshold=threshold,
        scenarios_above=scenarios_above,
        scenarios_below=scenarios_below,
        no_threshold=args.no_threshold,
        limit=args.limit,
        history_file=args.history_file
    )

    if not batch_requests:
        logger.info("No records to process. All records may have been processed in previous batches.")
        # Still aggregate and write output
        all_scenarios_by_record = aggregate_all_batch_results(temp_dir, all_records)
        write_output(
            output_file=args.output,
            all_records=all_records,
            scenarios_by_record=all_scenarios_by_record
        )
        return 0

    # Ask user for confirmation before submitting
    logger.info("=" * 80)
    logger.info("Batch request file created!")
    logger.info(f"Location: {batch_file_path}")
    logger.info(f"Total requests: {len(batch_requests)}")
    logger.info(f"Threshold: {threshold if not args.no_threshold else 'DISABLED'}")
    logger.info(f"Scenarios: {scenarios_above} (above) / {scenarios_below} (below)")
    if args.history_file:
        logger.info(f"History file: {args.history_file}")
    logger.info("=" * 80)

    # Show example prompts
    logger.info("\n" + "=" * 80)
    logger.info("Example Prompts (showing 3 samples):")
    logger.info("=" * 80)
    num_examples = min(3, len(batch_requests))
    for i in range(num_examples):
        request = batch_requests[i]
        key = request.get('key', 'unknown')
        prompt_text = request.get('request', {}).get('contents', [{}])[0].get('parts', [{}])[0].get('text', '')

        logger.info(f"\n{'-' * 80}")
        logger.info(f"[Example {i+1}] Request Key: {key}")
        logger.info(f"{'-' * 80}")
        logger.info(f"\n{prompt_text}...\n")  # Truncate long code
        logger.info("-" * 80)
    logger.info("=" * 80 + "\n")

    if args.skip_confirmation:
        logger.info("Skipping confirmation (--skip-confirmation flag set)")
        logger.info("Proceeding with batch submission...")
    else:
        logger.info("\nPlease review the batch file before submission.")
        logger.info("You can inspect the file to verify the requests are correct.\n")

        while True:
            user_input = input("Do you want to submit this batch job? (yes/no): ").strip().lower()
            if user_input in ['yes', 'y']:
                logger.info("User confirmed. Proceeding with batch submission...")
                break
            elif user_input in ['no', 'n']:
                logger.info("User cancelled. Cleaning up batch files...")
                cleanup_batch_files(temp_dir, batch_number)
                logger.info("Batch files deleted. Exiting without submission.")
                return 0
            else:
                logger.warning("Invalid input. Please enter 'yes' or 'no'.")

    # Submit batch job and process
    try:
        # Submit batch job
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
        scenarios_by_record = parse_batch_results(
            results=results,
            metadata_map=metadata_map
        )

        logger.info(f"Current batch generated {sum(len(v) for v in scenarios_by_record.values())} scenarios")

        # Save processed records to history (only after successful completion)
        if processed_record_info and args.history_file:
            logger.info(f"Saving {len(processed_record_info)} processed records to history...")
            save_processing_history(args.history_file, processed_record_info, mode='append')
            logger.info("History updated successfully")

        # Aggregate ALL results from all batches
        logger.info("=" * 80)
        logger.info("Aggregating results from all batches...")
        all_scenarios_by_record = aggregate_all_batch_results(temp_dir, all_records)

        # Write output with ALL aggregated scenarios
        write_output(
            output_file=args.output,
            all_records=all_records,
            scenarios_by_record=all_scenarios_by_record
        )

        logger.info("All done!")
        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        logger.error("Cleaning up batch files...")
        cleanup_batch_files(temp_dir, batch_number)
        logger.error("Batch files deleted due to failure.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
