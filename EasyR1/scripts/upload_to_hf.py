#!/usr/bin/env python3
"""
Script to upload model checkpoints to Hugging Face Hub.

Usage:
    # Standard upload
    python upload_to_hf.py --local_dir <path> --repo_id <hf_repo_id> [--repo_type model]
    
    # Fast upload (recommended for large models)
    # First install: pip install hf_transfer
    HF_HUB_ENABLE_HF_TRANSFER=1 python upload_to_hf.py --local_dir <path> --repo_id <hf_repo_id>
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os


def upload_to_huggingface(local_dir: str, repo_id: str, repo_type: str = "model"):
    """
    Upload a local directory to Hugging Face Hub.
    
    Args:
        local_dir: Path to the local directory containing the model
        repo_id: Repository ID on Hugging Face (e.g., 'username/model-name')
        repo_type: Type of repository ('model', 'dataset', or 'space')
    """
    # Verify local directory exists
    local_path = Path(local_dir)
    if not local_path.exists():
        raise ValueError(f"Local directory does not exist: {local_dir}")
    
    # Check if hf_transfer is enabled
    
    hf_transfer_enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"
    

    print(f"Uploading from: {local_dir}")
    print(f"Uploading to: {repo_id}")
    print(f"Repository type: {repo_type}")
    
    if hf_transfer_enabled:
        print("⚡ Fast upload enabled (hf_transfer)")
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("💡 Tip: Install 'hf_transfer' and set HF_HUB_ENABLE_HF_TRANSFER=1 for faster uploads")
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        print(f"\nCreating repository '{repo_id}' (if it doesn't exist)...")
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True
        )
        print("✓ Repository ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload folder
    print(f"\nUploading files...")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"\n✓ Upload completed successfully!")
        print(f"View your model at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload model checkpoints to Hugging Face Hub"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Path to the local directory containing the model"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repository (default: model)"
    )
    
    args = parser.parse_args()
    
    upload_to_huggingface(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type
    )


if __name__ == "__main__":
    main()

