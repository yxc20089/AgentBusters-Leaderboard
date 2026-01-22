#!/usr/bin/env python3
"""
Upload generated evaluation data to private storage.

This script uploads anonymized evaluation scenarios to private storage
for use by GitHub Actions during evaluation.

Supported backends:
- GitHub private repository (via gh CLI)
- HuggingFace private dataset
- S3/R2 bucket

Usage:
    # Upload to GitHub private repo
    python tools/upload_eval_data.py --source data/crypto/eval_hidden --backend github --repo your-org/agentbusters-eval-data

    # Upload to HuggingFace
    python tools/upload_eval_data.py --source data/crypto/eval_hidden --backend huggingface --repo your-org/agentbusters-eval-data

    # Upload to S3
    python tools/upload_eval_data.py --source data/crypto/eval_hidden --backend s3 --bucket your-bucket --prefix eval/crypto
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def upload_to_github(source_dir: Path, repo: str, branch: str = "main") -> bool:
    """Upload eval data to a GitHub private repository."""
    print(f"Uploading to GitHub repo: {repo}")

    # Create temp directory for git operations
    temp_dir = Path("temp_eval_upload")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        # Clone the repo (or init if new)
        result = subprocess.run(
            ["gh", "repo", "clone", repo, str(temp_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Try to create the repo if it doesn't exist
            print(f"Repo doesn't exist, creating private repo: {repo}")
            subprocess.run(
                ["gh", "repo", "create", repo, "--private", "--clone"],
                cwd=str(temp_dir.parent),
                check=True
            )

        # Copy evaluation data
        dest = temp_dir / "crypto"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_dir, dest)

        # Add metadata
        meta = {
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source_dir),
            "scenarios": len(list(source_dir.glob("scenario_*"))),
        }
        (temp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Commit and push
        subprocess.run(["git", "add", "."], cwd=str(temp_dir), check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Update eval data {datetime.now().strftime('%Y-%m-%d')}"],
            cwd=str(temp_dir),
            check=True
        )
        subprocess.run(["git", "push", "origin", branch], cwd=str(temp_dir), check=True)

        print(f"Successfully uploaded to {repo}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def upload_to_huggingface(source_dir: Path, repo: str, private: bool = True) -> bool:
    """Upload eval data to HuggingFace dataset."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    print(f"Uploading to HuggingFace: {repo}")

    api = HfApi()

    # Create repo if needed
    try:
        create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Upload all files
    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source_dir)
            print(f"  Uploading: {rel_path}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=f"crypto/{rel_path}",
                repo_id=repo,
                repo_type="dataset",
            )

    print(f"Successfully uploaded to https://huggingface.co/datasets/{repo}")
    return True


def upload_to_s3(source_dir: Path, bucket: str, prefix: str = "eval/crypto") -> bool:
    """Upload eval data to S3/R2 bucket."""
    print(f"Uploading to S3: s3://{bucket}/{prefix}")

    try:
        # Use AWS CLI for simplicity
        result = subprocess.run(
            ["aws", "s3", "sync", str(source_dir), f"s3://{bucket}/{prefix}"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False

        print(f"Successfully uploaded to s3://{bucket}/{prefix}")
        return True

    except FileNotFoundError:
        print("Error: AWS CLI not found. Install with: pip install awscli")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload evaluation data to private storage")
    parser.add_argument("--source", type=str, required=True, help="Source directory with eval data")
    parser.add_argument("--backend", type=str, choices=["github", "huggingface", "s3"], required=True)
    parser.add_argument("--repo", type=str, help="Repository name (github/huggingface)")
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument("--prefix", type=str, default="eval/crypto", help="S3 prefix")
    parser.add_argument("--branch", type=str, default="main", help="Git branch (github)")

    args = parser.parse_args()
    source = Path(args.source)

    if not source.exists():
        print(f"Error: Source directory not found: {source}")
        return 1

    if args.backend == "github":
        if not args.repo:
            print("Error: --repo required for github backend")
            return 1
        success = upload_to_github(source, args.repo, args.branch)
    elif args.backend == "huggingface":
        if not args.repo:
            print("Error: --repo required for huggingface backend")
            return 1
        success = upload_to_huggingface(source, args.repo)
    elif args.backend == "s3":
        if not args.bucket:
            print("Error: --bucket required for s3 backend")
            return 1
        success = upload_to_s3(source, args.bucket, args.prefix)
    else:
        print(f"Unknown backend: {args.backend}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
