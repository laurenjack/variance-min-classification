#!/usr/bin/env python3
"""Launch reward model training on SageMaker.

This script handles all SageMaker and S3 interaction:
- Checks if training data exists in S3
- If not, prepares it locally and uploads
- Submits the training job with appropriate paths

Run from project root:
    source ~/.cursor_bootstrap.sh && source venv/bin/activate
    python -m jl.reward_model.launch_sagemaker
"""

import os
import shutil
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# =============================================================================
# SageMaker Configuration (all S3/SageMaker-specific values live here)
# =============================================================================
ROLE_ARN = "arn:aws:iam::100611042793:role/SageMakerRewardModelRole"
REGION = "eu-central-1"
INSTANCE_TYPE = "ml.g6e.xlarge"
S3_BUCKET = "sagemaker-reward-model-100611042793"
S3_DATA_PREFIX = "data/hh-rlhf-tokenized"

# SageMaker paths (where data/model appear on the training instance)
SAGEMAKER_TRAIN_PATH = "/opt/ml/input/data/training"
SAGEMAKER_OUTPUT_PATH = "/opt/ml/model"

# Find project root (where jl/ directory is)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Files/dirs to include in the source package
INCLUDE_PATTERNS = [
    "jl/__init__.py",
    "jl/reward_model/__init__.py",
    "jl/reward_model/reward_config.py",
    "jl/reward_model/main.py",
    "jl/reward_model/model.py",
    "jl/reward_model/trainer.py",
    "jl/reward_model/load_data.py",
    "jl/reward_model/prep_data.py",
    "requirements.txt",
]


def prepare_source_dir():
    """Create minimal source directory for SageMaker."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="sagemaker_src_"))
    print(f"Preparing source in: {tmp_dir}")

    for pattern in INCLUDE_PATTERNS:
        src = PROJECT_ROOT / pattern
        dst = tmp_dir / pattern
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  + {pattern}")
        else:
            print(f"  ! Missing: {pattern}")

    return tmp_dir


def check_s3_data_exists(s3_client, bucket: str, prefix: str) -> bool:
    """Check if pre-staged training data exists in S3."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return response.get('KeyCount', 0) > 0
    except Exception:
        return False


def upload_directory_to_s3(s3_client, local_path: Path, bucket: str, prefix: str):
    """Upload a local directory to S3."""
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{prefix}/{relative_path}"
            s3_client.upload_file(str(file_path), bucket, s3_key)


def prepare_and_upload_data(s3_client, bucket: str, prefix: str):
    """Prepare training data locally and upload to S3."""
    from jl.reward_model.reward_config import RewardConfig
    from jl.reward_model.prep_data import prepare_dataset
    
    # Prepare data in a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="reward_data_"))
    local_data_path = tmp_dir / "tokenized"
    
    try:
        print("Preparing training data locally...")
        config = RewardConfig()
        prepare_dataset(config, str(local_data_path))
        
        # Upload to S3
        s3_uri = f"s3://{bucket}/{prefix}/"
        print(f"Uploading to {s3_uri}...")
        upload_directory_to_s3(s3_client, local_data_path, bucket, prefix)
        print("Upload complete!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    # Create SageMaker session in the correct region
    boto_session = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_session, default_bucket=S3_BUCKET)
    s3_client = boto_session.client('s3')

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Region: {REGION}")
    print(f"Instance type: {INSTANCE_TYPE}")
    print(f"S3 bucket: {S3_BUCKET}")
    print()

    # Check if training data exists in S3, prepare if not
    s3_data_uri = f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/"
    if not check_s3_data_exists(s3_client, S3_BUCKET, S3_DATA_PREFIX):
        print(f"Training data not found at {s3_data_uri}")
        print("Preparing and uploading data...")
        print()
        prepare_and_upload_data(s3_client, S3_BUCKET, S3_DATA_PREFIX)
        print()
    
    print(f"✓ Training data available at {s3_data_uri}")
    inputs = {'training': s3_data_uri}
    print()

    # Prepare minimal source directory
    source_dir = prepare_source_dir()
    print()

    # PyTorch Estimator - will install HuggingFace packages via requirements.txt
    pytorch_estimator = PyTorch(
        entry_point="jl/reward_model/main.py",
        source_dir=str(source_dir),
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="2.4.0",
        py_version="py311",
        sagemaker_session=sess,
        disable_profiler=True,
        hyperparameters={
            "train-path": SAGEMAKER_TRAIN_PATH,
            "output-path": SAGEMAKER_OUTPUT_PATH,
        },
        environment={
            "HF_HOME": "/tmp/hf_cache",
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
        },
        base_job_name="reward-model-smoke-test",
    )

    print("Starting SageMaker training job...")
    print(f"  Train path: {SAGEMAKER_TRAIN_PATH}")
    print(f"  Output path: {SAGEMAKER_OUTPUT_PATH}")
    pytorch_estimator.fit(inputs=inputs, wait=True, logs="All")
    print("Training complete!")
    print(f"Model artifacts saved to: s3://{S3_BUCKET}/reward-model-smoke-test-*/output/model.tar.gz")

    # Cleanup
    shutil.rmtree(source_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
