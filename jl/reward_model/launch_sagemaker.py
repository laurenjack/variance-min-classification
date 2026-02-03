#!/usr/bin/env python3
"""Launch reward model training on SageMaker.

Run from project root:
    source ~/.cursor_bootstrap.sh
    python -m jl.reward_model.launch_sagemaker
"""

import shutil
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# Configuration
ROLE_ARN = "arn:aws:iam::100611042793:role/SageMakerRewardModelRole"
REGION = "eu-central-1"
INSTANCE_TYPE = "ml.g6e.xlarge"
S3_BUCKET = "sagemaker-reward-model-100611042793"

# Find project root (where jl/ directory is)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Files/dirs to include in the source package
INCLUDE_PATTERNS = [
    "jl/__init__.py",
    "jl/reward_model/__init__.py",
    "jl/reward_model/main.py",
    "jl/reward_model/model.py",
    "jl/reward_model/trainer.py",
    "jl/reward_model/data_downloader.py",
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


def main():
    # Create SageMaker session in the correct region
    boto_session = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_session, default_bucket=S3_BUCKET)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Region: {REGION}")
    print(f"Instance type: {INSTANCE_TYPE}")
    print(f"S3 bucket: {S3_BUCKET}")
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
        framework_version="2.1.0",
        py_version="py310",
        sagemaker_session=sess,
        disable_profiler=True,
        environment={
            "HF_HOME": "/tmp/hf_cache",
        },
        base_job_name="reward-model-smoke-test",
    )

    print("Starting SageMaker training job...")
    pytorch_estimator.fit(wait=True, logs="All")
    print("Training complete!")

    # Cleanup
    shutil.rmtree(source_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
