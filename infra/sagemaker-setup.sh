#!/bin/bash
# SageMaker infrastructure setup for reward model training
# Requires AWS credentials to be configured (see CLAUDE.md)

set -euo pipefail

REGION="eu-central-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="sagemaker-reward-model-${ACCOUNT_ID}"
ROLE_NAME="SageMakerRewardModelRole"

echo "=== SageMaker Infrastructure Setup ==="
echo "Region: ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "Bucket: ${BUCKET_NAME}"
echo "Role: ${ROLE_NAME}"
echo ""

# Create S3 bucket (ignore error if exists)
echo "Creating S3 bucket..."
if aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null; then
    echo "  Bucket already exists"
else
    aws s3 mb "s3://${BUCKET_NAME}" --region "${REGION}"
    echo "  Bucket created"
fi

# Create IAM role (ignore error if exists)
echo "Creating IAM role..."
TRUST_POLICY=$(cat <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)

if aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
    echo "  Role already exists"
else
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document "${TRUST_POLICY}" \
        --description "Execution role for reward model training"
    echo "  Role created"
fi

# Attach policies (idempotent)
echo "Attaching policies..."
aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess 2>/dev/null || true
aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true
echo "  Policies attached"

echo ""
echo "=== Setup Complete ==="
echo "Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "S3 Bucket: s3://${BUCKET_NAME}"
echo ""
echo "To launch training (with AWS credentials and venv activated):"
echo "  python -m jl.reward_model.launch_sagemaker"
