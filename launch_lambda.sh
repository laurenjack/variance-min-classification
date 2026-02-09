#!/bin/bash
# launch_lambda.sh - Launch H100 instance on Lambda Labs and run reward model training
#
# Prerequisites:
#   - LAMBDA_API_KEY environment variable set
#   - SSH private key at ~/.ssh/jacklaurenson
#   - SSH public key "jacklaurenson" registered with Lambda Labs
#
# Usage:
#   ./launch_lambda.sh

set -euo pipefail

# Configuration
REGION="us-east-1"
SSH_KEY_NAME="jacklaurenson"
SSH_KEY_PATH="$HOME/.ssh/jacklaurenson"
API_BASE="https://cloud.lambdalabs.com/api/v1"
INSTANCE_NAME="reward-model-training"
REPO_URL="https://github.com/YOUR_USERNAME/variance-min-classification.git"  # TODO: Update this

# Instance types to try (in order of preference)
INSTANCE_TYPES=("gpu_1x_h100_pcie" "gpu_1x_h100_sxm")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
        log_error "LAMBDA_API_KEY environment variable not set"
        exit 1
    fi

    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        log_error "SSH private key not found at $SSH_KEY_PATH"
        exit 1
    fi

    chmod 600 "$SSH_KEY_PATH" 2>/dev/null || true
    log_info "Prerequisites OK"
}

# API helper function
lambda_api() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"

    if [[ -n "$data" ]]; then
        curl -s -X "$method" -u "$LAMBDA_API_KEY:" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "${API_BASE}${endpoint}"
    else
        curl -s -X "$method" -u "$LAMBDA_API_KEY:" \
            "${API_BASE}${endpoint}"
    fi
}

# Get available instance types in the region
get_available_instances() {
    log_info "Querying available instance types in $REGION..."
    lambda_api GET "/instance-types" | jq -r ".data | to_entries[] | select(.value.regions_with_capacity_available | map(.name) | index(\"$REGION\")) | .key"
}

# Find first available instance type from our preference list
find_instance_type() {
    local available
    available=$(get_available_instances)

    for instance_type in "${INSTANCE_TYPES[@]}"; do
        if echo "$available" | grep -q "^${instance_type}$"; then
            echo "$instance_type"
            return 0
        fi
    done

    log_error "No H100 instances available in $REGION"
    log_info "Available instances:"
    echo "$available"
    return 1
}

# Launch instance
launch_instance() {
    local instance_type="$1"
    log_info "Launching $instance_type in $REGION..."

    local response
    response=$(lambda_api POST "/instance-operations/launch" "{
        \"region_name\": \"$REGION\",
        \"instance_type_name\": \"$instance_type\",
        \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
        \"name\": \"$INSTANCE_NAME\"
    }")

    local instance_id
    instance_id=$(echo "$response" | jq -r '.data.instance_ids[0] // empty')

    if [[ -z "$instance_id" ]]; then
        log_error "Failed to launch instance"
        echo "$response" | jq .
        return 1
    fi

    echo "$instance_id"
}

# Wait for instance to be ready
wait_for_instance() {
    local instance_id="$1"
    local max_wait=600  # 10 minutes
    local elapsed=0
    local interval=10

    log_info "Waiting for instance $instance_id to be ready (timeout: ${max_wait}s)..."

    while [[ $elapsed -lt $max_wait ]]; do
        local status ip
        local response
        response=$(lambda_api GET "/instances/$instance_id")
        status=$(echo "$response" | jq -r '.data.status // empty')
        ip=$(echo "$response" | jq -r '.data.ip // empty')

        case "$status" in
            "active")
                if [[ -n "$ip" && "$ip" != "null" ]]; then
                    log_info "Instance is active at $ip"
                    echo "$ip"
                    return 0
                fi
                ;;
            "booting")
                echo -n "."
                ;;
            "unhealthy"|"terminated")
                log_error "Instance entered $status state"
                return 1
                ;;
        esac

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log_error "Timeout waiting for instance"
    return 1
}

# Wait for SSH to be available
wait_for_ssh() {
    local ip="$1"
    local max_wait=120
    local elapsed=0

    log_info "Waiting for SSH to be available..."

    while [[ $elapsed -lt $max_wait ]]; do
        if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
            "ubuntu@$ip" "echo 'SSH OK'" 2>/dev/null; then
            log_info "SSH connection established"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done

    log_error "Timeout waiting for SSH"
    return 1
}

# Run training on the instance
run_training() {
    local ip="$1"

    log_info "Starting training on $ip..."

    # Create a setup script to run on the instance
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "ubuntu@$ip" bash <<'REMOTE_SCRIPT'
set -ex

echo "=== Setting up environment ==="
cd ~

# Clone the repo (or pull if exists)
if [[ -d "variance-min-classification" ]]; then
    cd variance-min-classification
    git pull
else
    git clone https://github.com/YOUR_USERNAME/variance-min-classification.git  # TODO: Update this
    cd variance-min-classification
fi

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch transformers datasets accelerate

# Set HuggingFace token if needed (for gated models)
# export HF_TOKEN="your_token_here"

echo "=== Starting training ==="
mkdir -p output data

# Run training (data will be downloaded automatically if not present)
python -m jl.reward_model.main \
    --train-path ./data/tokenized \
    --output-path ./output

echo "=== Training complete ==="
ls -la output/
REMOTE_SCRIPT

    log_info "Training script completed"
}

# Copy artifacts back to local machine
copy_artifacts() {
    local ip="$1"
    local local_output="./lambda_output"

    log_info "Copying model artifacts to $local_output..."
    mkdir -p "$local_output"

    scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no -r \
        "ubuntu@$ip:~/variance-min-classification/output/*" \
        "$local_output/"

    log_info "Artifacts saved to $local_output"
    ls -la "$local_output"
}

# Terminate instance
terminate_instance() {
    local instance_id="$1"

    log_info "Terminating instance $instance_id..."

    local response
    response=$(lambda_api POST "/instance-operations/terminate" "{
        \"instance_ids\": [\"$instance_id\"]
    }")

    local terminated
    terminated=$(echo "$response" | jq -r '.data.terminated_instances[0].id // empty')

    if [[ "$terminated" == "$instance_id" ]]; then
        log_info "Instance terminated successfully"
    else
        log_warn "Could not confirm termination. Please check Lambda console."
        echo "$response" | jq .
    fi
}

# Cleanup function for trap
cleanup() {
    local exit_code=$?
    if [[ -n "${INSTANCE_ID:-}" ]]; then
        log_warn "Cleaning up due to error or interrupt..."
        terminate_instance "$INSTANCE_ID"
    fi
    exit $exit_code
}

# Main execution
main() {
    log_info "=== Lambda Labs Training Launcher ==="

    check_prerequisites

    # Find available instance type
    local instance_type
    instance_type=$(find_instance_type)
    log_info "Selected instance type: $instance_type"

    # Launch instance
    INSTANCE_ID=$(launch_instance "$instance_type")
    log_info "Launched instance: $INSTANCE_ID"

    # Set up cleanup trap
    trap cleanup EXIT INT TERM

    # Wait for instance to be ready
    local instance_ip
    instance_ip=$(wait_for_instance "$INSTANCE_ID")

    # Wait for SSH
    wait_for_ssh "$instance_ip"

    # Run training
    run_training "$instance_ip"

    # Copy artifacts
    copy_artifacts "$instance_ip"

    # Terminate (will be done by trap, but let's be explicit)
    trap - EXIT INT TERM  # Remove trap
    terminate_instance "$INSTANCE_ID"

    log_info "=== All done! ==="
}

main "$@"
