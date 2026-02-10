#!/bin/bash
# lambda_launch.sh - Launch H100 instance on Lambda Labs
#
# Prerequisites:
#   - LAMBDA_API_KEY environment variable set
#   - SSH key registered with Lambda Labs
#
# Usage:
#   ./lambda_launch.sh
#
# Outputs instance IP to stdout on success

set -euo pipefail

# Configuration
SSH_KEY_NAME="jacklaurenson"
API_BASE="https://cloud.lambdalabs.com/api/v1"
INSTANCE_NAME="reward-model-training"

INSTANCE_TYPE="gpu_1x_h100_sxm5"

# Colors for output (to stderr so stdout stays clean for IP)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
        log_error "LAMBDA_API_KEY environment variable not set"
        exit 1
    fi

    log_info "Prerequisites OK"
}

# API helper function - uses explicit Basic auth header
lambda_api() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"
    local auth_header="Authorization: Basic $(echo -n "$LAMBDA_API_KEY:" | base64)"

    if [[ -n "$data" ]]; then
        curl -s -X "$method" \
            -H "$auth_header" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "${API_BASE}${endpoint}"
    else
        curl -s -X "$method" \
            -H "$auth_header" \
            "${API_BASE}${endpoint}"
    fi
}

# Find available region for the instance type
find_region() {
    log_info "Querying availability for $INSTANCE_TYPE..."
    local response
    response=$(lambda_api GET "/instance-types")

    local available_region
    available_region=$(echo "$response" | jq -r ".data[\"$INSTANCE_TYPE\"].regions_with_capacity_available[0].name // empty")

    if [[ -n "$available_region" ]]; then
        log_info "Found $INSTANCE_TYPE available in $available_region"
        echo "$available_region"
        return 0
    fi

    log_error "No $INSTANCE_TYPE instances available in any region"
    return 1
}

# Launch instance
launch_instance() {
    log_info "Launching $INSTANCE_TYPE in $REGION..."

    local response
    response=$(lambda_api POST "/instance-operations/launch" "{
        \"region_name\": \"$REGION\",
        \"instance_type_name\": \"$INSTANCE_TYPE\",
        \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
        \"name\": \"$INSTANCE_NAME\"
    }")

    local instance_id
    instance_id=$(echo "$response" | jq -r '.data.instance_ids[0] // empty')

    if [[ -z "$instance_id" ]]; then
        log_error "Failed to launch instance"
        echo "$response" | jq . >&2
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
                echo -n "." >&2
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
    local ssh_key_path="${2:-$HOME/.ssh/jacklaurenson}"
    local max_wait=120
    local elapsed=0

    log_info "Waiting for SSH to be available..."

    while [[ $elapsed -lt $max_wait ]]; do
        if ssh -i "$ssh_key_path" -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
            "ubuntu@$ip" "echo 'SSH OK'" 2>/dev/null >&2; then
            log_info "SSH connection established"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "." >&2
    done

    log_error "Timeout waiting for SSH"
    return 1
}

# Main execution
main() {
    log_info "=== Lambda Labs Instance Launcher ==="

    check_prerequisites

    # Find available region
    REGION=$(find_region)
    log_info "Selected: $INSTANCE_TYPE in $REGION"

    # Launch instance
    local instance_id
    instance_id=$(launch_instance)
    log_info "Launched instance: $instance_id"

    # Wait for instance to be ready
    local instance_ip
    instance_ip=$(wait_for_instance "$instance_id")

    # Wait for SSH
    wait_for_ssh "$instance_ip"

    # Output instance info
    log_info "=== Instance Ready ==="
    log_info "Instance ID: $instance_id"
    log_info "Instance IP: $instance_ip"
    log_info "Instance Type: $INSTANCE_TYPE"
    log_info "Region: $REGION"

    # Output just the IP to stdout for scripting
    echo "$instance_ip"
}

main "$@"
