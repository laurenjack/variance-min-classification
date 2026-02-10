#!/bin/bash
# lambda_terminate.sh - Terminate Lambda Labs instance(s)
#
# Prerequisites:
#   - LAMBDA_API_KEY environment variable set
#
# Usage:
#   ./lambda_terminate.sh                    # Terminate all instances
#   ./lambda_terminate.sh <instance_id>      # Terminate specific instance

set -euo pipefail

API_BASE="https://cloud.lambdalabs.com/api/v1"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
    log_error "LAMBDA_API_KEY environment variable not set"
    exit 1
fi

# API helper function
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

# List instances
list_instances() {
    lambda_api GET "/instances" | jq -r '.data[] | "\(.id) | \(.status) | \(.ip) | \(.instance_type.name) | $\(.instance_type.price_cents_per_hour/100)/hr"'
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
        log_info "Instance $instance_id terminated successfully"
        return 0
    else
        log_warn "Could not confirm termination"
        echo "$response" | jq .
        return 1
    fi
}

# Main
if [[ $# -eq 1 ]]; then
    # Terminate specific instance
    terminate_instance "$1"
else
    # List and terminate all
    log_info "Current instances:"
    instances=$(lambda_api GET "/instances" | jq -r '.data[].id')

    if [[ -z "$instances" ]]; then
        log_info "No instances running"
        exit 0
    fi

    list_instances

    echo ""
    read -p "Terminate all instances? (y/N) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for id in $instances; do
            terminate_instance "$id"
        done
    else
        log_info "Cancelled"
    fi
fi
