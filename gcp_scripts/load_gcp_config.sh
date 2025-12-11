#!/bin/bash
# Load GCP configuration for all GCP scripts
# This ensures consistent configuration across all scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load GCP configuration
# Priority: 1) gcp_config.sh (local, gitignored), 2) Environment variables, 3) Defaults
if [ -f "$SCRIPT_DIR/gcp_config.sh" ]; then
    source "$SCRIPT_DIR/gcp_config.sh"
elif [ -n "$INSTANCE_NAME" ] && [ -n "$PROJECT" ] && [ -n "$ZONE" ]; then
    # Use environment variables if set
    : # Variables already set
else
    # Default values (for local testing - users should create gcp_config.sh)
    INSTANCE_NAME="${INSTANCE_NAME:-your-instance-name}"
    PROJECT="${PROJECT:-your-gcp-project-id}"
    ZONE="${ZONE:-your-zone}"
    REMOTE_DIR="${REMOTE_DIR:-~/SigRelayST}"
    EPOCHS="${EPOCHS:-40000}"
    
    if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
        # Only warn if sourced (not if run directly)
        echo "Warning: Using default/placeholder values. Create gcp_scripts/gcp_config.sh with your GCP settings." >&2
        echo "See gcp_scripts/gcp_config.example.sh for template." >&2
    fi
fi

