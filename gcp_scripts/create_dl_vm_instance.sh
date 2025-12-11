#!/bin/bash
# Setup Deep Learning VM instance: Create VM and install dependencies
# This script handles one-time setup. Run this before using run_training_comparison.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_gcp_config.sh"

echo "=========================================="
echo "GCP VM Setup: Instance and Dependencies"
echo "=========================================="
echo "This will:"
echo "1. Check/create Deep Learning VM instance"
echo "2. Verify/install dependencies"
echo ""

# Function to check if instance exists and is running
check_instance() {
    if gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT --zone=$ZONE &>/dev/null; then
        STATUS=$(gcloud compute instances describe $INSTANCE_NAME \
            --project=$PROJECT \
            --zone=$ZONE \
            --format="get(status)" 2>/dev/null)
        
        if [ "$STATUS" == "RUNNING" ]; then
            echo "Instance $INSTANCE_NAME exists and is RUNNING"
            return 0
        elif [ "$STATUS" == "TERMINATED" ] || [ "$STATUS" == "STOPPED" ]; then
            echo "Instance $INSTANCE_NAME exists but is $STATUS. Starting..."
            gcloud compute instances start $INSTANCE_NAME \
                --project=$PROJECT \
                --zone=$ZONE
            echo "Waiting for instance to be ready..."
            sleep 30
            return 0
        else
            echo "Instance $INSTANCE_NAME exists but status is $STATUS"
            return 1
        fi
    else
        echo "Instance $INSTANCE_NAME does not exist"
        return 1
    fi
}

# Function to check if PyTorch is installed
check_pytorch() {
    echo "Checking if PyTorch is installed..."
    gcloud compute ssh $INSTANCE_NAME \
        --project=$PROJECT \
        --zone=$ZONE \
        --command="python3 -c 'import torch; print(torch.__version__)'" &>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "PyTorch is installed"
        return 0
    else
        echo "PyTorch is not installed"
        return 1
    fi
}

# Function to install dependencies
install_dependencies() {
    echo ""
    echo "=========================================="
    echo "Installing Dependencies"
    echo "=========================================="
    echo "This will take 10-15 minutes..."
    echo "Note: Deep Learning VM should have most packages pre-installed"
    
    gcloud compute ssh $INSTANCE_NAME \
        --project=$PROJECT \
        --zone=$ZONE \
        --command="
        # Deep Learning VM uses conda environment, activate it
        source /opt/conda/bin/activate || true
        # Try system pip3 if conda not available
        which pip3 || which pip || echo 'pip not found'
        # Install missing packages using system pip (Deep Learning VM should have internet via Cloud NAT)
        pip3 install --user torch-geometric torch-sparse torch-scatter 2>&1 || \
        pip install --user torch-geometric torch-sparse torch-scatter 2>&1 || \
        echo 'Note: Some packages may already be installed or need internet access'
        # Verify what's available
        python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")' 2>&1 || echo 'PyTorch check failed'
        python3 -c 'import torch_geometric; print(f\"PyTorch Geometric: {torch_geometric.__version__}\")' 2>&1 || echo 'PyG not found'
        python3 -c 'import scipy; print(f\"SciPy: {scipy.__version__}\")' 2>&1 || echo 'SciPy not found'
        " 2>&1 | tee /tmp/install_deps.log
    
    # Check if at least PyTorch is available
    gcloud compute ssh $INSTANCE_NAME \
        --project=$PROJECT \
        --zone=$ZONE \
        --command="python3 -c 'import torch; import scipy; import numpy; import pandas; print(\"Core packages available\")'" &>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "Core dependencies are available"
        return 0
    else
        echo "Warning: Some dependencies may be missing, but continuing..."
        return 0  # Continue anyway, Deep Learning VM should have most packages
    fi
}

# Step 1: Check or create instance
echo "Step 1/2: Checking instance..."
if ! check_instance; then
    echo ""
    echo "Creating new Deep Learning VM instance..."
    
    # Check if instance exists but creation script will fail
    if gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT --zone=$ZONE &>/dev/null; then
        echo "Instance exists but may not be in a usable state. Attempting to start..."
        gcloud compute instances start $INSTANCE_NAME --project=$PROJECT --zone=$ZONE
        sleep 30
        if check_instance; then
            echo "Instance is now running"
        else
            echo "Error: Instance exists but cannot be started. Please check manually."
            exit 1
        fi
    else
        # Instance doesn't exist, create it
        echo "Creating instance $INSTANCE_NAME..."
        gcloud compute instances create $INSTANCE_NAME \
            --project=$PROJECT \
            --zone=$ZONE \
            --machine-type=n1-standard-4 \
            --accelerator=count=1,type=nvidia-tesla-t4 \
            --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
            --image-project=deeplearning-platform-release \
            --maintenance-policy=TERMINATE \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-balanced \
            --network-interface=stack-type=IPV4_ONLY,subnet=default,no-address \
            --service-account=891805737159-compute@developer.gserviceaccount.com \
            --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
            --no-shielded-secure-boot \
            --shielded-vtpm \
            --shielded-integrity-monitoring \
            --labels=goog-ops-agent-policy=v2-x86-template-1-4-0,goog-ec-src=vm_add-gcloud \
            --reservation-affinity=any
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create instance"
            exit 1
        fi
        
        echo "Waiting for instance to be ready (2 minutes)..."
        sleep 120
    fi
fi

# Step 2: Check dependencies and install if needed
echo ""
echo "Step 2/2: Checking dependencies..."
if ! check_pytorch; then
    if ! install_dependencies; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
else
    # Still check for torch-geometric
    gcloud compute ssh $INSTANCE_NAME \
        --project=$PROJECT \
        --zone=$ZONE \
        --command="python3 -c 'import torch_geometric'" &>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "PyTorch Geometric not found. Installing..."
        install_dependencies
    fi
fi

echo ""
echo "=========================================="
echo "VM Setup Complete!"
echo "=========================================="
echo ""
echo "Instance $INSTANCE_NAME is ready."
echo "You can now run: bash gcp_scripts/run_training_comparison.sh"
echo ""
