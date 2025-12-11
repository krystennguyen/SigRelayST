#!/bin/bash
# Transfer files to Deep Learning VM instance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_gcp_config.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Transferring files to VM instance..."
echo "Source: $PROJECT_ROOT"
echo "Destination: $INSTANCE_NAME:$REMOTE_DIR"

# Create remote directory structure
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="mkdir -p $REMOTE_DIR/{data,input_graph,metadata,model,embedding_data,output,logs,database,gcp_scripts}" \
    --tunnel-through-iap

# Transfer Python scripts
echo "Transferring Python scripts..."
gcloud compute scp \
    --recurse \
    --project=$PROJECT \
    --zone=$ZONE \
    "$PROJECT_ROOT"/*.py \
    $INSTANCE_NAME:$REMOTE_DIR/ \
    --tunnel-through-iap

# Transfer GCP scripts
echo "Transferring GCP scripts..."
gcloud compute scp \
    --recurse \
    --project=$PROJECT \
    --zone=$ZONE \
    "$PROJECT_ROOT/gcp_scripts/"* \
    $INSTANCE_NAME:$REMOTE_DIR/gcp_scripts/ \
    --tunnel-through-iap

# Transfer database files
echo "Transferring database files..."
gcloud compute scp \
    --recurse \
    --project=$PROJECT \
    --zone=$ZONE \
    "$PROJECT_ROOT/database/"* \
    $INSTANCE_NAME:$REMOTE_DIR/database/ \
    --tunnel-through-iap

# Transfer data directory (if exists)
if [ -d "$PROJECT_ROOT/data" ]; then
    echo "Transferring data files..."
    gcloud compute scp \
        --recurse \
        --project=$PROJECT \
        --zone=$ZONE \
        "$PROJECT_ROOT/data/"* \
        $INSTANCE_NAME:$REMOTE_DIR/data/ \
        --tunnel-through-iap
fi

echo "File transfer complete!"

