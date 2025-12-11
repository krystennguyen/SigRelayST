#!/bin/bash
# Preprocess and train with/without signature bias
# Assumes VM is already set up (run create_dl_vm_instance.sh first)


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


cd "$PROJECT_ROOT"


source "$SCRIPT_DIR/load_gcp_config.sh"

echo "=========================================="
echo "SigRelayST Pipeline: Preprocess and Train"
echo "=========================================="
echo "This will:"
echo "1. Transfer files to instance"
echo "2. Preprocess WITH signature bias"
echo "3. Preprocess WITHOUT signature bias"
echo "4. Train both models ($EPOCHS epochs each)"
echo ""
echo "Expected total time: 8-10 hours"
echo ""
echo "Note: Ensure VM is set up first by running: bash gcp_scripts/create_dl_vm_instance.sh"
echo "=========================================="
echo ""

# Quick check that instance exists and is running
if ! gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT --zone=$ZONE &>/dev/null; then
    echo "Error: Instance $INSTANCE_NAME does not exist."
    echo "Please run: bash gcp_scripts/create_dl_vm_instance.sh first"
    exit 1
fi

STATUS=$(gcloud compute instances describe $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --format="get(status)" 2>/dev/null)

if [ "$STATUS" != "RUNNING" ]; then
    echo "Instance $INSTANCE_NAME is $STATUS. Starting..."
    gcloud compute instances start $INSTANCE_NAME --project=$PROJECT --zone=$ZONE
    echo "Waiting for instance to be ready..."
    sleep 30
fi

# Step 1: Transfer files
echo ""
echo "Step 1/3: Transferring files to instance..."
bash "$SCRIPT_DIR/transfer_to_dl_instance.sh"

if [ $? -ne 0 ]; then
    echo "Warning: Some files may not have transferred. Continuing anyway..."
fi

# Step 2: Preprocessing
echo ""
echo "Step 2/3: Running preprocessing..."
echo ""

# Preprocess WITH bias
echo "2a. Preprocessing WITH signature bias..."
bash "$SCRIPT_DIR/run_preprocess_with_bias.sh"

if [ $? -ne 0 ]; then
    echo "Error: Preprocessing with bias failed"
    exit 1
fi

# Preprocess WITHOUT bias
echo ""
echo "2b. Preprocessing WITHOUT signature bias..."
bash "$SCRIPT_DIR/run_preprocess_without_bias.sh"

if [ $? -ne 0 ]; then
    echo "Error: Preprocessing without bias failed"
    exit 1
fi

# Step 3: Training comparison
echo ""
echo "Step 3/3: Starting training comparison ($EPOCHS epochs each)..."
echo ""

# Launch WITH bias training in background
echo "Starting training WITH signature bias..."
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        nohup python3 run_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial \
            --training_data=input_graph/V1_Human_Lymph_Node_spatial/ \
            --embedding_path=embedding_data/V1_Human_Lymph_Node_spatial/ \
            --model_path=model/V1_Human_Lymph_Node_spatial/ \
            --metadata_to=metadata/V1_Human_Lymph_Node_spatial/ \
            --model_name=SigRelayST_with_bias \
            --run_id=1 \
            --num_epoch=$EPOCHS \
            --hidden=512 \
            --heads=1 \
            --dropout=0.0 \
            --lr_rate=0.00001 \
            --total_subgraphs=1 \
            --manual_seed=yes --seed=1 \
            > logs/training_with_bias.log 2>&1 &" \
    --tunnel-through-iap

if [ $? -ne 0 ]; then
    echo "Error: Failed to start training WITH bias"
    exit 1
fi


sleep 5

# Launch WITHOUT bias training in background
echo "Starting training WITHOUT signature bias..."
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        nohup python3 run_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial_no_bias \
            --training_data=input_graph/V1_Human_Lymph_Node_spatial_no_bias/ \
            --embedding_path=embedding_data/V1_Human_Lymph_Node_spatial_no_bias/ \
            --model_path=model/V1_Human_Lymph_Node_spatial_no_bias/ \
            --metadata_to=metadata/V1_Human_Lymph_Node_spatial_no_bias/ \
            --model_name=SigRelayST_no_bias \
            --run_id=1 \
            --num_epoch=$EPOCHS \
            --hidden=512 \
            --heads=1 \
            --dropout=0.0 \
            --lr_rate=0.00001 \
            --total_subgraphs=1 \
            --manual_seed=yes --seed=1 \
            > logs/training_without_bias.log 2>&1 &" \
    --tunnel-through-iap

if [ $? -ne 0 ]; then
    echo "Error: Failed to start training WITHOUT bias"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Started Successfully!"
echo "=========================================="
echo ""
echo "Both training jobs are running in the background."
echo "Training will take approximately 6-8 hours each."
echo ""
echo "Monitor progress with:"
echo "  gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$ZONE --command='tail -f $REMOTE_DIR/logs/training_with_bias.log'"
echo "  gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$ZONE --command='tail -f $REMOTE_DIR/logs/training_without_bias.log'"
echo ""
echo "Check if processes are running:"
echo "  gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$ZONE --command='ps aux | grep run_SigRelayST'"
echo ""
echo "Check GPU usage:"
echo "  gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$ZONE --command='nvidia-smi'"
echo ""

