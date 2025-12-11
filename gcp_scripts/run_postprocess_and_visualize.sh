#!/bin/bash
# Postprocess and visualize results for both models (with and without signature bias)
# This script should be run after training completes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_gcp_config.sh"

TOTAL_RUNS=1

echo "=========================================="
echo "Postprocessing and Visualizing Results"
echo "=========================================="
echo "This will postprocess and visualize both models"
echo ""

# Check if training is still running
echo "Checking if training is complete..."
TRAINING_RUNNING=$(gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="ps aux | grep run_SigRelayST | grep -v grep | wc -l" \
    --tunnel-through-iap 2>&1 | tail -1 | tr -d ' ')

if [ "$TRAINING_RUNNING" != "0" ]; then
    echo "Warning: Training processes are still running."
    echo "Postprocessing can run in parallel, but results may be incomplete."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Postprocess WITH bias
echo ""
echo "=========================================="
echo "Postprocessing WITH signature bias"
echo "=========================================="
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        python3 output_postprocess_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial \
            --model_name=SigRelayST_with_bias \
            --total_runs=$TOTAL_RUNS \
            --top_percent=20 \
            --output_all=1 \
        2>&1 | tee logs/postprocess_with_bias_\$(date +%Y%m%d_%H%M%S).log" \
    --tunnel-through-iap

if [ $? -eq 0 ]; then
    echo "Postprocessing WITH bias completed!"
else
    echo "Postprocessing WITH bias failed"
fi

# Postprocess WITHOUT bias
echo ""
echo "=========================================="
echo "Postprocessing WITHOUT signature bias"
echo "=========================================="
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        python3 output_postprocess_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial_no_bias \
            --model_name=SigRelayST_no_bias \
            --total_runs=$TOTAL_RUNS \
            --top_percent=20 \
            --output_all=1 \
        2>&1 | tee logs/postprocess_without_bias_\$(date +%Y%m%d_%H%M%S).log" \
    --tunnel-through-iap

if [ $? -eq 0 ]; then
    echo "Postprocessing WITHOUT bias completed!"
else
    echo "Postprocessing WITHOUT bias failed"
fi

# Visualize WITH bias
echo ""
echo "=========================================="
echo "Visualizing WITH signature bias"
echo "=========================================="
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        python3 output_visualization_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial \
            --model_name=SigRelayST_with_bias \
\
            --top_percent=20 \
        2>&1 | tee logs/visualize_with_bias_\$(date +%Y%m%d_%H%M%S).log" \
    --tunnel-through-iap

if [ $? -eq 0 ]; then
    echo "Visualization WITH bias completed!"
else
    echo "Visualization WITH bias failed"
fi

# Visualize WITHOUT bias
echo ""
echo "=========================================="
echo "Visualizing WITHOUT signature bias"
echo "=========================================="
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        python3 output_visualization_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial_no_bias \
            --model_name=SigRelayST_no_bias \
\
            --top_percent=20 \
        2>&1 | tee logs/visualize_without_bias_\$(date +%Y%m%d_%H%M%S).log" \
    --tunnel-through-iap

if [ $? -eq 0 ]; then
    echo "Visualization WITHOUT bias completed!"
else
    echo "Visualization WITHOUT bias failed"
fi

echo ""
echo "=========================================="
echo "Postprocessing and Visualization Complete!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  WITH bias:    $REMOTE_DIR/output/V1_Human_Lymph_Node_spatial/"
echo "  WITHOUT bias: $REMOTE_DIR/output/V1_Human_Lymph_Node_spatial_no_bias/"
echo ""
echo "To download results:"
echo "  gcloud compute scp --recurse $INSTANCE_NAME:$REMOTE_DIR/output/ ./output/ --project=$PROJECT --zone=$ZONE"
echo ""

