#!/bin/bash
# Run relay confidence analysis for SigRelayST results on GCP
# This script should be run after postprocessing is complete

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_gcp_config.sh"

ORGANISM="human"  
DATABASE_DIR="database"  # Directory containing PPI and TF-target databases

echo "=========================================="
echo "Relay Confidence Analysis"
echo "=========================================="
echo "This will run:"
echo "1. Relay pattern extraction (if needed)"
echo "2. Relay confidence scoring"
echo ""

# Check if postprocessing is complete
echo "Checking if postprocessing is complete..."
WITH_BIAS_FILE=$(gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="ls $REMOTE_DIR/output/V1_Human_Lymph_Node_spatial/SigRelayST_with_bias_top20percent.csv 2>/dev/null" \
    --tunnel-through-iap 2>&1 | tail -1)

if [ -z "$WITH_BIAS_FILE" ]; then
    echo "Warning: Postprocessing output not found. Running postprocessing first..."
    bash gcp_scripts/run_postprocess_and_visualize.sh
fi

# Step 1: Extract relay patterns (if not already done)
echo ""
echo "=========================================="
echo "Step 1: Extracting Relay Patterns"
echo "=========================================="

# WITH bias
echo "Extracting relay patterns WITH bias..."
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        python3 extract_relay_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial \
            --top_ccc_file=output/V1_Human_Lymph_Node_spatial/SigRelayST_with_bias_top20percent.csv \
            --output_path=output/V1_Human_Lymph_Node_spatial/ \
        2>&1 | tee logs/extract_relay_with_bias_\$(date +%Y%m%d_%H%M%S).log" \
    --tunnel-through-iap

# WITHOUT bias
echo "Extracting relay patterns WITHOUT bias..."
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        python3 extract_relay_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial_no_bias \
            --top_ccc_file=output/V1_Human_Lymph_Node_spatial_no_bias/SigRelayST_no_bias_top20percent.csv \
            --output_path=output/V1_Human_Lymph_Node_spatial_no_bias/ \
        2>&1 | tee logs/extract_relay_without_bias_\$(date +%Y%m%d_%H%M%S).log" \
    --tunnel-through-iap

# Step 2: Calculate relay confidence scores
echo ""
echo "=========================================="
echo "Step 2: Relay Confidence Scoring"
echo "=========================================="

# Check if database files exist
echo "Checking for database files..."
DB_CHECK=$(gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="ls $REMOTE_DIR/$DATABASE_DIR/${ORGANISM}_tf_target.csv $REMOTE_DIR/$DATABASE_DIR/${ORGANISM}_signaling_ppi.csv 2>/dev/null | wc -l" \
    --tunnel-through-iap 2>&1 | tail -1 | tr -d ' ')

if [ "$DB_CHECK" != "2" ]; then
    echo "Warning: Database files not found in $REMOTE_DIR/$DATABASE_DIR/"
    echo "Please ensure the following files exist:"
    echo "  - ${ORGANISM}_tf_target.csv"
    echo "  - ${ORGANISM}_signaling_ppi.csv"
    echo "Skipping relay confidence scoring."
else
    # WITH bias
    RELAY_FILE_WITH="$REMOTE_DIR/output/V1_Human_Lymph_Node_spatial/SigRelayST_V1_Human_Lymph_Node_spatial_relay_pattern_count.csv"
    if gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$ZONE \
        --command="test -f $RELAY_FILE_WITH" --tunnel-through-iap 2>/dev/null; then
        echo "Calculating relay confidence WITH bias..."
        gcloud compute ssh $INSTANCE_NAME \
            --project=$PROJECT \
            --zone=$ZONE \
            --command="cd $REMOTE_DIR && \
                python3 relay_confidence_SigRelayST.py \
                    --input_path=$RELAY_FILE_WITH \
                    --database_dir=$DATABASE_DIR \
                    --organism=$ORGANISM \
                    --output_path=output/V1_Human_Lymph_Node_spatial/SigRelayST_V1_Human_Lymph_Node_spatial_relay_confidence.csv \
                    --activation_only=1 \
                2>&1 | tee logs/relay_confidence_with_bias_\$(date +%Y%m%d_%H%M%S).log" \
            --tunnel-through-iap
    fi
    
    # WITHOUT bias
    RELAY_FILE_WITHOUT="$REMOTE_DIR/output/V1_Human_Lymph_Node_spatial_no_bias/SigRelayST_V1_Human_Lymph_Node_spatial_no_bias_relay_pattern_count.csv"
    if gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$ZONE \
        --command="test -f $RELAY_FILE_WITHOUT" --tunnel-through-iap 2>/dev/null; then
        echo "Calculating relay confidence WITHOUT bias..."
        gcloud compute ssh $INSTANCE_NAME \
            --project=$PROJECT \
            --zone=$ZONE \
            --command="cd $REMOTE_DIR && \
                python3 relay_confidence_SigRelayST.py \
                    --input_path=$RELAY_FILE_WITHOUT \
                    --database_dir=$DATABASE_DIR \
                    --organism=$ORGANISM \
                    --output_path=output/V1_Human_Lymph_Node_spatial_no_bias/SigRelayST_V1_Human_Lymph_Node_spatial_no_bias_relay_confidence.csv \
                    --activation_only=1 \
                2>&1 | tee logs/relay_confidence_without_bias_\$(date +%Y%m%d_%H%M%S).log" \
            --tunnel-through-iap
    fi
fi

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  WITH bias:    $REMOTE_DIR/output/V1_Human_Lymph_Node_spatial/"
echo "  WITHOUT bias: $REMOTE_DIR/output/V1_Human_Lymph_Node_spatial_no_bias/"
echo ""
echo "To download results:"
echo "  gcloud compute scp --recurse $INSTANCE_NAME:$REMOTE_DIR/output/ ./output/ --project=$PROJECT --zone=$ZONE"
echo ""

