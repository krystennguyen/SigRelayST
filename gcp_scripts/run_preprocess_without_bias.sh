#!/bin/bash
# Preprocess data WITHOUT signature bias

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_gcp_config.sh"

echo "Preprocessing WITHOUT signature bias..."

gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command="cd $REMOTE_DIR && \
        python3 data_preprocess_SigRelayST.py \
            --data_name=V1_Human_Lymph_Node_spatial_no_bias \
            --data_from=data/V1_Human_Lymph_Node_spatial/ \
            --filter_min_cell=1 \
            --threshold_gene_exp=98 \
            --distance_measure=knn --k=50 \
            --data_to=input_graph/ \
            --metadata_to=metadata/ \
        2>&1 | tee logs/preprocessing_without_bias_\$(date +%Y%m%d_%H%M%S).log" \
    --tunnel-through-iap

if [ $? -eq 0 ]; then
    echo "Preprocessing WITHOUT bias completed!"
    exit 0
else
    echo "Preprocessing WITHOUT bias failed"
    exit 1
fi

