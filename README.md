# SigRelayST: Signature-based Relay for Spatial Transcriptomics 

SigRelayST extends CellNEST with signature-based bias terms derived from the Lignature database to improve cell-cell communication inference from spatial transcriptomics data.



## Outputs (no bias; with-bias equivalents present in sibling folder)
- `SigRelayST_V1_Human_Lymph_Node_spatial_no_bias_relay_confidence.csv`
- `SigRelayST_V1_Human_Lymph_Node_spatial_no_bias_relay_pattern_{count,cell_info}.csv`
- `SigRelayST_V1_Human_Lymph_Node_spatial_no_bias_relay_pattern_histograms.html`
- `SigRelayST_no_bias_{top20percent,allCCC,ccc_list_top1500}.csv`
- `SigRelayST_no_bias_{attention_score_distribution,component_plot_1500,mygraph_top1500}.html`
- `SigRelayST_no_bias_histogram_byFrequency_{plot_top1500,table_top1500}.(html|csv)`

## Key scripts
- `data_preprocess_SigRelayST.py` — build input graph (with or without signature bias)
- `run_SigRelayST.py` (+ `CCC_gat.py`, `CCC_gat_split.py`, `GATv2Conv_SigRelayST.py`) — training
- `output_postprocess_SigRelayST.py`, `output_visualization_SigRelayST.py` — postprocess & visualize
- `extract_relay_SigRelayST.py` — relay pattern extraction
- `relay_confidence_SigRelayST.py` — relay confidence scoring

## Databases
- `database/CellNEST_database.csv`
- `database/sigrelay_LR_pair_bias_capped.csv` (signature bias)
- `database/human_tf_target.csv`, `database/human_signaling_ppi.csv`

## System requirements
- Python 3.10+
- Recommended: >=32 GB RAM; GPU 16–24 GB if split training works; CPU-only works but is slow.
- Dependencies: torch, torch-geometric, scanpy, numpy, pandas, scipy, altair, networkx, pyarrow, qnorm (see `env/requirements.txt` or `env/environment.yml`).

## Local run (bash)
From repo root:
```bash
# 1) Preprocess
python3 data_preprocess_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --data_from=data/V1_Human_Lymph_Node_spatial/ \
  --signature_database=database/sigrelay_LR_pair_bias_capped.csv \
  --signature_bias_type=scaled_bias \
  --filter_min_cell=1 \
  --threshold_gene_exp=98 \
  --distance_measure=knn --k=50

# (no-bias variant: omit --signature_database/--signature_bias_type)

# 2) Train (adjust total_subgraphs and hidden for memory; CPU example)
CUDA_VISIBLE_DEVICES="" python3 run_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --training_data=input_graph/ \
  --embedding_path=embedding_data/ \
  --model_path=model/ \
  --metadata_to=metadata/ \
  --model_name=SigRelayST_visium_hd_with_bias \
  --run_id=1 \
  --num_epoch=50000 \
  --hidden=512 \
  --heads=1 \
  --dropout=0.0 \
  --lr_rate=0.00001 \
  --total_subgraphs=16 \
  --manual_seed=yes --seed=1

# 3) Postprocess
python3 output_postprocess_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --model_name=SigRelayST_visium_hd_with_bias \
  --total_runs=1 \
  --embedding_path=embedding_data/ \
  --metadata_from=metadata/ \
  --data_from=input_graph/ \
  --output_path=output/

# 4) Visualize
python3 output_visualization_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --model_name=SigRelayST_visium_hd_with_bias \
  --top_percent=20 \
  --output_path=output/

# 5) Relay extraction
python3 extract_relay_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --top_ccc_file=output/V1_Human_Lymph_Node_spatial/SigRelayST_with_bias_top20percent.csv \
  --output_path=output/V1_Human_Lymph_Node_spatial/

# 6) Relay confidence
python3 relay_confidence_SigRelayST.py \
  --input_path=output/V1_Human_Lymph_Node_spatial/SigRelayST_V1_Human_Lymph_Node_spatial_relay_pattern_count.csv \
  --database_dir=database/ \
  --organism=human \
  --output_path=output/V1_Human_Lymph_Node_spatial/SigRelayST_V1_Human_Lymph_Node_spatial_relay_confidence.csv
```
Repeat steps with the no-bias model by switching `model_name` and using the no-bias `top20percent` and relay pattern files.

## Cloud (anonymized example)
Use only these scripts in `cloud_scripts/`:
- `create_dl_vm_instance.sh` — create a DL VM
- `run_full_pipeline_40000.sh` — end-to-end preprocess + train (adjust data/model names if needed)
- `run_postprocess_and_visualize.sh` — download embeddings and run local postprocess/visualization
- `run_relay_confidence_pvalue_analysis.sh` — run relay extraction/confidence on the cloud

Typical flow:
```bash
# 0) Create VM
bash cloud_scripts/create_dl_vm_instance.sh

# 1) Run full pipeline (adjust paths, project/zone inside the script)
bash cloud_scripts/run_full_pipeline_40000.sh

# 2) Download results and postprocess locally
bash cloud_scripts/run_postprocess_and_visualize.sh

# 3) Relay extraction/confidence on cloud
bash cloud_scripts/run_relay_confidence_pvalue_analysis.sh
```

## Notes
- `total_subgraphs` controls memory: increase (e.g., 16–32) for large graphs.
- CPU training works but is slow; GPU recommended if available and split works.
- Outputs for with/without bias are stored separately under `output/`.
