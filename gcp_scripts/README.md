# GCP Scripts to Run SigRelayST



### Setup Scripts

- **`create_dl_vm_instance.sh`** - One-time setup
  - Creates Deep Learning VM instance (if needed)
  - Installs dependencies (PyTorch, torch-geometric, etc.)
  - Run this once before using other scripts

### Preprocess+Training Scripts

- **`run_training_comparison.sh`** - Main pipeline script
  - Transfers files to VM
  - Preprocesses data (with/without signature bias)
  - Launches training for both models

- **`transfer_to_dl_instance.sh`** - File transfer only
  - Transfers Python scripts, data, and databases to VM

- **`run_preprocess_with_bias.sh`** - Preprocessing with signature bias
- **`run_preprocess_without_bias.sh`** - Preprocessing without signature bias

### Post-Processing Scripts

- **`run_postprocess_and_visualize.sh`** - Postprocess and visualize results
  - Downloads model outputs
  - Runs postprocessing for both models
  - Generates visualizations
  - **Run after training completes**

- **`run_relay_confidence.sh`** - Relay confidence analysis
  - Extracts relay patterns
  - Calculates confidence scores
  - **Run after postprocessing**

## Workflow

### 1. Initial Setup 

```bash
# Configure GCP settings
cp gcp_scripts/gcp_config.example.sh gcp_scripts/gcp_config.sh
# Edit gcp_config.sh

# Create VM and install dependencies
bash gcp_scripts/create_dl_vm_instance.sh
```

### 2. Run Training Pipeline

```bash
# Full pipeline: transfer + preprocess + train
bash gcp_scripts/run_training_comparison.sh
```

This will:
1. Check/start VM instance
2. Transfer files to VM
3. Preprocess data (with and without bias)
4. Launch training for both models in parallel

### 3. Monitor Training

```bash
# Check training logs
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command='tail -f ~/SigRelayST/logs/training_with_bias.log' \
    --tunnel-through-iap

# Check if processes are running
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --command='ps aux | grep run_SigRelayST' \
    --tunnel-through-iap
```

### 4. Post-Processing (After Training)

```bash
# Postprocess and visualize
bash gcp_scripts/run_postprocess_and_visualize.sh

# Relay confidence analysis
bash gcp_scripts/run_relay_confidence.sh
```


### Download Results

```bash
# Download all outputs
gcloud compute scp --recurse \
    $INSTANCE_NAME:~/SigRelayST/output/ \
    ./output/ \
    --project=$PROJECT --zone=$ZONE \
    --tunnel-through-iap

```


## File Structure on VM

```
~/SigRelayST/
├── *.py                    # Python scripts
├── data/                    # Input data
├── input_graph/            # Preprocessed graphs
├── metadata/               # Metadata files
├── model/                  # Trained models
├── embedding_data/         # Embeddings
├── output/                 # Results
├── logs/                   # Log files
├── database/               # Database files
└── gcp_scripts/            # GCP scripts
```



