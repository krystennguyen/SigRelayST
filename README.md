# SigRelayST: Signature-based Relay for Spatial Transcriptomics 

SigRelayST extends CellNEST with signature-based bias terms derived from the Lignature database to improve cell-cell communication inference from spatial transcriptomics data.

## Requirements: install dependencies from `env/requirements.txt` or `environment.yml`:

```bash
pip install -r env/requirements.txt

# Or conda 
conda env create -f env/environment.yml
conda activate sigrelayst
```

**Note**: `torch-sparse` and `torch-scatter` may need to be installed from PyTorch Geometric wheels matching your PyTorch and CUDA versions.

## Input Data

### Required Files
- **Spatial Transcriptomics Data**: 
  - Visium format: `filtered_feature_bc_matrix.h5` and `spatial/` folder
  - AnnData format: `.h5ad` file with `obsm['spatial']` coordinates
  - Alternative: `.mtx` file + tissue position file

### Database Files
- `database/CellNEST_database.csv` - Ligand-receptor pair database (required)
- `database/SigRelayST_LR_pair_bias.csv` - Signature bias database (required for WITH bias model)
- `database/human_tf_target.csv` - TF-target database (required for relay confidence)
- `database/human_signaling_ppi.csv` - PPI database (required for relay confidence)

## File Descriptions

### Core Scripts

#### `data_preprocess_SigRelayST.py`
**Purpose**: Preprocess spatial transcriptomics data and build input graph for training.

**Key Parameters**:
- `--data_name` (required): Name identifier for the dataset
- `--data_from` (required): Path to input data directory (Space Ranger output or .h5ad file)
- `--data_type`: Data format (`visium` or `anndata`, default: `visium`)
- `--signature_database`: Path to signature bias CSV (optional, enables WITH bias model)
- `--signature_bias_type`: Type of signature bias (`scaled_bias`, default)
- `--filter_min_cell`: Minimum cells expressing a gene for filtering (default: 1)
- `--threshold_gene_exp`: Percentile threshold for active genes (default: 98)
- `--distance_measure`: Neighborhood method (`knn` or `fixed`, default: `fixed`)
- `--k`: Number of nearest neighbors if using k-NN (default: 50)
- `--data_to`: Output path for input graph (default: `input_graph/`)
- `--metadata_to`: Output path for metadata (default: `metadata/`)

**Outputs**:
- `input_graph/{data_name}/{data_name}_adjacency_records` - Graph edges and attributes (gzipped pickle)
- `input_graph/{data_name}/{data_name}_cell_vs_gene_quantile_transformed` - Normalized expression matrix (gzipped pickle)
- `metadata/{data_name}/{data_name}_barcode_info` - Cell metadata (gzipped pickle)
- `metadata/{data_name}/{data_name}_self_loop_record` - Self-loop information (gzipped pickle)

## Project Structure

```
SigRelayST/
├── *.py                    # Python scripts
├── data/                   # Input data
├── input_graph/            # Preprocessed graphs
├── metadata/               # Metadata files
├── model/                  # Trained models
├── embedding_data/        # Embeddings
├── output/                 # Results
├── logs/                   # Log files
├── database/               # Database files
└── gcp_scripts/            # GCP scripts (see gcp_scripts/README.md)
```

#### `run_SigRelayST.py`
**Purpose**: Train SigRelayST model using Deep Graph Infomax (DGI) with GATv2Conv layers.

**Key Parameters**:
- `--data_name` (required): Dataset name (must match preprocessing)
- `--model_name` (required): Model identifier
- `--run_id` (required): Run number (1, 2, 3, ... for ensemble)
- `--num_epoch`: Number of training epochs (default: 60000)
- `--hidden`: Hidden layer dimension / embedding size (default: 512)
- `--heads`: Number of attention heads (default: 1)
- `--dropout`: Dropout rate (default: 0.0)
- `--lr_rate`: Learning rate (default: 0.00001)
- `--total_subgraphs`: Number of graph splits for memory management (default: 1, increase for large graphs)
- `--manual_seed`: Use fixed random seed (`yes` or `no`, default: `no`)
- `--seed`: Random seed value (required if `--manual_seed=yes`)
- `--training_data`: Path to input graph (default: `input_graph/`)
- `--embedding_path`: Path to save embeddings (default: `embedding_data/`)
- `--model_path`: Path to save model checkpoints (default: `model/`)
- `--load`: Load previous checkpoint (0 or 1, default: 0)
- `--load_model_name`: Name of model to load (if `--load=1`)

**Outputs**:
- `model/{data_name}/DGI_{model_name}_r{run_id}.pth.tar` - Model checkpoint
- `embedding_data/{data_name}/{model_name}_r{run_id}_Embed_X` - Node embeddings (gzipped pickle)
- `embedding_data/{data_name}/{model_name}_r{run_id}_attention` - Attention scores (gzipped pickle)
- `logs/DGI_{model_name}_r{run_id}_loss_curve.csv` - Training loss curve

#### `output_postprocess_SigRelayST.py`
**Purpose**: Extract and rank cell-cell communications from trained model.

**Key Parameters**:
- `--data_name` (required): Dataset name
- `--model_name` (required): Model name (must match training)
- `--total_runs` (required): Number of model runs (for ensemble, default: 5)
- `--top_percent`: Top N% of CCCs to extract (default: 20)
- `--output_all`: Output all communications (0 or 1, default: 1)
- `--embedding_path`: Path to embeddings (default: `embedding_data/`)
- `--metadata_from`: Path to metadata (default: `metadata/`)
- `--data_from`: Path to input graph (default: `input_graph/`)
- `--output_path`: Path to save results (default: `output/`)

**Outputs**:
- `output/{data_name}/{model_name}_top20percent.csv` - Top-ranked CCCs
- `output/{data_name}/{model_name}_allCCC.csv` - All CCCs with attention scores
- `output/{data_name}/{model_name}_ccc_list_top1500.csv` - Top 1500 CCCs for visualization

#### `output_visualization_SigRelayST.py`
**Purpose**: Generate interactive visualizations of CCC results.

**Key Parameters**:
- `--data_name` (required): Dataset name
- `--model_name` (required): Model name
- `--top_percent`: Top N% to visualize (default: 20)
- `--top_edge_count`: Number of top edges to plot (default: 1500, use -1 for all)
- `--output_path`: Path to save visualizations (default: `output/`)
- `--metadata_from`: Path to metadata (default: `metadata/`)

**Outputs**:
- `output/{data_name}/{model_name}_attention_score_distribution.html` - Attention score histogram
- `output/{data_name}/{model_name}_component_plot_1500.html` - Spatial component visualization
- `output/{data_name}/{model_name}_mygraph_top1500.html` - Interactive network graph
- `output/{data_name}/{model_name}_histogram_byFrequency_plot_top1500.html` - Frequency histograms

#### `extract_relay_SigRelayST.py`
**Purpose**: Extract multi-step communication relay patterns from top CCCs.

**Key Parameters**:
- `--data_name` (required): Dataset name
- `--top_ccc_file` (required): Path to top CCC CSV from postprocessing
- `--output_path`: Path to save relay patterns (default: `output/`)
- `--metadata`: Path to metadata directory (default: `metadata/`)

**Outputs**:
- `output/{data_name}/SigRelayST_{data_name}_relay_pattern_count.csv` - Relay patterns with counts
- `output/{data_name}/SigRelayST_{data_name}_relay_pattern_cell_info` - Cell-level relay information
- `output/{data_name}/SigRelayST_{data_name}_relay_pattern_histograms.html` - Relay frequency visualization

#### `relay_confidence_SigRelayST.py`
**Purpose**: Validate relay patterns against TF-target and PPI databases.

**Key Parameters**:
- `--input_path` (required): Path to relay pattern CSV from extraction
- `--database_dir` (required): Directory containing TF-target and PPI databases
- `--organism` (required): Organism (`human` or `mouse`)
- `--output_path` (required): Path to save confidence scores
- `--activation_only`: Only consider activating TFs (0 or 1, default: 1)

**Outputs**:
- `{output_path}` - CSV with relay patterns, confidence scores, paths, and TFs

### Supporting Files

- `CCC_gat.py` - Training logic for non-split graphs
- `CCC_gat_split.py` - Training logic for split graphs (memory-efficient)
- `GATv2Conv_SigRelayST.py` - GATv2Conv layer with signature bias support
- `altairThemes.py` - Visualization theme configuration

## Local Run 

### Step 1: Preprocess Data

**WITH signature bias:**
```bash
python3 data_preprocess_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --data_from=data/V1_Human_Lymph_Node_spatial/ \
  --signature_database=database/SigRelayST_LR_pair_bias.csv \
  --signature_bias_type=scaled_bias \
  --filter_min_cell=1 \
  --threshold_gene_exp=98 \
  --distance_measure=knn --k=50
```

**WITHOUT signature bias:**
```bash
python3 data_preprocess_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial_no_bias \
  --data_from=data/V1_Human_Lymph_Node_spatial/ \
  --filter_min_cell=1 \
  --threshold_gene_exp=98 \
  --distance_measure=knn --k=50
```

### Step 2: Train Model
```bash
nohup python3 run_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --training_data=input_graph/ \
  --embedding_path=embedding_data/ \
  --model_path=model/ \
  --metadata_to=metadata/ \
  --model_name=SigRelayST_with_bias \
  --run_id=1 \
  --num_epoch=40000 \
  --hidden=512 \
  --heads=1 \
  --dropout=0.0 \
  --lr_rate=0.00001 \
  --total_subgraphs=1 \
  --manual_seed=yes --seed=1 \
  > logs/training_with_bias.log 2>&1 &
```

### Step 3: Postprocess

```bash
python3 output_postprocess_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --model_name=SigRelayST_with_bias \
  --total_runs=1 \
  --top_percent=20 \
  --output_all=1 \
  --embedding_path=embedding_data/ \
  --metadata_from=metadata/ \
  --data_from=input_graph/ \
  --output_path=output/
```

### Step 4: Visualize

```bash
python3 output_visualization_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --model_name=SigRelayST_with_bias \
  --top_percent=20 \
  --output_path=output/
```

### Step 5: Extract Relay Patterns

```bash
python3 extract_relay_SigRelayST.py \
  --data_name=V1_Human_Lymph_Node_spatial \
  --top_ccc_file=output/V1_Human_Lymph_Node_spatial/SigRelayST_with_bias_top20percent.csv \
  --output_path=output/V1_Human_Lymph_Node_spatial/
```

### Step 6: Calculate Relay Confidence

```bash
python3 relay_confidence_SigRelayST.py \
  --input_path=output/V1_Human_Lymph_Node_spatial/SigRelayST_V1_Human_Lymph_Node_spatial_relay_pattern_count.csv \
  --database_dir=database/ \
  --organism=human \
  --output_path=output/V1_Human_Lymph_Node_spatial/SigRelayST_V1_Human_Lymph_Node_spatial_relay_confidence.csv \
  --activation_only=1
```

**Repeat steps 2-6** for the WITHOUT bias model by changing `--data_name` and `--model_name` accordingly.

## GCP Usage

For running the pipeline on Google Cloud Platform, see the detail in [`gcp_scripts/README.md`](gcp_scripts/README.md).

## Output Files

Output files are saved in the `output/{data_name}/` directory. See individual script descriptions above.
