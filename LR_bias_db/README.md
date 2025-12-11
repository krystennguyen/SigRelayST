# LR Bias Database Construction Workflow

This directory contains the workflow for constructing the ligand-receptor (LR) pair bias database from the Lignature database. The bias scores derived from Lignature transcriptomic signatures are integrated into SigRelayST's GAT attention mechanism to improve cell-cell communication inference.


### Integration with SigRelayST

The LR bias database is a **prerequisite** for running SigRelayST with signature bias. The workflow transforms raw Lignature signatures into normalized bias scores that are:

1. **Loaded during preprocessing** (`data_preprocess_SigRelayST.py`):
   - Creates a lookup dictionary: `(ligand, receptor) → scaled_bias`
   - Adds `scaled_bias` as the 4th dimension to edge attributes: `[spatial_weight, coexpression_score, relation_id, signature_bias]`

2. **Used in GAT attention** (`GATv2Conv_SigRelayST.py`):
   - Signature bias is normalized and added to attention scores: `alpha = alpha + signature_bias_scale * signature_bias_normalized`
   - The learnable `signature_bias_scale` parameter allows the model to adaptively weight the bias contribution

3. **Enables comparison**:
   - Models trained WITH bias vs WITHOUT bias can be compared to assess the impact of signature information

## Steps

```
Lignature Database (siglist.RData, sigmeta.xlsx)
    ↓
[Step 1] Consensus PCA → ligand_consensus_matrix.csv
    ↓
[Step 2] Summary & Receptor Mapping → SigRelayST_signature_all.csv
    ↓
[Step 3] LR Pair Bias Calculation → database/SigRelayST_LR_pair_bias.csv
```

---

## Step 1: Consensus Signature Construction

**File**: `1. consensus_PCA.r`

**Purpose**: Create consensus ligand signatures from multiple Lignature signatures per ligand using PCA.

### Input Files
- `siglist.RData`: R list object containing all Lignature signatures (log fold change vectors)
- `sigmeta.xlsx`: Excel file with metadata for each signature (ligand, organism, cell type, etc.)

### Process
1. **Filter to human signatures**: Extracts only `Homo sapiens` signatures from metadata
2. **Group by ligand**: Organizes signatures by ligand name
3. **Create consensus**:
   - **Single signature**: Uses the signature directly
   - **Multiple signatures**: Performs PCA on the signature matrix and uses the first principal component as consensus
   - Filters genes with zero variance across signatures
   - Requires minimum 10 genes for PCA consensus

### Output Files
- **`ligand_consensus_matrix.csv`**: Gene × ligand matrix where each column is a consensus signature (log fold changes)
- **`ligand_signature_summary.csv`**: Metadata table with columns:
  - `sigid`: Signature ID (e.g., "consensus_CCL19")
  - `ligand`: Ligand gene symbol
  - `n_signatures`: Number of input signatures used
  - `type`: Consensus method ("single" or "pca")


## Step 2: Ligand Signature Summary & Receptor Mapping

**File**: `2. ligand_consensus_summary.r`

**Purpose**: Compute summary statistics for each ligand signature and map ligands to receptors from the CellNEST database.

### Input Files
- `ligand_consensus_matrix.csv`: Consensus signatures from Step 1
- `CellNEST_database.csv`: Ligand-receptor pair database (must be in working directory or path specified)

### Process
1. **Load consensus matrix**: Reads gene × ligand expression matrix
2. **Compute signature statistics** (for each ligand):
   - `n_upregulated`: Number of genes with LFC > 1
   - `n_downregulated`: Number of genes with LFC < -1
   - `n_significant_genes`: Total genes with |LFC| > 1
   - `total_strength`: Sum of absolute LFC values for all significant genes
3. **Map to receptors**: Links ligands to their receptors from CellNEST database
4. **Generate visualizations**: Creates plots for:
   - Top ligands by transcriptional strength
   - Ligand receptor complexity
   - Receptor count vs transcriptional strength

### Output Files
- **`SigRelayST_signature_all.csv`**: Comprehensive ligand signature table with columns:
  - `sigid`: Signature ID
  - `ligand`: Ligand gene symbol (uppercase)
  - `perturbed_receptors`: Comma-separated list of receptors from CellNEST database
  - `n_upregulated`: Number of upregulated genes
  - `n_downregulated`: Number of downregulated genes
  - `n_significant_genes`: Total significant genes
  - `total_strength`: Total signature strength
  - `n_signatures`: Number of signatures (always 1 after consensus)


---

## Step 3: LR Pair Bias Calculation

**File**: `3. SigRelayST_LR_pair_bias.r`

**Purpose**: Calculate normalized bias scores for each ligand-receptor pair.

### Input Files
- `SigRelayST_signature_all.csv`: Ligand signatures with receptor mappings from Step 2

### Process
1. **Filter valid entries**: Removes rows with:
   - Non-finite values (NaN, Inf)
   - Empty or missing receptor information
2. **Calculate signature score**: 
   - `signature_score = total_strength / n_significant_genes`
   - Average strength per significant gene
3. **Normalize by median**: 
   - `signature_bias = signature_score / median_score`
   - Creates relative bias scores centered around 1.0
4. **Expand to LR pairs**: 
   - Splits comma-separated receptor lists into individual rows
   - One row per ligand-receptor pair
5. **Clean bias values**:
   - Handles non-finite values by replacing with `max_finite * 1.1`
   - Fills remaining NaN with 1.0
   - Creates `lr_bias_clean` column
6. **Apply log2 scaling**:
   - `scaled_bias = log2(lr_bias_clean + 1.0)`
   - Optimizes values for neural network training
7. **Output to database folder**:
   - Saves final database directly to `../database/SigRelayST_LR_pair_bias.csv`
   - Ready for immediate use in SigRelayST

### Output Files
- **`../database/SigRelayST_LR_pair_bias.csv`**: Final LR bias database (used by SigRelayST) with columns:
  - `ligand`: Ligand gene symbol
  - `receptor`: Receptor gene symbol
  - `lr_bias`: Normalized signature bias score (relative to median)
  - `lr_bias_clean`: Cleaned bias values (non-finite values handled)
  - `scaled_bias`: Log2-transformed bias (`log2(lr_bias_clean + 1.0)`) - used in SigRelayST


## Using the Database in SigRelayST

Once Step 3 completes, the final database is automatically saved to `database/SigRelayST_LR_pair_bias.csv` with all required columns including `scaled_bias`. This file is ready to use in SigRelayST preprocessing:

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

The `scaled_bias` column is automatically loaded and integrated into edge features during graph construction.

---

## Understanding Bias Scores

### Bias Score Interpretation

- **`lr_bias`**: Normalized relative to median (median = 1.0)
  - Values > 1.0: Stronger than average transcriptional signature
  - Values < 1.0: Weaker than average transcriptional signature
  - Range: Typically 0.05 to 77 (before scaling)

- **`scaled_bias`**: Log2-transformed for numerical stability
  - Formula: `log2(lr_bias_clean + 1.0)`
  - Range: Typically 0.08 to 6.3
  - Used directly in GAT attention mechanism

### Example Values

From the database:
- **A2M → LRP1**: `lr_bias = 2.995`, `scaled_bias = 1.998` (strong signature)
- **ADM → CALCR**: `lr_bias = 0.205`, `scaled_bias = 0.269` (weak signature)


