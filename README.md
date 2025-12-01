# SigRelayST: Signature-based Relay for Spatial Transcriptomics

SigRelayST extends CellNEST with signature-based bias terms derived from the Lignature database to improve cell-cell communication inference from spatial transcriptomics data.

## Citation

This work extends CellNEST and uses the Lignature database:

**CellNEST Citation:**
Zohora, F. T., et al. "CellNEST: A Graph Neural Network Framework for Cell-Cell Communication Inference from Spatial Transcriptomics Data."

**Lignature Citation:**
Xin, Y., et al. "Lignature: A Comprehensive Database of Ligand Signatures to Predict Cell-Cell Communication."

## Workflow

1. **Create Signature Database**: `create_signature_database.R`
   - Extracts signature information from Lignature RData files
   - Output: `database/signatures_all.csv`

2. **Preprocess Data**: `data_preprocess_SigRelayST.py`
   - Constructs graph with signature bias in edge features
   - Output: `input_graph/{data_name}/{data_name}_adjacency_records`

3. **Train Model**: `run_SigRelayST.py`
   - Trains GATv2 with DGI using signature bias
   - Output: `model/{data_name}/DGI_{model_name}_r{run_id}.pth.tar`

4. **Postprocess**: `output_postprocess_SigRelayST.py`
   - Generates communication lists and attention scores
   - Output: `output/{data_name}/{model_name}_allCCC.csv`

5. **Visualize**: `output_visualization_SigRelayST.py`
   - Creates interactive visualizations
   - Output: HTML files in `output/{data_name}/`

## Key Features

- Signature-based bias terms from Lignature database
- Tunable bias scaling parameter in GATv2 attention mechanism
- Compatible with CellNEST workflow and outputs

