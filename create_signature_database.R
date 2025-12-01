#!/usr/bin/env Rscript
# Create a database of all signatures with their scores
# Output: CSV file with sigid, ligand, significantly perturbed receptors, and statistics
# All information extracted from siglist.RData
# 
# This script is part of SigRelayST, which extends CellNEST (Zohora et al.) 
# with signature-based bias terms derived from the Lignature database.
# 
# CellNEST Citation:
# Zohora, F. T., et al. "CellNEST: A Graph Neural Network Framework for 
# Cell-Cell Communication Inference from Spatial Transcriptomics Data."
#
# Lignature Citation:
# Xin, Y., et al. "Lignature: A Comprehensive Database of Ligand Signatures to 
# Predict Cell-Cell Communication."
# 
# Example: Rscript create_signature_database.R ../LignatureData database/signatures_all.csv 0.05

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript create_signature_database.R <lignature_dir> <output_csv> [p_threshold]")
}

lignature_dir <- args[1]
output_csv <- args[2]
p_threshold <- if (length(args) >= 3) as.numeric(args[3]) else 0.05

cat("Loading Lignature RData files...\n")

# Load RData files
load(file.path(lignature_dir, "siglist.RData"))
load(file.path(lignature_dir, "sigmeta.RData"))
load(file.path(lignature_dir, "lr_network_hg.RData"))

cat(sprintf("Loaded %d signatures\n", length(siglist)))
cat(sprintf("Loaded %d signature metadata entries\n", nrow(sigmeta)))
cat(sprintf("Loaded %d ligand-receptor pairs\n", nrow(lr_network)))

# Get all unique receptor genes from LR network (for matching perturbed receptors)
all_receptor_genes <- unique(toupper(unlist(strsplit(as.character(lr_network$Rgene), "_"))))
cat(sprintf("Found %d unique receptor genes in LR network\n", length(all_receptor_genes)))

# Initialize results list
results_list <- list()

cat("\nProcessing signatures...\n")
pb <- txtProgressBar(min = 0, max = length(siglist), style = 3)

sig_count <- 0
processed_count <- 0

for (sig_id in names(siglist)) {
  sig_count <- sig_count + 1
  setTxtProgressBar(pb, sig_count)
  
  
  # Get signature data
  sig_data <- siglist[[sig_id]]
  
  if (is.null(sig_data)) {
    next
  }

  # Get ligand from sigL field
  if (!"sigL" %in% names(sig_data)) {
    next
  }
  ligand <- sig_data$sigL
  
  # Extract lfc and padj vectors (they are named vectors with gene symbols as names)  
  if (!"lfc" %in% names(sig_data) || !"padj" %in% names(sig_data)) {
    next
  }
  
  lfc_vec <- sig_data$lfc
  padj_vec <- sig_data$padj
  
  # Get gene names from the names attribute of lfc/padj vectors
  gene_names <- names(lfc_vec)
  
  if (length(lfc_vec) == 0 || length(padj_vec) == 0 || length(gene_names) == 0) {
    next
  }
  
  if (length(lfc_vec) != length(padj_vec) || length(lfc_vec) != length(gene_names)) {
    next
  }
  
  # Filter by p-value threshold (handle NA values)
  sig_idx <- !is.na(padj_vec) & padj_vec < p_threshold
  
  if (sum(sig_idx, na.rm = TRUE) == 0) {
    next
  }
  
  # Get significantly perturbed genes (gene names are in the names attribute)
  sig_genes <- toupper(gene_names[sig_idx])
  lfc_filtered <- lfc_vec[sig_idx]
  padj_filtered <- padj_vec[sig_idx]
  
  # Get all unique receptor genes from LR network (for matching)
  all_receptor_genes <- unique(toupper(unlist(strsplit(as.character(lr_network$Rgene), "_"))))
  
  # Find which of these genes are receptors (match against receptor gene list)
  perturbed_receptors <- intersect(sig_genes, all_receptor_genes)
  
  # Calculate statistics
  n_up <- sum(lfc_filtered > 0)
  n_down <- sum(lfc_filtered < 0)
  strength <- sum(abs(lfc_filtered))
  n_sig <- length(lfc_filtered)
  
  # Store perturbed receptors as comma-separated string (or empty if none)
  receptors_str <- if (length(perturbed_receptors) > 0) {
    paste(sort(perturbed_receptors), collapse = ",")
  } else {
    ""
  }
  
  # Add to results
  processed_count <- processed_count + 1
  results_list[[processed_count]] <- data.frame(
    sigid = sig_id,
    ligand = ligand,
    perturbed_receptors = receptors_str,
    n_upregulated = n_up,
    n_downregulated = n_down,
    total_strength = strength,
    n_significant_genes = n_sig,
    stringsAsFactors = FALSE
  )
}

close(pb)

# Combine results
if (length(results_list) > 0) {
  results <- do.call(rbind, results_list)
} else {
  results <- data.frame(
    sigid = character(),
    ligand = character(),
    perturbed_receptors = character(),
    n_upregulated = integer(),
    n_downregulated = integer(),
    total_strength = numeric(),
    n_significant_genes = integer(),
    stringsAsFactors = FALSE
  )
}

cat(sprintf("\nProcessed %d signatures\n", sig_count))
cat(sprintf("Created %d signature entries (with significant genes)\n", nrow(results)))

# Calculate n_signatures per ligand - how many signatures are there for this ligand?
if (nrow(results) > 0) {
  ligand_counts <- table(results$ligand)
  results$n_signatures <- as.integer(ligand_counts[results$ligand])
} else {
  results$n_signatures <- integer(0)
}

# Reorder columns
results <- results[, c("sigid", "ligand", "perturbed_receptors", "n_upregulated", 
                       "n_downregulated", "total_strength", "n_significant_genes", "n_signatures")]

# Save to CSV
write.csv(results, output_csv, row.names = FALSE)
cat(sprintf("\nSaved to: %s\n", output_csv))

# Print summary
cat("\nSummary:\n")
cat(sprintf("  Unique signatures: %d\n", length(unique(results$sigid))))
cat(sprintf("  Unique ligands: %d\n", length(unique(results$ligand))))
cat(sprintf("  Signatures with perturbed receptors: %d\n", sum(results$perturbed_receptors != "")))
cat(sprintf("  Total entries: %d\n", nrow(results)))

# Show example
if (nrow(results) > 0) {
  cat("\nExample entry:\n")
  example_idx <- min(which(results$perturbed_receptors != ""), 1)
  if (is.na(example_idx)) example_idx <- 1
  if (example_idx <= nrow(results)) {
    cat(sprintf("  sigid: %s\n", results$sigid[example_idx]))
    cat(sprintf("  ligand: %s\n", results$ligand[example_idx]))
    cat(sprintf("  perturbed_receptors: %s\n", results$perturbed_receptors[example_idx]))
    cat(sprintf("  n_upregulated: %d\n", results$n_upregulated[example_idx]))
    cat(sprintf("  n_downregulated: %d\n", results$n_downregulated[example_idx]))
    cat(sprintf("  total_strength: %.2f\n", results$total_strength[example_idx]))
    cat(sprintf("  n_signatures (for this ligand): %d\n", results$n_signatures[example_idx]))
  }
}

