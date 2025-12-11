library(dplyr)
library(readr)
library(ggplot2)
library(ggrepel)

## Load consensus matrix
raw <- read.csv("ligand_consensus_matrix.csv", check.names = FALSE)

genes <- raw[[1]]
missing <- which(is.na(genes) | genes == "")
if (length(missing) > 0) {
  genes[missing] <- paste0("GENE_FIX_", seq_along(missing))
}

rownames(raw) <- genes
consensus_matrix <- raw[, -1, drop = FALSE]

cat("Rows =", nrow(consensus_matrix),
    "| Unique genes =", length(unique(rownames(consensus_matrix))), "\n")

## Load CellNEST ligand–receptor database
lr_db <- read.csv("CellNEST_database.csv")

lr_db <- lr_db %>%
  mutate(
    Ligand = toupper(Ligand),
    Receptor = toupper(Receptor)
  )

lr_map <- lr_db %>%
  group_by(Ligand) %>%
  summarise(
    perturbed_receptors = paste(unique(Receptor), collapse = ","),
    n_receptors = n_distinct(Receptor),
    .groups = "drop"
  )

## Ligand-level summary metrics
LFC_TH <- 1

compute_summary <- function(v) {
  list(
    n_upregulated = sum(v >  LFC_TH, na.rm = TRUE),
    n_downregulated = sum(v < -LFC_TH, na.rm = TRUE),
    n_significant_genes = sum(abs(v) > LFC_TH, na.rm = TRUE),
    total_strength = sum(abs(v), na.rm = TRUE)
  )
}

signature_all <- lapply(colnames(consensus_matrix), function(lig) {
  v <- consensus_matrix[, lig]
  stats <- compute_summary(v)
  
  receptors <- lr_map$perturbed_receptors[lr_map$Ligand == toupper(lig)]
  if (length(receptors) == 0) receptors <- ""
  
  data.frame(
    sigid = paste0("consensus_", lig),
    ligand = toupper(lig),
    perturbed_receptors = receptors,
    n_upregulated = stats$n_upregulated,
    n_downregulated = stats$n_downregulated,
    n_significant_genes = stats$n_significant_genes,
    total_strength = stats$total_strength,
    n_signatures = 1,
    stringsAsFactors = FALSE
  )
}) %>% bind_rows()

write.csv(signature_all,
          "SigRelayST_signature_all.csv",
          row.names = FALSE)

## Plot 1: Top ligands by transcriptional strength
sig_top <- signature_all %>%
  arrange(desc(total_strength)) %>%
  slice(1:30) %>%
  mutate(ligand = factor(ligand, levels = ligand))

ggplot(sig_top, aes(x = ligand, y = total_strength)) +
  geom_col(fill = "#00A8A8") +
  coord_flip() +
  theme_bw(base_size = 12) +
  labs(
    title = "Per-ligand Total Transcriptional Strength",
    x = "Ligand",
    y = "Total Strength (Σ |LFC|)"
  )

## Plot 2: Ligand receptor complexity
lr_top <- lr_map %>%
  arrange(desc(n_receptors)) %>%
  slice(1:30)

ggplot(lr_top,
       aes(x = reorder(Ligand, n_receptors),
           y = n_receptors)) +
  geom_col(fill = "orange") +
  coord_flip() +
  theme_bw(base_size = 12) +
  labs(
    title = "Top 30 Ligands by Receptor Count",
    x = "Ligand",
    y = "# Receptors"
  )

## Plot 3: Receptor count vs transcriptional strength
merged <- lr_map %>%
  rename(ligand = Ligand) %>%
  inner_join(
    signature_all %>% select(ligand, total_strength),
    by = "ligand"
  )

merged$label <- ifelse(
  merged$n_receptors >= 25 | merged$total_strength >= 2e7,
  merged$ligand,
  ""
)

ggplot(merged, aes(x = n_receptors, y = total_strength)) +
  geom_point(color = "#D100D1", size = 3) +
  geom_text_repel(
    aes(label = label),
    size = 3,
    max.overlaps = Inf,
    segment.color = "grey70"
  ) +
  theme_bw(base_size = 12) +
  labs(
    title = "Ligand Receptor Count vs Transcriptional Strength",
    x = "# Receptors",
    y = "Total Strength"
  )

