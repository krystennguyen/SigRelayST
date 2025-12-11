library(tidyverse)

# Read input from Step 2
df <- read_csv("SigRelayST_signature_all.csv")

df_clean <- df %>%
  filter(
    is.finite(n_upregulated),
    is.finite(n_downregulated),
    is.finite(total_strength),
    !is.na(n_significant_genes),
    perturbed_receptors != "",
    !is.na(perturbed_receptors)
  )

df_clean <- df_clean %>%
  mutate(
    signature_score = total_strength / n_significant_genes
  )

median_score <- median(df_clean$signature_score)

df_clean <- df_clean %>%
  mutate(
    signature_bias = signature_score / median_score
  )

lr_bias_table <- df_clean %>%
  select(ligand, perturbed_receptors, signature_bias) %>%
  separate_rows(perturbed_receptors, sep = ",") %>%
  rename(
    receptor = perturbed_receptors,
    lr_bias = signature_bias
  )  

# Write final database directly to database/ folder
write_csv(
  lr_bias_table %>% select(ligand, receptor, lr_bias, lr_bias_clean, scaled_bias),
  "../database/SigRelayST_LR_pair_bias.csv"
)
