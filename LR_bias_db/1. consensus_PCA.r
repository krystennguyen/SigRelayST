library(readxl)
library(dplyr)

meta <- read_excel("sigmeta.xlsx")
load("siglist.RData")

meta_h <- meta %>% filter(Organism == "Homo sapiens")
siglist_h <- siglist[names(siglist) %in% meta_h$sig_id]

ligand_groups <- meta_h %>%
  group_by(Ligand) %>%
  summarise(sig_ids = list(sig_id), .groups = "drop")

extract_lfc <- function(id) {
  if (!id %in% names(siglist_h)) return(NULL)
  entry <- siglist_h[[id]]
  if (!"lfc" %in% names(entry)) return(NULL)
  vec <- entry$lfc
  if (is.null(names(vec))) return(NULL)
  vec
}

make_consensus <- function(sig_ids) {
  if (length(sig_ids) == 1) {
    vec <- extract_lfc(sig_ids[[1]])
    if (is.null(vec)) return(NULL)
    return(list(type = "single", sig = vec, n = 1))
  }
  
  mlist <- lapply(sig_ids, extract_lfc)
  mlist <- mlist[!sapply(mlist, is.null)]
  if (length(mlist) == 0) return(NULL)
  
  genes <- sort(unique(unlist(lapply(mlist, names))))
  
  mats <- sapply(mlist, function(v) {
    x <- numeric(length(genes))
    names(x) <- genes
    x[names(v)] <- v
    x
  })
  
  keep <- apply(mats, 1, sd) > 0
  mats <- mats[keep, , drop = FALSE]
  if (nrow(mats) < 10) return(NULL)
  
  pca <- prcomp(t(mats), scale. = TRUE)
  w <- pca$x[, 1]
  
  consensus <- as.numeric(mats %*% w)
  names(consensus) <- rownames(mats)
  
  list(type = "pca", sig = consensus, n = ncol(mats))
}

results <- list()

for (i in seq_len(nrow(ligand_groups))) {
  lig <- ligand_groups$Ligand[i]
  ids <- ligand_groups$sig_ids[[i]]
  out <- make_consensus(ids)
  if (!is.null(out)) results[[lig]] <- out
}

signature_table <- do.call(
  rbind,
  lapply(names(results), function(lig) {
    data.frame(
      sigid = paste0("consensus_", lig),
      ligand = lig,
      n_signatures = results[[lig]]$n,
      type = results[[lig]]$type,
      stringsAsFactors = FALSE
    )
  })
)

sig_list <- lapply(results, function(x) x$sig)
all_genes <- sort(unique(unlist(lapply(sig_list, names))))

consensus_matrix <- do.call(
  cbind,
  lapply(sig_list, function(v) {
    x <- numeric(length(all_genes))
    names(x) <- all_genes
    x[names(v)] <- v
    x
  })
)

rownames(consensus_matrix) <- all_genes
colnames(consensus_matrix) <- names(results)

write.csv(signature_table, "ligand_signature_summary.csv", row.names = FALSE)
write.csv(consensus_matrix, "ligand_consensus_matrix.csv")
