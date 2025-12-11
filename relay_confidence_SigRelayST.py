# SigRelayST: Relay confidence scoring
# Adapted from CellNEST relay_confidence.py
# Written By: Deisha Paliwal (original), Adapted for SigRelayST

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from collections import deque
import networkx as nx
import os
import argparse

def query_tf(
        grn_db: pd.DataFrame, 
        target_gene: str,
        activation_only: bool,
    ) -> Dict[float, List[str]]:
    """Query transcription factors for a target gene from GRN database."""
    if activation_only:
        target_df = grn_db[(grn_db["target"] == target_gene) & (grn_db["mode"] == 1)]
    else:
        target_df = grn_db[grn_db["target"] == target_gene]
    if target_df.empty:
        return {}
    tf_scores = {}
    for tf, score in zip(target_df["source"], target_df["confidence_score"]):
        tf_scores.setdefault(score, []).append(tf) 

    return tf_scores

def first_weighted_path(
        graph: nx.Graph, 
        start: str, 
        end_nodes: List[str],
    ) -> Tuple[Union[float, None], Union[List[str], None]]:
    """Find first weighted path from start to any end node using BFS."""
    if start not in graph:
        return None, None
    queue = deque([(start, [start], 1.0)])  # (current node, path, cumulative score)
    while queue:
        current_node, path, current_score = queue.popleft()
        if current_node in end_nodes:
            return current_score, path
        visited = set(path)
        for neighbor in graph[current_node]:
            if neighbor not in visited:  # prevent cycles
                weight = graph[current_node][neighbor]['weight']
                new_score = current_score * weight
                new_path = path + [neighbor]
                queue.append((neighbor, new_path, new_score))

    return None, None  # no valid path found

def query_relay_network(
        input_path: str, 
        database_dir: str,
        organism: str, 
        output_path: str,
        activation_only: bool = True
    ) -> None:
    """Query confidence scores for relay network patterns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    relay_network = pd.read_csv(input_path)
    
    # Check if 'Relay Patterns' column exists
    if 'Relay Patterns' not in relay_network.columns:
        # Try alternative column names
        if 'Relay Pattern' in relay_network.columns:
            relay_network['Relay Patterns'] = relay_network['Relay Pattern']
        else:
            raise ValueError("Input file must contain 'Relay Patterns' column")
    
    grn_db = pd.read_csv(os.path.join(database_dir, f"{organism}_tf_target.csv"))
    ppi_db = pd.read_csv(os.path.join(database_dir, f"{organism}_signaling_ppi.csv"))
    
    # Extract ligand-receptor pairs from relay patterns
    # Format: "LIGAND1-RECEPTOR1 to LIGAND2-RECEPTOR2"
    relay_network[['ligand_1', 'receptor_1', 'ligand_2', 'receptor_2']] = \
        relay_network["Relay Patterns"].str.extract(r'(\w+)-(\w+) to (\w+)-(\w+)')
    
    if organism == "mouse":
        for col in ['ligand_1', 'receptor_1', 'ligand_2', 'receptor_2']:
            relay_network[col] = relay_network[col].str.capitalize()
    
    grn_targets = set(grn_db["target"])
    ppi_sources = set(ppi_db["source"])
    all_results = []
    
    for _, row in relay_network.iterrows():
        receptor_1 = row["receptor_1"]
        ligand_2 = row["ligand_2"]
        source_gene = receptor_1 if receptor_1 in ppi_sources else f"{row['ligand_1']}-{receptor_1}"
        target_gene = ligand_2 if ligand_2 in grn_targets else f"{ligand_2}-{row['receptor_2']}"
        
        result = {
            "relay_pattern": row["Relay Patterns"],
            "receptor_1": source_gene,
            "ligand_2": target_gene,
            "combined_score": None,
            "tf": None,
            "path": ""
        }
        
        if target_gene not in grn_targets:
            result["path"] += "ligand 2 not found; "
        if source_gene not in ppi_sources:
            result["path"] += "receptor 1 not found; "
            
        if target_gene in grn_targets and source_gene in ppi_sources:
            tf_dict = query_tf(grn_db, target_gene, activation_only)
            if tf_dict:
                best_score = 0.0
                best_path = None
                all_tfs = [tf for tfs in tf_dict.values() for tf in tfs]
                for weight in np.arange(0.5, 0.0, -0.1):
                    filtered_ppi_db = ppi_db[ppi_db["experimental_score"] >= weight]
                    G = nx.DiGraph()
                    for _, ppi_row in filtered_ppi_db.iterrows():
                        G.add_edge(ppi_row["source"], ppi_row["target"], weight=ppi_row["experimental_score"])
                    ppi_score, putative_path = first_weighted_path(G, source_gene, all_tfs)
                    if putative_path and ppi_score is not None:
                        grn_score = next((score for score, tfs in tf_dict.items() if putative_path[-1] in tfs), None)
                        if grn_score is not None:
                            total_score = ppi_score * grn_score
                            if total_score > best_score:
                                best_score = total_score
                                best_path = putative_path
                if best_path:
                    result.update({
                        "path": "; ".join(best_path),
                        "combined_score": best_score,
                        "tf": best_path[-1]
                    })
                else:
                    result["path"] += "No valid path found."
            else:
                result["path"] += "No TFs found."
                
        pd.DataFrame([result]).to_csv(
            output_path,
            mode='a',
            header=not os.path.exists(output_path),
            index=False
        )

def main():
    parser = argparse.ArgumentParser(description="Query confidence score of a relay network for SigRelayST")
    parser.add_argument('--input_path', type=str, required=True, 
                       help="Path to csv file containing relay network outputs from SigRelayST. Must contain column 'Relay Patterns'")
    parser.add_argument('--database_dir', type=str, required=True, 
                       help="Directory containing PPI and TF-target gene databases")
    parser.add_argument('--organism', type=str, required=True, 
                       help="Organism profiled in spatial transcriptomics experiment", 
                       choices=["human", "mouse"])
    parser.add_argument('--output_path', type=str, required=True, 
                       help="Path to csv file to write confidence scoring output")
    parser.add_argument('--activation_only', type=int, default=1,
                       help="Only consider activating TFs (1) or all TFs (0)")
    args = parser.parse_args()

    query_relay_network(
        input_path=args.input_path,
        database_dir=args.database_dir,
        organism=args.organism, 
        output_path=args.output_path,
        activation_only=bool(args.activation_only)
    )

if __name__ == "__main__":
    main()

