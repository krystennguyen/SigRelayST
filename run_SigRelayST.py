# SigRelayST: Signature-based Relay for Spatial Transcriptomics
# This script extends CellNEST with signature-based bias terms derived from the Lignature database.
#
# CellNEST Citation:
# Zohora, F. T., et al. "CellNEST: A Graph Neural Network Framework for 
# Cell-Cell Communication Inference from Spatial Transcriptomics Data."
#

import os
import sys
import numpy as np
from datetime import datetime 
import time
import random
import argparse
import torch
from torch_geometric.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument( '--data_name', type=str, help='Name of the dataset') #default='PDAC_64630', 
    parser.add_argument( '--model_name', type=str, help='Provide a model name')
    parser.add_argument( '--run_id', type=int, help='Please provide a running ID, for example: 0, 1, 2, etc. Five runs are recommended.' )
    #=========================== default is set ======================================
    parser.add_argument( '--num_epoch', type=int, default=60000, help='Number of epochs or iterations for model training')
    parser.add_argument( '--model_path', type=str, default='model/', help='Path to save the model state') # We do not need this for output generation  
    parser.add_argument( '--embedding_path', type=str, default='embedding_data/', help='Path to save the node embedding and attention scores') 
    parser.add_argument( '--hidden', type=int, default=512, help='Hidden layer dimension (dimension of node embedding)')
    parser.add_argument( '--training_data', type=str, default='input_graph/', help='Path to input graph. ')
    parser.add_argument( '--heads', type=int, default=1, help='Number of heads in the attention model')
    parser.add_argument( '--dropout', type=float, default=0)
    parser.add_argument( '--lr_rate', type=float, default=0.00001)
    parser.add_argument( '--manual_seed', type=str, default='no')
    parser.add_argument( '--seed', type=int )
    #parser.add_argument( '--split', type=int, default=0)
    parser.add_argument( '--total_subgraphs', type=int, default=1)
    parser.add_argument( '--metadata_to', type=str, default='metadata/', help='Path to save the metadata')
    #=========================== optional ======================================
    parser.add_argument( '--load', type=int, default=0, help='Load a previously saved model state')  
    parser.add_argument( '--load_model_name', type=str, default='None' , help='Provide the model name that you want to reload')
    #============================================================================
    args = parser.parse_args() 

    #parser.add_argument( '--options', type=str)
    #parser.add_argument( '--withFeature', type=str, default='r1') 
    #parser.add_argument( '--workflow_v', type=int, default=1)
    #parser.add_argument( '--datatype', type=str)

    '''
    if args.total_subgraphs > 1 :
        args.training_data = args.training_data + args.data_name + '/' + args.data_name + '_' + 'graph_bag'
    else:
        args.training_data = args.training_data + args.data_name + '/' + args.data_name + '_' + 'adjacency_records'
    '''

    args.training_data = args.training_data + args.data_name + '/' + args.data_name + '_' + 'adjacency_records'

    if args.total_subgraphs > 1 :
        node_id_sorted = args.metadata_to + args.data_name + '/'+ args.data_name+'_'+'node_id_sorted_xy'

    args.embedding_path = args.embedding_path + args.data_name +'/'
    args.model_path = args.model_path + args.data_name +'/'
    args.model_name = args.model_name + '_r' + str(args.run_id)



    print(args.data_name+', '+str(args.heads)+', '+args.training_data+', '+str(args.hidden) )

    if args.manual_seed == 'yes':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path) 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path) 

    print ('------------------------Model and Training Details--------------------------')
    print(args) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.total_subgraphs == 1:
        from CCC_gat import get_graph, train_SigRelayST
        # data preparation
        # Try to find expression matrix automatically
        training_dir = os.path.dirname(args.training_data)
        data_name_base = os.path.basename(args.training_data).replace('_adjacency_records', '')
        expression_matrix_path = os.path.join(training_dir, data_name_base + '_cell_vs_gene_quantile_transformed')
        if not os.path.exists(expression_matrix_path):
            expression_matrix_path = ''
        data_loader, num_feature, edge_dim = get_graph(args.training_data, expression_matrix_path=expression_matrix_path)
        # Update encoder edge_dim based on actual data
        from CCC_gat import Encoder
        # Create a temporary encoder to update edge_dim (will be recreated in train_SigRelayST)
        # Actually, we'll pass edge_dim to train_SigRelayST
        # train the model
        DGI_model = train_SigRelayST(args, data_loader=data_loader, in_channels=num_feature, edge_dim=edge_dim)
        # training done
    elif args.total_subgraphs > 1:
        from CCC_gat_split import get_split_graph, train_SigRelayST #_v2
        # data preparation
        # Try to find expression matrix automatically
        training_dir = os.path.dirname(args.training_data)
        data_name_base = os.path.basename(args.training_data).replace('_adjacency_records', '')
        expression_matrix_path = os.path.join(training_dir, data_name_base + '_cell_vs_gene_quantile_transformed')
        if not os.path.exists(expression_matrix_path):
            expression_matrix_path = ''
        graph_bag, num_feature, edge_dim = get_split_graph(args.training_data, node_id_sorted, args.total_subgraphs, expression_matrix_path=expression_matrix_path)    
        # train the model
        DGI_model = train_SigRelayST(args, graph_bag=graph_bag, in_channels=num_feature, edge_dim=edge_dim)
        # training done


    




    
