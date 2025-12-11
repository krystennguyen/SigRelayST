
# SigRelayST: Signature-based Relay for Spatial Transcriptomics
# This script extends CellNEST with signature-based bias terms derived from the Lignature database.
#
# CellNEST Citation:
# Zohora, F. T., et al. "CellNEST: A Graph Neural Network Framework for 
# Cell-Cell Communication Inference from Spatial Transcriptomics Data."
#

from scipy import sparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax #Linear, 
from torch_geometric.data import Data, DataLoader
import gzip
import os

from GATv2Conv_SigRelayST import GATv2Conv

def get_graph(training_data, expression_matrix_path=''):
    """Add Statement of Purpose
    Args:
        training_data: Path to the input graph    
        expression_matrix_path: Path to expression matrix (optional)
    Returns:
        List of torch_geometric.data.Data type: Loaded input graph
        Integer: Dimension of node embedding
        Integer: Dimension of edge attributes (3 or 4)
    """
    
    f = gzip.open(training_data , 'rb')
    row_col, edge_weight, lig_rec, num_cell = pickle.load(f)

    datapoint_size = num_cell
    #print(edge_weight)
    if expression_matrix_path == '':
        training_dir = os.path.dirname(training_data)
        data_name = os.path.basename(training_data).replace('_adjacency_records', '')
        potential_path = os.path.join(training_dir, data_name + '_cell_vs_gene_quantile_transformed')
        if os.path.exists(potential_path):
            expression_matrix_path = potential_path
            print(f"Found expression matrix at: {expression_matrix_path}")
    
    if expression_matrix_path != '' and os.path.exists(expression_matrix_path):
        f = gzip.open(expression_matrix_path, 'rb')
        X = pickle.load(f)
    else:
        # For large datasets, use sparse one-hot encoding instead of full identity matrix
        if datapoint_size > 50000:
            print(f"Warning: Large dataset ({datapoint_size} cells). Using sparse one-hot encoding instead of full identity matrix.")
            # Create sparse one-hot encoding: each node gets a unique feature index
            X = np.arange(datapoint_size).reshape(-1, 1)
            np.random.shuffle(X)
            # Convert to one-hot sparse representation
            from scipy.sparse import csr_matrix
            X_sparse = csr_matrix((np.ones(datapoint_size), (np.arange(datapoint_size), X.flatten())), shape=(datapoint_size, datapoint_size))
            X = X_sparse.toarray()  # This is still dense, but we'll use the expression matrix instead
            print("Consider using expression matrix for large datasets to avoid memory issues.")
        else:
            # one hot vector used as node feature vector
            X = np.eye(datapoint_size, datapoint_size)
            np.random.shuffle(X)

    X_data = X # node feature vector
    num_feature = X_data.shape[1]
    
    # Detect edge attribute dimension
    if len(edge_weight) > 0:
        edge_dim = len(edge_weight[0])
    else:
        edge_dim = 3  # default
    
    print('Node feature matrix: X has dimension ', X_data.shape)
    print("Total number of edges in the input graph is %d"%len(row_col))
    print("Edge attribute dimension: %d" % edge_dim)
    

    ###########

    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X_data, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    graph_bags.append(graph)

    print('Input graph generation done')

    data_loader = DataLoader(graph_bags, batch_size=1) # moved to get_graph
    
    return data_loader, num_feature, edge_dim


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, dropout, edge_dim=3):
        """Add Statement of Purpose
        Args: 
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            heads: Number of attention heads
            dropout: Dropout rate
            edge_dim: Edge attribute dimension (3 or 4)
        Returns: [to be]
    
        """

        
        super(Encoder, self).__init__()
        print('incoming channel %d'%in_channels)
        print('edge attribute dimension %d'%edge_dim)

        heads = heads
        self.edge_dim = edge_dim
        self.conv =  GATv2Conv(in_channels, hidden_channels, edge_dim=self.edge_dim, heads=heads, concat = False)#, dropout=dropout)
        self.conv_2 =  GATv2Conv(hidden_channels, hidden_channels, edge_dim=self.edge_dim, heads=heads, concat = False)#, dropout=0)

        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'

        self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'

        #self.prelu = nn.Tanh(hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)


    def forward(self, data):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """

        # layer 1
        x, attention_scores, attention_scores_unnormalized = self.conv(data.x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights = True)
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores_unnormalized


        # layer 2
        x, attention_scores, attention_scores_unnormalized  = self.conv_2(x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights = True)  # <---- ***
        self.attention_scores_mine = attention_scores #self.attention_scores_mine_l1 #attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized #self.attention_scores_mine_unnormalized_l1 #attention_scores_unnormalized

        ###############################
        x = self.prelu(x)

        return x #, attention_scores

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data):
    """Add Statement of Purpose
    Args: [to be]
           
    Returns: [to be]

    """
    #print('inside corruption function')
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)


def train_SigRelayST(args, data_loader, in_channels, edge_dim=3):
    """Add Statement of Purpose
    Args: 
        args: Training arguments
        data_loader: Data loader for graph data
        in_channels: Input feature dimension
        edge_dim: Edge attribute dimension (3 or 4, default 3)
    Returns: 
        Trained DGI model
    """
    loss_curve = np.zeros((args.num_epoch//100+1))
    loss_curve_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden, heads=args.heads, dropout = args.dropout, edge_dim=edge_dim),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    
    #print('initialized DGI model')
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate) #1e-5)#5 #6 #DGI_optimizer = torch.optim.RMSprop(DGI_model.parameters(), lr=1e-5)
    DGI_filename = args.model_path+'DGI_'+ args.model_name  +'.pth.tar'

    if args.load == 1:
        print('loading model')
        checkpoint = torch.load(DGI_filename, weights_only=False)
        DGI_model.load_state_dict(checkpoint['model_state_dict'])
        DGI_model.to(device)
        DGI_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        min_loss = checkpoint['loss']
        print('min_loss was %g'%min_loss)
    else:
        print('Saving init model state ...')
        torch.save({
            'epoch': 0,
            'model_state_dict': DGI_model.state_dict(),
            'optimizer_state_dict': DGI_optimizer.state_dict(),
            #'loss': loss,
            }, args.model_path+'DGI_init_model_optimizer_'+ args.model_name  + '.pth.tar')
        min_loss = 10000
        epoch_start = 0
        
    import datetime
    start_time = datetime.datetime.now()


    for epoch in range(epoch_start, args.num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        DGI_all_loss = []

        for data in data_loader:
            data = data.to(device)
            pos_z, neg_z, summary = DGI_model(data=data)
            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            DGI_loss.backward()
            DGI_all_loss.append(DGI_loss.item())
            DGI_optimizer.step()

        if ((epoch)%100) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))
            loss_curve[loss_curve_counter] = np.mean(DGI_all_loss)
            loss_curve_counter = loss_curve_counter + 1

            if np.mean(DGI_all_loss)<min_loss:

                min_loss=np.mean(DGI_all_loss)

                ######## save the current model state ########
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': DGI_model.state_dict(),
                    'optimizer_state_dict': DGI_optimizer.state_dict(),
                    'loss': min_loss,
                    }, DGI_filename)

                ##################################################
                # save the node embedding
                X_embedding = pos_z
                X_embedding = X_embedding.cpu().detach().numpy()
                X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' #.npy
                with gzip.open(X_embedding_filename, 'wb') as fp:  
                    pickle.dump(X_embedding, fp)
                                    
                # save the attention scores
                X_attention_index = DGI_model.encoder.attention_scores_mine[0]
                X_attention_index = X_attention_index.cpu().detach().numpy()

                # layer 1
                X_attention_score_normalized_l1 = DGI_model.encoder.attention_scores_mine_l1[1]
                X_attention_score_normalized_l1 = X_attention_score_normalized_l1.cpu().detach().numpy()
                # layer 1 unnormalized
                X_attention_score_unnormalized_l1 = DGI_model.encoder.attention_scores_mine_unnormalized_l1
                X_attention_score_unnormalized_l1 = X_attention_score_unnormalized_l1.cpu().detach().numpy()

                # layer 2
                X_attention_score_normalized = DGI_model.encoder.attention_scores_mine[1]
                X_attention_score_normalized = X_attention_score_normalized.cpu().detach().numpy()
                # layer 2 unnormalized
                X_attention_score_unnormalized = DGI_model.encoder.attention_scores_mine_unnormalized
                X_attention_score_unnormalized = X_attention_score_unnormalized.cpu().detach().numpy()

                print('making the bundle to save')
                X_attention_bundle = [X_attention_index, X_attention_score_normalized_l1, X_attention_score_unnormalized, X_attention_score_unnormalized_l1, X_attention_score_normalized]
                X_attention_filename =  args.embedding_path + args.model_name + '_attention' #.npy
                # np.save(X_attention_filename, X_attention_bundle) # this is deprecated
                with gzip.open(X_attention_filename, 'wb') as fp:  
                    pickle.dump(X_attention_bundle, fp)

                # Ensure logs directory exists
                log_dir = 'logs/'
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                logfile=open(log_dir+'DGI_'+ args.model_name+'_loss_curve.csv', 'wb')
                np.savetxt(logfile,loss_curve, delimiter=',')
                logfile.close()

               

    end_time = datetime.datetime.now()

    print('Training time in seconds: ', (end_time-start_time).seconds)

    checkpoint = torch.load(DGI_filename, weights_only=False)
    DGI_model.load_state_dict(checkpoint['model_state_dict'])
    DGI_model.to(device)
    DGI_model.eval()
    print("debug loss")
    DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
    print("debug loss latest tupple %g"%DGI_loss.item())

    return DGI_model

