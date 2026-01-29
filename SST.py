import torch.nn as nn
import torch
import torch.optim as optim
import tqdm

class SST(nn.Module):
    """
    SST model for sensitive attribute prediction from user embeddings.
    """
    def __init__(self, config):
        super().__init__()
        embedding_dim = config.emb_size
        hidden_dim = config.sst_hidden_sizes
        n_classes = len(config.s_ratios)
        
        self.first_layer = nn.Linear(embedding_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x):
        temp = self.first_layer(x)
        temp = nn.ReLU()(temp)
        temp = self.second_layer(temp)
        return temp