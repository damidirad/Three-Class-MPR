import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MatrixFactorization(nn.Module):
    """
    Matrix Factorization model for collaborative filtering.
    
    Predicts user-item ratings using latent factor embeddings and bias terms:
        rating = dot(user_embedding, item_embedding) + user_bias + item_bias
    
    Args:
        num_users (int): Total number of users in the dataset.
        num_items (int): Total number of items in the dataset.
        emb_size (int): Dimensionality of the latent embedding space.
    
    Attributes:
        user_emb (nn.Embedding): User embedding matrix (num_users × emb_size).
        item_emb (nn.Embedding): Item embedding matrix (num_items × emb_size).
        user_bias (nn.Embedding): Per-user bias terms.
        item_bias (nn.Embedding): Per-item bias terms.
    """
    
    def __init__(self, num_users, num_items, emb_size):
        super(MatrixFactorization, self).__init__()
        
        # Embedding layers
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Initialize embeddings and biases
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)

        self.output_activation = nn.Sigmoid()
        
    def forward(self, user_ids, item_ids):
        """
        Forward pass to compute predicted ratings.
        
        Args:
            user_ids (torch.Tensor): User IDs, shape (batch_size,).
            item_ids (torch.Tensor): Item IDs, shape (batch_size,).
        
        Returns:
            torch.Tensor: Predicted ratings, shape (batch_size,).
        """
        user_embs = self.user_emb(user_ids)
        item_embs = self.item_emb(item_ids)
        
        user_biases = self.user_bias(user_ids).squeeze()
        item_biases = self.item_bias(item_ids).squeeze()

        rating = self.output_activation((user_embs * item_embs).sum(1) +  user_biases  + item_biases)

        return rating