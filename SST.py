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

# def train_sst(sst_model, mf_model, known_user_ids, known_labels, config):
#     """
#     Train the SST model using known sensitive attribute labels.
    
#     Args:
#         sst_model: Instance of SST
#         mf_model: Pre-trained MF model
#         known_user_ids: User indices with ground-truth sensitive labels
#         known_labels: Sensitive attribute labels (0, 1, ..., n_classes-1)
#         config: Config dataclass instance
#     """
#     device = next(sst_model.parameters()).device
#     known_user_ids = known_user_ids.to(device)
#     known_labels = known_labels.to(device)
    
#     sst_model.train()
#     optimizer = optim.Adam(sst_model.parameters(), lr=config.sst_lr, weight_decay=config.weight_decay)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in tqdm(range(config.sst_epochs), desc="[SST] Training SST Model"):
#         # get embeddings from mf model
#         with torch.no_grad():
#             embeddings = mf_model.user_emb(known_user_ids).detach()

#         logits = sst_model(embeddings)
#         loss = criterion(logits, known_labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if epoch % 10 == 0:
#             preds = torch.argmax(logits, dim=1)
#             acc = (preds == known_labels).float().mean()
#             print(f"SST Epoch {epoch} | Loss: {loss.item():.4f} | Internal Acc: {acc:.2%}")