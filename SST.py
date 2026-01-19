import torch.nn as nn
import torch
import torch.optim as optim
import tqdm

class SST(nn.Module):
    """
    SST
    """
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        # 3 Layer Feedforward Network   
        self.net = nn.Sequential(
            # Layer 1: Expand and Normalize
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting on biased data
            
            # Layer 2: Feature Refinement
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            # Layer 3: Binary Logits
            nn.Linear(hidden_dim // 2, 2) 
        )
        
        # Temperature Scaling
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.net(x)
        # Soften logits with temperature scaling
        return logits / self.temperature

def train_sst(sst_model, mf_model, known_user_ids, known_labels, epochs=30, lr=1e-3):
    """
    Args:
        sst_model: Your SST class instance
        mf_model: The trained/partially trained MF model
        known_user_ids: Tensor of user IDs we have labels for
        known_labels: Tensor of 0s and 1s
    """
    sst_model.train()
    # We detach the embeddings because we don't want to change the MF model here
    optimizer = optim.Adam(sst_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="[SST] Training SST Model"):
        # get embeddings from mf model
        with torch.no_grad():
            embeddings = mf_model.user_emb(known_user_ids).detach()

        # forward pass
        logits = sst_model(embeddings)
        loss = criterion(logits, known_labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"SST Epoch {epoch} | Loss: {loss.item():.4f}")