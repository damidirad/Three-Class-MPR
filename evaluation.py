"""
Evaluation metrics for fair recommender systems.

Authors: Daniel Amidirad, Kagan Sert
Date: January 2026
"""

import numpy as np
import torch
from config import Config

def evaluate_direct_parity(model, df_eval, df_sensitive_attr, device, config: Config, disclosed_ids=None):
    """
    Evaluate DP 
    
    If disclosed_ids is provided, it acts as the partial knowledge validator.
    If None, it acts as the complete knowledge tester.
    """
    model.eval()
    with torch.no_grad():
        # predictions
        user_tensor = torch.tensor(df_eval["user_id"].values, dtype=torch.long, device=device)
        item_tensor = torch.tensor(df_eval["item_id"].values, dtype=torch.long, device=device)
        y_true = torch.tensor(df_eval["label"].values, dtype=torch.float32, device=device)

        # sigmoid for bceloss
        pred = torch.sigmoid(model(user_tensor, item_tensor))
        
        # 2. RMSE Calculation
        rmse = float(torch.sqrt(torch.mean((pred - y_true) ** 2)).cpu())

        # 3. Sensitive Attribute Mapping
        # Use config.s_attr to dynamically select 'gender', 'age', etc.
        user_to_sens_attr = df_sensitive_attr.set_index("user_id")[config.s_attr].to_dict()
        user_sens_attr = torch.tensor([user_to_sens_attr[uid] for uid in df_eval["user_id"]], 
                                       dtype=torch.long, device=device)

        # 4. Filter for Partial Knowledge (if applicable)
        if disclosed_ids is not None:
            mask = torch.isin(user_tensor, disclosed_ids)
            pred = pred[mask]
            user_sens_attr = user_sens_attr[mask]

        # 5. Group Fairness Calculation
        unique_attrs = torch.unique(user_sens_attr)
        group_means = []
        for attr in unique_attrs:
            group_mask = (user_sens_attr == attr)
            group_means.append(pred[group_mask].mean())
        
        group_means_tensor = torch.stack(group_means)

        # Max-Min-Gap for binary or multiclass
        max_min_gap = float(torch.max(group_means_tensor) - torch.min(group_means_tensor))
        
        # Standard deviation
        std_unfairness = float(torch.std(group_means_tensor)) if len(group_means) > 1 else 0.0

    return rmse, max_min_gap, std_unfairness



def evaluate_robust_metrics(model, df_eval, df_sensitive_attr, device, config: Config):
    """
    Wrapper specifically for the Robust MPR framework testing.
    """
    return evaluate_direct_parity(model, df_eval, df_sensitive_attr, device, config, disclosed_ids=None)