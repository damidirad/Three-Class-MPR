"""
Evaluation metrics for fair recommender systems.

This module provides functions to evaluate:
- Fairness violations on validation set with partial knowledge (validate_fairness)
- Fairness violations on test set with complete knowledge (test_fairness)

Author: FACT-AI Group 21
Date: January 2026
"""

import numpy as np
import torch

def validate_fairness(model, df_val, df_sensitive_attr, s0_known, s1_known, device, sensitive_attr="gender"):
    """
    Evaluate RMSE and fairness on validation set (partial sensitive attribute knowledge).
    
    Only computes fairness metrics for users with known sensitive attributes.
    
    Args:
        model: Matrix factorization model
        df_val: Validation dataframe
        df_sensitive_attr: Sensitive attributes dataframe
        s0_known: Known group 0 user IDs
        s1_known: Known group 1 user IDs
        device: Torch device
        sensitive_attr: Sensitive attribute column name
        
    Returns:
        tuple: (RMSE, demographic parity violation)
    """
    model.eval()
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"]), dtype=torch.long, device=device)
        test_item_total = torch.tensor(np.array(df_val["item_id"]), dtype=torch.long, device=device)

        logits_total = model(test_user_total, test_item_total)
        pred_total = torch.sigmoid(logits_total)
        
        y_true = torch.tensor(np.array(df_val["label"]), dtype=torch.float32, device=device)
        
        known_users = set(s0_known) | set(s1_known)
        known_mask = torch.tensor([uid in known_users for uid in df_val["user_id"]], 
                                   dtype=torch.bool, device=device)
        
        user_to_sens_attr = df_sensitive_attr.set_index("user_id")[sensitive_attr].to_dict()
        user_sens_attr = torch.tensor([user_to_sens_attr[uid] for uid in df_val["user_id"]], 
                                       dtype=torch.long, device=device)
        
        pred_known = pred_total[known_mask]
        sens_attr_known = user_sens_attr[known_mask]
        
        unique_attrs = torch.unique(sens_attr_known)
        sens_attr_means = {}
        for attr in unique_attrs:
            mask = (sens_attr_known == attr)
            sens_attr_means[attr.item()] = pred_known[mask].mean()
        

        attr_mean_values = list(sens_attr_means.values())
        naive_unfairness = float(torch.max(torch.stack(attr_mean_values)) - torch.min(torch.stack(attr_mean_values)))

        rmse = float(torch.sqrt(torch.mean((pred_total - y_true) ** 2)).cpu())
        
    return rmse, naive_unfairness

def test_fairness(model, df_val, df_sensitive_attr, device, sensitive_attr="gender"):
    """
    Evaluate RMSE and fairness on test set (complete sensitive attribute knowledge).
    
    Computes fairness metrics for all users.
    
    Args:
        model: Matrix factorization model
        df_val: Test dataframe
        df_sensitive_attr: Sensitive attributes dataframe
        device: Torch device
        sensitive_attr: Sensitive attribute column name
        
    Returns:
        tuple: (RMSE, demographic parity violation)
    """
    model.eval()
    with torch.no_grad():
        test_user_total = torch.tensor(np.array(df_val["user_id"]), dtype=torch.long, device=device)
        test_item_total = torch.tensor(np.array(df_val["item_id"]), dtype=torch.long, device=device)

        logits_total = model(test_user_total, test_item_total)
        pred_total = torch.sigmoid(logits_total)

        y_true = torch.tensor(np.array(df_val["label"]), dtype=torch.float32, device=device)

        user_to_sens_attr = df_sensitive_attr.set_index("user_id")[sensitive_attr].to_dict()
        user_sens_attr = torch.tensor([user_to_sens_attr[uid] for uid in df_val["user_id"]],
                                       dtype=torch.long, device=device)
        
        unique_attrs = torch.unique(user_sens_attr)
        sens_attr_means = {}
        for attr in unique_attrs:
            mask = (user_sens_attr == attr)
            sens_attr_means[attr.item()] = pred_total[mask].mean()

        attr_mean_values = list(sens_attr_means.values())
        naive_unfairness = float(torch.max(torch.stack(attr_mean_values)) - torch.min(torch.stack(attr_mean_values)))

        rmse = float(torch.sqrt(torch.mean((pred_total - y_true) ** 2)).cpu())
        
    return rmse, naive_unfairness