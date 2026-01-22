"""
Evaluation metrics for fair recommender systems.

Authors: Daniel Amidirad, Kagan Sert
Date: January 2026
"""

import torch
from config import Config
import torch.nn as nn
from helpers import calculate_rmse
import pandas as pd

def evaluate_direct_parity(
    model: nn.Module, 
    df_eval: pd.DataFrame, 
    df_sensitive_attr: pd.DataFrame, 
    device: torch.device, 
    config: Config, 
    disclosed_ids=None
) -> tuple[float, float, float]:
    """
    Evaluate main metrics for Direct Parity MPR framework.
    
    If disclosed_ids is provided, it acts as the partial knowledge validator.
    If None, it acts as the complete knowledge tester.

    Args:
        model: The trained MPR model.
        df_eval: DataFrame for evaluation (validation or test).
        df_sensitive_attr: DataFrame containing sensitive attributes.
        device: Torch device (CPU or GPU).
        config: Configuration dataclass instance.
        disclosed_ids: Optional tensor of user IDs with known sensitive attributes.
    Returns:
        rmse: Root Mean Square Error on eval set.
        max_min_gap: Max-Min Gap unfairness metric.
        std_unfairness: Standard deviation of group means.
    """
    model.eval()
    with torch.no_grad():
        # predictions
        user_tensor = torch.tensor(df_eval["user_id"].values, dtype=torch.long, device=device)
        item_tensor = torch.tensor(df_eval["item_id"].values, dtype=torch.long, device=device)
        y_true = torch.tensor(df_eval["label"].values, dtype=torch.float32, device=device)

        # sigmoid for BCELoss compatibility
        pred = torch.sigmoid(model(user_tensor, item_tensor))
        
        # rmse calculation
        rmse = calculate_rmse(y_true, pred)

        # sensitive attribute extraction
        user_to_sens_attr = df_sensitive_attr.set_index("user_id")[config.s_attr].to_dict()
        user_sens_attr = torch.tensor([user_to_sens_attr[uid] for uid in df_eval["user_id"]], 
                                       dtype=torch.long, device=device)

        # filtering for disclosed IDs if provided
        if disclosed_ids is not None:
            mask = torch.isin(user_tensor, disclosed_ids)
            pred = pred[mask]
            user_sens_attr = user_sens_attr[mask]

        # group means calculation
        unique_attrs = torch.unique(user_sens_attr)
        group_means = []
        for attr in unique_attrs:
            group_means.append(pred[user_sens_attr == attr].mean())
        
        group_means_tensor = torch.stack(group_means)

        # max-min gap unfairness
        max_min_gap = float(torch.max(group_means_tensor) - torch.min(group_means_tensor))
        
        # standard deviation
        std_unfairness = float(torch.std(group_means_tensor)) if len(group_means) > 1 else 0.0

    return rmse, max_min_gap, std_unfairness



def evaluate_robust_metrics(
    model: nn.Module, 
    df_eval: pd.DataFrame, 
    df_sensitive_attr: pd.DataFrame, 
    device: torch.device, 
    config: Config
) -> tuple[float, float, float]:
    """
    Wrapper for Robust MPR framework testing.

    Args:
        model: The trained MPR model.
        df_eval: DataFrame for evaluation (validation or test).
        df_sensitive_attr: DataFrame containing sensitive attributes.
        device: Torch device (CPU or GPU).
        config: Configuration dataclass instance.
    Returns:
        rmse: Root Mean Square Error on eval set.
        max_min_gap: Max-Min Gap unfairness metric.
        std_unfairness: Standard deviation of group means.
    """
    return evaluate_direct_parity(model, df_eval, df_sensitive_attr, device, config, disclosed_ids=None)