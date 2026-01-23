import torch
import numpy as np
from typing import Tuple
import pandas as pd
from config import Config
import torch.nn as nn

def evaluate_direct_parity(
    model: nn.Module, 
    df_eval: pd.DataFrame, 
    df_sensitive_attr: pd.DataFrame, 
    device: torch.device, 
    config: Config, 
    disclosed_ids=None
) -> tuple[float, float, float]:
    """
    Direct (Demographic) Parity with a single multiclass metric:
    - Unfairness = max(mean_g) - min(mean_g) over sensitive groups g.
      This equals binary DP MAD when K=2 and natively extends to K>2.

    Returns:
        rmse: Root Mean Square Error on eval set.
        gap:  Max–min gap across group means.
        std_unfairness: Std of group means.
    """
    model.eval()
    with torch.no_grad():
        user_ids_np = df_eval["user_id"].values
        item_ids_np = df_eval["item_id"].values
        labels_np   = df_eval["label"].values

        user_tensor = torch.tensor(user_ids_np, dtype=torch.long, device=device)
        item_tensor = torch.tensor(item_ids_np, dtype=torch.long, device=device)
        y_true      = torch.tensor(labels_np,   dtype=torch.float32, device=device)

        logits = model(user_tensor, item_tensor).view(-1)
        pred   = torch.sigmoid(logits)

        # RMSE inline
        rmse = float(torch.sqrt(torch.mean((pred - y_true) ** 2)))

        # Sensitive attribute mapping
        s_attr_col = getattr(config, "s_attr", "gender")
        user_to_sens = df_sensitive_attr.set_index("user_id")[s_attr_col].to_dict()
        sens_vals = [user_to_sens[uid] for uid in user_ids_np]
        user_sens_attr = torch.tensor(sens_vals, dtype=torch.long, device=device)

        # Optional filtering by disclosed IDs 
        if disclosed_ids is not None:
            if isinstance(disclosed_ids, dict):
                parts = []
                for _, ids in disclosed_ids.items():
                    if isinstance(ids, torch.Tensor):
                        parts.append(ids.to(device).long())
                    else:
                        parts.append(torch.tensor(np.array(ids), dtype=torch.long, device=device))
                disclosed_union = torch.unique(torch.cat(parts)) if len(parts) else torch.empty(0, dtype=torch.long, device=device)
            else:
                disclosed_union = (
                    disclosed_ids.to(device).long()
                    if isinstance(disclosed_ids, torch.Tensor)
                    else torch.tensor(np.array(disclosed_ids), dtype=torch.long, device=device)
                )
            mask = torch.isin(user_tensor, disclosed_union)
            pred = pred[mask]
            user_sens_attr = user_sens_attr[mask]

        if pred.numel() == 0:
            return rmse, 0.0, 0.0

        # Group means and max–min gap
        unique_attrs = torch.unique(user_sens_attr)
        means = []
        for g in unique_attrs:
            m = (user_sens_attr == g)
            if m.any():
                means.append(pred[m].mean())

        if len(means) <= 1:
            gap = 0.0
            std_unfairness = 0.0
        else:
            means_t = torch.stack(means)
            gap = float(means_t.max() - means_t.min())
            std_unfairness = float(torch.std(means_t))

    return rmse, gap, std_unfairness

def evaluate_sst(data, label, model, n_classes):
    """
    Multiclass evaluator. 
    
    Returns:
      acc_percent: accuracy in percentage.
      pred_class_ratios: list of predicted class ratios.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(1)
        label = label.long()
        acc = round(((preds == label).float().mean().item()) * 100, 2)
        total = max(len(preds), 1)
        pred_class_ratios = []
        for c in range(n_classes):
            pred_class_ratios.append(float((preds == c).sum().item()) / float(total))
    return acc, pred_class_ratios