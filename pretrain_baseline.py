import copy
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from helpers import set_random_seed
from config import Config
from evaluation import evaluate_direct_parity

def train_partial_disclosed_dp_multiclass(
    model: nn.Module,
    df_train: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    sensitive_attr: pd.DataFrame,
    disclosed_ids_dict: Dict[int, np.ndarray],
    config: Config,
    device: torch.device,
) -> Tuple[float, float, float, float, int, nn.Module]:
    """
    Multiclass disclosed-only demographic parity regularization.
    Args:
        model: Recommendation model (MF).
        df_train: Training data DataFrame.
        valid_data: Validation data DataFrame.
        test_data: Test data DataFrame.
        sensitive_attr: DataFrame with sensitive attributes.
        disclosed_ids_dict: Dict mapping class index to known user IDs.
        config: Configuration object with hyperparameters.
    Returns:
        best_val_rmse: Best validation RMSE achieved.
        best_test_rmse: Corresponding test RMSE at best validation.
        best_val_gap: Best validation demographic parity gap.
        best_test_gap: Corresponding test demographic parity gap.
        best_epoch: Epoch at which best validation RMSE was achieved.
        best_model: Model state at best validation RMSE.    
    """

    if getattr(config, "seed", None) is not None:
        set_random_seed(config.seed)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.mf_lr, weight_decay=config.weight_decay)

    # Preconvert disclosed ID arrays to tensors
    disclosed_ids_t: Dict[int, torch.Tensor] = {
        c: torch.as_tensor(ids, dtype=torch.long, device=device)
        for c, ids in disclosed_ids_dict.items()
        if ids is not None and len(ids) > 0
    }

    best_val_rmse = float("inf")
    best_test_rmse = float("inf")
    best_val_gap = 0.0
    best_test_gap = 0.0
    best_epoch = 0
    best_model = copy.deepcopy(model)
    stall_counter = 0

    model.train()
    for epoch in tqdm.tqdm(range(config.mf_epochs), desc="[Partial-DP Multiclass] Training"):
        indices = np.arange(len(df_train))
        if getattr(config, "seed", None) is not None:
            rng = np.random.default_rng(config.seed + epoch)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        for start_idx in range(0, len(df_train), config.batch_size):
            batch_idx = indices[start_idx : start_idx + config.batch_size]
            batch = df_train.iloc[batch_idx]

            u_in = torch.as_tensor(batch["user_id"].values, dtype=torch.long, device=device)
            i_in = torch.as_tensor(batch["item_id"].values, dtype=torch.long, device=device)
            y = torch.as_tensor(batch["label"].values, dtype=torch.float32, device=device)

            logits = model(u_in, i_in).view(-1) 
            probs = torch.sigmoid(logits)

            base_loss = criterion(logits, y.view(-1))

            # Multiclass disclosed-only DP penalty
            means = []
            for c, ids_t in disclosed_ids_t.items():
                in_class_disclosed = torch.isin(u_in, ids_t)
                if in_class_disclosed.any():
                    means.append(probs[in_class_disclosed].mean())

            if len(means) <= 1:
                fair_penalty = torch.tensor(0.0, device=device)
            else:
                means_t = torch.stack(means)
                gap = means_t.max() - means_t.min()
                fair_penalty = gap * config.fair_reg

            loss = base_loss + fair_penalty

            optimizer.zero_grad()
            loss.backward()
            if getattr(config, "grad_clip", None) is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        if epoch % config.evaluation_interval == 0:
            model.eval()
            with torch.no_grad():
                rmse_val, gap_val, _ = evaluate_direct_parity(
                    model=model,
                    df_eval=valid_data,
                    df_sensitive_attr=sensitive_attr,
                    device=device,
                    config=config,
                    disclosed_ids=disclosed_ids_dict,  # disclosed-only
                )
                rmse_test, gap_test, _ = evaluate_direct_parity(
                    model=model,
                    df_eval=test_data,
                    df_sensitive_attr=sensitive_attr,
                    device=device,
                    config=config,
                    disclosed_ids=None,  # full/oracle
                )

            if rmse_val < best_val_rmse:
                best_val_rmse = rmse_val
                best_test_rmse = rmse_test
                best_val_gap = gap_val
                best_test_gap = gap_test
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                stall_counter = 0
            else:
                stall_counter += 1
                if stall_counter >= config.early_stopping_patience:
                    model.train()
                    break

            model.train()

    return best_val_rmse, best_test_rmse, best_val_gap, best_test_gap, best_epoch, best_model