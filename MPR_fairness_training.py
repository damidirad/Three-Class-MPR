import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import tqdm
from typing import Dict, Tuple
import pandas as pd
from config import Config
from evaluation import evaluate_direct_parity

def fairness_training(
    model: nn.Module, 
    df_train: pd.DataFrame, 
    valid_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    sensitive_attr: pd.DataFrame, 
    disclosed_ids_dict: Dict[int, np.ndarray],
    predicted_sensitive_attr_dict: Dict[float, Dict[int, pd.DataFrame]],
    config: Config,
    device: torch.device,
    rmse_thresh: float, # ADDED
) -> Tuple[float, float, float, float, int, nn.Module]:
    """
    Train with MPR-style robust multiclass demographic parity.

    Args:
        model: Recommendation model (MF).
        df_train: Training data DataFrame.
        valid_data: Validation data DataFrame.
        test_data: Test data DataFrame.
        sensitive_attr: DataFrame with sensitive attributes.
        disclosed_ids_dict: Dict mapping class index to known user IDs.
        predicted_sensitive_attr_dict: Dict mapping prior keys to dicts of seed to DataFrame with predicted sensitive attributes.
        config: Configuration object with hyperparameters.
    Returns:
        best_val_rmse: Best validation RMSE achieved.
        best_test_rmse: Corresponding test RMSE at best validation.
        best_val_gap: Best validation demographic parity gap.
        best_test_gap: Corresponding test demographic parity gap.
        best_epoch: Epoch at which best validation RMSE was achieved.
        best_model: Model state at best validation RMSE.    
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.mf_lr, weight_decay=config.weight_decay)

    best_val_rmse = 100.0
    best_test_rmse = 100.0
    best_val_gap = float('inf') # changed from 0.0 to inf
    best_test_gap = 0.0
    best_epoch = 0
    best_model = copy.deepcopy(model)
    # stall_counter = 0

    # Concatenate disclosed IDs for evaluator
    all_disclosed_ids = torch.cat(
        [torch.LongTensor(ids) for ids in disclosed_ids_dict.values() if len(ids) > 0]
    ).to(device)
    
    model.train()
    for epoch in tqdm.tqdm(range(config.mf_epochs), desc="[MPR] Training"):
        # Optional deterministic shuffle
        indices = np.arange(len(df_train))
        if hasattr(config, "seed"):
            rng = np.random.default_rng(config.seed + epoch)
            rng.shuffle(indices)

        for start_idx in range(0, len(df_train), config.batch_size):
            batch_idx = indices[start_idx : start_idx + config.batch_size]
            batch = df_train.iloc[batch_idx]

            # Inputs
            u_in = torch.LongTensor(batch["user_id"].values).to(device)
            i_in = torch.LongTensor(batch["item_id"].values).to(device)
            labels = torch.FloatTensor(batch["label"].values).to(device)  

            # Model output: logits -> probs
            probs = model(u_in, i_in).view(-1)     
            # probs  = torch.sigmoid(logits)         

            # Reconstruction loss
            base_loss = criterion(probs, labels.view(-1))

            # MPR multiclass DP penalty across resamples with log-sum-exp
            reg_list = []
            beta = getattr(config, "beta", 1e-3)

            for prior_key, seed_map in predicted_sensitive_attr_dict.items():
                for seed, resample_df in seed_map.items():
                    # Sensitive labels 
                    sst = torch.LongTensor(
                        np.array(resample_df.iloc[np.array(batch["user_id"])][config.s_attr])
                    ).to(device)  

                    uniq = torch.unique(sst)
                    means = []
                    for g in uniq:
                        m = (sst == g)
                        if m.any():
                            means.append(probs[m].mean())

                    if len(means) <= 1:
                        gap = torch.tensor(0.0, device=device)
                    else:
                        means_t = torch.stack(means)        
                        gap = means_t.max() - means_t.min()    # max–min: DP MAD for K=2

                    reg_list.append(gap)

            if len(reg_list) == 0:
                fair_penalty = torch.tensor(0.0, device=device)
            else:
                regs = torch.stack(reg_list) 
                if beta == 0:
                    fair_penalty = regs.max() * config.fair_reg
                else:
                    scaled = regs / beta
                    Cstab = scaled.max().detach()
                    acc = torch.exp(scaled - Cstab).sum()
                    lse = beta * (torch.log(acc) + Cstab)
                    fair_penalty = lse * config.fair_reg

            total_loss = base_loss + fair_penalty

            optimizer.zero_grad()
            total_loss.backward()
            if hasattr(config, "grad_clip") and config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Evaluate periodically 
        if epoch % config.evaluation_interval == 0:
            model.eval()
            with torch.no_grad():
                rmse_val, val_gap, _ = evaluate_direct_parity(
                    model, valid_data, sensitive_attr, device, config, disclosed_ids=all_disclosed_ids
                )
                rmse_test, test_gap, _ = evaluate_direct_parity(
                    model, test_data, sensitive_attr, device, config, disclosed_ids=None
                )

                print(f"Epoch {epoch}: Val RMSE={rmse_val:.5f}, Val Gap={val_gap:.5f}, Test RMSE={rmse_test:.5f}, Test Gap={test_gap:.5f}")
                
                # Only save if RMSE meets threshold
                if rmse_val < rmse_thresh:
                    if val_gap < best_val_gap:
                        best_val_rmse = rmse_val
                        best_test_rmse = rmse_test
                        best_epoch = epoch
                        best_model = copy.deepcopy(model)
                        best_val_gap = val_gap
                        best_test_gap = test_gap
                        print(f" --> New best model (Gap improved: {val_gap:.5f})")

            model.train()
            
    return best_val_rmse, best_test_rmse, best_val_gap, best_test_gap, best_epoch, best_model