import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Dict, Tuple

import tqdm
from config import Config
from evaluation import evaluate_direct_parity

def train_multiclass_fairness(
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
    Train Matrix Factorization with multiclass fairness regularization.
    
    Args:
        model: The MPR model to be trained.
        df_train: Training data DataFrame.
        valid_data: Validation data DataFrame.
        test_data: Test data DataFrame.
        sensitive_attr: DataFrame containing sensitive attributes.
        disclosed_ids_dict: Dictionary mapping sensitive attribute values to known user IDs.
        config: Configuration dataclass instance.
        device: Torch device (CPU or GPU).
    """
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.mf_lr, 
        weight_decay=config.weight_decay
    )
    
    best_val_rmse = 100.0
    test_rmse_at_best = 0.0
    best_epoch = 0
    val_unfairness_at_best = 0.0
    test_unfairness_at_best = 0.0
    stall_counter = 0
    
    # pre-calculate disclosed IDs for validation efficiency
    all_disclosed_ids = torch.cat(
        [torch.LongTensor(ids) for ids in disclosed_ids_dict.values() if len(ids) > 0]
        ).to(device)
    
    model.train()
    
    for epoch in tqdm.tqdm(range(config.mf_epochs), desc="[MF] Training MPR with fairness"):
        loss_total = 0.0
        fair_reg_total = 0.0
        num_batches = 0
        
        # shuffle training indices
        indices = np.arange(len(df_train))

        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(indices)
        
        for start_idx in range(0, len(df_train), config.batch_size):
            batch_indices = indices[start_idx : start_idx + config.batch_size]
            data_batch = df_train.iloc[batch_indices].reset_index(drop=True)
            
            # prepare tensors
            train_ratings = torch.FloatTensor(data_batch["label"].values).to(device)
            train_user_input = torch.LongTensor(data_batch["user_id"].values).to(device)
            train_item_input = torch.LongTensor(data_batch["item_id"].values).to(device)
            
            # forward pass
            y_hat = model(train_user_input, train_item_input)
            
            # reconstruction loss
            base_loss = criterion(y_hat, train_ratings.view(-1))
            
            # max-min gap fairness regularization
            group_means = []
            for class_idx, disclosed_ids in disclosed_ids_dict.items():
                if len(disclosed_ids) == 0:
                    continue
                mask = torch.isin(train_user_input, torch.tensor(disclosed_ids, device=device))
                if mask.any():
                    group_means.append(y_hat[mask].mean())
                    
            if len(group_means) > 1:
                group_means_tensor = torch.stack(group_means)
                # gap between max and min group means
                fair_penalty = (group_means_tensor.max() - group_means_tensor.min()) * config.fair_reg
            else:
                fair_penalty = torch.tensor(0.0, device=device)
            
            total_loss = base_loss + fair_penalty
            
            # backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            
            loss_total += base_loss.item()
            fair_reg_total += fair_penalty.item()
            num_batches += 1
            
        avg_loss = loss_total / num_batches
        avg_fair = fair_reg_total / num_batches
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Fair Reg: {avg_fair:.4f}")
        
        # evaluation cycle
        if epoch % config.evaluation_interval == 0:
            model.eval()
            with torch.no_grad():
                # validation and test evaluation, using disclosed IDs for validation
                rmse_val, val_mmg, _ = evaluate_direct_parity(
                    model, valid_data, sensitive_attr, device, config, disclosed_ids=all_disclosed_ids
                )
                rmse_test, test_mmg, _ = evaluate_direct_parity(
                    model, test_data, sensitive_attr, device, config, disclosed_ids=None
                )
            
            print(f"[Evaluation] Val RMSE: {rmse_val:.4f} (Max-min gap: {val_mmg:.4f}) | Test RMSE: {rmse_test:.4f}")
            
            # early stopping and checkpointing
            if rmse_val < best_val_rmse:
                best_val_rmse = rmse_val
                test_rmse_at_best = rmse_test
                val_unfairness_at_best = val_mmg
                test_unfairness_at_best = test_mmg
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                stall_counter = 0
            else:
                stall_counter += 1
                if stall_counter >= config.early_stopping_patience:
                    print("[Log] Early stopping triggered.")
                    break
            model.train()
            
    return (
        best_val_rmse, 
        test_rmse_at_best, 
        val_unfairness_at_best, 
        test_unfairness_at_best, 
        best_epoch, 
        best_model
    )