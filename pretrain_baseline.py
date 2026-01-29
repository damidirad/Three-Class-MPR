import copy
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from config import Config
from helpers import set_random_seed
from evaluation import evaluate_direct_parity

def pretrain_baseline(
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
        
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.mf_lr, weight_decay=config.weight_decay)

    u_all = df_train["user_id"].to_numpy()
    i_all = df_train["item_id"].to_numpy()
    y_all = df_train["label"].to_numpy(dtype=np.float32)
    n_train = len(u_all)


    # Preconvert disclosed ID arrays to tensors
    num_users = int(max(u_all.max(), sensitive_attr["user_id"].max())) + 1 
    disclosed_class_of_user = torch.full((num_users,), -1, dtype=torch.long, device=device)

    for c, ids in disclosed_ids_dict.items():
        if ids is None or len(ids) == 0:
            continue
        disclosed_class_of_user[torch.as_tensor(ids, dtype=torch.long, device=device)] = int(c)

    best_val_rmse = float("inf")
    best_test_rmse = float("inf")
    best_val_gap = 0.0
    best_test_gap = 0.0
    best_epoch = 0
    best_model = copy.deepcopy(model)
    stall_counter = 0

    model.train()
    for epoch in tqdm.tqdm(range(config.mf_epochs), desc="[Partial-DP Multiclass] Training"):
        indices = np.arange(n_train)
        if getattr(config, "seed", None) is not None:
            rng = np.random.default_rng(config.seed + epoch)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        for start_idx in range(0, n_train, config.batch_size):
            batch_idx = indices[start_idx : start_idx + config.batch_size]

            u_in = torch.as_tensor(u_all[batch_idx], dtype=torch.long, device=device)
            i_in = torch.as_tensor(i_all[batch_idx], dtype=torch.long, device=device)
            y    = torch.as_tensor(y_all[batch_idx], dtype=torch.float32, device=device)
            y_hat = model(u_in, i_in).view(-1) 

            base_loss = criterion(y_hat, y.view(-1))

            # Multiclass disclosed-only DP penalty
            g = disclosed_class_of_user[u_in]  # group id for each user in batch
            mask = g >= 0

            if mask.any():
                g2 = g[mask]
                p2 = y_hat[mask]

                uniq, inv = torch.unique(g2, return_inverse=True)

                sums = torch.zeros(len(uniq), device=device).scatter_add_(0, inv, p2)
                cnts = torch.zeros(len(uniq), device=device).scatter_add_(0, inv, torch.ones_like(p2))

                means = sums / cnts
                if means.numel() <= 1:
                    fair_penalty = torch.tensor(0.0, device=device)
                else:
                    fair_penalty = (means.max() - means.min()) * config.fair_reg
            else:
                fair_penalty = torch.tensor(0.0, device=device)

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

if __name__ == "__main__":
    import argparse
    from MF import MF
    from helpers import get_device, setup_paths, load_data, build_disclosed_ids

    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="Lastfm-360K-synthetic", help= "Specify task type: ml-1m/ml-1m-synthetic/Lastfm-360K/Lastfm-360K-synthetic")
    parser.add_argument("--s_attr", type=str, default="gender", help= "Specify sensitive attribute name.")
    parser.add_argument(
        "--s_ratios",
        type=float,
        nargs="+", 
        default=[1.0, 1.0, 1.0],  # default for binary cases
        help="Known ratios for training each sensitive group. Example: --s_ratios 0.5 0.1 0.1"
    )
    args = parser.parse_args()

    config = Config(
        task_type=args.task_type,
        s_attr=args.s_attr,
        fair_reg=0,
        s_ratios=args.s_ratios,
    )

    paths = setup_paths(config)
    data = load_data(paths)

    num_users = int(data["train"]["user_id"].max()) + 1
    num_items = int(data["train"]["item_id"].max()) + 1

    disclosed_ids = build_disclosed_ids(
        data["known_sensitive"], config.s_attr, config.s_ratios, seed=config.seed
    )
    
    model = MF(num_users=num_users, num_items=num_items, emb_size=config.emb_size).to(device)
    best_val_rmse, best_test_rmse, best_val_gap, best_test_gap, best_epoch, best_model = pretrain_baseline(
        model=model,
        df_train=data["train"],
        valid_data=data["valid"],
        test_data=data["test"],
        sensitive_attr=data["true_sensitive"],
        disclosed_ids_dict=disclosed_ids,
        config=config,
        device=device,
    )

    print(best_val_rmse, best_test_rmse, best_val_gap, best_test_gap, best_epoch)
    torch.save(best_model.state_dict(), f"./pretrained_models/{config.task_type}/MF_orig_model")