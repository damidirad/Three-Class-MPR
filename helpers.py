import torch
from pathlib import Path
import random
import numpy as np
import pandas as pd

def setup_paths(args):
    try:
        base_path = Path.cwd() / "datasets" / args.task_type
        
        required_files = {
            "train": base_path / "train.csv",
            "valid": base_path / "valid.csv",
            "test": base_path / "test.csv",
            "true_sensitive": base_path / "sensitive_attribute.csv",
            "known_sensitive": base_path / "sensitive_attribute_random.csv"
        }

        if not base_path.exists():
            raise FileNotFoundError(f"Dataset directory missing at: {base_path}")

        for name, file_path in required_files.items():
            if not file_path.is_file():
                raise FileNotFoundError(f"Required {name} file missing: {file_path}")

        print(f"Dataset paths set successfully for {args.task_type}.")
        return required_files

    except FileNotFoundError as e:
        print(f"Configuration Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def load_data(paths):
    data = {}
    for name, path in paths.items():
        data[name] = pd.read_csv(path)
    return data


def get_partial_group_ids(df, attr_name, target_value, ratio, shuffle=True, seed=42):
    """
    Extracts a specific percentage of user IDs for a given attribute value.
    """
    # Mask to filter rows with the target attribute value
    mask = (df[attr_name] == target_value)
    group_ids = df.loc[mask, "user_id"].values
    
    # Shuffle if required
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(group_ids)
        
    # Calculate number of IDs to select
    n_subset = int(ratio * len(group_ids))
    return group_ids[:n_subset]

def set_random_seed(state=1):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on device: MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on device: CUDA")
    else:
        device = torch.device("cpu")
        print("Running on device: CPU")
    return device

def set_rmse_thresh(config):
    task_type = config.task_type
    seed = config.seed

    if task_type == "Lastfm-360K":
        if seed == 1:
            return 0.327087092 / 0.98
        elif seed == 2:
            return 0.327050738 / 0.98
        elif seed ==3:
            return 0.327054454 / 0.98
    elif task_type == "ml-1m":
        if seed == 1:
            return 0.412740352 / 0.98
        elif seed == 2:
            return 0.412416265 / 0.98
        elif seed ==3:
            return 0.412392938 / 0.98
    else:
        raise ValueError("No RMSE threshold specified for this dataset.")

def set_resample_range(args):
    if args.s2_ratio is not None:
        return [
            # balanced
            "0.33_0.33_0.34", 
            
            # extreme skew
            "0.70_0.15_0.15", # 0 skewed
            "0.15_0.70_0.15", # 1 skewed
            "0.15_0.15_0.70", # 2 skewed
            
            # pair dominant
            "0.45_0.45_0.10", # 0 & 1 vs. 2
            "0.45_0.10_0.45", # 1 & 3 vs. 2
            "0.10_0.45_0.45", # 2 & 3 vs. 1
            
            # empirical shift
            "0.20_0.50_0.30", 
            "0.50_0.30_0.20"
        ]
    else:
        # 37 priors from Jizhi et. al. (2025)
        return ["0.1", "0.105", "0.11", "0.12", 
                "0.125", "0.13", "0.14", "0.15", 
                "0.17", "0.18", "0.2", "0.22", 
                "0.25", "0.29", "0.33", "0.4", 
                "0.5", "0.67", "1.0", "1.5", "2.0", 
                "2.5", "3.0", "3.5", "4.0", "4.5", 
                "5.0", "5.5", "6.0", "6.5", "7.0",
                "7.5", "8.0", "8.5", "9.0", "9.5", "10.0"]
 
def compute_robust_fairness_loss(y_hat, user_ids, priors_dict, config, device):
    """Replaces the nested loops in the Robust training function."""
    fair_violations = []
    C = torch.tensor([0.0], device=device)

    # Vectorized check of all priors in the ensemble
    for ratio_key, seeds in priors_dict.items():
        for seed, resample_df in seeds.items():
            # Quick lookup: using .values is faster than .iloc for large batches
            batch_attrs = torch.from_numpy(resample_df[config.s_attr].values[user_ids.cpu()]).to(device)
            
            unique_g = torch.unique(batch_attrs)
            if len(unique_g) > 1:
                group_means = torch.stack([y_hat[batch_attrs == g].mean() for g in unique_g])
                violation = group_means.max() - group_means.min()
                fair_violations.append(violation)
                C = torch.max(C, violation / config.beta)

    if not fair_violations:
        return torch.tensor(0.0, device=device)

    # Log-Sum-Exp trick for stability
    lse = torch.stack([torch.exp((v / config.beta) - C.detach()) for v in fair_violations]).sum()
    return config.beta * (torch.log(lse) + C.detach())