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
    # mask to filter rows with the target attribute value
    mask = (df[attr_name] == target_value)
    group_ids = df.loc[mask, "user_id"].values
    
    # shuffle if required
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(group_ids)
        
    # calculate number of IDs to select
    n_subset = int(ratio * len(group_ids))
    return group_ids[:n_subset]

def set_random_seed(state=1):
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def compute_loss(user_embeddings, classifier, target_priors, beta=0.01):
    """
    user_embeddings: [batch_size, dim]
    classifier: Calibrated classifier
    target_priors: Tensor of shape [num_priors] (e.g., torch.tensor([0.1, 0.5, 0.9]))
    """
    # Get observed probs (Shape: [batch_size])
    p_female_obs = classifier(user_embeddings).softmax(dim=1)[:, 1]
    p_male_obs = 1 - p_female_obs

    # Reshape for broadcasting
    # p_female_obs: [batch_size, 1]
    # target_priors: [1, num_priors]
    p_f = p_female_obs.unsqueeze(1)
    p_m = p_male_obs.unsqueeze(1)
    p_t = target_priors.unsqueeze(0)

    # Weight calculation
    # Shape: [batch_size, num_priors]
    w_female = p_t / (p_f.mean(dim=0, keepdim=True) + 1e-9)
    w_male = (1 - p_t) / (p_m.mean(dim=0, keepdim=True) + 1e-9)

    # Weighted means
    # Shape: [num_priors]
    weighted_f_means = (p_f * w_female).mean(dim=0)
    weighted_m_means = (p_m * w_male).mean(dim=0)

    # Unfairness per prior
    # Shape: [num_priors]
    unfairness_per_prior = torch.abs(weighted_f_means - weighted_m_means)
    
    # Log-Sum-Exp across the prior dimension
    mpr_loss = beta * torch.logsumexp(unfairness_per_prior / beta, dim=0)

    return mpr_loss

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
