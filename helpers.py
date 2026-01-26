import torch
from pathlib import Path
import random
import numpy as np
import pandas as pd

from config import Config

def setup_paths(config: Config) -> dict[str, Path]:
    """
    Set up dataset paths based on task type.
    
    Args:
        config: Configuration object containing task_type.
    Returns:
        A dictionary with paths to train, valid, test, true_sensitive, and known_sensitive files
    """
    try:
        base_path = Path.cwd() / "datasets" / config.task_type
        
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

        print(f"Dataset paths set successfully for {config.task_type}.")
        return required_files

    except FileNotFoundError as e:
        print(f"Configuration Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def load_data(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """
    Load datasets from specified paths into pandas DataFrames.

    Args:
        paths: Dictionary with dataset file paths.
    Returns:    
        A dictionary with loaded DataFrames for each dataset.
    """
    data = {}
    for name, path in paths.items():
        data[name] = pd.read_csv(path)
    return data


def get_partial_group_ids(df: pd.DataFrame, 
                          attr_name: str, 
                          target_value, 
                          ratio: float, 
                          shuffle: bool = True, 
                          seed: int = 23) -> np.ndarray:
    """
    Extracts a specific percentage of user IDs for a given attribute value.

    Args:
        df: DataFrame containing user IDs and sensitive attributes.
        attr_name: Name of the sensitive attribute column.
        target_value: The specific value of the sensitive attribute to filter by.
        ratio: Proportion of user IDs to extract (between 0 and 1).
        shuffle: Whether to shuffle the selected IDs.
        seed: Random seed for shuffling.
    Returns:
        A numpy array of user IDs corresponding to the specified attribute value.
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

def set_random_seed(state: int = 1) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        state: The seed value to set.
    """
    gens = (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def get_device() -> torch.device:
    """
    Determine the available device (MPS, CUDA, or CPU) for PyTorch operations.

    Returns:
        A torch.device object representing the selected device.
    """
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

def set_rmse_thresh(config: Config) -> float:
    """
    Set RMSE threshold based on task type and seed.

    Args:
        config: Configuration object containing task_type and seed.
    Returns:
        The RMSE threshold value.
    """
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

# ensure resample range exists for given s_ratios
def set_resample_range(config: Config) -> list[float]:
    """
    Set resample ranges based on the number of sensitive attribute classes.
    
    Args:
        config: Configuration object containing s_ratios.
    Returns:
        A list of resample ranges.
    """
    ranges = {3: 
        # 37 priors for 3-class sensitive attribute
        [
            # 1. Balanced
            0.333, 0.333, 0.334,
            
            # 2-11. NB vulnerable, male≈female (symmetric minority)
            0.35, 0.35, 0.30,
            0.375, 0.375, 0.25,
            0.40, 0.40, 0.20,
            0.425, 0.425, 0.15,
            0.45, 0.45, 0.10,
            0.46, 0.46, 0.08,
            0.47, 0.47, 0.06,
            0.475, 0.475, 0.05,
            0.48, 0.48, 0.04,
            0.49, 0.49, 0.02,
            
            # 12-21. Male dominant (nb vulnerable, female<male)
            0.70, 0.20, 0.10,
            0.70, 0.15, 0.15,
            0.60, 0.30, 0.10,
            0.75, 0.15, 0.10,
            0.80, 0.10, 0.10,
            0.65, 0.25, 0.10,
            0.55, 0.35, 0.10,
            0.75, 0.20, 0.05,
            0.85, 0.10, 0.05,
            0.68, 0.22, 0.10,
            
            # 22-31. Female dominant (nb vulnerable, male<female)
            0.20, 0.70, 0.10,
            0.15, 0.70, 0.15,
            0.30, 0.60, 0.10,
            0.15, 0.75, 0.10,
            0.10, 0.80, 0.10,
            0.25, 0.65, 0.10,
            0.35, 0.55, 0.10,
            0.20, 0.75, 0.05,
            0.10, 0.85, 0.05,
            0.22, 0.68, 0.10,
            
            # 32-37. NB dominant & asymmetric scenarios
            0.20, 0.20, 0.60,   # NB dominant, male=female vulnerable
            0.15, 0.15, 0.70,   # NB very dominant
            0.10, 0.10, 0.80,   # NB extremely dominant
            0.50, 0.30, 0.20,   # Male>female, NB moderate
            0.30, 0.50, 0.20,   # Female>male, NB moderate
            0.40, 0.35, 0.25,   # Moderate asymmetry, all represented
        ],
            2:
        # 37 priors for 2-class sensitive attribute
        # from Jizhi et. al. (2025)
        [ 
            0.1, 0.105, 0.11, 0.12, 
            0.125, 0.13, 0.14, 0.15, 
            0.17, 0.18, 0.2, 0.22, 
            0.25, 0.29, 0.33, 0.4, 
            0.5, 0.67, 1.0, 1.5, 2.0, 
            2.5, 3.0, 3.5, 4.0, 4.5, 
            5.0, 5.5, 6.0, 6.5, 7.0,
            7.5, 8.0, 8.5, 9.0, 9.5, 10.0
        ]
        }
    
    num_classes = len(config.s_ratios)
    if num_classes in ranges:
        return ranges[num_classes]
    else:
        raise ValueError("No resample range specified for this number of sensitive attribute classes.")

def calculate_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate RMSE between true and predicted values.
    
    Args:
        y_true: Tensor of true values.
        y_pred: Tensor of predicted values.
    Returns:
        RMSE value as a float.
    """
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))
 
def build_disclosed_ids(df_sensitive: pd.DataFrame,
                                 s_attr: str,
                                 s_ratios: list[float]) -> dict[int, np.ndarray]:
    """
    Build a dictionary of disclosed user IDs per sensitive attribute class.

    Args:
        df_sensitive: DataFrame containing user IDs and sensitive attributes.
        s_attr: Name of the sensitive attribute column.
        s_ratios: List of ratios for each sensitive group.
    Returns:
        A dictionary mapping class index to numpy array of disclosed user IDs.
    """
    disclosed = {}
    classes = sorted(df_sensitive[s_attr].unique().tolist())
    for c in classes:
        ids = df_sensitive.loc[df_sensitive[s_attr] == c, "user_id"].to_numpy()
        k = int(s_ratios[c] * len(ids))
        disclosed[c] = ids[:k]
    return disclosed

def get_prior_configurations(resample_range: list[float], n_classes: int) -> list[list[float]]:
    """
    Split flattened resample range into individual prior configurations.
    
    Args:
        resample_range: Flattened list from set_resample_range().
        n_classes: Number of classes (2 or 3).
        
    Returns:
        List of prior configurations.
        For 2-class: [[0.1], [0.105], [0.11], ...]  (37 configs)
        For 3-class: [[0.333, 0.333, 0.334], [0.35, 0.35, 0.30], ...]  (21 configs)
    """
    if n_classes == 2:
        # Each ratio is a separate configuration
        return [[r] for r in resample_range]
    else:
        # Group every n_classes values into one configuration
        configs = []
        for i in range(0, len(resample_range), n_classes):
            configs.append(resample_range[i:i+n_classes])
        return configs
    
def resample_multiclass(disclosed_ids: dict[int, np.ndarray],
                                  priors: list[float],
                                  seed: int) -> dict[int, np.ndarray]:
    priors = np.asarray(priors, dtype=np.float64)
    if priors.ndim != 1:
        raise ValueError("resample_priors must be a 1D list")
    if np.any(priors < 0):
        raise ValueError("resample_priors must be non-negative")
    if priors.sum() <= 0:
        raise ValueError("resample_priors must sum to > 0")

    priors = priors / priors.sum()

    available = {c: len(disclosed_ids.get(c, [])) for c in range(len(priors))}
    # limiting total N (same logic as your resample_ids_to_priors, but keep numpy sampling like old)
    ratios = []
    for c in range(len(priors)):
        if priors[c] > 0:
            ratios.append(available[c] / priors[c])
    if not ratios:
        return {c: np.array([], dtype=np.int64) for c in range(len(priors))}
    N = int(min(ratios))

    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    for c in range(len(priors)):
        target = int(N * priors[c])
        ids = np.asarray(disclosed_ids.get(c, np.array([], dtype=np.int64)), dtype=np.int64)
        if target <= 0 or ids.size == 0:
            out[c] = np.array([], dtype=np.int64)
        elif target <= ids.size:
            out[c] = rng.choice(ids, size=target, replace=False)
        else:
            out[c] = ids.copy()
    return out


def make_sst_tensors(user_embedding, disclosed_ids, device, *, prior_key=None):
    # If nested dict, select one prior
    if disclosed_ids and isinstance(next(iter(disclosed_ids.values())), dict):
        if prior_key is None:
            prior_key = next(iter(disclosed_ids.keys()))  # pick first by default
        disclosed_ids = disclosed_ids[prior_key]

    embedding_list, label_list = [], []
    for class_idx in sorted(disclosed_ids.keys()):
        user_ids = disclosed_ids[class_idx]
        user_ids = torch.as_tensor(np.asarray(user_ids), dtype=torch.long, device=user_embedding.device)
        if user_ids.numel() > 0:
            embedding_list.append(user_embedding[user_ids])
            label_list.append(torch.full((user_ids.numel(),), class_idx, dtype=torch.long, device=device))

    if not embedding_list:
        emb_dim = user_embedding.shape[1]
        return torch.empty(0, emb_dim, device=device), torch.empty(0, device=device, dtype=torch.long)

    return torch.cat(embedding_list, 0).to(device), torch.cat(label_list, 0)


def resample_ids_to_priors(disclosed_ids: dict[int, np.ndarray], 
                           resample_range: list[float], 
                           n_classes: int,
                           seed: int = 23) -> dict:
    """
    Resample disclosed IDs for ALL prior configurations.

    Args:
        disclosed_ids: dict[int, np.ndarray] mapping class_idx -> disclosed user_ids.
        resample_range: Flattened list of all prior configurations from set_resample_range().
        n_classes: Number of classes (2 or 3).
        seed: int, Base RNG seed for reproducibility.
        
    Returns:
        For 2-class: {ratio_key: {0: user_ids_male, 1: user_ids_female}, ...}
        For 3-class: {prior_idx: {0: user_ids_male, 1: user_ids_female, 2: user_ids_nb}, ...}
    """
    # Split flattened range into individual prior configurations
    prior_configs = get_prior_configurations(resample_range, n_classes)
    
    resampled_dict = {}
    
    for prior_idx, prior in enumerate(prior_configs):
        # Use different seed for each prior to avoid identical samples
        prior_seed = seed + prior_idx
        rng = np.random.default_rng(prior_seed)
        
        # Handle 2-class ratio format
        if n_classes == 2 and len(prior) == 1:
            ratio = prior[0]
            normalized_priors = [ratio / (1 + ratio), 1 / (1 + ratio)]
            prior_key = ratio  # Use ratio as key for 2-class
        else:
            # Normalize priors
            prior_sum = sum(prior)
            normalized_priors = [p / prior_sum for p in prior]
            prior_key = prior_idx  # Use index as key for 3-class
        
        # Find limiting class
        ratios = []
        for class_idx in range(n_classes):
            available = len(disclosed_ids[class_idx])
            if normalized_priors[class_idx] > 0:
                ratios.append(available / normalized_priors[class_idx])
        
        if len(ratios) == 0:
            resampled_dict[prior_key] = {i: np.array([], dtype=np.int64) for i in range(n_classes)}
            continue
        
        max_total_samples = int(min(ratios))
        
        # Resample each class
        resampled_ids = {}
        for class_idx in range(n_classes):
            target_count = int(max_total_samples * normalized_priors[class_idx])
            available_ids = disclosed_ids[class_idx]
            
            if target_count <= len(available_ids):
                resampled_ids[class_idx] = rng.choice(
                    available_ids, 
                    size=target_count, 
                    replace=False
                )
            else:
                resampled_ids[class_idx] = available_ids.copy()
        
        resampled_dict[prior_key] = resampled_ids
    
    return resampled_dict
