import torch
from pathlib import Path
import random
import numpy as np
import pandas as pd

from config import Config

def setup_paths(args: Config) -> dict[str, Path]:
    """
    Set up dataset paths based on task type.
    
    Args:
        args: Parsed command-line arguments containing task_type.
    Returns:
        A dictionary with paths to train, valid, test, true_sensitive, and known_sensitive files
    """
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
        # 10 priors for 3-class sensitive attribute
        [
            # balanced
            0.33, 0.33, 0.34, 
            
            # extreme skew
            0.70, 0.15, 0.15, # 0 skewed
            0.15, 0.70, 0.15, # 1 skewed
            0.15, 0.15, 0.70, # 2 skewed
            
            # pair dominant
            0.45, 0.45, 0.10, # 0 & 1 vs. 2
            0.45, 0.10, 0.45, # 0 & 2 vs. 1
            0.10, 0.45, 0.45, # 1 & 2 vs. 0
            
            # empirical shift
            0.20, 0.50, 0.30, 
            0.50, 0.30, 0.20
        ],
            2:
        # 37 priors for 2-class sensitive attribute
        # from Jizhi et. al. (2025)
        [0.1, 0.105, 0.11, 0.12, 
                0.125, 0.13, 0.14, 0.15, 
                0.17, 0.18, 0.2, 0.22, 
                0.25, 0.29, 0.33, 0.4, 
                0.5, 0.67, 1.0, 1.5, 2.0, 
                2.5, 3.0, 3.5, 4.0, 4.5, 
                5.0, 5.5, 6.0, 6.5, 7.0,
                7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    
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
                        s_ratios: list[float], 
                        seed: int = 23) -> dict[int, np.ndarray]:
    """
    Build a dict with known users per class,
    using the provided ratios. Deterministically shuffles before slicing.

    Args:
        df_sensitive: DataFrame with columns ['user_id', s_attr].
        s_attr: str, sensitive attribute column name (e.g., 'gender').
        s_ratios: list[float], known ratio per class index (0..K-1).
        seed: int, RNG seed for reproducibility.

    Returns:
        disclosed: dict[int, np.ndarray] mapping class_idx -> disclosed user_ids.
    """
    disclosed = {}
    rng = np.random.default_rng(seed)
    classes = sorted(df_sensitive[s_attr].unique().tolist())

    for class_idx in classes:
        class_users = df_sensitive[df_sensitive[s_attr] == class_idx]["user_id"].to_numpy()
        rng.shuffle(class_users)

        # If s_ratios shorter than number of classes, fallback to uniform for missing indices
        ratio = s_ratios[class_idx] if class_idx < len(s_ratios) else (1.0 / len(classes))
        k = int(ratio * len(class_users))

        disclosed[class_idx] = class_users[:k]

    return disclosed

