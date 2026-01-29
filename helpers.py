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
    mask = (df[attr_name] == target_value)
    group_ids = df.loc[mask, "user_id"].values
    
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(group_ids)
        
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

    if task_type == "Lastfm-360K" or task_type == "Lastfm-360K-synthetic": #TEMPORARY
        if seed == 1:
            return 0.327087092 / 0.98
        elif seed == 2:
            return 0.327050738 / 0.98
        elif seed == 3:
            return 0.327054454 / 0.98
    elif task_type == "ml-1m":
        if seed == 1:
            return 0.412740352 / 0.98
        elif seed == 2:
            return 0.412416265 / 0.98
        elif seed == 3:
            return 0.412392938 / 0.98
    elif task_type == "ml-1m-synthetic":
        if seed == 1:
            return 0.362311214 / 0.98
        elif seed == 2:
            return 0.362311214 / 0.98
        elif seed == 3:
            return 0.362311214 / 0.98
    else:
        raise ValueError("No RMSE threshold specified for this dataset.")

def set_resample_range(config: Config) -> list[float]:
    """
    Set resample ranges based on the number of sensitive attribute classes.
    
    Args:
        config: Configuration object containing s_ratios.
    Returns:
        Resample range (37 standard ratios from Jizhi et al. 2025).
    """
    return [ 
        0.1, 0.105, 0.11, 0.12, 
        0.125, 0.13, 0.14, 0.15, 
        0.17, 0.18, 0.2, 0.22, 
        0.25, 0.29, 0.33, 0.4, 
        0.5, 0.67, 1.0, 1.5, 2.0, 
        2.5, 3.0, 3.5, 4.0, 4.5, 
        5.0, 5.5, 6.0, 6.5, 7.0,
        7.5, 8.0, 8.5, 9.0, 9.5, 10.0
    ]
    
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
 
def build_disclosed_ids(sensitive_attr_df, s_attr, s_ratios):
    """
    Build dictionary of disclosed user IDs per class.
    
    Args:
        sensitive_attr_df: DataFrame with user_id and sensitive attribute columns
        s_attr: Name of sensitive attribute column (e.g., "gender")
        s_ratios: List of disclosure ratios per class
        
    Returns:
        Dict mapping class_idx -> np.array of disclosed user_ids
    """
    n_classes = len(s_ratios)
    disclosed_ids = {}
    
    for class_idx in range(n_classes):
        class_users = sensitive_attr_df[sensitive_attr_df[s_attr] == class_idx]["user_id"].to_numpy()
        n_disclosed = int(s_ratios[class_idx] * len(class_users))
        disclosed_ids[class_idx] = class_users[:n_disclosed]
    
    return disclosed_ids

def get_prior_configurations(resample_range: list[float], n_classes: int) -> list[list[float]]:
    """
    Generate prior configurations from resample_range.
    
    Each configuration applies the same ratio to all non-reference classes.
    The last class is always the reference (ratio = 1.0).
    
    Examples:
        n_classes=2, r=0.5 → [0.5, 1.0] (class0 is 50% of class1)
        n_classes=3, r=0.5 → [0.5, 0.5, 1.0] (class0 and class1 are each 50% of class2)
        n_classes=4, r=2.0 → [2.0, 2.0, 2.0, 1.0] (all non-ref classes are 2× reference)
    
    Args:
        resample_range: List of ratio values
        n_classes: Number of classes
        
    Returns:
        List of prior configurations, one per value in resample_range
    """
    return [[r] * (n_classes - 1) + [1.0] for r in resample_range]

def make_tensors_from_disclosed(user_embedding, disclosed_ids_dict):
    """
    Create training tensors from disclosed user IDs.
    
    Args:
        user_embedding: Tensor of user embeddings [num_users, emb_dim]
        disclosed_ids_dict: Dict mapping class_idx -> np.array of user_ids
        device: torch device
        
    Returns:
        Tuple of (embeddings_tensor, labels_tensor)
    """
    embeddings_list = []
    labels_list = []
    
    for class_idx, user_ids in disclosed_ids_dict.items():
        if len(user_ids) > 0:
            class_embeddings = user_embedding[user_ids]
            class_labels = torch.full((len(user_ids),), class_idx, dtype=torch.long)
            
            embeddings_list.append(class_embeddings)
            labels_list.append(class_labels)
    
    if len(embeddings_list) == 0:
        raise ValueError("No disclosed IDs to create tensors from")
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return embeddings, labels

def resample_ids_to_prior(disclosed_ids_dict, prior_ratios, seed):
    """
    Resample disclosed IDs to match target prior distribution.
    Uses the last class as reference and downsamples others.
    
    Args:
        disclosed_ids_dict: Dict mapping class_idx -> np.array of user_ids
        prior_ratios: List of ratios [r0, r1, ..., r(n-2), 1.0]
                      where ri = class_i / class_(n-1)
        seed: Random seed
        
    Returns:
        Dict mapping class_idx -> resampled user_ids (np.ndarray)
    """
    np.random.seed(seed)
    
    n_classes = len(prior_ratios)
    counts = {c: len(ids) for c, ids in disclosed_ids_dict.items() if len(ids) > 0}
    
    print(f"\n[DEBUG resample_ids_to_prior]")
    print(f"  Input prior_ratios: {prior_ratios}")
    print(f"  Available counts: {counts}")
    
    if len(counts) == 0:
        raise ValueError("No disclosed IDs available for resampling")
    
    # Last class is always the reference (should have ratio = 1.0)
    reference_class = n_classes - 1
    
    if reference_class not in counts:
        raise ValueError(f"Reference class {reference_class} has no disclosed IDs")
    
    reference_count = counts[reference_class]
    print(f"  Reference class: {reference_class} with {reference_count} samples (ratio=1.0)")
    
    resampled = {}
    for c in range(n_classes):
        if c not in disclosed_ids_dict or len(disclosed_ids_dict[c]) == 0:
            resampled[c] = np.array([], dtype=np.int64)
            print(f"  Class {c}: EMPTY (not in disclosed_ids)")
            continue
        
        if c == reference_class:
            # Keep all samples from reference class
            resampled[c] = disclosed_ids_dict[c]
            print(f"  Class {c}: REFERENCE - using all {len(resampled[c])} samples")
        else:
            # Downsample to match ratio relative to reference
            target_count = int(reference_count * prior_ratios[c])
            available_count = counts[c]
            
            if target_count >= available_count:
                # Use all available if target exceeds availability
                resampled[c] = disclosed_ids_dict[c]
                print(f"  Class {c}: ratio={prior_ratios[c]:.3f} -> target={target_count} >= available={available_count} -> using all")
            else:
                # Randomly sample without replacement
                resampled[c] = np.random.choice(
                    disclosed_ids_dict[c], target_count, replace=False
                )
                print(f"  Class {c}: ratio={prior_ratios[c]:.3f} -> target={target_count} < available={available_count} -> sampled {target_count}")
    
    final_counts = [len(resampled.get(c, [])) for c in range(n_classes)]
    print(f"  Final resampled counts: {final_counts}\n")
    
    return resampled