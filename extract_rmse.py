# Add this script to your project root: extract_rmse.py

import torch
import numpy as np
from pathlib import Path
from config import Config
from helpers import setup_paths, load_data, get_device, calculate_rmse
from MF import MF

def extract_pretrained_rmse(task_type: str, model_path: str, seed: int = 1):
    """
    Load a pretrained model and calculate its validation RMSE.
    
    Args:
        task_type: Dataset name (e.g., "ml-1m-synthetic")
        model_path: Path to pretrained model file
        seed: Seed used during training
    """
    device = get_device()
    
    # Create minimal config just for loading data
    config = Config(
        task_type=task_type,
        s_attr="gender",
        unfair_model=model_path,
        s_ratios=[0.5, 0.5, 0.1],
        seed=seed
    )
    
    paths = setup_paths(config)
    data = load_data(paths)
    
    # Get dimensions
    num_users = max(data["train"].user_id.max(), data["valid"].user_id.max()) + 1
    num_items = max(data["train"].item_id.max(), data["valid"].item_id.max()) + 1
    
    # Load model
    model = MF(np.int64(num_users), np.int64(num_items), config.emb_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Calculate validation RMSE - use 'label' column instead of 'rating'
    valid_users = torch.LongTensor(data["valid"].user_id.values).to(device)
    valid_items = torch.LongTensor(data["valid"].item_id.values).to(device)
    valid_labels = torch.FloatTensor(data["valid"].label.values).to(device)
    
    with torch.no_grad():
        predictions = model(valid_users, valid_items)
        rmse = calculate_rmse(valid_labels, predictions)
    
    print(f"\nTask: {task_type}, Seed: {seed}")
    print(f"Validation RMSE: {rmse:.9f}")
    print(f"RMSE / 0.98: {rmse / 0.98:.9f}")
    print(f"\nAdd to helpers.py set_rmse_thresh():")
    print(f'    if seed == {seed}:')
    print(f'        return {rmse:.9f} / 0.98')
    
    return rmse

if __name__ == "__main__":
    # Extract RMSE for all synthetic datasets
    datasets = [
        ("ml-1m-synthetic", "./pretrained_models/ml-1m-synthetic/MF_orig_model"),
        ("Lastfm-360K-synthetic", "./pretrained_models/Lastfm-360K-synthetic/MF_orig_model")
    ]
    
    for task_type, model_path in datasets:
        print("\n" + "="*60)
        for seed in [1, 2, 3]:
            try:
                extract_pretrained_rmse(task_type, model_path, seed)
            except Exception as e:
                print(f"Error for {task_type} seed {seed}: {e}")