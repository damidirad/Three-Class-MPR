# -*- coding: utf-8 -*-
"""
SST Prediction with Multi-Class Prior Resampling
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path 
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from config import Config
from SST import SST
from helpers import *

parser = argparse.ArgumentParser(description='SST Prediction with Resampling')
parser.add_argument("--task_type", type=str, default="Lastfm-360K", 
                    help="Specify task type: ml-1m/Lastfm-360K/ml-1m-synthetic/Lastfm-360K-synthetic")
parser.add_argument("--s_attr", type=str, default="gender", 
                    help="Specify sensitive attribute name.")
parser.add_argument("--unfair_model", type=str, 
                    default="./pretrained_model/Lastfm-360K/MF_orig_model",
                    help="Path to pretrained unfair model")
parser.add_argument(
    "--s_ratios",
    type=float,
    nargs="+",
    default=[0.5, 0.2],
    help="Known ratios for each sensitive group. Example: --s_ratios 0.5 0.1 0.1"
)
parser.add_argument("--seed", type=int, default=1, 
                    help="Seed for reproducibility.")
parser.add_argument("--sst_train_epochs", type=int, default=100, 
                    help="Number of epochs for SST training")
parser.add_argument("--prior_resample_idx", type=int, default=0,
                    help="Index into resample_range for prior resampling")
parser.add_argument("--batch_size", type=int, default=128, 
                    help="Batch size for training")
parser.add_argument("--saving_path", type=str, 
                    default="./predict_sst_diff_seed_batch/",
                    help="Directory to save predicted sensitive attributes")

args = parser.parse_args()

config = Config(
    task_type=args.task_type,
    s_attr=args.s_attr,
    unfair_model=args.unfair_model,
    s_ratios=args.s_ratios,
    seed=args.seed,
)

set_random_seed(config.seed)
device = get_device()

resample_range = set_resample_range(config)
n_classes = len(config.s_ratios)

paths = setup_paths(config)
data = load_data(paths)

num_users = len(data["true_sensitive"])
train_sensitive_attr = data["known_sensitive"][:np.int64(0.8 * num_users)] 
test_sensitive_attr = data["known_sensitive"][np.int64(0.8 * num_users):]

disclosed_ids_train = build_disclosed_ids(
    train_sensitive_attr,
    config.s_attr, 
    config.s_ratios, 
)

disclosed_ids_test = build_disclosed_ids(
    test_sensitive_attr, 
    config.s_attr, 
    config.s_ratios, 
)

disclosed_ids_full = build_disclosed_ids(
    data["true_sensitive"], 
    config.s_attr, 
    config.s_ratios
)

print(f"Loading pretrained model from {config.unfair_model}")
orig_model = torch.load(config.unfair_model, map_location=torch.device("cpu"))
user_embedding = orig_model['user_emb.weight'].detach()

user_embedding = user_embedding.to(device)
num_model_users = user_embedding.shape[0]

for d in [disclosed_ids_train, disclosed_ids_test, disclosed_ids_full]:
    for c in d:
        d[c] = d[c][d[c] < num_model_users]

classifier_model = SST(config).to(device)

prior_configs = get_prior_configurations(resample_range, n_classes)

if args.prior_resample_idx >= len(prior_configs):
    raise ValueError(
        f"prior_resample_idx {args.prior_resample_idx} out of range. "
        f"Max index for {n_classes} classes: {len(prior_configs) - 1}"
    )

prior_ratios = prior_configs[args.prior_resample_idx]

print(f"\nUsing prior ratios: {prior_ratios}")
print(f"Original disclosed counts (train): {[len(disclosed_ids_train.get(c, [])) for c in range(n_classes)]}")

resampled_train_ids = resample_ids_to_prior(
    disclosed_ids_train, prior_ratios, seed=config.seed
)

print(f"Resampled counts (train): {[len(resampled_train_ids.get(c, [])) for c in range(n_classes)]}")

# ✅ Build tensors - embeddings already on device, labels will be moved
train_embeddings_list = []
train_labels_list = []

for class_idx in range(n_classes):
    user_ids = resampled_train_ids.get(class_idx, np.array([]))
    if len(user_ids) > 0:
        train_embeddings_list.append(user_embedding[user_ids])
        train_labels_list.append(torch.full((len(user_ids),), class_idx, dtype=torch.float32))

train_tensor = torch.cat(train_embeddings_list, dim=0)  # Already on device
train_label = torch.cat(train_labels_list, dim=0).to(device)

n_train = len(train_label)
print(f"Total training samples: {n_train}")

print("\nBuilding test sets...")

# Reshuffled test set
sensitive_attr_reshuffled = data["true_sensitive"].sample(
    frac=1, random_state=config.seed
).reset_index(drop=True)

disclosed_ids_reshuffled = build_disclosed_ids(
    sensitive_attr_reshuffled, 
    config.s_attr, 
    config.s_ratios
)

for c in disclosed_ids_reshuffled:
    disclosed_ids_reshuffled[c] = disclosed_ids_reshuffled[c][disclosed_ids_reshuffled[c] < num_model_users]

test_embeddings_list = []
test_labels_list = []

for class_idx in range(n_classes):
    user_ids = disclosed_ids_reshuffled.get(class_idx, np.array([]))
    if len(user_ids) > 0:
        test_embeddings_list.append(user_embedding[user_ids])
        test_labels_list.append(torch.full((len(user_ids),), class_idx, dtype=torch.float32))

test_tensor = torch.cat(test_embeddings_list, dim=0)
test_label = torch.cat(test_labels_list, dim=0).to(device)

print(f"Test set (reshuffled) size: {len(test_label)}")

# Unseen test set
test_embeddings_unseen_list = []
test_labels_unseen_list = []

for class_idx in range(n_classes):
    user_ids = disclosed_ids_test.get(class_idx, np.array([]))
    if len(user_ids) > 0:
        test_embeddings_unseen_list.append(user_embedding[user_ids])
        test_labels_unseen_list.append(torch.full((len(user_ids),), class_idx, dtype=torch.float32))

test_tensor_unseen = torch.cat(test_embeddings_unseen_list, dim=0)
test_label_unseen = torch.cat(test_labels_unseen_list, dim=0).to(device)

print(f"Test set (unseen 20%) size: {len(test_label_unseen)}")

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = CustomDataset(train_tensor, train_label)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

optimizer = torch.optim.Adam(classifier_model.parameters(), lr=config.sst_lr)
criterion = torch.nn.CrossEntropyLoss()

print(f"\nTraining SST for {args.sst_train_epochs} epochs...")

# Training loop
for i in tqdm(range(args.sst_train_epochs)):
    classifier_model.train()  # Set to training mode
    
    for train_input, labels in train_dataloader:
        train_pred = classifier_model(train_input)

        loss_train = criterion(train_pred, labels.type(torch.LongTensor).to(device))
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

print("\nFinal Evaluation")
print("-" * 60)

classifier_model.eval()
with torch.no_grad():
    train_preds = classifier_model(train_tensor).argmax(1)
    train_acc = (train_preds == train_label.long()).float().mean().item()
    
    test_preds = classifier_model(test_tensor).argmax(1)
    test_acc = (test_preds == test_label.long()).float().mean().item()
    
    test_preds_unseen = classifier_model(test_tensor_unseen).argmax(1)
    test_acc_unseen = (test_preds_unseen == test_label_unseen.long()).float().mean().item()
    
    train_class_dist = [(train_preds == c).sum().item() / len(train_preds) 
                        for c in range(n_classes)]
    test_class_dist = [(test_preds == c).sum().item() / len(test_preds)
                       for c in range(n_classes)]
    test_class_dist_unseen = [(test_preds_unseen == c).sum().item() / len(test_preds_unseen)
                              for c in range(n_classes)]

print(f"Train Accuracy: {train_acc:.4f}")
print(f"  Predicted class distribution: {[f'{p:.3f}' for p in train_class_dist]}")

print(f"\nTest Accuracy (Reshuffled): {test_acc:.4f}")
print(f"  Predicted class distribution: {[f'{p:.3f}' for p in test_class_dist]}")

print(f"\nTest Accuracy (Unseen 20%): {test_acc_unseen:.4f}")
print(f"  Predicted class distribution: {[f'{p:.3f}' for p in test_class_dist_unseen]}")

print("\nPredicting All User Labels")
print("-" * 60)

classifier_model.eval()
with torch.no_grad():
    pred_all_label = classifier_model(user_embedding).max(1).indices

# Override with known labels
for class_idx, user_ids in disclosed_ids_full.items():
    if len(user_ids) > 0:
        pred_all_label[user_ids] = class_idx
        print(f"  Class {class_idx}: {len(user_ids)} users set to ground truth")

pred_sensitive_attr = pd.DataFrame({
    "user_id": list(range(num_model_users)),
    config.s_attr: pred_all_label.cpu().tolist()
})

ratio_str = "_".join([f"{r}" for r in config.s_ratios])
subdir = (
    f"{config.task_type}_ratios_{ratio_str}_"
    f"epochs_{args.sst_train_epochs}_prior_{args.prior_resample_idx}"
)

save_filename = f"seed_{config.seed}.csv"
save_dir = Path(args.saving_path) / subdir
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / save_filename

pred_sensitive_attr.to_csv(save_path, index=False)

print("\nResults Saved")
print("-" * 60)
print(f"Output directory: {save_dir}")
print(f"Output file: {save_filename}")
print(f"Full path: {save_path}")
print(f"\nTotal users: {len(pred_sensitive_attr)}")
print(f"Predicted class distribution: {pred_sensitive_attr[config.s_attr].value_counts().sort_index().to_dict()}")