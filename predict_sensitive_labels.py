# -*- coding: utf-8 -*-
"""
SST Prediction with Multi-Class Prior Resampling

Trains a Sensitive Attribute Classifier (SST) on user embeddings from a pretrained
recommendation model, with support for multi-class sensitive attributes and 
configurable prior resampling.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path 
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

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
print(f"(Format: [class0/class{n_classes-1}, class1/class{n_classes-1}, ..., 1.0])")
print(f"Original disclosed counts (train): {[len(disclosed_ids_train.get(c, [])) for c in range(n_classes)]}")

resampled_train_ids = resample_ids_to_prior(
    disclosed_ids_train, prior_ratios, seed=config.seed
)

print(f"Resampled counts (train): {[len(resampled_train_ids.get(c, [])) for c in range(n_classes)]}")

train_embeddings, train_labels = make_tensors_from_disclosed(
    user_embedding, resampled_train_ids, device
)

n_train = len(train_labels)
print(f"Total training samples: {n_train}")

print("\nBuilding test sets...")

sensitive_attr_reshuffled = data["true_sensitive"].sample(
    frac=1, random_state=config.seed + 2
).reset_index(drop=True)

disclosed_ids_reshuffled = build_disclosed_ids(
    sensitive_attr_reshuffled, 
    config.s_attr, 
    config.s_ratios
)

for c in disclosed_ids_reshuffled:
    disclosed_ids_reshuffled[c] = disclosed_ids_reshuffled[c][disclosed_ids_reshuffled[c] < num_model_users]

test_embeddings_reshuffled, test_labels_reshuffled = make_tensors_from_disclosed(
    user_embedding, disclosed_ids_reshuffled, device
)
test_embeddings_reshuffled = test_embeddings_reshuffled.to(device)
test_labels_reshuffled = test_labels_reshuffled.to(device)

print(f"Test set (reshuffled) size: {len(test_labels_reshuffled)}")

test_embeddings_unseen, test_labels_unseen = make_tensors_from_disclosed(
    user_embedding, disclosed_ids_test, device
)
test_embeddings_unseen = test_embeddings_unseen.to(device)
test_labels_unseen = test_labels_unseen.to(device)

print(f"Test set (unseen 20%) size: {len(test_labels_unseen)}")

train_dataset = TensorDataset(train_embeddings, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

optimizer = torch.optim.Adam(
    classifier_model.parameters(), 
    lr=config.sst_lr, 
    weight_decay=config.weight_decay
)
criterion = torch.nn.CrossEntropyLoss()

print(f"\nTraining SST for {args.sst_train_epochs} epochs...")
classifier_model.train()

for epoch in tqdm(range(args.sst_train_epochs), desc="[SST] Training"):
    epoch_loss = 0.0
    n_batches = 0
    
    for batch_embeddings, batch_labels in train_dataloader:
        batch_embeddings = batch_embeddings.to(device)
        batch_labels = batch_labels.to(device)
        
        logits = classifier_model(batch_embeddings)
        loss = criterion(logits, batch_labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    if epoch % 10 == 0 or epoch == args.sst_train_epochs - 1:
        avg_loss = epoch_loss / n_batches
        
        classifier_model.eval()
        with torch.no_grad():
            train_logits = classifier_model(train_embeddings.to(device))
            train_preds = torch.argmax(train_logits, dim=1)
            train_acc = (train_preds == train_labels.to(device).long()).float().mean().item()
        classifier_model.train()
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

print("\nFinal Evaluation")
print("-" * 60)

classifier_model.eval()
with torch.no_grad():
    train_logits = classifier_model(train_embeddings.to(device))
    train_preds = torch.argmax(train_logits, dim=1)
    train_acc = (train_preds == train_labels.to(device).long()).float().mean().item()
    
    test_logits_reshuffled = classifier_model(test_embeddings_reshuffled)
    test_preds_reshuffled = torch.argmax(test_logits_reshuffled, dim=1)
    test_acc_reshuffled = (test_preds_reshuffled == test_labels_reshuffled.long()).float().mean().item()
    
    test_logits_unseen = classifier_model(test_embeddings_unseen)
    test_preds_unseen = torch.argmax(test_logits_unseen, dim=1)
    test_acc_unseen = (test_preds_unseen == test_labels_unseen.long()).float().mean().item()
    
    train_class_dist = [(train_preds == c).sum().item() / len(train_preds) 
                        for c in range(n_classes)]
    test_class_dist_reshuffled = [(test_preds_reshuffled == c).sum().item() / len(test_preds_reshuffled)
                                   for c in range(n_classes)]
    test_class_dist_unseen = [(test_preds_unseen == c).sum().item() / len(test_preds_unseen)
                              for c in range(n_classes)]

print(f"Train Accuracy: {train_acc:.4f}")
print(f"  Predicted class distribution: {[f'{p:.3f}' for p in train_class_dist]}")

print(f"\nTest Accuracy (Reshuffled): {test_acc_reshuffled:.4f}")
print(f"  Predicted class distribution: {[f'{p:.3f}' for p in test_class_dist_reshuffled]}")

print(f"\nTest Accuracy (Unseen 20%): {test_acc_unseen:.4f}")
print(f"  Predicted class distribution: {[f'{p:.3f}' for p in test_class_dist_unseen]}")

print("\nPredicting All User Labels")
print("-" * 60)

classifier_model.eval()
with torch.no_grad():
    all_predictions = []
    batch_size_pred = 1024
    
    for start_idx in range(0, len(user_embedding), batch_size_pred):
        end_idx = min(start_idx + batch_size_pred, len(user_embedding))
        batch_emb = user_embedding[start_idx:end_idx].to(device)
        batch_logits = classifier_model(batch_emb)
        batch_preds = torch.argmax(batch_logits, dim=1)
        all_predictions.append(batch_preds.cpu())
    
    pred_all_labels = torch.cat(all_predictions)

print("Overriding predictions with known labels...")
for class_idx, user_ids in disclosed_ids_full.items():
    if len(user_ids) > 0:
        pred_all_labels[user_ids] = class_idx
        print(f"  Class {class_idx}: {len(user_ids)} users set to ground truth")

pred_sensitive_attr = pd.DataFrame({
    "user_id": list(range(num_model_users)),
    config.s_attr: pred_all_labels.tolist()
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