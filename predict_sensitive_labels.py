import argparse
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from SST import SST
from tqdm import tqdm
from config import Config
from helpers import *
from evaluation import evaluate_sst

parser = argparse.ArgumentParser()

parser.add_argument("--task_type",type=str,default="Lastfm-360K",help="Specify task type: ml-1m/lastfm-360K") 
parser.add_argument("--s_attr",type=str,default="gender",help="Specify sensitive attribute name.")
parser.add_argument("--unfair_model", type=str, default= "./pretrained_model/Lastfm-360K/MF_orig_model")
parser.add_argument(
    "--s_ratios",
    type=float,
    nargs="+", 
    default=[0.5, 0.1],  # default for binary cases
    help="Known ratios for training each sensitive group. Example: --s_ratios 0.5 0.1 0.1"
)
parser.add_argument("--seed", type=int, default= 1, help= "Seed for reproducibility.")

args = parser.parse_args()

config = Config(
    task_type=args.task_type,
    s_attr=args.s_attr,
    unfair_model=args.unfair_model,
    s_ratios=args.s_ratios,
    seed=args.seed,
)

resample_range = set_resample_range(config)

# Seed all RNGs
set_random_seed(config.seed)

device = get_device()

# Load sensitive attributes 
paths = setup_paths(args)
data = load_data(paths)

# 80/20 train/test split of sensitive attributes
num_users = len(data["true_sensitive"])
train_sensitive_attr = data["true_sensitive"][:np.int64(0.8 * num_users)]
test_sensitive_attr = data["true_sensitive"][np.int64(0.8 * num_users):]

# Known users per class for train/test based on config.s_ratios
disclosed_ids_train = build_disclosed_ids(train_sensitive_attr, config.s_attr, config.s_ratios, seed=config.seed)
disclosed_ids_test = build_disclosed_ids(test_sensitive_attr, config.s_attr, config.s_ratios, seed=config.seed + 1)

# Load pretrained MF user embeddings
orig_model = torch.load(str(config.unfair_model), map_location=torch.device("cpu")) # adjust based on your setup 
user_embedding = orig_model['user_emb.weight'].detach().to(device)

# Initialize SST classifier
classifier_model = SST(config).to(device)

# Build training tensors with resampled class priors (multiclass)
resampled_train_ids = resample_ids_to_priors(disclosed_ids_train, resample_range, seed=config.seed)
train_tensor, train_label = make_sst_tensors(user_embedding, resampled_train_ids, device)

# Construct evaluation tensors
# Shuffle full sensitive_attr deterministically for building a test subset per class
sensitive_attr_reshuffled = data["true_sensitive"].sample(frac=1, random_state=config.seed).reset_index(drop=True)
disclosed_ids_eval = build_disclosed_ids(sensitive_attr_reshuffled, config.s_attr, config.s_ratios, seed=config.seed)
test_tensor, test_label = make_sst_tensors(user_embedding, disclosed_ids_eval, device)

# 20% unseen users for final evaluation
test_tensor_unseen, test_label_unseen = data["true_sensitive"](user_embedding, disclosed_ids_test, device)

# Custom dataset for SST training
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = CustomDataset(train_tensor, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

optimizer_for_classifier = torch.optim.Adam(classifier_model.parameters(), lr=config.sst_lr)
loss_for_classifier = torch.nn.CrossEntropyLoss()

# Train SST classifier
for i in tqdm(range(config.sst_epochs), desc="[SST] Training SST classifier"):
    for train_input, labels in train_dataloader:
        train_pred = classifier_model(train_input)
        loss_train = loss_for_classifier(train_pred, labels.type(torch.LongTensor).to(device))
        optimizer_for_classifier.zero_grad()
        loss_train.backward()
        optimizer_for_classifier.step()

# Evaluate SST (multiclass accuracy + predicted class ratios)
train_acc, train_pred_ratio = evaluate_sst(train_tensor, train_label, classifier_model, config.n_classes, device)
test_acc, test_pred_ratio = evaluate_sst(test_tensor, test_label, classifier_model, config.n_classes, device)
test_unseen_acc, test_pred_ratio_unseen = evaluate_sst(test_tensor_unseen, test_label_unseen, classifier_model, config.n_classes, device)

print("[SST] Test accuracy on unseen users:" + str(test_unseen_acc) + "\n")
print("[SST] Test accuracy on disclosed users:" + str(test_acc) + "\n")
print("[SST] Test predicted class ratio on unseen users:" + str(test_pred_ratio_unseen) + "\n")
print("[SST] Test predicted class ratio on disclosed users:" + str(test_pred_ratio) + "\n")

# Predict labels for ALL users, then overwrite with ground truth for disclosed users
pred_all_label = classifier_model(user_embedding).max(1).indices.cpu()

# Build disclosures on full population to overwrite with ground truth
full_disclosed_ids = build_disclosed_ids(data["true_sensitive"], config.s_attr, config.s_ratios, seed=config.seed)
for class_idx, ids in full_disclosed_ids.items():
    if len(ids) > 0:
        pred_all_label[ids] = class_idx

pred_sensitive_attr = pd.DataFrame(list(zip(list(range(num_users))), list(pred_all_label.tolist())), \
    columns=["user_id", config.s_attr])

# Save predictions
ratio_str = "_".join([str(r) for r in config.s_ratios])
subdir = args.task_type + "_" + ratio_str + "_gender_train_epoch_" + str(args.gender_train_epoch)
os.makedirs(os.path.join(args.saving_path, subdir), exist_ok=True)

# If priors are provided, include them in filename for clarity
if config.resample_priors is not None:
    prior_str = "_".join([str(p) for p in config.resample_priors])
    save_csv = "priors_" + prior_str + "_seed" + str(config.seed) + ".csv"
else:
    save_csv = "seed" + str(config.seed) + ".csv"

pred_sensitive_attr.to_csv(os.path.join(args.saving_path, subdir, save_csv), index=None)