from pathlib import Path
import pandas as pd
import numpy as np

# --- Paths ---
data_dir = Path("/Users/danie/Documents/projects/NewMPR/NewMPR/datasets/Lastfm-360K/Lastfm-360K-3C")  
users_path = data_dir / "users.tsv"
items_path = data_dir / "items.tsv"

# --- Load users ---
users = pd.read_csv(
    users_path,
    sep="\t",
    header=None,
    names=["uid", "gender", "age", "country", "signup_date"]
)[["uid", "gender"]].dropna()

# --- Add 'other' gender (10% of men/women) ---
rng = np.random.default_rng(seed=42)
users_f = users[users["gender"] == "f"]
users_m = users[users["gender"] == "m"]

n_f_other = int(0.10 * len(users_f))
n_m_other = int(0.10 * len(users_m))

f_other_idx = rng.choice(users_f.index, size=n_f_other, replace=False)
m_other_idx = rng.choice(users_m.index, size=n_m_other, replace=False)

users.loc[f_other_idx, "gender"] = "other"
users.loc[m_other_idx, "gender"] = "other"

# --- Load items ---
items = pd.read_csv(
    items_path,
    sep="\t",
    header=None,
    names=["uid", "artist_id", "item_id", "plays"]
)[["uid", "item_id", "plays"]]

# --- Filter users with >=50 unique artists ---
user_artist_counts = items.groupby("uid")["item_id"].nunique()
active_users = user_artist_counts[user_artist_counts >= 50].index

items = items[items["uid"].isin(active_users)]
users = users[users["uid"].isin(active_users)]

print("Unique users after user filter:", users['uid'].nunique())
print("Items shape after user filter:", items.shape)

# --- Filter artists listened to by >=20 users ---
min_user_per_item = 20
item_user_counts = items.groupby("item_id")["uid"].nunique()
popular_items = item_user_counts[item_user_counts >= min_user_per_item].index

items = items[items["item_id"].isin(popular_items)]


# Remove users with no remaining items
active_users_after_item_filter = items["uid"].unique()
users = users[users["uid"].isin(active_users_after_item_filter)]
items = items[items["uid"].isin(users["uid"])]

print("Unique users after item filter:", users['uid'].nunique())
print("Items shape after item filter:", items.shape)

# --- Log-transform plays ---
items["log_plays"] = np.log1p(items["plays"])

# --- Min-max normalize to [1,5] ---
min_log = items["log_plays"].min()
max_log = items["log_plays"].max()
items["rating"] = 1 + (items["log_plays"] - min_log) * 4 / (max_log - min_log)
items.drop(columns=["log_plays"], inplace=True)

# --- Binarize labels (~55% positives) ---
threshold = 2.5
print("Threshold for label=1:", threshold)

items["label"] = (items["rating"] >= threshold).astype(int)

print(items["label"].value_counts())

unique_uids = users["uid"].unique()
uid_map = {old: new for new, old in enumerate(unique_uids)}

users["uid"] = users["uid"].map(uid_map)
items["uid"] = items["uid"].map(uid_map)

print("Users uid range:", users["uid"].min(), "-", users["uid"].max())
print("Items uid range:", items["uid"].min(), "-", items["uid"].max())
print("Users consistent with items:", items["uid"].isin(users["uid"]).all())
gender_map = {'m': 0, 'f': 1, 'other': 2}
users["gender"] = users["gender"].map(gender_map)

users.to_csv('/Users/danie/Documents/projects/NewMPR/NewMPR/datasets/Lastfm-360K/Lastfm-360K-3C/sensitive_attribute.csv', index=False)
users.sample(frac=1, random_state=rng.integers(1e6)).reset_index(drop=True).to_csv('/Users/danie/Documents/projects/NewMPR/NewMPR/datasets/Lastfm-360K/Lastfm-360K-3C/sensitive_attribute_random.csv', index=False)

items = items.drop(columns=["plays", "rating"], errors='ignore')
# --- Map item_ids to consecutive integers ---
unique_items = items["item_id"].unique()
item_map = {old: new for new, old in enumerate(unique_items)}

items["item_id"] = items["item_id"].map(item_map)

# Optional: print range check
print("Items id range:", items["item_id"].min(), "-", items["item_id"].max())

# Split items per user
train_list = []
val_list = []
test_list = []

for uid, user_items in items.groupby("uid"):
    user_items = user_items.sample(frac=1, random_state=rng.integers(1e6))  # shuffle user's items
    n = len(user_items)
    
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val  # remaining goes to test
    
    train_list.append(user_items.iloc[:n_train])
    val_list.append(user_items.iloc[n_train:n_train+n_val])
    test_list.append(user_items.iloc[n_train+n_val:])

# Concatenate back
train = pd.concat(train_list).reset_index(drop=True)
val = pd.concat(val_list).reset_index(drop=True)
test = pd.concat(test_list).reset_index(drop=True)

# Check shapes
print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)

# Optional: check user coverage
print("Users in train:", train["uid"].nunique())
print("Users in val:", val["uid"].nunique())
print("Users in test:", test["uid"].nunique())

dfs = {"train": train, "val": val, "test": test}

for name, df in dfs.items():
    df.to_csv(data_dir / f"{name}.csv", index=False)