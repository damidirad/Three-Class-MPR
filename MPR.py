from config import Config
from SST import SST, train_sst
from MF import MF
import argparse
from helpers import *

parser = argparse.ArgumentParser()

parser.add_argument("--task_type",type=str,default="Lastfm-360K",help="Specify task type: ml-1m/tenrec/lastfm-1K/lastfm-360K")
parser.add_argument("--s_attr",type=str,default="gender",help="Specify sensitive attribute name.")
parser.add_argument("--unfair_model", type=str, default= "./pretrained_model/Lastfm-360K/MF_orig_model")
parser.add_argument("--s0_ratio", type=float, default= 0.5, help= "Known ratio for training sensitive atrribute s0.")
parser.add_argument("--s1_ratio", type=float, default= 0.1, help= "Known ratio for training sensitive attribute s1.")
parser.add_argument("--seed", type=int, default= 1, help= "Seed for reproducibility.")

args = parser.parse_args()

config = Config(
    task_type = args.task_type,
    s_attr = args.s_attr,
    unfair_model = args.unfair_model,
    s0_ratio = args.s0_ratio,
    s1_ratio = args.s1_ratio,
    seed = args.seed
)

if __name__ == "__main__":
    device = get_device()
    paths = setup_paths(args)
    data = load_data(paths) # train, valid, test, true_sensitive, known_sensitive

    disclosed_s0 = get_partial_group_ids(data["true_sensitive"], config.s_attr, 0, config.s0_ratio)
    disclosed_s1 = get_partial_group_ids(data["true_sensitive"], config.s_attr, 1, config.s1_ratio)

    known_user_ids = torch.cat([
        torch.LongTensor(disclosed_s0),
        torch.LongTensor(disclosed_s1)
    ]).to(device)

    known_labels = torch.cat([
        torch.zeros(len(disclosed_s0), dtype=torch.long),
        torch.ones(len(disclosed_s1), dtype=torch.long)
    ]).to(device)

    num_users = max(data["train"].user_id) + 1
    num_items = max(data["train"].item_id) + 1

    sst_model = SST(embedding_dim=config.emb_size, hidden_dim=config.sst_hidden_sizes[0]).to(device)
    mf_model = MF(np.int64(num_users), np.int64(num_items), config.emb_size).to(device)

    if Path(config.unfair_model).exists():
        print(f"Loading pre-trained MF weights from {config.unfair_model}")
        mf_model.load_state_dict(torch.load(config.unfair_model, map_location=device))
    
    rmse_thresh = set_rmse_thresh(config)

    train_sst(sst_model=sst_model, mf_model=mf_model, known_user_ids=known_user_ids, known_labels=known_labels)
