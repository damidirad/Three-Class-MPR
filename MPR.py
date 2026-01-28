from config import Config
from SST import SST #, train_sst
from MF import MF
import argparse
from helpers import *

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
parser.add_argument("--fair_reg", type=float, default=12.0, help= "Fairness regularization coefficient.")
parser.add_argument("--beta", type=float, default=0.0005, help= "Regularization constraint.")

args = parser.parse_args()

config = Config(
    task_type = args.task_type,
    s_attr = args.s_attr,
    unfair_model = args.unfair_model,
    s_ratios = args.s_ratios,
    seed = args.seed,
    fair_reg = args.fair_reg,
    beta = args.beta,
)

if __name__ == "__main__":
    device = get_device()
    paths = setup_paths(config)
    data = load_data(paths) # train, valid, test, true_sensitive, known_sensitive

    set_random_seed(config.seed)

    known_user_ids = []
    known_labels = []

    known_user_ids = []
    known_labels = []

    for class_idx, ratio in enumerate(config.s_ratios):
        disclosed = get_partial_group_ids(
            data["true_sensitive"], config.s_attr, class_idx, ratio, seed=config.seed
        )
        if len(disclosed) == 0:
            continue

        known_user_ids.append(torch.LongTensor(disclosed))
        known_labels.append(torch.full((len(disclosed),), class_idx, dtype=torch.long))

    known_user_ids = torch.cat(known_user_ids).to(device)
    known_labels = torch.cat(known_labels).to(device)

    num_users = max(data["train"].user_id) + 1
    num_items = max(data["train"].item_id) + 1

    sst_model = SST(config).to(device)
    mf_model = MF(np.int64(num_users), np.int64(num_items), config.emb_size).to(device)

    if Path(config.unfair_model).exists():
        print(f"Loading pre-trained MF weights from {config.unfair_model}")
        mf_model.load_state_dict(torch.load(config.unfair_model, map_location=device))
    
    rmse_thresh = set_rmse_thresh(config)
    resample_range = set_resample_range(config)

    # WRITE FUNCTION TO GET PREDICTED SENSITIVE ATTRIBUTES FOR GIVEN RESAMPLE RATIO AND SEED SAMPLE
    #######################################################
    # Load predicted sensitive attributes from CSVs
    predicted_sensitive_attr_dict = {}
    ratio_str = "_".join([f"{r}" for r in config.s_ratios])
    main_dir = f"./predict_sst_diff_seed_batch/{config.task_type}/{config.task_type}_ratios_{ratio_str}_epochs_1000"

    for prior_idx, prior_ratio in enumerate(resample_range):
        predicted_sensitive_attr_dict[prior_ratio] = {}
        for seed in [1, 2, 3]:
            csv_path = f"{main_dir}_prior_{prior_idx}/seed_{seed}.csv"
            if Path(csv_path).exists():
                predicted_sensitive_attr_dict[prior_ratio][seed] = pd.read_csv(csv_path)
            else:
                print(f"Warning: Missing {csv_path}")

    print(f"Loaded predictions for {len(predicted_sensitive_attr_dict)} priors × {len([1,2,3])} seeds")
    print(f"\nLoaded {len(predicted_sensitive_attr_dict)} priors × 3 seeds")

    # Build disclosed_ids_dict from known user IDs
    disclosed_ids_dict = {}
    for class_idx, ratio in enumerate(config.s_ratios):
        disclosed = get_partial_group_ids(
            data["true_sensitive"], config.s_attr, class_idx, ratio, seed=config.seed
        )
        disclosed_ids_dict[class_idx] = disclosed

    # Call fairness training with loaded predictions
    from MPR_fairness_training import fairness_training

    best_val_rmse, best_test_rmse, best_val_gap, best_test_gap, best_epoch, best_model = fairness_training(
        model=mf_model,
        df_train=data["train"],
        valid_data=data["valid"],
        test_data=data["test"],
        sensitive_attr=data["true_sensitive"],
        disclosed_ids_dict=disclosed_ids_dict,
        predicted_sensitive_attr_dict=predicted_sensitive_attr_dict,
        config=config,
        device=device
    )

    print(f"\nBest Results:")
    print(f"Val RMSE: {best_val_rmse:.4f}, Test RMSE: {best_test_rmse:.4f}")
    print(f"Val Gap: {best_val_gap:.4f}, Test Gap: {best_test_gap:.4f}")
    print(f"Best Epoch: {best_epoch}")
    #######################################################
    #train_sst(sst_model=sst_model, mf_model=mf_model, known_user_ids=known_user_ids, known_labels=known_labels)