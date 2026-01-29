#!/usr/bin/env bash
set -euo pipefail

TASK_TYPE="ml-1m"
UNFAIR_MODEL="./pretrained_model/ml-1m/MF_orig_model"
S_ATTR="gender"
SAVING_PATH="./scripts/predict_sst_diff_seed_batch/ml-1m/"
SST_EPOCHS=1000
BATCH_SIZE=128

SEEDS=(1 2 3)
FEMALE_RATIOS=(0.1 0.2 0.3 0.4)
MALE_RATIO=0.5

# 37 prior indices (0 to 36)
for PRIOR_IDX in {0..36}; do
  for SEED in "${SEEDS[@]}"; do
    for FEMALE_RATIO in "${FEMALE_RATIOS[@]}"; do
      echo "============================================================"
      echo "Running: prior_idx=${PRIOR_IDX} seed=${SEED} s_ratios=[${MALE_RATIO}, ${FEMALE_RATIO}]"
      echo "============================================================"

      python predict_sensitive_labels.py \
        --task_type "${TASK_TYPE}" \
        --s_attr "${S_ATTR}" \
        --unfair_model "${UNFAIR_MODEL}" \
        --seed "${SEED}" \
        --s_ratios "${MALE_RATIO}" "${FEMALE_RATIO}" \
        --prior_resample_idx "${PRIOR_IDX}" \
        --sst_train_epochs "${SST_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --saving_path "${SAVING_PATH}"
    done
  done
done

echo "✅ Done. Total: 444 CSVs (37 priors × 3 seeds × 4 disclosure ratios)"