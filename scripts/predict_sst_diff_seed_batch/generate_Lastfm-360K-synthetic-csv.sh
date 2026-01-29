#!/usr/bin/env bash
set -euo pipefail

TASK_TYPE="Lastfm-360K-synthetic"
UNFAIR_MODEL="./pretrained_models/Lastfm-360K-synthetic/MF_orig_model"
S_ATTR="gender"
SAVING_PATH="./deliverables/Lastfm-360K-synthetic/"
SST_EPOCHS=1000
BATCH_SIZE=128

SEEDS=(1 2 3)
NON_BINARY_RATIOS=(0.1 0.2 0.3 0.4)
FEMALE_RATIO=0.5
MALE_RATIO=0.5

for PRIOR_IDX in {0..36}; do
  for SEED in "${SEEDS[@]}"; do
    for NON_BINARY_RATIO in "${NON_BINARY_RATIOS[@]}"; do
      RATIO_STR="${MALE_RATIO}_${FEMALE_RATIO}_${NON_BINARY_RATIO}"
      OUTPUT_DIR="${SAVING_PATH}generated_csv/${TASK_TYPE}_ratios_${RATIO_STR}_epochs_${SST_EPOCHS}_prior_${PRIOR_IDX}"
      OUTPUT_FILE="${OUTPUT_DIR}/seed_${SEED}.csv"
      
      if [ -f "${OUTPUT_FILE}" ]; then
        echo "⏭️  Skipping: prior_idx=${PRIOR_IDX} seed=${SEED} s_ratios=[${MALE_RATIO}, ${FEMALE_RATIO}, ${NON_BINARY_RATIO}] (already exists)"
        continue
      fi
      
      echo "============================================================"
      echo "Running: prior_idx=${PRIOR_IDX} seed=${SEED} s_ratios=[${MALE_RATIO}, ${FEMALE_RATIO}, ${NON_BINARY_RATIO}]"
      echo "============================================================"

      python predict_sensitive_labels.py \
        --task_type "${TASK_TYPE}" \
        --s_attr "${S_ATTR}" \
        --unfair_model "${UNFAIR_MODEL}" \
        --seed "${SEED}" \
        --s_ratios "${MALE_RATIO}" "${FEMALE_RATIO}" "${NON_BINARY_RATIO}" \
        --prior_resample_idx "${PRIOR_IDX}" \
        --sst_train_epochs "${SST_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --saving_path "${SAVING_PATH}"
    done
  done
done

echo "✅ Done. Total: 444 CSVs (37 priors × 3 seeds × 4 disclosure ratios)"