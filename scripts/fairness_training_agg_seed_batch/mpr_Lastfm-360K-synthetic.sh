#!/usr/bin/env bash
set -euo pipefail

TASK_TYPE="Lastfm-360K-synthetic"
UNFAIR_MODEL="./pretrained_models/Lastfm-360K-synthetic/MF_orig_model"
S_ATTR="gender"

SEEDS=(1 2 3)
NON_BINARY_RATIOS=(0.1 0.2 0.3 0.4)
FEMALE_RATIO=0.5
MALE_RATIO=0.5

for SEED in "${SEEDS[@]}"; do
  for NON_BINARY_RATIO in "${NON_BINARY_RATIOS[@]}"; do
    RATIO_STR="${MALE_RATIO}_${FEMALE_RATIO}_${NON_BINARY_RATIO}"
    MODEL_PATH="./deliverables/${TASK_TYPE}/MPR_model_checkpoint/MPR_model_ratios_${RATIO_STR}_seed_${SEED}.pt"
    
    if [ -f "${MODEL_PATH}" ]; then
      echo "⏭️  Skipping: seed=${SEED} s_ratios=[${MALE_RATIO}, ${FEMALE_RATIO}, ${NON_BINARY_RATIO}] (checkpoint exists)"
      continue
    fi

    echo "============================================================"
    echo "MPR Training: seed=${SEED} s_ratios=[${MALE_RATIO}, ${FEMALE_RATIO}, ${NON_BINARY_RATIO}]"
    echo "============================================================"

    python MPR.py \
      --task_type "${TASK_TYPE}" \
      --s_attr "${S_ATTR}" \
      --unfair_model "${UNFAIR_MODEL}" \
      --seed "${SEED}" \
      --s_ratios "${MALE_RATIO}" "${FEMALE_RATIO}" "${NON_BINARY_RATIO}" \
      --fair_reg 1.0 \
      --beta 0.005 \
      --weight_decay 1e-5
  done
done

echo "✅ Done. Trained 12 models (3 seeds × 4 non-binary disclosure ratios)"
echo "Each model uses 111 prediction variants (37 priors × 3 SST seeds)"