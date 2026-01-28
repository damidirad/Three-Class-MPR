#!/usr/bin/env bash
set -euo pipefail

TASK_TYPE="ml-1m"
UNFAIR_MODEL="./pretrained_model/ml-1m/MF_orig_model"
S_ATTR="gender"

SEEDS=(1 2 3)
FEMALE_RATIOS=(0.1 0.2 0.3 0.4)
MALE_RATIO=0.5

for SEED in "${SEEDS[@]}"; do
  for FEMALE_RATIO in "${FEMALE_RATIOS[@]}"; do
    echo "============================================================"
    echo "MPR Training: seed=${SEED} s_ratios=[${MALE_RATIO}, ${FEMALE_RATIO}]"
    echo "============================================================"

    python MPR.py \
      --task_type "${TASK_TYPE}" \
      --s_attr "${S_ATTR}" \
      --unfair_model "${UNFAIR_MODEL}" \
      --seed "${SEED}" \
      --s_ratios "${MALE_RATIO}" "${FEMALE_RATIO}" \
      --fair_reg 12.0 \
      --beta 0.0005
  done
done

echo "✅ Done. Trained 12 models (3 seeds × 4 disclosure ratios)"