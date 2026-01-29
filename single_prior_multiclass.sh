#!/bin/bash

# The list of multiclass priors
priors=(0.1 0.105 0.11 0.12 
        0.125 0.13 0.14 0.15 
        0.17 0.18 0.2 0.22 
        0.25 0.29 0.33 0.4 
        0.5 0.67 1.0 1.5 2.0 
        2.5 3.0 3.5 4.0 4.5 
        5.0 5.5 6.0 6.5 7.0
        7.5 8.0 8.5 9.0 9.5 10.0)

# Experiment settings (customize as needed)
SEED=1
S_ATTR="gender"
S_RATIOS="0.5 0.3 0.2"
TASK_TYPE="ml-1m-synthetic"
UNFAIR_MODEL="./pretrained_models/ml-1m-synthetic/MF_orig_model"
FAIR_REG=1e-1
BETA=0.005
WEIGHT_DECAY=1e-7
LEARNING_RATE=1e-3

for PRIOR in "${priors[@]}"
do
  echo "Running for prior: $PRIOR"

  python3 mpr.py \
    --task_type "$TASK_TYPE" \
    --s_attr "$S_ATTR" \
    --unfair_model "$UNFAIR_MODEL" \
    --s_ratios $S_RATIOS \
    --prior "$PRIOR" \
    --seed "$SEED" \
    --fair_reg "$FAIR_REG" \
    --beta "$BETA" \
    --weight_decay "$WEIGHT_DECAY" \
    --learning_rate "$LEARNING_RATE" 
done