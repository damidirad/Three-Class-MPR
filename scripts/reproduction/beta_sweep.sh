#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate

gpu_id=0
epoch=200
learning_rate=1e-3
weight_decay=1e-7
fair_reg=1e-1
gender_train_epoch=1000
partial_ratio_male=0.5
partial_ratio_females=(0.1 0.2 0.3)
betas=("0.00316227766" "0.01" "0.0316227766" "0.1" "0.316227766" "1" "3.16227766" "10")
tasks=("Lastfm-360K" "ml-1m")
out_root="./deliverables/fig6_beta"

echo "Starting MPR Beta Sweep Experiment.."
echo "Output root: ${out_root}"
echo "Tasks: ${tasks[*]}"
echo "Partial ratio females: ${partial_ratio_females[*]}"
echo "Betas: ${betas[*]}"

for task in "${tasks[@]}"; do
  case "$task" in
    Lastfm-360K) pred_sst_dir="./deliverables/Lastfm-360K/generated_csv";;
    ml-1m)       pred_sst_dir="./deliverables/ml-1m/generated_csv";;
    *) echo "Unknown task: $task"; exit 1;;
  esac

  echo "Running for task: $task.."
  for partial_female in "${partial_ratio_females[@]}"; do
    echo "  ..Partial ratio (female): $partial_female"
    for seed in 1 2 3; do
      echo "    ..Seed: $seed"
      for beta in "${betas[@]}"; do
        out_dir="${out_root}/${task}/male${partial_ratio_male}_female${partial_female}/beta${beta}/seed${seed}"
        mkdir -p "${out_dir}"
        echo "      >> [task=${task}] [male=${partial_ratio_male}] [female=${partial_female}] [seed=${seed}] [beta=${beta}]"
        echo "         Log file: ${out_dir}/train.log"
        python -u mpr.py \
          --task_type "${task}" \
          --s_attr gender \
          --unfair_model "./pretrained_models/${task}/MF_orig_model" \
          --s_ratios "${partial_ratio_male}" "${partial_female}" \
          --seed "${seed}" \
          --fair_reg "${fair_reg}" \
          --beta "${beta}" \
          --weight_decay "${weight_decay}" \
          --learning_rate "${learning_rate}" \
          --prior_count 37 \
          > "${out_dir}/train.log" 2>&1
      done
    done
  done
done

echo "All experiments completed."