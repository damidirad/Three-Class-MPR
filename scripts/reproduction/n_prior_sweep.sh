#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate

gpu_id=0
task="Lastfm-360K"
epoch=200
learning_rate=1e-3
weight_decay=1e-7
fair_reg=1e-1
gender_train_epoch=1000
partial_ratio_male=0.5
partial_ratio_female="0.4"
num_priors_list=(27 29 31 33 35 37)
seeds=(2 3)
out_root="./deliverables/fig7_numpriors"
pred_sst_dir="./deliverables/Lastfm-360K/generated_csv"

echo "Starting MPR Prior Count Sweep Experiments (Figure 7 style).."
echo "Output root: ${out_root}"
echo "Task: $task"
echo "Partial ratio male: $partial_ratio_male"
echo "Partial ratio female: $partial_ratio_female"
echo "Num priors: ${num_priors_list[*]}"
echo "Seeds: ${seeds[*]}"

for num_priors in "${num_priors_list[@]}"; do
  echo "Testing with num_priors: $num_priors"
  for seed in "${seeds[@]}"; do
    out_dir="${out_root}/${task}/male${partial_ratio_male}_female${partial_ratio_female}/priors${num_priors}/seed${seed}"
    mkdir -p "${out_dir}"
    echo "      >> [task=${task}] [male=${partial_ratio_male}] [female=${partial_ratio_female}] [seed=${seed}] [num_priors=${num_priors}]"
    echo "         Log file: ${out_dir}/train.log"
    python -u mpr.py \
      --task_type "${task}" \
      --s_attr gender \
      --unfair_model "./pretrained_models/${task}/MF_orig_model" \
      --s_ratios "${partial_ratio_male}" "${partial_ratio_female}" \
      --seed "${seed}" \
      --fair_reg "${fair_reg}" \
      --beta "0.005" \
      --weight_decay "${weight_decay}" \
      --learning_rate "${learning_rate}" \
      --prior_count "${num_priors}" \
      > "${out_dir}/train.log" 2>&1
  done
done

echo "All prior count sweep experiments completed."