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
tasks=("Lastfm-360K" "ml-1m")
partial_females=("0.1" "0.2" "0.3")
seeds=("1" "2" "3")
priors_Lastfm_360K=("0.1" "0.2" "0.5" "1.0" "2.0" "3.5" "5.0" "7.5" "10.0")
priors_ml_1m=("0.1" "0.2" "0.5" "1.0" "2.0" "2.5" "5.0" "7.5" "10.0")
out_root="./deliverables/fig5_single_prior"

echo "Starting MPR Single Prior Sweep Experiments (Figure 5 style).."
echo "Output root: ${out_root}"
echo "Tasks: ${tasks[*]}"
echo "Partial ratio females: ${partial_females[*]}"
echo "Seeds: ${seeds[*]}"
echo "Lastfm priors: ${priors_Lastfm_360K[*]}"
echo "ml-1m priors: ${priors_ml_1m[*]}"

for task in "${tasks[@]}"; do
  [[ "$task" == "Lastfm-360K" ]] && priors=("${priors_Lastfm_360K[@]}") || priors=("${priors_ml_1m[@]}")
  echo "Running for task: $task.."
  for partial_female in "${partial_females[@]}"; do
    echo "  ..Partial ratio (female): $partial_female"
    for seed in "${seeds[@]}"; do
      echo "    ..Seed: $seed"
      for prior in "${priors[@]}"; do
        out_dir="${out_root}/${task}/male${partial_ratio_male}_female${partial_female}/prior${prior}/seed${seed}"
        mkdir -p "${out_dir}"
        echo "      >> [task=${task}] [male=${partial_ratio_male}] [female=${partial_female}] [seed=${seed}] [prior=${prior}]"
        echo "         Log file: ${out_dir}/train.log"
        python -u mpr.py \
          --task_type "${task}" \
          --s_attr gender \
          --unfair_model "./pretrained_models/${task}/MF_orig_model" \
          --s_ratios "${partial_ratio_male}" "${partial_female}" \
          --prior "${prior}" \
          --seed "${seed}" \
          --fair_reg "${fair_reg}" \
          --beta "0.005" \
          --weight_decay "${weight_decay}" \
          --learning_rate "${learning_rate}" \
          --prior_count 1 \
          > "${out_dir}/train.log" 2>&1
      done
    done
  done
done

echo "All single prior sweep experiments completed."