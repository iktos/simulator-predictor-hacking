#!/bin/bash

chids='irak4_bayer irak4_merck irak4_pfizer'
ga_steps=100 
power=6
targets=$(seq 0 14)
seeds_ga=$(seq 0 14)
model_types='lr rf'

for seed_ga in $seeds_ga
do
for chid in $chids
do
for target in $targets
do
for model_type in $model_types
do
echo "Running $chid $target $seed"
python my_run_goal_directed.py --chid $chid --results_dir my_res_dir --optimizer graph_ga --model_type $model_type --use_train_cs 1 --target_names 'target_2targs_power'$power'_seed'$target'_targid0' 'target_2targs_power'$power'_seed'$target'_targid1' --seed $seed_ga --steps $ga_steps
done
done
done
done