#!/usr/bin/env bash
# run_cppo_parallel.sh
# Runs CPPO training with different CVaR alpha values in parallel (max 3 concurrent jobs)

# Base directory of the safe_locomotion project
BASE_DIR="$HOME/Documents/safe_locomotion"

# Change to project directory
cd "$BASE_DIR" || { echo "Failed to change directory to $BASE_DIR"; exit 1; }

# List of alpha values to sweep
# alphas=(0.05 0.1 0.25 0.5 0.75)
alphas=(0.05)
seeds=(0 8 9 15 18 42 44 46 83)

# Maximum number of concurrent jobs
max_jobs=5

for seed in "${seeds[@]}"; do
  for alpha in "${alphas[@]}"; do
    echo "Starting run with cvar_alpha=$alpha"
    PYTHONPATH="$PWD:$PYTHONPATH" \
      python scripts/train/train_cvar_ppo.py \
        agent.cvar_alpha="$alpha" \
        seed="$seed" \
        agent.lam_max=0 &

    # If number of running jobs reaches max, wait for any to finish
    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
      sleep 1
    done
  done
done

# Wait for all background jobs to complete
wait
echo "All runs completed."
