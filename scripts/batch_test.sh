#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------
# Static configuration
# ------------------------------------------------------------------------
CHECKPOINT_DIR_BASE="/home/danny/Documents/safe_locomotion/runs/cvar_ppo"
RESULT_DIR_BASE="/home/danny/Documents/safe_locomotion/results"
SEEDS=(42 44 46 83)          # <‑‑‑ add / remove seeds here
MAX_JOBS=4                     # max concurrent python jobs

ENV_IDS=(
  "Safe-Locomotion-Base-v0"
  "Safe-Locomotion-Incline-v0"
  "Safe-Locomotion-Incline-v1"
  "Safe-Locomotion-Incline-v2"
  "Safe-Locomotion-Rough-v0"
  "Safe-Locomotion-Gain-v0"
  "Safe-Locomotion-Jitter-v0"
  "Safe-Locomotion-Delay-v0"
  "Safe-Locomotion-Delay-v1"
  "Safe-Locomotion-Delay-v2"
  "Safe-Locomotion-Command-v0"
  "Safe-Locomotion-Command-v1"
  "Safe-Locomotion-Brownian-v0"
  "Safe-Locomotion-Brownian-v1"
  "Safe-Locomotion-Push-v0"
  "Safe-Locomotion-Push-v1"
  "Safe-Locomotion-Push-v2"
  "Safe-Locomotion-Friction-v0"
  "Safe-Locomotion-Friction-v1"
  "Safe-Locomotion-Friction-v2"
  "Safe-Locomotion-Restitution-v0"
  "Safe-Locomotion-Restitution-v1"
)

# ------------------------------------------------------------------------
# One‑time setup
# ------------------------------------------------------------------------
cd /home/danny/Documents/safe_locomotion
export PYTHONPATH="$PWD:$PYTHONPATH"

# ------------------------------------------------------------------------
# Main loops
# ------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  CHECKPOINT_DIR="${CHECKPOINT_DIR_BASE}/${SEED}"
  RESULT_DIR="${RESULT_DIR_BASE}/${SEED}"
  mkdir -p "$RESULT_DIR"                       # ensure result folder exists

  # Discover experiments in this seed’s checkpoint folder
  mapfile -t EXPS < <(
    find "$CHECKPOINT_DIR" -maxdepth 1 -mindepth 1 -type d \
      \( -name "*warmup*" -o -name "*ppo*" \) \
      -printf "%f\n"
  )

  # Skip this seed if no experiments found
  [[ ${#EXPS[@]} -eq 0 ]] && { echo "No experiments in $CHECKPOINT_DIR – skipping"; continue; }

  for ENV in "${ENV_IDS[@]}"; do
    for EXP in "${EXPS[@]}"; do
      echo "Launching: seed=$SEED  env=$ENV  exp=$EXP"

      python scripts/test/test_policy.py \
        --env_id         "$ENV" \
        --exp            "$EXP" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --result_dir     "$RESULT_DIR" \
        --agent_name     "best_agent.pt" \
        --seed           "$SEED" &

      # Throttle to at most $MAX_JOBS background tasks
      while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
      done
    done
  done
done

wait
echo "All done!"
