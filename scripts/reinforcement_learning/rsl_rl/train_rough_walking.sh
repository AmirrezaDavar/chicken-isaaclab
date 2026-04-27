#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../" && pwd)"

cd "${REPO_ROOT}"

if [[ -f "env_isaacsim/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "env_isaacsim/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-15000}"
SEED="${SEED:-42}"
TASK="${TASK:-Isaac-Velocity-Rough-ClassHumanoid-v0}"

ARGS=(
  --task "${TASK}"
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERATIONS}"
  --seed "${SEED}"
)

if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
  ARGS+=(--experiment_name "${EXPERIMENT_NAME}")
fi

if [[ -n "${RUN_NAME:-}" ]]; then
  ARGS+=(--run_name "${RUN_NAME}")
fi

exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py "${ARGS[@]}" "$@"
