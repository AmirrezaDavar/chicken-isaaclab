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

NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1500}"

ARGS=(
  --task Isaac-Step-ClassHumanoid-v0
  --num_envs "${NUM_ENVS}"
  --video
  --video_length "${VIDEO_LENGTH}"
  --headless
)

if [[ -n "${CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
elif [[ -n "${LOAD_RUN:-}" ]]; then
  ARGS+=(--load_run "${LOAD_RUN}")
fi

exec ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py "${ARGS[@]}" "$@"
