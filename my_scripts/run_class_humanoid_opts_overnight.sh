#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # pip/.venv flow
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f "env_isaaclab/bin/activate" ]]; then
  # fallback for existing env name
  # shellcheck disable=SC1091
  source env_isaaclab/bin/activate
else
  echo "[ERROR] Python env not found (.venv or env_isaaclab)."
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
SEED="${SEED:-42}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-class_humanoid_opts}"
RUN_PREFIX="${RUN_PREFIX:-overnight_$(date +%Y%m%d_%H%M%S)}"

declare -a RUN_SPECS=(
  "opt1_contact Isaac-Velocity-Rough-ClassHumanoid-ContactPenalty-v0"
  "opt2_bad_orientation Isaac-Velocity-Rough-ClassHumanoid-BadOrientation-v0"
  "opt3_extended_base_contact Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0"
)

LOG_DIR="logs/rsl_rl/${EXPERIMENT_NAME}/batch_${RUN_PREFIX}"
mkdir -p "${LOG_DIR}"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] NUM_ENVS=${NUM_ENVS}, MAX_ITERATIONS=${MAX_ITERATIONS}, SEED=${SEED}"
echo "[INFO] EXPERIMENT_NAME=${EXPERIMENT_NAME}, RUN_PREFIX=${RUN_PREFIX}"
echo "[INFO] Per-run logs: ${LOG_DIR}"

fail_count=0

for spec in "${RUN_SPECS[@]}"; do
  run_suffix="${spec%% *}"
  task_id="${spec#* }"
  run_name="${RUN_PREFIX}_${run_suffix}"
  log_file="${LOG_DIR}/${run_name}.log"

  echo ""
  echo "============================================================"
  echo "[START] run_name=${run_name}"
  echo "[START] task=${task_id}"
  echo "[START] log=${log_file}"
  echo "============================================================"

  start_ts="$(date +%s)"

  if ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --headless \
      --num_envs "${NUM_ENVS}" \
      --max_iterations "${MAX_ITERATIONS}" \
      --seed "${SEED}" \
      --experiment_name "${EXPERIMENT_NAME}" \
      --run_name "${run_name}" \
      --task "${task_id}" 2>&1 | tee "${log_file}"; then
    end_ts="$(date +%s)"
    echo "[DONE] ${run_name} finished in $((end_ts - start_ts)) sec."
  else
    end_ts="$(date +%s)"
    echo "[FAIL] ${run_name} failed after $((end_ts - start_ts)) sec."
    fail_count=$((fail_count + 1))
  fi
done

echo ""
echo "============================================================"
if [[ "${fail_count}" -eq 0 ]]; then
  echo "[ALL DONE] 3/3 runs completed successfully."
  exit 0
else
  echo "[PARTIAL] ${fail_count} run(s) failed. Check logs in ${LOG_DIR}."
  exit 1
fi
