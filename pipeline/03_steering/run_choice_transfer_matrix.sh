#!/usr/bin/env bash
set -euo pipefail

# Run the configured choice-transfer matrix.
#
# Axes:
#   fp, task, advice, advice_daily
#
# Targets:
#   fp choice, task choice, advice-daily choice, advice-task choice
#
# Run this on a GPU node with the sae conda environment activated, e.g.:
#   cd /mnt/shared-storage-user/liuhaoze/llm_behavior_test/LLM-Behavioral-Personality
#   source /root/miniconda3/etc/profile.d/conda.sh
#   conda activate sae
#   CUDA_VISIBLE_DEVICES=0 bash pipeline/03_steering/run_choice_transfer_matrix.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-bfi2c_organization}"
SCRIPT="$SCRIPT_DIR/steer.py"
SUMMARY_SCRIPT="$REPO_ROOT/pipeline/04_results/summarize_choice_transfer_matrix.py"
COUNT="${COUNT:-50}"
LAYER="${LAYER:-11}"
COEFFS="${COEFFS:--3:3:1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-400}"
AXES="${AXES:-both}"
AXIS_NAMES="${AXIS_NAMES:-fp task advice advice_daily}"
TARGET_NAMES="${TARGET_NAMES:-fp_choice task_choice advice_daily_choice advice_task_choice}"

AXIS_SCENARIO_DIR="${AXIS_SCENARIO_DIR:-$(PYTHONPATH="$REPO_ROOT/pipeline" python - <<PY
from common_org import config_axis_scenario_dir, load_config
print(config_axis_scenario_dir(load_config("$CONFIG")))
PY
)}"
RESULT_DIR="${RESULT_DIR:-$(PYTHONPATH="$REPO_ROOT/pipeline" python - <<PY
from common_org import config_steering_dir, load_config
print(config_steering_dir(load_config("$CONFIG")))
PY
)}"

declare -A AXIS_SOURCE=(
  [fp]="$AXIS_SCENARIO_DIR/axis_fp.json"
  [task]="$AXIS_SCENARIO_DIR/axis_task.json"
  [advice]="$AXIS_SCENARIO_DIR/axis_advice.json"
  [advice_daily]="$AXIS_SCENARIO_DIR/axis_advice_daily.json"
)

mkdir -p "$RESULT_DIR"

range_label="$(PYTHONPATH="$REPO_ROOT" python - <<PY
from pipeline.common_org import coeff_label, parse_coeffs
print(coeff_label(parse_coeffs("$COEFFS")))
PY
)"

for axis_name in $AXIS_NAMES; do
  axis_file="${AXIS_SOURCE[$axis_name]}"
  if [[ ! -f "$axis_file" ]]; then
    echo "Missing axis file: $axis_file" >&2
    exit 1
  fi

  for target_name in $TARGET_NAMES; do
    out="$RESULT_DIR/${axis_name}_axes_to_${target_name}_l${LAYER}_${range_label}.json"
    echo "==> axis=${axis_name} target=${target_name}"
    python "$SCRIPT" \
      --config "$CONFIG" \
      --axis-source "$axis_file" \
      --axes "$AXES" \
      --target "$target_name" \
      --first-n "$COUNT" \
      --coeffs="$COEFFS" \
      --layer "$LAYER" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --out "$out"
    if [[ ! -f "$out" ]]; then
      echo "Expected output missing: $out" >&2
      exit 1
    fi
  done
done

python "$SUMMARY_SCRIPT" "$RESULT_DIR"/*_l${LAYER}_${range_label}.json \
  --out "$RESULT_DIR/choice_transfer_matrix_summary.json"

echo "Finished choice-transfer matrix."
