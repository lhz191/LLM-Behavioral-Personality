#!/usr/bin/env bash
set -euo pipefail

# Steer individual SAE features selected from thought/response shared-feature summaries.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${CONFIG:-bfi2c_organization}"
LAYER="${LAYER:-11}"
THRESHOLD_FRAC="${THRESHOLD_FRAC:-0.5}"
COEFFS="${COEFFS:--3:3:1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"
FIRST_N="${FIRST_N:-40}"
REGISTERS="${REGISTERS:-fp task advice advice_daily}"
EXCLUDE_FEATURES="${EXCLUDE_FEATURES:-123331}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SAE_RESULTS_DIR="${SAE_RESULTS_DIR:-$(PYTHONPATH="$REPO_ROOT/pipeline" python - <<PY
from common_org import config_sae_dir, load_config
print(config_sae_dir(load_config("$CONFIG")))
PY
)}"
TEST_SCENARIO_DIR="${TEST_SCENARIO_DIR:-$(PYTHONPATH="$REPO_ROOT/pipeline" python - <<PY
from common_org import config_test_scenario_dir, load_config
print(config_test_scenario_dir(load_config("$CONFIG")))
PY
)}"
STEER_RESULT_DIR="${STEER_RESULT_DIR:-$REPO_ROOT/data/$CONFIG/results/steer}"
LOG_DIR="${LOG_DIR:-$STEER_RESULT_DIR/logs}"
RANGE_LABEL="$(PYTHONPATH="$REPO_ROOT/pipeline" python - <<PY
from common_org import coeff_label, parse_coeffs
print(coeff_label(parse_coeffs("$COEFFS")))
PY
)"

IFS=',' read -r -a DEVICE_IDS <<< "$CUDA_DEVICES"
for i in "${!DEVICE_IDS[@]}"; do
  DEVICE_IDS[$i]="$(echo "${DEVICE_IDS[$i]}" | xargs)"
done
if [[ "${#DEVICE_IDS[@]}" -eq 0 || -z "${DEVICE_IDS[0]}" ]]; then
  echo "CUDA_DEVICES must contain at least one device id" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
declare -a DEVICE_PIDS
declare -a DEVICE_LABELS
FAILED=0

find_free_device() {
  while true; do
    for idx in "${!DEVICE_IDS[@]}"; do
      local pid="${DEVICE_PIDS[$idx]:-}"
      if [[ -z "$pid" ]]; then
        echo "$idx"
        return
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        if ! wait "$pid"; then
          echo "Job failed: ${DEVICE_LABELS[$idx]}" >&2
          FAILED=1
        fi
        DEVICE_PIDS[$idx]=""
        DEVICE_LABELS[$idx]=""
        echo "$idx"
        return
      fi
    done
    sleep 5
  done
}

wait_for_all_jobs() {
  for idx in "${!DEVICE_IDS[@]}"; do
    local pid="${DEVICE_PIDS[$idx]:-}"
    if [[ -n "$pid" ]]; then
      if ! wait "$pid"; then
        echo "Job failed: ${DEVICE_LABELS[$idx]}" >&2
        FAILED=1
      fi
      DEVICE_PIDS[$idx]=""
      DEVICE_LABELS[$idx]=""
    fi
  done
}

feature_ids_for_axis() {
  local features_file="$1"
  local axis_name="$2"
  PYTHONPATH="$REPO_ROOT/pipeline" python - "$features_file" "$axis_name" "$THRESHOLD_FRAC" "$EXCLUDE_FEATURES" <<'PY'
import json
import sys

features_file, axis_name, threshold_frac, exclude = sys.argv[1:5]
excluded = {int(x) for x in exclude.replace(",", " ").split() if x.strip()}
payload = json.load(open(features_file, encoding="utf-8"))
rows = payload["axes"][axis_name]["shared_features"]
selected = []
for row in rows:
    fid = int(row["feature"])
    if fid in excluded:
        continue
    if row["freq"] / row["n"] >= float(threshold_frac):
        selected.append(str(fid))
print(" ".join(selected))
PY
}

prompt_file_for_register() {
  case "$1" in
    fp) echo "$TEST_SCENARIO_DIR/choice_fp.json" ;;
    task) echo "$TEST_SCENARIO_DIR/choice_task.json" ;;
    advice) echo "$TEST_SCENARIO_DIR/choice_advice_task.json" ;;
    advice_daily) echo "$TEST_SCENARIO_DIR/choice_advice_daily.json" ;;
    *) echo "Unknown register: $1" >&2; exit 1 ;;
  esac
}

system_for_register() {
  case "$1" in
    fp) echo "fp" ;;
    task|advice|advice_daily) echo "task" ;;
    *) echo "Unknown register: $1" >&2; exit 1 ;;
  esac
}

for register in $REGISTERS; do
  features_file="$SAE_RESULTS_DIR/${CONFIG}_axis_${register}_sae_axis_features_layer${LAYER}.json"
  prompt_file="$(prompt_file_for_register "$register")"
  system_name="$(system_for_register "$register")"
  if [[ ! -f "$features_file" ]]; then
    echo "Missing SAE features file: $features_file" >&2
    exit 1
  fi
  if [[ ! -f "$prompt_file" ]]; then
    echo "Missing prompt file: $prompt_file" >&2
    exit 1
  fi

  mkdir -p "$STEER_RESULT_DIR/$register"
  for axis_name in thought response; do
    ids="$(feature_ids_for_axis "$features_file" "$axis_name")"
    if [[ -z "$ids" ]]; then
      echo "No $register $axis_name features selected at threshold $THRESHOLD_FRAC"
      continue
    fi
    for fid in $ids; do
      out="$STEER_RESULT_DIR/$register/${axis_name}_feature_${fid}_l${LAYER}_${RANGE_LABEL}.json"
      log="$LOG_DIR/${register}_${axis_name}_feature_${fid}_l${LAYER}_${RANGE_LABEL}.log"
      if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
        echo "skip existing: $out"
        continue
      fi

      device_idx="$(find_free_device)"
      device="${DEVICE_IDS[$device_idx]}"
      label="register=$register axis=$axis_name feature=$fid device=$device"
      echo "==> $label"
      (
        CUDA_VISIBLE_DEVICES="$device" python -u "$SCRIPT_DIR/sae_steer_single_features.py" \
          --config "$CONFIG" \
          --prompt-file "$prompt_file" \
          --features-file "$features_file" \
          --feature-ids "$fid" \
          --system "$system_name" \
          --first-n "$FIRST_N" \
          --layer "$LAYER" \
          --coeffs="$COEFFS" \
          --max-new-tokens "$MAX_NEW_TOKENS" \
          --out "$out"
      ) >"$log" 2>&1 &
      DEVICE_PIDS[$device_idx]="$!"
      DEVICE_LABELS[$device_idx]="$label"
    done
  done
done

wait_for_all_jobs
if [[ "$FAILED" != "0" ]]; then
  echo "One or more single-feature steering jobs failed. Check logs in $LOG_DIR" >&2
  exit 1
fi

echo "Finished single-feature SAE steering suite."
