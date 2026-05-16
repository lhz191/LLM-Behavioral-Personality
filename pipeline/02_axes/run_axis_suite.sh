#!/usr/bin/env bash
set -euo pipefail

# Run axis-quality diagnostics for configured behavior-axis scenario files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${CONFIG:-bfi2c_organization}"
LAYER="${LAYER:-11}"
AXIS_FILES="${AXIS_FILES:-axis_fp axis_task axis_advice axis_advice_daily}"
FIELDS="${FIELDS:-both}"
PCA_MODE="${PCA_MODE:-both}"
N_COMPONENTS="${N_COMPONENTS:-10}"
SPLIT_HALF_SAMPLES="${SPLIT_HALF_SAMPLES:-5000}"

AXIS_SCENARIO_DIR="${AXIS_SCENARIO_DIR:-$(PYTHONPATH="$REPO_ROOT/pipeline" python - <<PY
from common_org import config_axis_scenario_dir, load_config
print(config_axis_scenario_dir(load_config("$CONFIG")))
PY
)}"

for axis_name in $AXIS_FILES; do
  axis_stem="${axis_name%.json}"
  scenario_file="$AXIS_SCENARIO_DIR/${axis_stem}.json"
  if [[ ! -f "$scenario_file" ]]; then
    echo "Missing axis scenario file: $scenario_file" >&2
    exit 1
  fi

  echo "==> diagnostics: $scenario_file"
  python "$SCRIPT_DIR/axis_diagnostics.py" \
    --config "$CONFIG" \
    --scenario-file "$scenario_file" \
    --fields "$FIELDS" \
    --pca-mode "$PCA_MODE" \
    --layer "$LAYER" \
    --n-components "$N_COMPONENTS" \
    --split-half-samples "$SPLIT_HALF_SAMPLES"
done

echo "Finished axis diagnostics."
