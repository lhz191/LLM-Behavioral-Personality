#!/usr/bin/env bash
set -euo pipefail

# Find thought/response SAE features for each maintained axis scenario file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${CONFIG:-bfi2c_organization}"
LAYER="${LAYER:-11}"
THRESHOLD="${THRESHOLD:-0.5}"
AXIS_FILES="${AXIS_FILES:-axis_fp axis_task axis_advice axis_advice_daily}"

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

  echo "==> sae compare: $scenario_file"
  python "$SCRIPT_DIR/sae_compare_axes.py" \
    --config "$CONFIG" \
    --scenario-file "$scenario_file" \
    --layer "$LAYER" \
    --threshold "$THRESHOLD"
done

echo "Finished SAE axis comparison suite."
