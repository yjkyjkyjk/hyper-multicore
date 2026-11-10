#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 REPO PYTHON DEVICES NEW_OUTPUT PORT_BASE" >&2
    exit 2
fi
repo=$(cd "$1" && pwd)
python=$2
devices=$3
mkdir -p "$4"
output=$(cd "$4" && pwd)
port=$5
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -e "$output/confirmation" || -e "$output/monitored_profile" ]]; then
    echo "Refusing to overwrite an existing confirmation or profile directory." >&2
    exit 2
fi

export HP_EXPERIMENT_WARMUP=20 HP_EXPERIMENT_MEASURED=100
for label in baseline_before both cube vector baseline_after; do
    mode=$label
    if [[ $label == baseline_* ]]; then mode=baseline; fi
    export HP_EXPERIMENT_TRIAL="confirmation_$label"
    "$python" "$script_dir/run_monitored.py" "$repo" "$python" "$devices" \
        "$mode" performance "$output/confirmation/$label" "$port"
    port=$((port + 160))
done
for mode in baseline both; do
    export HP_EXPERIMENT_TRIAL=monitored_profile
    "$python" "$script_dir/run_monitored.py" "$repo" "$python" "$devices" \
        "$mode" profile "$output/monitored_profile/$mode" "$port"
    port=$((port + 160))
done
