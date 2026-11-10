#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "Usage: $0 REPO PYTHON DEVICES MODE PHASE OUTPUT PORT_BASE [SHAPE]" >&2
  exit 2
fi
repo=$1
python=$2
devices=$3
mode=$4
phase=$5
output=$6
port_base=$7
shape=${8:-representative}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${ASCEND_HOME_PATH:?source the experiment CANN environment first}"
[[ -f "$repo/build/native/payload/hyper_multicore/lib/set_env.bash" ]]
if [[ -f "$output/result.json" ]]; then
  echo "Refusing to overwrite an existing experiment result: $output/result.json" >&2
  exit 2
fi
mkdir -p "$output"
IFS=, read -r -a cards <<< "$devices"
unset ASCEND_CUSTOM_OPP_PATH CANN_VENDOR_HYPER_PARALLEL CANN_VENDOR_HYPER_MULTICORE
unset RANK_TABLE_FILE RANK_ID RANK_SIZE
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
source "$repo/build/native/payload/hyper_multicore/lib/set_env.bash"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="$devices"
export HCCL_IF_BASE_PORT="$((port_base + 64))"
export HCCL_NPU_SOCKET_PORT_RANGE="$((port_base + 64))-$((port_base + 127))"
export HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT="tcp://127.0.0.1:$((port_base + 1))"
unset HYPER_MULTICORE_SHMEM_HEAP_SIZE
cd "$repo"
set +e
"$python" -m torch.distributed.run --nnodes=1 --nproc_per_node="${#cards[@]}" \
  --master_addr=127.0.0.1 --master_port="$port_base" \
  "$script_dir/core_partition_experiment.py" \
  --mode "$mode" --phase "$phase" --shape "$shape" \
  --dispatch-mode "${HP_EXPERIMENT_DISPATCH_MODE:-push}" \
  --warmup "${HP_EXPERIMENT_WARMUP:-10}" --measured "${HP_EXPERIMENT_MEASURED:-50}" \
  --trial "${HP_EXPERIMENT_TRIAL:-r0}" --output "$output" > "$output/run.log" 2>&1
status=$?
printf '%s\n' "$status" > "$output/exit_code.txt"
tail -n 40 "$output/run.log"
exit "$status"
