# Sourced by train.sh and train_sequence_renyi.sh, not run on its own.
# run_with_resources <run_dir> <gpus> <log_file> <command...> runs the command (output copied to the log file) while a
# background sampler records the memory in use on the listed GPUs, then writes <run_dir>/resources.json:
# wall seconds of the job, GPU hours (wall x GPUs / 3600) and the per-GPU peak of NVML memory.used in GiB.
# The sampling interval is GPU_SAMPLE_SEC seconds (default 15). The raw samples stay in <run_dir>/gpu_mem.csv.
GPU_SAMPLE_SEC=${GPU_SAMPLE_SEC:-15}
SAMPLER_PID=""

write_resources() {  # csv, resources.json, start epoch, end epoch, command seconds, number of GPUs, GPU list
  python - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$GPU_SAMPLE_SEC" <<'PY' || true
import csv, json, sys
src, dst, t0, t1, train_s, ngpu, gpus, every = sys.argv[1:9]
peak = {}
n = 0
try:
    rows = list(csv.DictReader(open(src)))
except FileNotFoundError:
    rows = []
for row in rows:
    try:
        mib = float(row["memory_used_mib"])
    except (KeyError, ValueError, TypeError):
        continue
    n += 1
    peak[row["index"]] = max(peak.get(row["index"], 0.0), mib)
wall = int(t1) - int(t0)
out = {
    "definition": "wall = whole training job, gpu_hours = wall x GPUs / 3600, "
                  f"peak = max over GPUs of NVML memory.used sampled every {every} s",
    "wall_seconds": wall,
    "wall_hms": f"{wall // 3600}:{wall % 3600 // 60:02d}:{wall % 60:02d}",
    "train_command_seconds": int(train_s) if train_s else None,
    "gpus": gpus, "ngpus": int(ngpu),
    "gpu_hours": round(wall * int(ngpu) / 3600, 2),
    "per_gpu_peak_memory_gib": {k: round(v / 1024, 4) for k, v in sorted(peak.items())},
    "peak_gpu_memory_gib": round(max(peak.values()) / 1024, 2) if peak else None,
    "sample_interval_seconds": int(every), "samples": n,
}
json.dump(out, open(dst, "w"), indent=2)
PY
}

run_with_resources() {
  local out=$1 gpus=$2 log=$3 t0 tc rc=0
  local -a g
  shift 3
  IFS=, read -r -a g <<< "$gpus"
  echo "timestamp,index,memory_used_mib,utilization_gpu" > "$out/gpu_mem.csv"
  t0=$(date +%s)
  (
    set +e
    while true; do
      ts=$(date +%s)
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpus" 2>/dev/null \
        | sed "s/^/$ts,/; s/ //g" >> "$out/gpu_mem.csv"
      sleep "$GPU_SAMPLE_SEC"
    done
  ) > /dev/null 2>&1 &
  SAMPLER_PID=$!
  trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT
  tc=$(date +%s)
  "$@" 2>&1 | tee "$log" || rc=$?
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
  trap - EXIT
  write_resources "$out/gpu_mem.csv" "$out/resources.json" "$t0" "$(date +%s)" "$(($(date +%s) - tc))" "${#g[@]}" "$gpus"
  return "$rc"
}
