#!/bin/bash
# usage: free_gpu.sh MIN_MB -> the GPU with the most free memory (at least MIN_MB), skipping the GPUs in gpu_exclude
EXCLUDE=$(cat "$(dirname "$0")/gpu_exclude" 2>/dev/null)
# gpu_min_override (optional, MB): lowers MIN_MB for every queue, e.g. while the free GPUs sit just below 38 GB
OVERRIDE=$(cat "$(dirname "$0")/gpu_min_override" 2>/dev/null)
[ -n "$OVERRIDE" ] && [ "$OVERRIDE" -lt "$1" ] && set -- "$OVERRIDE"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr |
  awk -F', ' -v min="$1" -v ex=" $EXCLUDE " '$2 >= min && index(ex, " " $1 " ") == 0 {print $1; exit}'
