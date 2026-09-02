#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p result

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

(
  export CUDA_VISIBLE_DEVICES=0
  echo "[$(date '+%F %T')] start group1 on GPU 0"
  bash run_group1_linear133311.sh
  echo "[$(date '+%F %T')] finished group1 on GPU 0"

  echo "[$(date '+%F %T')] start group2 on GPU 0"
  bash run_group2_proxyquad133311.sh
  echo "[$(date '+%F %T')] finished group2 on GPU 0"
) > result/gpu0_group1_then_group2.log 2>&1 &
pid_gpu0=$!

(
  export CUDA_VISIBLE_DEVICES=1
  echo "[$(date '+%F %T')] start group3 on GPU 1"
  bash run_group3_linear234421.sh
  echo "[$(date '+%F %T')] finished group3 on GPU 1"
) > result/gpu1_group3.log 2>&1 &
pid_gpu1=$!

echo "GPU0 queue pid: ${pid_gpu0}"
echo "GPU1 group3 pid: ${pid_gpu1}"
echo "logs:"
echo "  result/gpu0_group1_then_group2.log"
echo "  result/gpu1_group3.log"

wait "${pid_gpu0}" "${pid_gpu1}"
