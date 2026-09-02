#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python main.py \
  --device cuda \
  --preset three_groups \
  --sample-sizes 1000,2000,3000,4000 \
  --experiments 1000 \
  --dtype float64 \
  --splits 5 \
  --output-dir result \
  --save-every 10
