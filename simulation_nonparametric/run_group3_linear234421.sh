#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python main.py \
  --device cuda \
  --preset single \
  --group group234421_linear_fixed035 \
  --dims 2,3,4,4,2,1 \
  --sample-sizes 1000,2000,3000,4000 \
  --experiments 1000 \
  --dtype float64 \
  --splits 5 \
  --output-dir result \
  --save-every 10 \
  --fixed-weights \
  --fixed-weight-scale 0.35 \
  --proxy-strength 1.5 \
  --proxy-noise 0.25 \
  --treatment-proxy-strength 1.1 \
  --outcome-proxy-strength 1.1 \
  --proxy-square-strength 0.0 \
  --outcome-square-strength 0.0
