#!/bin/zsh
set -euo pipefail
RELEASE_ROOT="${0:A:h}"
RELEASE_PYTHON="${RELEASE_PYTHON:-/usr/bin/python3}"
cd "${RELEASE_ROOT}/lesserdnn_cifar10_rigorous_eval"
"${RELEASE_PYTHON}" run_experiment.py \
  --seeds 0 \
  --target-epochs 2 \
  --lr-schedule 1:0.01,2:0.003 \
  --max-gradient-batches 2 \
  --data-root "${RELEASE_ROOT}/data" \
  --ldnn-path "${RELEASE_ROOT}/lesserdnn_cifar10_ablation_200/ldnn_runtime" \
  --output-root "${RELEASE_ROOT}/runs/smoke_ldnn"
"${RELEASE_PYTHON}" run_pytorch_reference.py \
  --seeds 0 \
  --target-epochs 2 \
  --lr-schedule 2:0.01 \
  --max-gradient-batches 2 \
  --data-root "${RELEASE_ROOT}/data" \
  --output-root "${RELEASE_ROOT}/runs/smoke_pytorch"
