#!/bin/zsh
set -euo pipefail
RELEASE_ROOT="${0:A:h}"
RELEASE_PYTHON="${RELEASE_PYTHON:-/usr/bin/python3}"
cd "${RELEASE_ROOT}/lesserdnn_cifar10_rigorous_eval"
"${RELEASE_PYTHON}" run_pytorch_reference.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --target-epochs 50 \
  --lr-schedule 50:0.01 \
  --data-root "${RELEASE_ROOT}/data" \
  --output-root "${RELEASE_ROOT}/runs/pytorch_reference_10seeds"
