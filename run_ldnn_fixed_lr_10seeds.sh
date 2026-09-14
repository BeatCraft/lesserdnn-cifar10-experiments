#!/bin/zsh
set -euo pipefail
RELEASE_ROOT="${0:A:h}"
RELEASE_PYTHON="${RELEASE_PYTHON:-/usr/bin/python3}"
cd "${RELEASE_ROOT}/lesserdnn_cifar10_rigorous_eval"
"${RELEASE_PYTHON}" run_experiment.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --target-epochs 300 \
  --lr-schedule 300:0.01 \
  --data-root "${RELEASE_ROOT}/data" \
  --ldnn-path "${RELEASE_ROOT}/lesserdnn_cifar10_ablation_200/ldnn_runtime" \
  --output-root "${RELEASE_ROOT}/runs/ldnn_fixed_lr_10seeds"
