# Reproducibility notes

## Randomness

- Model/training seeds are 0--9.
- The stratified split seed is fixed to 20260826.
- Each epoch's training order is generated from the run seed and epoch number.
- All modes for a given seed see the same retained mini-batches in the same order.

## Data usage

- The original CIFAR-10 test set is not used for checkpoint selection.
- The fixed validation split contains exactly 500 examples from each class.
- The training subset contains 45,000 examples. Its final odd full batch and incomplete remainder are
  excluded, yielding 350 gradient batches and 44,800 processed images per epoch.
- No data augmentation is applied.

## LesserDNN weights

Forward and backward computation use the nine-level quantized LesserDNN weights. A float32 shadow
copy accumulates updates. After every optimizer step the shadow weights are clipped to the codebook range,
mapped to the nearest codebook value for the next forward pass, and retained in float32 for future updates.

## Update modes

- `single`: update from one mini-batch gradient.
- `two_batch_average`: evaluate two batches at the same quantized state and update using their mean gradient.
- `same_sign`: use the mean gradient only at coordinates where both nonzero batch gradients have the same sign.

For `epsilon=0`, exact zeros are excluded by the same-sign mask.

## Initialization

The 300-epoch Gradient-mode comparison uses `initialize_fan_in_aware_weights` in
`lesserdnn_cifar10_ablation_200/ldnn_cifar10_model.py`. This is part of the experimental setup,
not an update rule. For each layer, in layer order:

1. Set the target second moment `v = 2 / fan_in`.
2. Select `a`, the available positive weight value closest to `sqrt(v)`; ties use the first value.
3. Set `p = min(1, v / a²)`.
4. Draw an independent activation mask with probability `p` and independent equiprobable signs;
   assign `+a` or `-a` at active locations, otherwise `0`.

The generator is NumPy `default_rng(seed)`, drawing the full activation mask and then the full sign
mask for each layer. A single generator continues across Conv1, Conv2, Conv3 and FC. Thus seed,
draw order, codebook ordering and matrix shape are part of reproducibility.

For the paper's Q9 codebook, the nonzero magnitudes are 0.25, 0.125, 0.125 and 0.125, and the
activation probabilities are 1, 4/9, 2/9 and 1/64. The achieved second moment may differ from the
target when density is capped at one. Bias is zero and shadow weights start as exact float32 copies
of the quantized initial weights.

This is **not** the initializer used by the 80 timed runs. Those runs draw each index uniformly
from the entries with absolute value at most 0.125, independently per layer. Q8 draws from two
entries, Q9 three, Q10 four and Q11 five. See the [timed protocol](additional_experiments/README.md).

## Checkpoints

Each run saves its latest quantized weights, shadow weights, state, split indexes, and epoch history. The best
validation checkpoint is stored separately. Resume validation rejects mismatched configurations.

## Aggregate artifacts

The public `results/` directory contains only lightweight aggregate data. Raw checkpoints and test predictions
are omitted because they are large and are not required to verify the reported summary statistics.

`results/timed_codebooks_80runs/` contains 80 final rows and 560 initial/intermediate/endpoint
validation rows, plus sanitized original manifests and the recorded software/hardware environments.
The rejected, exploratory single Q11 300-epoch run is not included. The original 300-epoch
Gradient-mode results and the additional time-budget results use different initialization,
selection and test-usage protocols and must not be pooled into a single mean.
