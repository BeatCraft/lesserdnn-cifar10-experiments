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

## Checkpoints

Each run saves its latest quantized weights, shadow weights, state, split indexes, and epoch history. The best
validation checkpoint is stored separately. Resume validation rejects mismatched configurations.

## Aggregate artifacts

The public `results/` directory contains only lightweight aggregate data. Raw checkpoints and test predictions
are omitted because they are large and are not required to verify the reported summary statistics.
