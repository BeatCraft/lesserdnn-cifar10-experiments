# Published aggregate results

All values below are means over seeds 0--9. Checkpoints were selected using the fixed validation set; the
test set was evaluated once per seed and condition.

| Framework | LR schedule | Update mode | Epochs | Test accuracy mean ± SD | 95% CI |
| --- | --- | --- | ---: | ---: | ---: |
| LesserDNN | 0.01 → 0.005 → 0.001 | Single-batch | 300 | 69.794% ± 0.677% | [69.310%, 70.278%] |
| LesserDNN | 0.01 → 0.005 → 0.001 | Two-batch average | 300 | 69.159% ± 0.750% | [68.623%, 69.695%] |
| LesserDNN | 0.01 → 0.005 → 0.001 | Same-sign two-batch | 300 | 69.074% ± 0.678% | [68.589%, 69.559%] |
| LesserDNN | 0.01 fixed | Single-batch | 300 | 68.775% ± 0.527% | [68.398%, 69.152%] |
| LesserDNN | 0.01 fixed | Two-batch average | 300 | 68.081% ± 0.690% | [67.587%, 68.575%] |
| LesserDNN | 0.01 fixed | Same-sign two-batch | 300 | 69.303% ± 0.552% | [68.908%, 69.698%] |
| PyTorch | 0.01 fixed | Full-precision SGD | 50 | 69.040% ± 0.743% | [68.509%, 69.571%] |

The complete per-seed tables and machine-readable summaries are in:

- `results/ldnn_three_stage_10seeds/`
- `results/ldnn_fixed_lr_10seeds/`
- `results/pytorch_reference_10seeds/`

Because the PyTorch and LesserDNN runs use different epoch budgets and initialization representations,
cross-framework comparisons are descriptive. The controlled PyTorch reference is not intended as a
state-of-the-art CIFAR-10 baseline: bias, BatchNorm, augmentation, momentum, and weight decay are disabled.
