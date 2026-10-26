# Published aggregate results

This section reports the original 300-epoch Gradient-mode experiments and the separately provided
SGD/50-epoch full-precision reference. The additional 80-run comparison is below; it uses validation
endpoints rather than validation-selected test accuracy. See [EXPERIMENTS.md](EXPERIMENTS.md) for scope.

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

## Additional 30-minute comparison (80 runs; validation only)

| Levels | SA CE mean ± SD | SA accuracy mean ± SD | Gradient CE mean ± SD | Gradient accuracy mean ± SD |
|---|---:|---:|---:|---:|
| 8 | 14.8035 ± 1.1213 | 12.95% ± 0.83 pp | 1.5254 ± 0.0987 | 46.65% ± 2.62 pp |
| 9 | 7.1083 ± 0.6845 | 13.07% ± 0.64 pp | 1.1627 ± 0.0482 | 60.39% ± 1.32 pp |
| 10 | 6.1657 ± 0.4545 | 12.39% ± 0.58 pp | 1.1609 ± 0.0723 | 61.80% ± 1.95 pp |
| 11 | 4.2274 ± 0.3869 | 13.65% ± 1.17 pp | 1.0819 ± 0.0488 | 65.15% ± 1.27 pp |

The authoritative table and paired contrasts are regenerated from the 80 seed-level records in
[`results/timed_codebooks_80runs/SUMMARY.md`](results/timed_codebooks_80runs/SUMMARY.md).
`SA` is a label for temperature-free strict first-improvement search in this experiment.
The comparison varies both codebook and initial distribution; the codebook series were not
jointly randomized. The official test set was not evaluated.
