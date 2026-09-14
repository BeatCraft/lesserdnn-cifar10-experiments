# Combined rigorous CIFAR-10 evaluation (10 seeds)

| Method | Runs | Test accuracy mean ± SD | 95% CI | Mean best epoch |
| --- | ---: | ---: | ---: | ---: |
| Single-batch | 10 | 68.7750% ± 0.5275% | [68.3977%, 69.1523%] | 143.2 |
| Two-batch average | 10 | 68.0810% ± 0.6903% | [67.5872%, 68.5748%] | 207.9 |
| Same-sign two-batch | 10 | 69.3030% ± 0.5516% | [68.9084%, 69.6976%] | 272.8 |

## Paired comparisons

Difference is first method minus second method.

| First | Second | Seeds | Mean difference | 95% CI | Exact p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| Single-batch | Two-batch average | 10 | +0.6940% | [+0.1498%, +1.2382%] | 0.0117 |
| Single-batch | Same-sign two-batch | 10 | -0.5280% | [-1.1753%, +0.1193%] | 0.1172 |
| Two-batch average | Same-sign two-batch | 10 | -1.2220% | [-1.8596%, -0.5844%] | 0.0020 |
