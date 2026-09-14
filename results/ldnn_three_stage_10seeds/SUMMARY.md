# Combined rigorous CIFAR-10 evaluation (10 seeds)

| Method | Runs | Test accuracy mean ± SD | 95% CI | Mean best epoch |
| --- | ---: | ---: | ---: | ---: |
| Single-batch | 10 | 69.7940% ± 0.6767% | [69.3099%, 70.2781%] | 215.3 |
| Two-batch average | 10 | 69.1590% ± 0.7497% | [68.6227%, 69.6953%] | 219.4 |
| Same-sign two-batch | 10 | 69.0740% ± 0.6778% | [68.5892%, 69.5588%] | 230.7 |

## Paired comparisons

Difference is first method minus second method.

| First | Second | Seeds | Mean difference | 95% CI | Exact p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| Single-batch | Two-batch average | 10 | +0.6350% | [+0.2113%, +1.0587%] | 0.0098 |
| Single-batch | Same-sign two-batch | 10 | +0.7200% | [+0.2395%, +1.2005%] | 0.0098 |
| Two-batch average | Same-sign two-batch | 10 | +0.0850% | [-0.4208%, +0.5908%] | 0.7051 |
