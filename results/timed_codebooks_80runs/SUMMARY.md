# Time-budget codebook comparison: 80 completed runs

Validation metrics at the 30-minute endpoint, 10 seeds per condition. No test-set evaluation.

| Method | Levels | Validation CE mean ± SD | Validation accuracy mean ± SD |
|---|---:|---:|---:|
| sa | 8 | 14.8035 ± 1.1213 | 12.95% ± 0.83 pp |
| sa | 9 | 7.1083 ± 0.6845 | 13.07% ± 0.64 pp |
| sa | 10 | 6.1657 ± 0.4545 | 12.39% ± 0.58 pp |
| sa | 11 | 4.2274 ± 0.3869 | 13.65% ± 1.17 pp |
| gradient | 8 | 1.5254 ± 0.0987 | 46.65% ± 2.62 pp |
| gradient | 9 | 1.1627 ± 0.0482 | 60.39% ± 1.32 pp |
| gradient | 10 | 1.1609 ± 0.0723 | 61.80% ± 1.95 pp |
| gradient | 11 | 1.0819 ± 0.0488 | 65.15% ± 1.27 pp |

The `sa` label denotes strict first-improvement random search: no temperature or worsening-loss acceptance.
All codebooks use uniform initialization restricted to |w| ≤ 0.125. These initial distributions differ.
See `summary.json` for paired contrasts, SDs and 95% intervals; p-values are exploratory and unadjusted.
The three codebook series ran sequentially (Q9/Q10, Q11, Q8), so wall-time comparisons have an execution-order limitation.

Recompute: `python3 additional_experiments/analyze_results.py --plot` from the repository root.
