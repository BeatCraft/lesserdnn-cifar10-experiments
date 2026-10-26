# Experiment map

| Experiment | Initializer | Stopping / selection | Metric | Reproduction |
|---|---|---|---|---|
| Original Gradient-mode comparison, Q9 | Fan-in sparse | 300 epochs; highest validation accuracy, then lowest CE | Selected-checkpoint test accuracy and CE | `run_ldnn_three_stage_10seeds.sh`, `run_ldnn_fixed_lr_10seeds.sh` |
| Controlled full-precision reference supplied with the artifact | Kaiming normal, fan-in, ReLU | SGD, 50 epochs; validation selection | Test accuracy and CE | `run_pytorch_reference_10seeds.sh` |
| Codebook / optimizer comparison for the paper revision | Uniform entries with `|w| ≤ 0.125` | 30 minutes net training; endpoint, no checkpoint selection | Validation accuracy and CE, no test evaluation | `additional_experiments/run_experiment.py` |

The original manuscript's historical full-precision reference uses Adam for 20 epochs and reports
73.16% test accuracy. The supplied controlled SGD/50-epoch reference reports 69.04% across ten seeds.
They are different experiments. This repository does not claim that the SGD driver reproduces
the historical Adam run; its original checkpoint/predictions are not supplied here.

The 80-run dataset contains Q8, Q9, Q10 and Q11, strict search (SA label) and single-batch Gradient,
seeds 0–9 for each condition. The original three-mode comparison and the timed comparison must be
described separately because initialization, evaluation cadence, stopping and test usage differ.

Exploratory runs, failed/interrupted predecessor queues, and the rejected single Q11 300-epoch
follow-up are not part of the paper results exported in this working tree.

The added timed comparison is included in v1.1.0. `CITATION.cff` identifies this version;
the v1.0.0 version-specific DOI describes only the earlier archive. The concept DOI spans versions.
