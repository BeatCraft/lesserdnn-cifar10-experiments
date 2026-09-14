# Gradient-Guided LesserDNN: CIFAR-10 Experiments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22743842.svg)](https://doi.org/10.5281/zenodo.22743842)

This repository-ready package contains the code and aggregate results for the CIFAR-10 experiments in
**Gradient-Guided Learning for LesserDNN with Quantized Weights**. It includes the LesserDNN
Metal implementation, three gradient update modes, and a controlled full-precision PyTorch reference.

## What is included

- LesserDNN training with quantized forward weights and float32 shadow weights.
- Three update modes: single-batch, two-batch average, and same-sign two-batch.
- Fixed stratified train/validation split and one-time test evaluation.
- Resumable 10-seed experiment drivers.
- A full-precision PyTorch reference with the same bias-free CNN topology.
- Per-seed aggregate CSV files and statistical summaries used for reporting.

CIFAR-10 images, raw checkpoints, and per-example predictions are intentionally excluded.

## Repository layout

```text
.
├── lesserdnn_cifar10_ablation_200/
│   ├── ldnn_runtime/                 # LesserDNN CPU/Metal runtime
│   ├── ldnn_cifar10_model.py         # Quantized CNN definition
│   └── experiments/.../train.py      # Gradient and shadow-weight updates
├── lesserdnn_cifar10_rigorous_eval/
│   ├── train_one.py                  # One LesserDNN seed/mode
│   ├── run_experiment.py             # Multi-seed LesserDNN driver
│   ├── pytorch_reference_train.py    # CIFAR-10 PyTorch specialization
│   └── *_core.py                     # Self-contained PyTorch shared code
├── results/                          # Aggregate, lightweight published results
├── configs/                          # Machine-readable formal protocols
├── download_cifar10.py
├── run_smoke_tests.sh
├── run_ldnn_three_stage_10seeds.sh
├── run_ldnn_fixed_lr_10seeds.sh
└── run_pytorch_reference_10seeds.sh
```

## Model

Both frameworks use a bias-free network with 174,944 weights:

```text
3x32x32 input
Conv3x3(3 -> 32), ReLU, MaxPool2x2
Conv3x3(32 -> 64), ReLU, MaxPool2x2
Conv3x3(64 -> 128), ReLU
Linear(128*8*8 -> 10)
```

Batch normalization and data augmentation are disabled. Inputs are normalized with CIFAR-10 channel
means `(0.4914, 0.4822, 0.4465)` and standard deviations `(0.2470, 0.2435, 0.2616)`.

## Platform and environment

The LesserDNN experiments require macOS on Apple Silicon because their convolutional forward and
backward passes use Metal through PyObjC. The published run used:

- Python 3.9.6
- NumPy 2.0.2
- PyTorch 2.8.0
- torchvision 0.23.0
- PyObjC 11.1
- SciPy 1.13.1
- PyOpenCL 2025.1

Create an environment and install the recorded dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
export RELEASE_PYTHON="$PWD/.venv/bin/python3"
```

On a machine where `/usr/bin/python3` already has these dependencies, `RELEASE_PYTHON` may be omitted.

Run the lightweight source/result consistency check:

```bash
"${RELEASE_PYTHON:-/usr/bin/python3}" validate_release.py
```

## Dataset

Download CIFAR-10 through torchvision:

```bash
"${RELEASE_PYTHON:-/usr/bin/python3}" download_cifar10.py
```

This creates `data/cifar-10-batches-py/`. The dataset is excluded from Git.

## Preflight smoke tests

The following command runs two gradient batches per epoch and checks model construction, Metal
forward/backward execution, validation selection, checkpoint writing, one-time test evaluation, and the
PyTorch path:

```bash
./run_smoke_tests.sh
```

Smoke-test outputs go to `runs/` and must not be reported as experimental results.

## Reproduce the experiments

Three-stage LesserDNN schedule, 10 seeds, 300 epochs:

```bash
./run_ldnn_three_stage_10seeds.sh
```

The learning rate is 0.01 for epochs 1--100, 0.005 for 101--200, and 0.001 for 201--300.

Fixed-rate LesserDNN experiment, 10 seeds, 300 epochs:

```bash
./run_ldnn_fixed_lr_10seeds.sh
```

The learning rate is 0.01 throughout.

Full-precision PyTorch reference, 10 seeds, 50 epochs:

```bash
./run_pytorch_reference_10seeds.sh
```

The reference uses float32 SGD with learning rate 0.01, zero momentum, and zero weight decay. PyTorch uses
MPS when available and otherwise uses CPU.

All drivers resume at epoch boundaries. Re-run the same command after an interruption. Do not reuse an
output directory with changed arguments.

## Evaluation protocol

- Seeds: 0 through 9.
- Split seed: 20260826.
- CIFAR-10 training set: fixed class-stratified split into 45,000 training and 5,000 validation examples.
- Batch size: 128 examples per gradient.
- Each epoch: 350 gradient batches and 44,800 processed training examples.
- Checkpoint selection: highest validation accuracy, then lowest validation loss for ties.
- Test set: evaluated once, after training, using the validation-selected checkpoint.
- Reported uncertainty: sample SD and a t-based 95% confidence interval across seeds.
- Within-LesserDNN comparisons: paired differences and exact two-sided sign-flip tests.

Single-batch LesserDNN and PyTorch perform 350 optimizer updates per epoch. The two two-batch modes evaluate
the same 350 mini-batches but perform 175 updates per epoch.

## Results

| Framework / schedule / mode | Mean test accuracy |
| --- | ---: |
| PyTorch, fixed 0.01, 50 epochs | 69.040% |
| LesserDNN, three-stage, single-batch | 69.794% |
| LesserDNN, three-stage, two-batch average | 69.159% |
| LesserDNN, three-stage, same-sign | 69.074% |
| LesserDNN, fixed 0.01, single-batch | 68.775% |
| LesserDNN, fixed 0.01, two-batch average | 68.081% |
| LesserDNN, fixed 0.01, same-sign | 69.303% |

Detailed seed-level results and confidence intervals are under [`results/`](results/). The PyTorch run uses
50 epochs whereas LesserDNN uses 300; cross-framework values are therefore a descriptive comparison of
attained performance, not evidence that either framework is superior under an equal training budget.

## Citation

The archived `v1.0.0` release is available at
[doi:10.5281/zenodo.22743843](https://doi.org/10.5281/zenodo.22743843). To cite the repository across all
versions, use [doi:10.5281/zenodo.22743842](https://doi.org/10.5281/zenodo.22743842). Machine-readable
citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

The original source code and documentation in this repository are available under the
[MIT License](LICENSE). External dependencies and CIFAR-10 remain subject to their respective licenses;
see [`LICENSE-NOTICE.md`](LICENSE-NOTICE.md).
