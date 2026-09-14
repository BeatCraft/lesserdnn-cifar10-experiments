# CIFAR-10 PyTorchリファレンス実験：条件と結果

## 目的

量子化重みとfull-precision shadow weightを用いるLesserDNNの3更新方式と、同一の
ネットワーク構造を通常のfloat32重みで学習するPyTorchリファレンスを比較する。
本リファレンスは、augmentationや高度なoptimizerを用いた最高性能モデルではなく、
重み表現の違いを評価するための統制された基準である。

## PyTorch実験条件

| 項目 | 条件 |
| --- | --- |
| Dataset | CIFAR-10 |
| 学習／validation／test | 45,000／5,000／10,000 |
| 分割方法 | クラス層化固定分割 |
| Split seed | 20260826 |
| 学習seed | 0–9（10反復） |
| Epoch | 50 |
| Learning rate | 0.01固定 |
| Optimizer | SGD（momentum=0、weight decay=0） |
| Batch size | 128 |
| Gradient batches/epoch | 350 |
| 使用画像数/epoch | 44,800 |
| Precision | float32 |
| 初期化 | Kaiming normal（fan-in、ReLU） |
| Bias | なし |
| Batch normalization | なし |
| Data augmentation | なし |
| 入力正規化mean | (0.4914, 0.4822, 0.4465) |
| 入力正規化std | (0.2470, 0.2435, 0.2616) |
| 実行device | Apple MPS |
| Checkpoint選択 | validation accuracy最大、同率時はvalidation loss最小 |
| Test使用 | 選択完了後にbest checkpointを1回だけ評価 |

全seedで同じ学習／validation indexを使用した。学習indexのSHA-256は
`187d2962f1f83df973be736f3ab4441d78c79c93056bd4e2962a5a7c4d0bc35e`、
validation indexは
`98d69b62e4ebfc0710966c1546feb5c3cecde6662b6f51490924628dc4587c54`である。

## モデル構造

```text
Input 3×32×32
Conv 3×3, 3→32, padding=1, biasなし → ReLU → MaxPool 2×2
Conv 3×3, 32→64, padding=1, biasなし → ReLU → MaxPool 2×2
Conv 3×3, 64→128, padding=1, biasなし → ReLU
Flatten 128×8×8
Linear 8192→10, biasなし
```

総学習パラメータ数は174,944であり、LesserDNNモデルの重み数と一致する。

## seed別結果

| Seed | Best epoch | Best validation accuracy | Test accuracy | Test NLL |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 27 | 68.94% | 69.12% | 0.9850 |
| 1 | 35 | 68.24% | 68.81% | 1.1406 |
| 2 | 39 | 69.30% | 68.68% | 1.1961 |
| 3 | 23 | 68.94% | 69.16% | 0.9683 |
| 4 | 27 | 69.32% | 69.72% | 0.9867 |
| 5 | 26 | 68.70% | 68.96% | 1.0106 |
| 6 | 28 | 69.54% | 69.84% | 0.9819 |
| 7 | 20 | 68.72% | 68.50% | 0.9622 |
| 8 | 20 | 67.74% | 67.53% | 0.9734 |
| 9 | 24 | 68.96% | 70.08% | 0.9619 |

## 集約結果

| 指標 | 結果 |
| --- | ---: |
| Test accuracy（mean ± sample SD） | **69.040% ± 0.743%** |
| Test accuracy 95% CI | **[68.509%, 69.571%]** |
| Test accuracy範囲 | 67.53%–70.08% |
| Best validation accuracy平均 | 68.840% |
| Best epoch平均 | 26.9 |
| Best epoch範囲 | 20–39 |
| Test NLL平均 | 1.0167 |
| 10 runs合計学習時間 | 31分27秒 |

すべてのseedでbest checkpointはepoch 39以前に得られた。このモデルと固定LRでは、
50 epochで十分であり、300 epoch学習では訓練精度がほぼ100%となる一方でvalidation
lossが増加する過学習が確認された。

## LesserDNNとの記述的比較

LesserDNNは300 epoch、PyTorchは50 epochであるため、以下は同一seedを用いた記述的比較で
あり、同一学習予算による優劣検定ではない。

| 学習条件・方式 | Test accuracy（10-seed平均） | PyTorchとの差 |
| --- | ---: | ---: |
| PyTorch、LR=0.01固定、50 epoch | **69.040%** | — |
| LesserDNN、3段階LR、Single-batch | **69.794%** | +0.754ポイント |
| LesserDNN、3段階LR、Two-batch average | **69.159%** | +0.119ポイント |
| LesserDNN、3段階LR、Same-sign | **69.074%** | +0.034ポイント |
| LesserDNN、LR=0.01固定、Single-batch | **68.775%** | −0.265ポイント |
| LesserDNN、LR=0.01固定、Two-batch average | **68.081%** | −0.959ポイント |
| LesserDNN、LR=0.01固定、Same-sign | **69.303%** | +0.263ポイント |

## 解釈

PyTorchリファレンスとLesserDNNの多くの条件との差は小さい。特に3段階LRの
Two-batch averageとSame-signは、それぞれPyTorchとの差が+0.119、+0.034ポイントである。
この結果は、勾配とshadow weightを利用することで、離散的な量子化重みを直接用いる
LesserDNNでもfull-precision PyTorchと概ね同等水準のCIFAR-10性能を達成できるという
主張を支持する。

一方、PyTorchとLesserDNNでは初期化方式、重み精度、epoch数が異なる。したがって、
LesserDNNがPyTorchより優れている、または量子化が精度を改善するとは結論しない。
また、本PyTorchモデルはBatchNorm、augmentation、momentumを使用しない統制ベースラインで
あり、CIFAR-10におけるstate-of-the-art基準ではない。
