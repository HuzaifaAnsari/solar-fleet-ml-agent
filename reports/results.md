| Mode | Model | Accuracy | Macro-F1 | Worst-class recall |
|---|---|---|---|---|
| MPPT | Always 'healthy' (baseline) | 56% | 0.09 | 0% |
| MPPT | Gradient boosting, raw features | 84% | 0.83 | 75% |
| MPPT | Gradient boosting, site-baseline features | 83% | 0.82 | 76% |
| MPPT | 1D-CNN on raw signals | 67% | 0.70 | 47% |
| IPPT | Always 'healthy' (baseline) | 56% | 0.09 | 0% |
| IPPT | Gradient boosting, raw features | 83% | 0.74 | 0% |
| IPPT | Gradient boosting, site-baseline features | 81% | 0.74 | 0% |
| IPPT | 1D-CNN on raw signals | 42% | 0.51 | 0% |

| Mode | Unsupervised detector | Precision | Recall | False alarms on healthy |
|---|---|---|---|---|
| MPPT | Isolation Forest, raw features | 100% | 28% | 0% |
| MPPT | Control chart on site baseline, k=4 | 96% | 66% | 2% |
| MPPT | Control chart on site baseline, k=6 | 100% | 54% | 0% |
| MPPT | Control chart on site baseline, k=8 | 100% | 47% | 0% |
| IPPT | Isolation Forest, raw features | 0% | 0% | 0% |
| IPPT | Control chart on site baseline, k=4 | 95% | 68% | 3% |
| IPPT | Control chart on site baseline, k=6 | 99% | 58% | 1% |
| IPPT | Control chart on site baseline, k=8 | 99% | 55% | 1% |

Cross-mode (train on one operating mode, test on the other), gradient boosting, macro-F1 raw -> site-baseline features: train M test L = 0.11 -> 0.49, train L test M = 0.25 -> 0.52
