# Rolling-origin backtest

12 monthly folds (2024-07 to 2025-06), 7-day embargo between training and test data.

| Model | balanced_accuracy | roc_auc | recall | precision | brier |
|---|---|---|---|---|---|
| logistic | 0.826 | 0.902 | 0.813 | 0.529 | 0.123 |
| xgboost | 0.795 | 0.904 | 0.723 | 0.548 | 0.108 |
| rule | 0.590 | 0.590 | 0.729 | 0.228 | 0.498 |

**Selected for production: `logistic`** (best balanced accuracy; ties go to the simpler model).

## Balanced accuracy by fold

| Month | logistic | xgboost | Positives / rows |
|---|---|---|---|
| 2024-07 | 0.883 | 0.968 | 8 / 55 |
| 2024-08 | 0.844 | 0.870 | 7 / 84 |
| 2024-09 | 0.790 | 0.643 | 21 / 76 |
| 2024-10 | 0.967 | 0.961 | 7 / 83 |
| 2024-11 | 0.619 | 0.619 | 7 / 86 |
| 2024-12 | 0.783 | 0.771 | 3 / 92 |
| 2025-01 | 0.836 | 0.831 | 21 / 75 |
| 2025-02 | 0.641 | 0.631 | 16 / 63 |
| 2025-03 | nan | nan | 0 / 87 |
| 2025-04 | 0.852 | 0.746 | 32 / 67 |
| 2025-05 | 0.890 | 0.902 | 24 / 66 |
| 2025-06 | 0.781 | 0.797 | 20 / 78 |
