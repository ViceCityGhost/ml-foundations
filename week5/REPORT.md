# Week 5 — Hyperparameter Tuning & Cross-Validation

**Date:** 2025-09-06 19:43
**Datasets:** Breast Cancer (classification) & Diabetes (regression)

## Goal
Get confident with model selection: proper cross-validation, hyperparameter tuning (Grid/Randomized), and diagnostics (confusion matrix, ROC, learning curves). Save best models and summarize results.

---

## Methods
- **Cross-Validation:** StratifiedKFold (classification), KFold (regression)
- **Pipelines:** `StandardScaler` + estimator (preprocessing stays **inside** CV)
- **Tuning:** GridSearchCV for small spaces; RandomizedSearchCV (and optional Halving) for wide spaces
- **Metrics:** ROC-AUC / F1 / Accuracy (classification), RMSE (regression)

---

## Classification — Leaderboard (top first)
| model           |   cv_roc_auc_mean |   cv_roc_auc_std |   test_roc_auc |   test_f1 |   test_acc |
|:----------------|------------------:|-----------------:|---------------:|----------:|-----------:|
| Best_SVC        |            0.9972 |           0.0039 |         0.9983 |    0.9832 |      0.979 |
| Best_LogReg     |            0.9963 |           0.0036 |         0.9977 |    0.9889 |      0.986 |
| Baseline_LogReg |            0.9962 |           0.004  |         0.9977 |    0.9889 |      0.986 |
| Best_RF_or_HGB  |            0.9879 |           0.0113 |         0.9935 |    0.967  |      0.958 |

**Top model:** **Best_SVC**  
- CV ROC-AUC: 0.9972 ± 0.0039  
- Test ROC-AUC / F1 / Acc: 0.9983 / 0.9832 / 0.9790

---

## Regression — Leaderboard (sorted by lowest CV RMSE)
| model           |   cv_rmse_mean |   cv_rmse_std |   test_rmse |
|:----------------|---------------:|--------------:|------------:|
| Best_Lasso      |         56.056 |         3.196 |      53.282 |
| Best_Ridge      |         56.07  |         3.237 |      53.294 |
| Baseline_LinReg |         56.119 |         3.133 |      53.37  |
| Best_RF_or_HGB  |         59.22  |         2.406 |      52.722 |

**Top model:** **Best_Lasso**  
- CV RMSE: 56.056 ± 3.196  
- Test RMSE: 53.282

---

## Artifacts
- Notebook: `Week5_Hyperparameter_Tuning_CV.ipynb`
- Saved models: `models/<top_classifier>.pkl`, `models/<top_regressor>.pkl`
- Leaderboards: `models/classification_leaderboard.csv`, `models/regression_leaderboard.csv`

## Notes
- Keep preprocessing inside CV to avoid leakage.
- Use task-aligned metrics (ROC-AUC/F1 vs RMSE).
- RandomizedSearch is efficient for wide spaces; refine later if needed.
- Learning curves help diagnose bias/variance and whether more data could help.
