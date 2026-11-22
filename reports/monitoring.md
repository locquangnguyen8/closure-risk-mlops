# Production monitoring simulation

Drift is introduced from 2025-07: heavier rain and a stronger catchment response.

| Month | Model | Balanced accuracy | 28-day balanced accuracy | Recall | Max PSI (90 days) | Action |
|---|---|---|---|---|---|---|
| 2025-07 | v1 | 0.817 | 0.750 | 0.800 | 0.514 | train v2 |
| 2025-08 | v1 | 0.805 | 0.814 | 0.818 | 0.694 | discard v2 |
| 2025-09 | v1 | 0.740 | 0.784 | 0.524 | 0.582 | train v3 |
| 2025-10 | v1 | 0.802 | 0.802 | 0.688 | 0.498 | promote v3 |
| 2025-11 | v3 | 0.736 | 0.768 | 0.880 | 0.591 | train v4 |
| 2025-12 | v3 | 0.800 | 0.801 | 0.676 | 0.533 | promote v4 |
| 2026-01 | v4 | 0.854 | 0.836 | 0.792 | 0.177 |  |
| 2026-02 | v4 | 0.807 | 0.807 | 0.714 | 0.507 | train v5 |
| 2026-03 | v4 | 0.880 | 0.869 | 0.852 | 0.504 | promote v5 |
| 2026-04 | v5 | 0.911 | 0.884 | 0.893 | 0.251 | train v6 |
| 2026-05 | v5 | 0.841 | 0.831 | 0.773 | 0.398 | promote v6 |
| 2026-06 | v6 | 0.721 | 0.710 | 1.000 | 0.228 |  |
| 2026-07 | v6 | 0.722 | 0.967 | 0.944 | 0.341 | train v7 |
| 2026-08 | v6 | 0.808 | 0.813 | 0.700 | 0.535 | discard v7 |
| 2026-09 | v6 | 0.968 | 0.968 | 1.000 | 0.552 | train v8 |
| 2026-10 | v6 | 0.668 | 0.666 | 0.357 | 0.517 | discard v8 |

## Events

- 2025-07: champion v1 (logistic) deployed, trained through 2025-06-23
- 2025-07: retraining triggered (PSI 0.51 on flow_7d_mean > 0.25); challenger v2 goes to shadow next month
- 2025-08: kept v1; challenger v2 did not improve enough (+0.000)
- 2025-09: retraining triggered (PSI 0.58 on salinity_3d_min > 0.25); challenger v3 goes to shadow next month
- 2025-10: promoted v3 (shadow balanced accuracy 0.896 vs champion 0.802)
- 2025-11: retraining triggered (PSI 0.59 on flow_7d_mean > 0.25); challenger v4 goes to shadow next month
- 2025-12: promoted v4 (shadow balanced accuracy 0.810 vs champion 0.800)
- 2026-02: retraining triggered (PSI 0.51 on rain_7d > 0.25); challenger v5 goes to shadow next month
- 2026-03: promoted v5 (shadow balanced accuracy 0.895 vs champion 0.880)
- 2026-04: retraining triggered (PSI 0.25 on rain_7d > 0.25); challenger v6 goes to shadow next month
- 2026-05: promoted v6 (shadow balanced accuracy 0.864 vs champion 0.841)
- 2026-07: retraining triggered (PSI 0.34 on salinity_3d_min > 0.25); challenger v7 goes to shadow next month
- 2026-08: kept v6; challenger v7 did not improve enough (+0.000)
- 2026-09: retraining triggered (PSI 0.55 on rain_7d > 0.25); challenger v8 goes to shadow next month
- 2026-10: kept v6; challenger v8 did not improve enough (+0.000)
