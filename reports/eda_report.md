 EDA Report — Credit Risk Scorer

Dataset Overview
| Property       | Value                 |
|----------------|-----------------------|
| Rows           | 30,000 |
| Columns        | 24   |
| Missing Values | 0                     |
| Duplicates     | 0                     |
| Target Column  | default (0=No, 1=Yes) |

Class Distribution
| Class          | Count    | Percentage |
|----------------|----------|------------|
| No Default (0) | 23,364 | 77.9% |
| Default (1)    | 6,636 | 22.1% |
| Ratio          | 3.5:1 | |

Top Predictive Features
| Rank | Feature   | Correlation | Direction      |
|------|-----------|-------------|----------------|
| 1    | PAY_0     | +0.324      | Increases risk |
| 2    | PAY_2     | +0.263      | Increases risk |
| 3    | PAY_3     | +0.235      | Increases risk |
| 4    | PAY_4     | +0.217      | Increases risk |
| 5    | PAY_5     | +0.204      | Increases risk |
| 6    | PAY_6     | +0.187      | Increases risk |
| 7    | LIMIT_BAL | -0.153      | Reduces risk   |
| 8    | BILL_AMT1 | +0.147      | Increases risk |

Key Findings
- Payment delay history (PAY_0 to PAY_6) is the strongest signal
- Low credit limits strongly predict default
- Younger customers (21-30) have higher default rates
- EDUCATION has 345 rows with invalid codes (0,5,6)
- MARRIAGE has 54 rows with invalid code (0)
- BILL_AMT and PAY_AMT columns have significant outliers

Data Quality Issues for Phase 2
| Issue                   | Column      | Fix                    |
|-------------------------|-------------|------------------------|
| Invalid codes (0, 5, 6) | EDUCATION   | Remap to Others (4)    |
| Invalid code (0)        | MARRIAGE    | Remap to Others (3)    |
| Right-skewed            | PAY_AMT1-6  | Log transformation     |
| High outliers           | BILL_AMT1-6 | Cap at 99th percentile |

Figures Saved
- reports/figures/01_class_imbalance.png
- reports/figures/02_correlation_heatmap.png
- reports/figures/03_target_correlation.png
- reports/figures/04_distributions.png
- reports/figures/05_payment_status.png
- reports/figures/06_age_credit_analysis.png
- reports/figures/07_categorical_analysis.png
- reports/figures/08_boxplots.png
