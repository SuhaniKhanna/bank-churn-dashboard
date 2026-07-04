# Predictive Modeling and Risk Scoring for Bank Customer Churn

A churn intelligence system for a European retail bank, built to move retention strategy from reactive to proactive by scoring every customer's likelihood of churning before they leave.

**Live app:** _add your Streamlit Cloud link here after deployment_
**Research paper:** _add your Medium link here_

## Problem

Retail banks typically explain churn *after* it happens rather than predicting it *before* it happens, which makes retention efforts broad, reactive, and costly. This project builds a model that assigns each customer a churn probability, so retention campaigns can be targeted at the customers most likely to actually leave.

## Dataset

10,000 customers across France, Spain, and Germany, with 14 raw attributes including credit score, tenure, balance, product holdings, and activity status. Target variable: `Exited` (1 = churned, 0 = retained). The dataset is imbalanced — roughly 20% churn — which shaped both the modeling approach (class-weighted models) and the evaluation metrics used (ROC-AUC and F1 over raw accuracy).

## Approach

1. **Preprocessing** — dropped non-informative identifiers, checked for missing values and duplicates (none found), confirmed feature ranges.
2. **Feature engineering** — five derived features designed around customer *behavior* rather than demographics: Balance-to-Salary ratio, Product Density, an Engagement-Product interaction term, Age-Tenure interaction, and a zero-balance flag.
3. **Modeling** — four models across the brief's required tiers: Logistic Regression (baseline), Decision Tree and Random Forest (tree-based), Gradient Boosting (advanced). Stratified 80/20 split, 5-fold cross-validation, class-weighting to address imbalance.
4. **Evaluation** — Accuracy, Precision, Recall, F1, and ROC-AUC compared across all four models; Gradient Boosting selected as the primary model on ROC-AUC, with the precision/recall tradeoff against Random Forest discussed explicitly in the paper.
5. **Explainability** — SHAP values used to rank global feature importance and support the "why" behind individual predictions, since black-box churn scores aren't actionable for a retention team without a driver behind them.

## Repository structure

```
bank-churn/
├── data/                  # raw and processed datasets
├── notebooks/             # 01-04: EDA, feature engineering, modeling, SHAP
├── models/                # trained model artifacts
├── outputs/                # evaluation tables and SHAP plots
├── app/                   # Streamlit dashboard
│   ├── app.py
│   ├── utils.py
│   ├── assets/            # model + data files bundled for deployment
│   └── requirements.txt
└── README.md
```

## Running locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard modules

- **Churn Risk Calculator** — score any individual customer profile live
- **Probability Distribution** — how predicted risk spreads across the customer base, by geography and product count
- **Feature Importance** — what the model is actually weighting, and why
- **What-If Simulator** — adjust engagement/product variables on a baseline customer to test retention-offer scenarios

## Future Enhancements

A few directions this could be extended, noted here rather than left for a reviewer to point out:

- **Temporal churn signals** — this dataset is a single snapshot; adding transaction-frequency or engagement-trend data over time would let the model catch gradual disengagement, not just a static profile.
- **Cross-sell audit for the 3-4 product segment** — the partial dependence finding (higher product counts correlating with higher churn) points to a specific, checkable business question: were these products opted into or pushed through cross-sell targets. Worth a dedicated follow-up analysis with account-opening records.
- **Cost-sensitive threshold tuning** — right now the model outputs a probability; the next step is translating that into a deployment threshold based on the actual dollar cost of a false positive (wasted retention offer) versus a false negative (lost customer).
- **Geographic expansion** — validating whether the behavioral drivers found here (engagement, product density) hold in banking markets outside France, Spain, and Germany.

## Author

Suhani Khanna — Data Science Intern, Unified Mentor
