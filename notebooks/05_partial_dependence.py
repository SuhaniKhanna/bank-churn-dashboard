"""
Bank Customer Churn - Partial Dependence Plots

PDPs show the model's average predicted churn probability as ONE feature
varies, holding all others at their observed values. This complements SHAP
(which explains individual predictions) with a population-level view of
"what happens, on average, as this variable moves" -- the framing regulators
and business stakeholders actually want.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

df = pd.read_csv("../data/churn_features.csv")
X = df.drop(columns=["Exited"]).astype(float)
feature_names = joblib.load("../models/feature_names.pkl")
model = joblib.load("../models/gradient_boosting.pkl")

# Focus on the features that mattered most in the SHAP analysis, plus one
# engineered feature, rather than plotting all 16 -- keeps the paper/summary
# readable and ties directly back to the explainability findings.
features_to_plot = ["NumOfProducts", "Age", "IsActiveMember", "EngagementProductScore"]

fig, ax = plt.subplots(figsize=(11, 8))
PartialDependenceDisplay.from_estimator(
    model, X, features_to_plot, ax=ax, n_cols=2,
    kind="average", grid_resolution=30, response_method="predict_proba", method="brute",
)
fig.suptitle("Partial Dependence: Effect of Key Drivers on Predicted Churn Probability", y=1.02)
plt.tight_layout()
plt.savefig("../outputs/partial_dependence.png", dpi=150, bbox_inches="tight")
print("Saved partial dependence plot.")

# Also print the raw average-probability-by-value table for NumOfProducts and
# Age specifically -- these are the two easiest to describe in plain language
# for the executive summary.
for feat in ["NumOfProducts", "Age"]:
    disp = PartialDependenceDisplay.from_estimator(
        model, X, [feat], kind="average", grid_resolution=10,
        response_method="predict_proba", method="brute"
    )
    values = disp.pd_results[0]["grid_values"][0]
    avg_pred = disp.pd_results[0]["average"][0]
    print(f"\n{feat}:")
    for v, p in zip(values, avg_pred):
        print(f"  {v:.1f} -> avg predicted churn probability {p:.3f}")
    plt.close("all")
