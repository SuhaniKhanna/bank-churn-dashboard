"""
Bank Customer Churn - Model Explainability

SHAP values answer "why did the model predict this?" per customer, not just
"which features matter overall" -- this is the regulatory/business-trust
piece the brief calls out explicitly.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("../data/churn_features.csv")
X = df.drop(columns=["Exited"])
y = df["Exited"]

model = joblib.load("../models/gradient_boosting.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

# Use a sample for SHAP -- full 10k rows is unnecessary compute for a stable
# global importance picture, and TreeExplainer is exact for tree models anyway.
sample = X.sample(1000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Global feature importance (mean absolute SHAP value)
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print(importance.to_string(index=False))
importance.to_csv("../outputs/shap_importance.csv", index=False)

# Save a summary plot for the paper
plt.figure()
shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("../outputs/shap_summary.png", dpi=150, bbox_inches="tight")
print("\nSaved SHAP summary plot.")

# Also save native sklearn feature importance for comparison / the Streamlit dashboard
sk_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
sk_importance.to_csv("../outputs/sklearn_importance.csv", index=False)
