"""
Bank Customer Churn - Model Development & Evaluation

Trains 4 models per the brief's tiers:
  Baseline:    Logistic Regression
  Tree-based:  Decision Tree, Random Forest
  Advanced:    Gradient Boosting

Stratified split preserves the 80/20 churn ratio in both train and test sets --
without this, a random split could accidentally under/over-represent churners
in the test set and give misleading metrics.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

df = pd.read_csv("../data/churn_features.csv")

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train churn rate:", y_train.mean().round(3))
print("Test churn rate:", y_test.mean().round(3))

# Scale for Logistic Regression only -- tree-based models don't need it since
# they split on thresholds, not distances, and scaling would just add noise
# to their feature importance interpretation.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}
# class_weight="balanced" tells LogReg/DT/RF to weight the minority (churn) class
# more heavily during training -- directly addresses the 80/20 imbalance instead
# of letting the model get away with ignoring churners.

results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    Xtr = X_train_scaled if name == "Logistic Regression" else X_train
    Xte = X_test_scaled if name == "Logistic Regression" else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    cv_auc = cross_val_score(model, Xtr, y_train, cv=skf, scoring="roc_auc").mean()

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV ROC-AUC (5-fold)": cv_auc,
    }
    results.append(row)
    joblib.dump(model, f"../models/{name.replace(' ', '_').lower()}.pkl")

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n" + results_df.to_string(index=False))

results_df.to_csv("../outputs/model_comparison.csv", index=False)
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(list(X.columns), "../models/feature_names.pkl")

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model by ROC-AUC: {best_model_name}")

# Detailed report for the best model
best_model = models[best_model_name]
Xte = X_test_scaled if best_model_name == "Logistic Regression" else X_test
print("\nConfusion matrix:\n", confusion_matrix(y_test, best_model.predict(Xte)))
print("\nClassification report:\n", classification_report(y_test, best_model.predict(Xte)))
