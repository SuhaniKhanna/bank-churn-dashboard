"""
Bank Customer Churn - EDA & Preprocessing
Unified Mentor | Predictive Modeling and Risk Scoring for Bank Customer Churn

Goal: understand the shape of the data, check quality issues, and clean it
before any feature engineering happens.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("../data/European_Bank.csv")

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nChurn distribution:\n", df["Exited"].value_counts(normalize=True))

# --- Non-informative columns ---
# CustomerId and Surname carry no predictive signal (pure identifiers).
# 'Year' is constant across the dataset (single snapshot) so it adds nothing either.
print("\nYear unique values:", df["Year"].unique())

drop_cols = ["CustomerId", "Surname", "Year"]
df_clean = df.drop(columns=drop_cols)

# --- Sanity checks on ranges ---
print("\nAge range:", df_clean["Age"].min(), "-", df_clean["Age"].max())
print("CreditScore range:", df_clean["CreditScore"].min(), "-", df_clean["CreditScore"].max())
print("Balance == 0 count:", (df_clean["Balance"] == 0).sum(), f"({(df_clean['Balance']==0).mean()*100:.1f}%)")
print("Geography values:", df_clean["Geography"].unique())
print("Gender values:", df_clean["Gender"].unique())

df_clean.to_csv("../data/churn_clean.csv", index=False)
print("\nSaved cleaned data:", df_clean.shape)
