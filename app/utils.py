"""Shared helpers for the churn dashboard -- kept separate from app.py so the
main file stays readable and the feature-building logic isn't duplicated
across the calculator and simulator tabs."""

import pandas as pd
import numpy as np


def build_feature_row(credit_score, geography, gender, age, tenure, balance,
                       num_products, has_cr_card, is_active, salary, feature_names):
    """Takes raw customer inputs and produces the same engineered feature
    vector the model was trained on. Order must match feature_names exactly."""

    balance_salary_ratio = balance / (salary + 1)
    product_density = num_products / (tenure + 1)
    engagement_product_score = is_active * num_products
    age_tenure_interaction = age * tenure
    is_zero_balance = 1 if balance == 0 else 0

    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    row = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "BalanceSalaryRatio": balance_salary_ratio,
        "ProductDensity": product_density,
        "EngagementProductScore": engagement_product_score,
        "AgeTenureInteraction": age_tenure_interaction,
        "IsZeroBalance": is_zero_balance,
        "Geography_Germany": geo_germany,
        "Geography_Spain": geo_spain,
        "Gender_Male": gender_male,
    }
    return pd.DataFrame([row])[feature_names]


def risk_band(probability):
    """Maps a raw probability to a human-readable risk band with a colour,
    used consistently across the calculator and simulator tabs."""
    if probability < 0.30:
        return "Low", "#2E7D32"
    elif probability < 0.60:
        return "Moderate", "#F9A825"
    else:
        return "High", "#C62828"


def estimate_annual_value(balance, num_products, salary):
    """Rough, clearly-labeled illustrative estimate of a customer's annual
    value to the bank -- this dataset has no real revenue/fee data, so this
    is an assumption-driven proxy (net interest margin on balance + a flat
    per-product fee estimate), not a measured financial figure. Exposed as
    editable assumptions in the UI rather than presented as fact.
    """
    interest_margin_rate = 0.02   # assumed net interest margin on balance
    fee_per_product = 60          # assumed average annual fee per product held
    return balance * interest_margin_rate + num_products * fee_per_product


def batch_score(raw_df, model, feature_names):
    """Takes a raw customer dataframe (same schema as the original dataset,
    minus the target) and runs it through the same feature engineering used
    in training, then scores it. Kept separate from build_feature_row (which
    handles one manually-entered customer) since batch input arrives as a
    full dataframe rather than individual widget values.
    """
    df = raw_df.copy()

    required_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                      "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")

    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementProductScore"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

    df_encoded = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[feature_names]
    probabilities = model.predict_proba(X)[:, 1]

    result = raw_df.copy()
    result["ChurnProbability"] = probabilities
    result["RiskBand"] = result["ChurnProbability"].apply(lambda p: risk_band(p)[0])
    return result
