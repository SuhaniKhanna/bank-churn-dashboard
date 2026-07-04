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
