"""
Bank Customer Churn - Feature Engineering

Brief asks for:
- Balance-to-Salary ratio
- Product density indicator
- Engagement-product interaction
- Age-tenure interaction features

Each one is built with a stated business reason, not just because the brief
lists it -- that reasoning is what goes in the research paper.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("../data/churn_clean.csv")

# 1. Balance-to-Salary ratio
# Rationale: raw balance means little without context. A customer with 50k balance
# on a 40k salary is in a very different financial relationship with the bank than
# one with 50k balance on a 200k salary. Add 1 to salary to avoid div-by-zero edge cases.
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

# 2. Product density indicator
# Rationale: NumOfProducts alone doesn't say whether the customer is "into" the bank
# or just holds accounts passively. Combine with tenure to get products-per-year,
# a rough proxy for how quickly the relationship is deepening.
df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)

# 3. Engagement-product interaction
# Rationale: a customer with many products but who is NOT an active member is a
# classic churn-risk profile (paying for products they no longer use). Multiplying
# captures this combined effect rather than treating the two as independent.
df["EngagementProductScore"] = df["IsActiveMember"] * df["NumOfProducts"]

# 4. Age-tenure interaction
# Rationale: a 60-year-old with 2 years tenure and a 25-year-old with 2 years tenure
# are not the same risk profile -- older customers with short tenure may have
# switched from a competitor recently and could be more likely to switch again.
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

# 5. Zero-balance flag (informed by the EDA finding: 36% of customers carry a
# zero balance -- treating this as a binary signal rather than letting it get
# lost inside a continuous Balance column)
df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

# --- Encoding ---
# One-hot encode Geography and Gender per the brief. drop_first=True avoids the
# dummy variable trap (multicollinearity) for the linear baseline model.
df_encoded = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

print("Final feature set:", list(df_encoded.columns))
print("Shape:", df_encoded.shape)

df_encoded.to_csv("../data/churn_features.csv", index=False)
print("\nSaved engineered dataset.")
