"""Generates a styled table image of the model comparison results,
since Medium doesn't render markdown tables reliably (same workaround
used for the Project 1 paper)."""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../outputs/model_comparison.csv")
df = df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
    df[col] = df[col].map(lambda x: f"{x:.3f}")

fig, ax = plt.subplots(figsize=(9, 2.6))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.1)

# Style header row
for col_idx in range(len(df.columns)):
    cell = table[0, col_idx]
    cell.set_facecolor("#1a1a2e")
    cell.set_text_props(color="white", fontweight="bold")

# Highlight the best model's row (first row, already sorted by ROC-AUC)
for col_idx in range(len(df.columns)):
    cell = table[1, col_idx]
    cell.set_facecolor("#FFF3E0")

# Alternate row shading for the rest
for row_idx in range(2, len(df) + 1):
    for col_idx in range(len(df.columns)):
        cell = table[row_idx, col_idx]
        cell.set_facecolor("#F7F7F7" if row_idx % 2 == 0 else "#FFFFFF")

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#DDDDDD")

plt.title("Model Comparison: Bank Customer Churn Prediction", fontsize=13, fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("../outputs/model_comparison_table.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved table image.")
