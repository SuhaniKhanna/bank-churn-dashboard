"""Generates an original cover/banner image for the Medium article.
Abstract customer-network visualization: most customers in neutral tones,
a subset highlighted in red representing the high-risk segment the model
identifies, plus a rising risk curve motif. No stock photography, no
third-party assets, entirely generated for this project.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(11)

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor("#0d1b2a")
ax.set_facecolor("#0d1b2a")

# Scatter of "customers" -- most retained (steel blue), ~20% churned (red),
# matching the actual 20.4% churn rate found in the data.
n_points = 220
x = np.random.uniform(0, 16, n_points)
y = np.random.uniform(0, 8, n_points)
is_churned = np.random.rand(n_points) < 0.204

sizes = np.random.uniform(40, 160, n_points)

ax.scatter(x[~is_churned], y[~is_churned], s=sizes[~is_churned],
           color="#4A6FA5", alpha=0.55, edgecolors="none", zorder=2)
ax.scatter(x[is_churned], y[is_churned], s=sizes[is_churned] * 1.3,
           color="#E63946", alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)

# Subtle rising curve motif suggesting escalating risk, drawn behind the points
curve_x = np.linspace(0, 16, 200)
curve_y = 1.2 + 0.02 * curve_x**1.7
ax.plot(curve_x, curve_y, color="#F4A261", linewidth=2.5, alpha=0.6, zorder=1)

ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-0.5, 8.8)
ax.axis("off")

# Title text, baked into the image for feed/thumbnail recognizability
fig.text(0.06, 0.80, "Predictive Churn Risk Intelligence",
          fontsize=34, fontweight="bold", color="white", family="DejaVu Serif", ha="left")
fig.text(0.06, 0.70, "Modeling and Risk Scoring for Bank Customer Churn",
          fontsize=17, color="#B8C4D4", family="DejaVu Serif", ha="left", style="italic")

plt.tight_layout(pad=0)
plt.savefig("../outputs/medium_cover.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1b2a", pad_inches=0.3)
print("Saved cover image.")
