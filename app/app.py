"""
Predictive Modeling and Risk Scoring for Bank Customer Churn
Unified Mentor | Data Science Internship Project

Author: Suhani Khanna
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from utils import build_feature_row, risk_band

st.set_page_config(
    page_title="Bank Churn Risk Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
# Custom CSS to move away from Streamlit's default look -- flat colour cards,
# tighter spacing, a single accent colour rather than the default rainbow.
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-label { font-size: 13px; color: #666666; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 28px; font-weight: 600; color: #1a1a1a; }
    h1, h2, h3 { font-family: 'Georgia', serif; color: #1a1a2e; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
    .risk-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        color: white; font-weight: 600; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, "assets/gradient_boosting.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "assets/feature_names.pkl"))
    importance = pd.read_csv(os.path.join(BASE_DIR, "assets/sklearn_importance.csv"))
    comparison = pd.read_csv(os.path.join(BASE_DIR, "assets/model_comparison.csv"))
    sample_data = pd.read_csv(os.path.join(BASE_DIR, "assets/churn_sample.csv"))
    return model, feature_names, importance, comparison, sample_data


model, feature_names, importance_df, comparison_df, sample_df = load_assets()

# Pre-compute probabilities on the sample for the distribution tab
X_sample = sample_df[feature_names]
sample_df = sample_df.copy()
sample_df["ChurnProbability"] = model.predict_proba(X_sample)[:, 1]

# ---------- Header ----------
st.title("Bank Customer Churn Risk Intelligence")
st.caption("Predictive churn scoring for retail banking customers — built on customer engagement and product-usage signals rather than demographics alone.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write(
        "This dashboard predicts customer churn probability using a Gradient "
        "Boosting model trained on 10,000 retail banking customers across "
        "France, Spain, and Germany."
    )
    st.metric("Model ROC-AUC", f"{comparison_df.iloc[0]['ROC-AUC']:.3f}")
    st.metric("Dataset size", f"{len(sample_df):,} customers")
    st.divider()
    st.caption("Unified Mentor Data Science Internship — Project 2")

# ---------- KPI row ----------
col1, col2, col3, col4 = st.columns(4)
churn_rate = sample_df["Exited"].mean()
avg_risk_score = sample_df["ChurnProbability"].mean()
high_risk_count = (sample_df["ChurnProbability"] >= 0.6).sum()
active_pct = sample_df["IsActiveMember"].mean()

with col1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Actual Churn Rate</div>
    <div class="metric-value">{churn_rate:.1%}</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg. Predicted Risk</div>
    <div class="metric-value">{avg_risk_score:.1%}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">High-Risk Customers</div>
    <div class="metric-value">{high_risk_count:,}</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Active Members</div>
    <div class="metric-value">{active_pct:.1%}</div></div>""", unsafe_allow_html=True)

st.write("")

# ---------- Tabs: the 4 required modules ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "Churn Risk Calculator",
    "Probability Distribution",
    "Feature Importance",
    "What-If Simulator",
])

# ===================== TAB 1: Churn Risk Calculator =====================
with tab1:
    st.subheader("Individual Customer Risk Calculator")
    st.write("Enter a customer's profile to get a live churn probability from the trained model.")

    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 350, 850, 650, key="calc_cs")
        age = st.slider("Age", 18, 92, 40, key="calc_age")
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5, key="calc_tenure")
    with c2:
        balance = st.number_input("Account Balance", 0.0, 250000.0, 60000.0, step=1000.0, key="calc_bal")
        st.caption(f"${balance:,.2f}")
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0, step=1000.0, key="calc_sal")
        st.caption(f"${salary:,.2f}")
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1, key="calc_prod")
    with c3:
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"], key="calc_geo")
        gender = st.selectbox("Gender", ["Male", "Female"], key="calc_gender")
        has_cr_card = st.radio("Has Credit Card", ["Yes", "No"], horizontal=True, key="calc_card")
        is_active = st.radio("Active Member", ["Yes", "No"], horizontal=True, key="calc_active")

    with st.spinner("Calculating churn probability..."):
        row = build_feature_row(
            credit_score, geography, gender, age, tenure, balance, num_products,
            1 if has_cr_card == "Yes" else 0, 1 if is_active == "Yes" else 0,
            salary, feature_names
        )
        proba = model.predict_proba(row)[0, 1]
    band, colour = risk_band(proba)

    st.write("")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Probability</div>
            <div class="metric-value">{proba:.1%}</div>
            <span class="risk-badge" style="background-color:{colour};">{band} Risk</span>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colour},
                'steps': [
                    {'range': [0, 30], 'color': '#E8F5E9'},
                    {'range': [30, 60], 'color': '#FFF8E1'},
                    {'range': [60, 100], 'color': '#FFEBEE'},
                ],
            },
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 2: Probability Distribution =====================
with tab2:
    st.subheader("Churn Probability Distribution Across the Customer Base")
    st.write("How predicted risk is spread across all customers, and where it concentrates by segment.")

    fig_dist = px.histogram(
        sample_df, x="ChurnProbability", nbins=40, color="Exited",
        color_discrete_map={0: "#90A4AE", 1: "#C62828"},
        labels={"ChurnProbability": "Predicted Churn Probability", "Exited": "Actually Churned"},
        opacity=0.75,
    )
    fig_dist.update_layout(barmode="overlay", height=380)
    st.plotly_chart(fig_dist, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Note: can't use idxmax here -- when both dummy columns are 0 (France),
        # idxmax ties and silently picks the first column, mislabeling France
        # rows as Germany. np.select handles the "neither" case explicitly.
        geo_labels = np.select(
            [sample_df["Geography_Germany"] == 1, sample_df["Geography_Spain"] == 1],
            ["Germany", "Spain"],
            default="France",
        )
        geo_risk = sample_df.assign(Geography=geo_labels).groupby("Geography")["ChurnProbability"].mean().reset_index()
        geo_risk.columns = ["Geography", "AvgRisk"]
        fig_geo = px.bar(geo_risk, x="Geography", y="AvgRisk",
                          title="Average Predicted Risk by Geography",
                          color="AvgRisk", color_continuous_scale="Reds")
        fig_geo.update_layout(height=340, yaxis_tickformat=".0%")
        st.plotly_chart(fig_geo, use_container_width=True)
    with c2:
        prod_risk = sample_df.groupby("NumOfProducts")["ChurnProbability"].mean().reset_index()
        fig_prod = px.bar(prod_risk, x="NumOfProducts", y="ChurnProbability",
                           title="Average Predicted Risk by Number of Products",
                           color="ChurnProbability", color_continuous_scale="Reds")
        fig_prod.update_layout(height=340, yaxis_tickformat=".0%")
        st.plotly_chart(fig_prod, use_container_width=True)

# ===================== TAB 3: Feature Importance =====================
with tab3:
    st.subheader("What Drives the Model's Predictions")
    st.write("Feature importance from the trained Gradient Boosting model — ranked by contribution to churn prediction.")

    top_n = st.slider("Number of features to show", 5, len(importance_df), 10, key="imp_n")
    plot_df = importance_df.head(top_n).sort_values("importance")

    fig_imp = px.bar(
        plot_df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Blues",
        labels={"importance": "Relative Importance", "feature": ""},
    )
    fig_imp.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.caption(
        "Number of products and customer age are the strongest predictors, followed by "
        "geography and the engagement-product interaction feature — confirming that "
        "relationship depth and behaviour matter more than raw demographics alone."
    )

    with st.expander("Model comparison across all trained algorithms"):
        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
                "F1": "{:.3f}", "ROC-AUC": "{:.3f}", "CV ROC-AUC (5-fold)": "{:.3f}"
            }),
            use_container_width=True, hide_index=True
        )

# ===================== TAB 4: What-If Simulator =====================
with tab4:
    st.subheader("What-If Scenario Simulator")
    st.write(
        "Start from a baseline customer and adjust engagement/product variables to see "
        "how churn risk shifts in real time — useful for testing retention-offer scenarios."
    )

    base = sample_df.sample(1, random_state=7).iloc[0]

    st.markdown("**Baseline customer profile** (randomly selected from the dataset)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Age", int(base["Age"]))
    b2.metric("Tenure", int(base["Tenure"]))
    b3.metric("Products", int(base["NumOfProducts"]))
    b4.metric("Baseline Risk", f"{base['ChurnProbability']:.1%}")

    st.write("")
    st.markdown("**Adjust engagement & product variables:**")
    s1, s2 = st.columns(2)
    with s1:
        sim_products = st.slider("Number of Products", 1, 4, int(base["NumOfProducts"]), key="sim_prod")
        sim_active = st.radio("Active Member", ["Yes", "No"],
                               index=0 if base["IsActiveMember"] == 1 else 1,
                               horizontal=True, key="sim_active")
    with s2:
        sim_balance = st.slider("Account Balance", 0.0, 250000.0, float(base["Balance"]), step=5000.0, key="sim_bal")
        st.caption(f"${sim_balance:,.2f}")
        sim_card = st.radio("Has Credit Card", ["Yes", "No"],
                             index=0 if base["HasCrCard"] == 1 else 1,
                             horizontal=True, key="sim_card")

    geo = "Germany" if base["Geography_Germany"] == 1 else ("Spain" if base["Geography_Spain"] == 1 else "France")
    gender_val = "Male" if base["Gender_Male"] == 1 else "Female"

    with st.spinner("Recalculating scenario..."):
        sim_row = build_feature_row(
            base["CreditScore"], geo, gender_val, base["Age"], base["Tenure"],
            sim_balance, sim_products, 1 if sim_card == "Yes" else 0,
            1 if sim_active == "Yes" else 0, base["EstimatedSalary"], feature_names
        )
        sim_proba = model.predict_proba(sim_row)[0, 1]
    delta = sim_proba - base["ChurnProbability"]

    st.write("")
    d1, d2 = st.columns(2)
    with d1:
        st.metric("Simulated Churn Risk", f"{sim_proba:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
    with d2:
        band, colour = risk_band(sim_proba)
        st.markdown(f'<span class="risk-badge" style="background-color:{colour};">{band} Risk</span>',
                    unsafe_allow_html=True)

    if delta < -0.02:
        st.success("This scenario meaningfully lowers churn risk — a good candidate for a retention offer.")
    elif delta > 0.02:
        st.warning("This scenario increases churn risk relative to the baseline.")
    else:
        st.info("This scenario has minimal impact on churn risk.")
