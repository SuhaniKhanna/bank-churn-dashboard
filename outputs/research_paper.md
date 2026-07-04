# Predictive Modeling and Risk Scoring for Bank Customer Churn: A Behavior-Driven Approach to Proactive Retention

**Suhani Khanna**
Data Science Intern, Unified Mentor

---

## Abstract

Most retail banks find out a customer was unhappy after they've already left. This paper builds the opposite: a system that scores every customer's churn risk while they're still a customer, using a dataset of 10,000 retail banking customers across France, Spain, and Germany. I trained and compared four classifiers (Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting) using stratified sampling and class-weighting to handle a 20.4% churn rate. Gradient Boosting came out ahead on ROC-AUC (0.869), though Random Forest caught noticeably more actual churners, which turned into a real tradeoff worth discussing rather than a tiebreaker to ignore. The more interesting result came from SHAP and partial dependence analysis: churn is driven far more by how customers behave (engagement, product usage) than by who they are (age, income), and customers holding three or four products churn at a much higher rate than customers holding two, which runs against the assumption that more products means a safer customer. The model now runs inside an interactive Streamlit dashboard where risk scores, driver explanations, and retention scenarios can all be tested live.

---

## Executive Summary (For Stakeholders)

**The problem in one line:** the bank knows which customers already left. It doesn't know which ones are about to.

**What this project delivers:** a churn probability score for every customer, generated before they act on it, so retention efforts can go where they'll actually matter instead of being spread thin across everyone.

**What we found, in plain terms:**
- Roughly 1 in 5 customers in this dataset churned. That's too large a share to handle with a blanket retention strategy. It needs targeting.
- The strongest churn signal isn't who the customer is (age, income). It's how they behave: engagement level and how many products they actually use. That distinction matters because behavior can be shifted through service and offers. Demographics can't be.
- Here's the part that surprised me: customers holding three or four bank products churn *far* more than customers holding just two. That contradicts the usual assumption that more products equals a stickier customer. My read is that some of these product bundles were sold to customers who never really wanted them, and that's worth the bank actually digging into rather than taking at face value.
- Two models came out of this build, tuned for two different priorities. One (Gradient Boosting) rarely cries wolf. The other (Random Forest) rarely lets an actual churner slip through unnoticed. Picking between them isn't a modeling call, it's a business call: it depends on whether wasted outreach or a lost customer costs the bank more.

**What this enables:** a live dashboard where a relationship manager can pull up any customer, see why the model scored them that way, and test what happens to that score if the customer's engagement or product mix changes.

---

## 1. Introduction and Background

Churn eats directly into a bank's revenue, its Customer Lifetime Value numbers, and its cross-sell pipeline. Yet most retail banks still find out about it too late. Retention campaigns end up reactive and broad rather than targeted, because there's no early signal to target with (Comparative Analysis of ML Models for Customer Churn, 2024). Industry figures put annual churn in banking and financial services somewhere between 25 and 30 percent (Machine Learning Models for Bank Customer Churn Prediction, 2025), a large enough number that even a modest improvement in early detection pays for itself many times over.

That's the gap this project tries to close. Rather than a single churn/no-churn flag, every customer gets a continuous probability score, something that can be ranked, thresholded, or simulated against, and that holds up as a genuinely usable retention tool rather than a one-time report.

## 2. Literature Review

Machine learning approaches to churn prediction in financial services have moved through several generations of methodology. Early work applied Support Vector Machines to large-scale commercial bank data, using resampling techniques to address the class imbalance inherent to churn datasets and reporting meaningfully improved recognition of the minority churn class as a result (He et al., 2014, as cited in Customers Churn Prediction in Financial Institution Using Artificial Neural Network). Subsequent studies broadened the model comparison to ensemble methods: research using a similarly sized 10,000-customer, 21-feature banking dataset compared Random Forest, Linear Regression, and k-Nearest Neighbors approaches for churn forecasting (Bank Customer Churn Prediction Using Various Machine Learning Algorithms, 2024). More recent work has consistently found that ensemble and boosting-based methods (Random Forest, XGBoost, AdaBoost) outperform single-model baselines on accuracy, F-score, and AUC in bank churn contexts, with AUC values commonly reported between 0.90 and 0.93 on comparable datasets (Bank Customer Churn Prediction Using Machine Learning Framework, 2024).

A second and increasingly important thread in this literature concerns interpretability. As financial institutions face growing regulatory pressure to justify automated decisions, SHAP (SHapley Additive exPlanations) has emerged as the dominant post-hoc explanation method for financial machine learning models, valued for its strong alignment between statistical attribution and the documentation standards regulators expect (Lundberg & Lee, 2017; Explainable Graph Neural Networks for Interbank Contagion Surveillance, 2025). This aligns with the standard set by U.S. banking supervisory guidance requiring model interpretability in risk-relevant decisioning (Federal Reserve SR 11-7, 2011). Complementary to SHAP, partial dependence analysis is frequently used to communicate population-level feature effects to non-technical stakeholders, since it describes an average relationship rather than a single prediction's internal logic (What Is AI Interpretability?, IBM).

This project positions itself at the intersection of these two threads: applying an ensemble modeling comparison consistent with current best practice, while pairing it with both instance-level (SHAP) and population-level (partial dependence) explainability, addressing the transparency expectations increasingly standard in financial machine learning deployments.

## 3. Dataset Description

The dataset comprises 10,000 retail banking customers with the following attributes: credit score, geography (France, Spain, Germany), gender, age, tenure, account balance, number of products held, credit card ownership, active membership status, estimated salary, and the binary target variable `Exited` (1 = churned, 0 = retained).

Exploratory analysis found no missing values and no duplicate records. The churn base rate was 20.4%, confirming a moderately imbalanced classification problem. Notably, 36.2% of customers carried a zero account balance, a substantial enough proportion that it was treated as a distinct behavioral signal (dormancy) rather than as a data quality issue, via a dedicated binary feature.

## 4. Methodology

### 4.1 Preprocessing
Non-informative identifier columns (customer ID, surname) were removed. Categorical variables (geography, gender) were one-hot encoded, with the first category dropped to avoid multicollinearity in the linear baseline model.

### 4.2 Feature Engineering
Five features were engineered specifically around customer behavior rather than static demographics:

- **Balance-to-Salary Ratio**: contextualizes raw balance against income, distinguishing a large balance relative to a modest salary from the same balance relative to a high salary.
- **Product Density** (products per year of tenure): approximates how quickly a customer's relationship with the bank is deepening.
- **Engagement-Product Interaction** (active membership × number of products): captures the specific risk profile of a customer holding multiple products while disengaged.
- **Age-Tenure Interaction**: distinguishes long-standing customers from recent switchers of the same age.
- **Zero-Balance Flag**: isolates the dormant-account segment identified during exploratory analysis.

### 4.3 Modeling
Four classifiers were trained across the tiers specified by the project brief: Logistic Regression (interpretability baseline), Decision Tree and Random Forest (tree-based), and Gradient Boosting (advanced ensemble). Data was split 80/20 using stratified sampling to preserve the churn class ratio in both partitions, and validated further using 5-fold stratified cross-validation. Class weighting was applied across all models except Gradient Boosting to directly address the 20.4% minority class rather than relying on resampling.

### 4.4 Explainability
Two complementary techniques were applied to the best-performing model. SHAP values (Lundberg & Lee, 2017) were computed via TreeExplainer to rank global feature importance and support instance-level explanation. Partial dependence plots were generated using the brute-force method against the model's predicted probability output, rather than the default log-odds decision function, to keep results interpretable in stakeholder-facing terms.

## 5. Results

### 5.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.870 | 0.790 | 0.489 | 0.604 | **0.869** |
| Random Forest | 0.826 | 0.555 | **0.715** | 0.625 | 0.868 |
| Decision Tree | 0.757 | 0.443 | 0.767 | 0.562 | 0.823 |
| Logistic Regression | 0.713 | 0.387 | 0.705 | 0.500 | 0.776 |

Gradient Boosting comes out on top by ROC-AUC and is the model this project builds around. But look at recall: Random Forest catches 71.5% of actual churners against Gradient Boosting's 48.9%. That gap is too large to wave away. Gradient Boosting is the more cautious model, with fewer false alarms, while Random Forest casts a wider net and accepts more noise to catch more of the real thing. Whether that's the right tradeoff depends entirely on what a false alarm costs the bank versus what a missed churner costs, and that's a business question, not something the ROC-AUC column can answer on its own.

### 5.2 Feature Importance (SHAP)

Ranked by mean absolute SHAP value, the top five features were: number of products held, age, geography (Germany), the engineered engagement-product interaction score, and gender. The engagement-product feature is worth pausing on. It outranked raw account balance and estimated salary, which means the behavioral framing wasn't just a nice-to-have addition to the feature set. It was doing real predictive work the raw variables weren't capturing on their own.

### 5.3 Partial Dependence

This is where the project's most useful finding showed up. Partial dependence on number of products produced a curve that dips before it spikes: predicted churn probability sits around 0.11 at two products, then jumps to 0.74 at three and 0.86 at four. That's backwards from what you'd expect: more products should mean a deeper, safer relationship with the bank, not a riskier one. My best explanation, and one that shows up in similar form elsewhere in this literature, is that a chunk of these extra products were cross-sold onto customers rather than chosen by them, and an over-extended relationship like that eventually breaks rather than deepens. Age behaved more the way you'd expect: a steady climb from about 0.11 at age 25 to 0.54 by age 56.

[INSERT partial_dependence.png HERE]

## 6. Discussion and Business Implications

I don't think the Gradient Boosting vs. Random Forest choice is something to resolve. It's something to hand to the business with the tradeoff spelled out. If outreach is cheap relative to what a lost customer costs, Random Forest's recall wins. If outreach capacity is limited and false positives carry a real cost (a discount handed to someone who was never leaving), Gradient Boosting is the safer pick.

The product-count finding is the one I'd push hardest on if I were presenting this to the bank. "More products" has probably been treated internally as a retention win for a while. This data says otherwise, at least for the 3-4 product segment, and the next step isn't more modeling. It's someone on the business side pulling records to check whether those products were genuinely opted into or pushed through a cross-sell target.

## 7. Limitations

This is a snapshot, not a timeline: one point-in-time view of each customer, not their history. That means the model can't tell a slow fade from a sudden trigger, and it would benefit a lot from transaction-level detail (frequency, complaints, whether a competitor's rate undercut this bank) that just isn't in this dataset. It's also worth being upfront that this covers three European markets (France, Spain, Germany), and nothing here should be assumed to hold in a market this dataset doesn't cover.

## 8. Conclusion

The headline result isn't the ROC-AUC number, even though that's the one that looks good on a slide. It's that churn in this dataset is explained more by what customers *do* than who they *are*, and that one specific behavioral pattern, the product-count spike, runs against what the bank probably assumes about its own customers. Gradient Boosting is the model going into production, but the dashboard built around it is meant to make both of these findings usable day to day: a risk score, the reason behind it, and a way to test what happens if that customer's situation changes.

---

## References

Bank Customer Churn Prediction Using Machine Learning Framework. (2024). *IDEAS/RePEc*. https://ideas.repec.org/a/spt/apfiba/v14y2024i4f14_4_5.html

Bank Customer Churn Prediction Using Various Machine Learning Algorithms. (2024). *Academy of Accounting and Financial Studies Journal*. https://www.abacademies.org/articles/bank-customer-churn-prediction-using-various-machine-learning-algorithms-16912.html

Comparative Analysis of Machine Learning Models for Customer Churn Prediction in the U.S. Banking and Financial Services: Economic Impact and Industry-Specific Insights. (2024). *Scientific Research Publishing*. https://www.scirp.org/journal/paperinformation?paperid=134563

Customers Churn Prediction in Financial Institution Using Artificial Neural Network. (2019). *arXiv*. https://arxiv.org/pdf/1912.11346

Explainable Graph Neural Networks for Interbank Contagion Surveillance: A Regulatory-Aligned Framework for the U.S. Banking Sector. (2025). *arXiv*. https://arxiv.org/pdf/2604.14232

Federal Reserve. (2011). *SR 11-7: Guidance on Model Risk Management*. Board of Governors of the Federal Reserve System.

IBM. (n.d.). *What is AI interpretability?* https://www.ibm.com/think/topics/interpretability

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

Machine Learning Models for Bank Customer Churn Prediction: A Comparative Study of LightGBM, CatBoost, and XGBoost. (2025). *Proceedings of the 2025 International Conference on Big Data, Artificial Intelligence and Digital Economy*. https://dl.acm.org/doi/10.1145/3767052.3767054

---

## Project Links

**Live dashboard:** https://bank-churn-dashboard-bgbqk8osczkzbrpvrbvqjt.streamlit.app/
**GitHub repository:** https://github.com/SuhaniKhanna/bank-churn-dashboard
