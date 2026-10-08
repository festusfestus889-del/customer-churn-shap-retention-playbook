# Customer Churn Prediction with SHAP Explainability & Retention Playbooks

**Problem:** 28% churn rate → need to know who, why, and what to do.

**Pipeline:**
1. Generated 6000 customers (tenure, spend, tickets, usage, NPS, CLV)
2. RandomForest predicts churn 30d (AUC 0.88)
3. SHAP explainability: Top drivers = support_tickets, low usage, payment_delay, low NPS
4. Segmented High-Value High-Risk → Playbooks: VIP call, re-engagement, service recovery
5. ROI: Saving 40% of HV-HR → Net ₦18M savings, ROI 12x

**Stack:** Python, Scikit-learn, SHAP, Streamlit, Plotly, GitHub Actions

**Run:** streamlit run app.py
