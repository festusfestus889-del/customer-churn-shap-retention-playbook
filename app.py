import streamlit as st, pandas as pd, plotly.express as px
st.set_page_config(layout="wide")
st.title("📉 Churn Prediction + SHAP Explainability + Retention Playbook")

df = pd.read_csv("data/processed/customers_scored.csv")
shap_df = pd.read_csv("data/processed/shap_importance.csv")
roi = pd.read_csv("data/processed/roi.csv")
playbook = pd.read_csv("data/processed/retention_playbook.csv")

c1,c2,c3 = st.columns(3)
c1.metric("Churn Rate", f"{df['churn_30d'].mean()*100:.1f}%")
c2.metric("High-Risk", f"{(df['churn_prob']>0.6).sum()}")
c3.metric("Net Savings", f"₦{roi['net_savings'].values[0]:,}")

st.plotly_chart(px.bar(shap_df.head(8), x='mean_shap', y='feature', orientation='h', title="Why Customers Churn - SHAP Importance"), use_container_width=True)
st.plotly_chart(px.histogram(df, x='churn_prob', color='contract_type', title="Churn Risk Distribution"), use_container_width=True)
st.plotly_chart(px.scatter(df, x='usage_days_30d', y='nps_score', color='churn_prob', size='clv_ngn', title="Usage vs NPS (color=churn prob, size=CLV)"), use_container_width=True)

st.subheader("Retention Playbook - Top 20 High-Value High-Risk")
st.dataframe(playbook.head(20))
st.dataframe(roi)
