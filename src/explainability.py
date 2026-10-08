import pandas as pd, shap, pickle
from sklearn.ensemble import RandomForestClassifier
import os

df = pd.read_csv("data/processed/customers_scored.csv")
X = pd.get_dummies(df[['tenure_months','monthly_spend','support_tickets_90d','usage_days_30d','contract_type','payment_delay','nps_score']], drop_first=True)

# Train small model for SHAP speed
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X, df['churn_30d'])

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global SHAP importance
shap_importance = pd.DataFrame({
  'feature': X.columns,
  'mean_shap': abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(axis=0)
}).sort_values('mean_shap', ascending=False)

shap_importance.to_csv("data/processed/shap_importance.csv", index=False)
print("Top churn reasons (SHAP):")
print(shap_importance.head(8))

# Save sample explanations for high-risk customers
high_risk = df[df.churn_prob>0.7].head(5)
print("\nHigh-risk sample:")
print(high_risk[['customer_id','churn_prob','clv_ngn','support_tickets_90d','usage_days_30d','nps_score']])
