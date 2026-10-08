import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

df = pd.read_csv("data/processed/customers_scored.csv")
X = pd.get_dummies(df[['tenure_months','monthly_spend','support_tickets_90d','usage_days_30d','contract_type','payment_delay','nps_score']], drop_first=True)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X, df['churn_30d'])

# SHAP-like importance using feature importance (explainable, no heavy lib)
importance = pd.DataFrame({
  'feature': X.columns,
  'mean_shap': model.feature_importances_
}).sort_values('mean_shap', ascending=False)

importance.to_csv("data/processed/shap_importance.csv", index=False)
print("Top churn reasons (SHAP-style importance):")
print(importance.head(8))

high_risk = df[df.churn_prob>0.7].head(5)
print("\nHigh-risk sample:")
print(high_risk[['customer_id','churn_prob','clv_ngn','support_tickets_90d','usage_days_30d','nps_score']])
