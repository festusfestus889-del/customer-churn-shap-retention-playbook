import pandas as pd

df = pd.read_csv("data/processed/customers_scored.csv")
shap_imp = pd.read_csv("data/processed/shap_importance.csv")

# Segment high-risk
df['risk_segment'] = pd.cut(df['churn_prob'], bins=[0,0.3,0.6,1.0], labels=['Low','Medium','High'])
df['value_segment'] = pd.cut(df['clv_ngn'], bins=[0,50000,150000,1000000], labels=['Low-Value','Mid-Value','High-Value'])

playbook = []

# Rule-based playbook from SHAP drivers
for idx, row in df[df.risk_segment=='High'].iterrows():
  reasons = []
  if row['support_tickets_90d']>=3: reasons.append("high support tickets")
  if row['usage_days_30d']<=10: reasons.append("low usage")
  if row['nps_score']<=5: reasons.append("low NPS")
  if row['payment_delay']: reasons.append("payment delay")

  if row['clv_ngn']>150000:
    action = "VIP: Personal call + 20% discount + success manager"
    expected_save_rate = 0.45
  elif "low usage" in reasons:
    action = "Re-engagement: Tutorial + feature tips + check-in"
    expected_save_rate = 0.35
  elif "high support tickets" in reasons:
    action = "Service recovery: Apologize + fix + free month"
    expected_save_rate = 0.4
  else:
    action = "Standard: Discount + payment plan"
    expected_save_rate = 0.25

  playbook.append({
    "customer_id": row['customer_id'],
    "churn_prob": row['churn_prob'],
    "clv": row['clv_ngn'],
    "reasons": ", ".join(reasons) if reasons else "contract/spend",
    "action": action,
    "expected_save_rate": expected_save_rate
  })

pb_df = pd.DataFrame(playbook)
pb_df.to_csv("data/processed/retention_playbook.csv", index=False)

# ROI calculation
high_risk = df[df.risk_segment=='High']
high_value_high_risk = high_risk[high_risk.value_segment=='High-Value']
avg_clv_hv = high_value_high_risk['clv_ngn'].mean()
prevented = len(high_value_high_risk)*0.4 # save 40%
gross_savings = prevented*avg_clv_hv
cost = len(high_value_high_risk)*2000 # intervention cost

roi_df = pd.DataFrame([{
  'segment': 'High-Value High-Risk',
  'customers': len(high_value_high_risk),
  'avg_clv': int(avg_clv_hv),
  'prevented_churn': int(prevented),
  'gross_savings_ngn': int(gross_savings),
  'intervention_cost': int(cost),
  'net_savings': int(gross_savings-cost),
  'roi': round((gross_savings-cost)/cost,2) if cost>0 else 0
}])

roi_df.to_csv("data/processed/roi.csv", index=False)
print(roi_df.to_string(index=False))
print(f"\nPlaybook generated for {len(pb_df)} high-risk customers")
