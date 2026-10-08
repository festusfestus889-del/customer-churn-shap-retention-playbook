import pandas as pd, numpy as np, os, random
os.makedirs("data/raw", exist_ok=True)
np.random.seed(42)

rows=[]
for i in range(6000):
  tenure = np.random.exponential(12) # months
  monthly_spend = np.random.normal(8000, 2500)
  monthly_spend = max(1000, monthly_spend)
  support_tickets = np.random.poisson(1 + (0 if tenure>12 else 1.5))
  usage_days = int(np.random.normal(20, 8))
  usage_days = max(0, min(30, usage_days))
  contract = random.choice(["monthly","annual"])
  payment_delay = random.random()<0.2
  nps = np.random.normal(7, 2)

  # churn logic
  risk = 0
  risk += -0.05*tenure
  risk += 0.3*support_tickets
  risk += -0.1*usage_days
  risk += 0.8 if payment_delay else 0
  risk += -0.3*nps
  risk += 0.6 if contract=="monthly" else 0
  risk += 0.5 if monthly_spend>10000 else 0

  prob = 1/(1+np.exp(-risk+0.5))
  churn_30d = random.random() < prob
  clv = monthly_spend * tenure * 0.8

  rows.append({
    "customer_id": f"C_{i}",
    "tenure_months": round(tenure,1),
    "monthly_spend": int(monthly_spend),
    "support_tickets_90d": support_tickets,
    "usage_days_30d": usage_days,
    "contract_type": contract,
    "payment_delay": payment_delay,
    "nps_score": round(max(0,min(10,nps)),1),
    "clv_ngn": int(clv),
    "churn_30d": churn_30d
  })

pd.DataFrame(rows).to_csv("data/raw/customers.csv", index=False)
df=pd.DataFrame(rows)
print(f"Generated {len(df)} | Churn rate: {df['churn_30d'].mean()*100:.1f}% | Avg CLV: ₦{df['clv_ngn'].mean():,.0f}")
