import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os
os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/customers.csv")
X = pd.get_dummies(df[['tenure_months','monthly_spend','support_tickets_90d','usage_days_30d','contract_type','payment_delay','nps_score']], drop_first=True)
y = df['churn_30d'].astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced').fit(X_train,y_train)

print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")
print(classification_report(y_test, model.predict(X_test)))

# Save risk
df['churn_prob'] = model.predict_proba(X)[:,1]
df.to_csv("data/processed/customers_scored.csv", index=False)

# Feature importance
imp = pd.DataFrame({'feature':X.columns,'importance':model.feature_importances_}).sort_values('importance', ascending=False)
imp.to_csv("data/processed/feature_importance.csv", index=False)
print(imp)
