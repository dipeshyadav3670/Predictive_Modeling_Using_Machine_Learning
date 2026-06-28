"""
ML Dataset Generator
====================
Simulates a realistic telecom customer churn dataset.
Target: churn (0 = stayed, 1 = churned) — binary classification
Also produces a continuous target (monthly_spend) for regression demo.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

# ── Core features ──────────────────────────────────────────────────────────────
tenure          = np.random.exponential(scale=30, size=n).clip(1, 72).astype(int)
monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
num_products    = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.10])
support_calls   = np.random.poisson(lam=1.5, size=n).clip(0, 10)
contract_type   = np.random.choice(["Month-to-Month", "One Year", "Two Year"],
                                   n, p=[0.55, 0.25, 0.20])
internet_service= np.random.choice(["Fiber", "DSL", "None"],
                                   n, p=[0.45, 0.35, 0.20])
has_tech_support= np.random.choice([0, 1], n, p=[0.45, 0.55])
payment_method  = np.random.choice(
    ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
    n, p=[0.35, 0.22, 0.22, 0.21])
paperless_billing = np.random.choice([0, 1], n, p=[0.4, 0.6])
age             = np.random.randint(18, 80, n)
avg_monthly_gb  = np.round(np.random.gamma(3, 5, n), 1).clip(0.5, 80)

# ── Churn probability (logistic-style) ────────────────────────────────────────
contract_penalty = np.where(contract_type == "Month-to-Month", 0.4,
                   np.where(contract_type == "One Year", -0.1, -0.4))
churn_score = (
    -2.5
    - 0.03  * tenure
    + 0.012 * monthly_charges
    - 0.25  * num_products
    + 0.35  * support_calls
    + contract_penalty
    + 0.20  * (internet_service == "Fiber").astype(int)
    - 0.15  * has_tech_support
    + 0.10  * paperless_billing
    + np.random.normal(0, 0.3, n)   # noise
)
churn_prob   = 1 / (1 + np.exp(-churn_score))
churn        = (np.random.uniform(size=n) < churn_prob).astype(int)

# ── Continuous target: monthly_spend ──────────────────────────────────────────
monthly_spend = np.round(
    monthly_charges * num_products * 0.9
    + avg_monthly_gb * 0.5
    + np.random.normal(0, 8, n), 2
).clip(15, 600)

df = pd.DataFrame({
    "customer_id":       [f"C{str(i).zfill(5)}" for i in range(1, n+1)],
    "age":               age,
    "tenure_months":     tenure,
    "num_products":      num_products,
    "monthly_charges":   monthly_charges,
    "avg_monthly_gb":    avg_monthly_gb,
    "support_calls":     support_calls,
    "has_tech_support":  has_tech_support,
    "paperless_billing": paperless_billing,
    "contract_type":     contract_type,
    "internet_service":  internet_service,
    "payment_method":    payment_method,
    "monthly_spend":     monthly_spend,
    "churn":             churn,
})

df.to_csv("/home/claude/churn_dataset.csv", index=False)
print(f"Dataset saved: {df.shape}")
print(f"Churn rate   : {churn.mean()*100:.1f}%")
print(df.dtypes)
