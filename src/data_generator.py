import os
import sqlite3
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta

def generate_production_data(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["modeling"]["random_state"])
    n_cust = cfg["data_generation"]["num_customers"]
    start_date = datetime.strptime(cfg["data_generation"]["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(cfg["data_generation"]["end_date"], "%Y-%m-%d")

    db_dir = os.path.dirname(cfg["database"]["db_path"])
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    conn = sqlite3.connect(cfg["database"]["db_path"])
    cursor = conn.cursor()

    with open("schema.sql", "r") as f:
        cursor.executescript(f.read())

    # 1. Customers
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_cust + 1)]
    signup_dates = [start_date + timedelta(days=int(np.random.randint(0, 365))) for _ in range(n_cust)]
    segments = np.random.choice(["Retail", "Wealth", "SMB"], size=n_cust, p=[0.65, 0.15, 0.20])
    channels = np.random.choice(["Organic", "Paid Search", "Referral", "Branch"], size=n_cust)
    credit_scores = np.random.normal(700, 50, n_cust).astype(int)

    cust_df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
        "segment": segments,
        "acquisition_channel": channels,
        "initial_credit_score": credit_scores
    })
    cust_df.to_sql("customers", conn, if_exists="append", index=False)

    # 2. Activity & Transactions
    base_activity_rate = np.random.beta(2, 5, size=n_cust)
    tx_records, act_records = [], []
    tx_id, act_id = 1, 1
    categories = ["Groceries", "Utilities", "Travel", "Entertainment", "Transfer"]

    for idx, cid in enumerate(customer_ids):
        c_signup = signup_dates[idx]
        rate = base_activity_rate[idx]
        curr_dt = c_signup
        
        while curr_dt <= end_date:
            days_since_signup = (curr_dt - c_signup).days
            decay = np.exp(-0.002 * days_since_signup)
            
            if np.random.rand() < (rate * decay):
                act_records.append((
                    f"ACT_{act_id:08d}", cid, curr_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    round(float(np.random.gamma(2, 10)), 2),
                    int(np.random.poisson(5)),
                    np.random.choice(["Mobile", "Web", "Tablet"])
                ))
                act_id += 1

                if np.random.rand() < 0.7:
                    amount = round(float(np.random.lognormal(3.5, 0.8)), 2)
                    tx_records.append((
                        f"TX_{tx_id:08d}", cid, curr_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        amount, np.random.choice(categories)
                    ))
                    tx_id += 1
            curr_dt += timedelta(days=np.random.randint(1, 4))

    pd.DataFrame(tx_records, columns=["transaction_id", "customer_id", "transaction_date", "amount", "merchant_category"]).to_sql("transactions", conn, if_exists="append", index=False)
    pd.DataFrame(act_records, columns=["activity_id", "customer_id", "activity_date", "session_duration_mins", "actions_count", "device_type"]).to_sql("user_activity", conn, if_exists="append", index=False)

    # 3. A/B Experiment Data
    variants = np.random.choice(["control", "treatment"], size=n_cust, p=[0.5, 0.5])
    exp_records = [(cid, v, "2025-07-01", round(float(np.random.gamma(3, 50)), 2)) for cid, v in zip(customer_ids, variants)]
    pd.DataFrame(exp_records, columns=["customer_id", "variant", "exposure_date", "pre_period_avg_spend"]).to_sql("experiment_exposure", conn, if_exists="append", index=False)

    # 4. Marketing Campaign Data (Confounded for PSM/DiD)
    policy_records = []
    for idx, cid in enumerate(customer_ids):
        # FIX: Ensure lambda parameter for poisson distribution is strictly non-negative
        lam = max(0.0, 10.0 + (credit_scores[idx] - 700) / 10.0)
        pre_act = np.random.poisson(lam)
        
        prob_treatment = 1 / (1 + np.exp(-(pre_act - 10) / 3))
        t_assigned = 1 if np.random.rand() < prob_treatment else 0
        post_spend = 50 + 2.0 * pre_act + (25.0 * t_assigned) + np.random.normal(0, 10)
        policy_records.append((cid, "CAMP_2025_Q3", t_assigned, pre_act, round(float(max(0, post_spend)), 2)))

    pd.DataFrame(policy_records, columns=["customer_id", "campaign_id", "treatment_group", "pre_policy_activity", "post_policy_spend"]).to_sql("marketing_campaigns", conn, if_exists="append", index=False)
    conn.close()
    print("Database synthesized successfully.")

if __name__ == "__main__":
    generate_production_data()