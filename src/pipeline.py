import sqlite3
import numpy as np
import pandas as pd
import yaml
import joblib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from src.causal_psm import run_psm_analysis

class PointInTimeFeatureExtractor:
    def __init__(self, db_path):
        self.db_path = db_path

    def extract_dataset(self, cutoff_date, outcome_days=90):
        conn = sqlite3.connect(self.db_path)
        feature_sql = f"""
        SELECT 
            c.customer_id, c.segment, c.acquisition_channel, c.initial_credit_score,
            CAST(JULIANDAY('{cutoff_date}') - JULIANDAY(c.signup_date) AS INTEGER) AS account_age_days,
            COALESCE(COUNT(DISTINCT a.activity_id), 0) AS sessions_last_90d,
            COALESCE(SUM(a.session_duration_mins), 0.0) AS total_duration_last_90d,
            COALESCE(MAX(a.activity_date), '2020-01-01 00:00:00') AS last_active_date,
            COALESCE(COUNT(DISTINCT t.transaction_id), 0) AS tx_count_last_90d,
            COALESCE(SUM(t.amount), 0.0) AS tx_volume_last_90d,
            COALESCE(AVG(t.amount), 0.0) AS avg_tx_size_last_90d
        FROM customers c
        LEFT JOIN user_activity a ON c.customer_id = a.customer_id AND a.activity_date BETWEEN DATE('{cutoff_date}', '-90 days') AND '{cutoff_date} 23:59:59'
        LEFT JOIN transactions t ON c.customer_id = t.customer_id AND t.transaction_date BETWEEN DATE('{cutoff_date}', '-90 days') AND '{cutoff_date} 23:59:59'
        WHERE c.signup_date <= '{cutoff_date}'
        GROUP BY c.customer_id;
        """
        df_features = pd.read_sql_query(feature_sql, conn)
        
        # Uses format='mixed' to resolve timestamp parsing errors across mixed date/time strings
        df_features['recency_days'] = (
            pd.to_datetime(cutoff_date) - pd.to_datetime(df_features['last_active_date'], format='mixed')
        ).dt.days
        df_features.drop(columns=['last_active_date'], inplace=True)

        outcome_sql = f"""
        SELECT c.customer_id, CASE WHEN COUNT(a.activity_id) = 0 THEN 1 ELSE 0 END AS is_churned
        FROM customers c
        LEFT JOIN user_activity a ON c.customer_id = a.customer_id AND a.activity_date > '{cutoff_date} 23:59:59' AND a.activity_date <= DATE('{cutoff_date}', '+{outcome_days} days')
        WHERE c.signup_date <= '{cutoff_date}'
        GROUP BY c.customer_id;
        """
        df_outcome = pd.read_sql_query(outcome_sql, conn)
        conn.close()
        
        return pd.merge(df_features, df_outcome, on="customer_id")

def run_ab_test_analysis(db_path):
    conn = sqlite3.connect(db_path)
    sql = """
    SELECT e.customer_id, e.variant, CASE WHEN COUNT(a.activity_id) > 0 THEN 1 ELSE 0 END as retained_post
    FROM experiment_exposure e
    LEFT JOIN user_activity a ON e.customer_id = a.customer_id AND a.activity_date > e.exposure_date
    GROUP BY e.customer_id;
    """
    df = pd.read_sql_query(sql, conn)
    conn.close()

    ctrl = df[df['variant'] == 'control']['retained_post']
    trt = df[df['variant'] == 'treatment']['retained_post']
    p_ctrl, p_trt = ctrl.mean(), trt.mean()
    n_ctrl, n_trt = len(ctrl), len(trt)

    if (n_ctrl + n_trt) == 0:
        return {"control_rate": 0.0, "treatment_rate": 0.0, "absolute_lift": 0.0, "p_value": 1.0}

    p_pool = (ctrl.sum() + trt.sum()) / (n_ctrl + n_trt)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_trt)) if p_pool not in [0, 1] else 0.0
    z_stat = (p_trt - p_ctrl) / se if se > 0 else 0.0
    p_val = stats.norm.sf(abs(z_stat)) * 2 if se > 0 else 1.0

    return {"control_rate": p_ctrl, "treatment_rate": p_trt, "absolute_lift": p_trt - p_ctrl, "p_value": p_val}

def execute_pipeline(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    extractor = PointInTimeFeatureExtractor(cfg["database"]["db_path"])
    train_df = extractor.extract_dataset(cfg["modeling"]["train_cutoff"], cfg["modeling"]["train_outcome_days"])
    val_df = extractor.extract_dataset(cfg["modeling"]["val_cutoff"], cfg["modeling"]["val_outcome_days"])

    num_cols = [
        'initial_credit_score', 'account_age_days', 'sessions_last_90d', 
        'total_duration_last_90d', 'tx_count_last_90d', 'tx_volume_last_90d', 
        'avg_tx_size_last_90d', 'recency_days'
    ]
    
    cat_cols = ['segment', 'acquisition_channel']
    
    # Encode categorical features
    train_encoded = pd.get_dummies(train_df[cat_cols], drop_first=True)
    val_encoded = pd.get_dummies(val_df[cat_cols], drop_first=True)
    
    # Align validation dummy columns to match train features
    train_encoded, val_encoded = train_encoded.align(val_encoded, join='left', axis=1, fill_value=0)

    X_train_full = pd.concat([train_df[num_cols], train_encoded], axis=1)
    X_val_full = pd.concat([val_df[num_cols], val_encoded], axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_full)
    X_val = scaler.transform(X_val_full)

    clf = LogisticRegression(C=1.0, random_state=cfg["modeling"]["random_state"])
    clf.fit(X_train, train_df['is_churned'])

    val_preds = clf.predict_proba(X_val)[:, 1]
    joblib.dump(clf, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    val_df['propensity_score'] = val_preds
    val_df.to_csv("validation_predictions.csv", index=False)

    psm_res = run_psm_analysis(cfg["database"]["db_path"])
    psm_res['balance_diagnostics'].to_csv("psm_balance_diagnostics.csv", index=False)

    print("Pipeline Execution Complete.")
    print(f"ROC-AUC: {roc_auc_score(val_df['is_churned'], val_preds):.4f}")
    print(f"PSM Confounded Naive Lift: ${psm_res['naive_att']:.2f} | PSM Unbiased ATT: ${psm_res['psm_att']:.2f}")

if __name__ == "__main__":
    execute_pipeline()