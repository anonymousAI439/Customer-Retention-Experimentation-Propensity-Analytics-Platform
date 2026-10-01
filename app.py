import os
import sqlite3
import yaml
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from statsmodels.stats.power import NormalIndPower

st.set_page_config(page_title="JPMorgan - Analytics Platform", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

@st.cache_data
def load_app_data():
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        db_path = cfg.get("database", {}).get("db_path", "data.db")
    else:
        cfg = {}
        db_path = "data.db"

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        customers = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()
    else:
        customers = pd.DataFrame()

    val_preds = pd.read_csv("validation_predictions.csv") if os.path.exists("validation_predictions.csv") else pd.DataFrame()
    balance_df = pd.read_csv("psm_balance_diagnostics.csv") if os.path.exists("psm_balance_diagnostics.csv") else pd.DataFrame()
    return cfg, customers, val_preds, balance_df

cfg, customers, val_preds, balance_df = load_app_data()

st.title("🏦 Customer Retention, Experimentation & Propensity Platform")

# Sidebar - API Health Status Monitor
st.sidebar.header("System Status")
try:
    health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        if health_data.get("status") == "healthy":
            st.sidebar.success("FastAPI Service: Connected & Healthy")
        else:
            st.sidebar.warning("FastAPI Service: Degraded State")
    else:
        st.sidebar.error(f"FastAPI Status Code: {health_resp.status_code}")
except Exception:
    st.sidebar.error("FastAPI Service: Offline / Unreachable")

tabs = st.tabs([
    "📊 Executive Overview", 
    "🎯 Model & Lift", 
    "🔮 Real-Time API Scoring",
    "🧪 Experimentation & Power", 
    "📈 PSM & Economics"
])

# TAB 1: EXECUTIVE OVERVIEW
with tabs[0]:
    st.header("Portfolio Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(customers):,}" if not customers.empty else "0")
    c2.metric("Validation Churn Rate", f"{val_preds['is_churned'].mean():.2%}" if not val_preds.empty and 'is_churned' in val_preds.columns else "N/A")
    c3.metric("Avg Credit Score", f"{int(customers['initial_credit_score'].mean())}" if not customers.empty and 'initial_credit_score' in customers.columns else "N/A")

# TAB 2: MODEL & LIFT
with tabs[1]:
    st.header("Decile Lift Profile")
    if not val_preds.empty and 'propensity_score' in val_preds.columns and 'is_churned' in val_preds.columns:
        val_preds['decile'] = 10 - pd.qcut(val_preds['propensity_score'], q=10, labels=False, duplicates='drop')
        lift = val_preds.groupby('decile').agg(total=('is_churned', 'count'), churns=('is_churned', 'sum')).reset_index()
        lift['churn_rate'] = lift['churns'] / lift['total']
        lift['lift'] = lift['churn_rate'] / val_preds['is_churned'].mean()
        fig = px.bar(lift, x='decile', y='lift', title="Decile Lift Factor", labels={'decile': 'Decile Top (10=Highest Risk)'})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Validation predictions data not available. Run the pipeline to populate.")

# TAB 3: REAL-TIME API SCORING (FASTAPI INTERACTION)
with tabs[2]:
    st.header("Propensity Scoring via FastAPI Endpoint")
    
    scoring_mode = st.radio("Select Inference Mode", ["Single Customer", "Batch Upload (CSV)"], horizontal=True)

    if scoring_mode == "Single Customer":
        st.subheader("Single Customer Payload")
        with st.form("single_scoring_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                customer_id = st.text_input("Customer ID", value="CUST-1001")
                initial_credit_score = st.slider("Credit Score", 300, 850, 700)
                segment = st.selectbox("Segment", ["Standard", "SMB", "Wealth"])
            with col2:
                account_age_days = st.number_input("Account Age (Days)", min_value=0, value=180)
                sessions_last_90d = st.number_input("Sessions (Last 90d)", min_value=0, value=12)
                total_duration_last_90d = st.number_input("Total Duration (Mins)", min_value=0.0, value=120.0)
            with col3:
                tx_count_last_90d = st.number_input("Tx Count (Last 90d)", min_value=0, value=5)
                tx_volume_last_90d = st.number_input("Tx Volume ($)", min_value=0.0, value=450.0)
                recency_days = st.number_input("Recency (Days Inactive)", min_value=0, value=15)
                acquisition_channel = st.selectbox("Acquisition Channel", ["Organic", "Paid Search", "Referral"])

            submit_btn = st.form_submit_button("Submit Request to FastAPI")

        if submit_btn:
            avg_tx_size = tx_volume_last_90d / tx_count_last_90d if tx_count_last_90d > 0 else 0.0
            
            payload = {
                "customer_id": customer_id,
                "initial_credit_score": int(initial_credit_score),
                "account_age_days": int(account_age_days),
                "sessions_last_90d": int(sessions_last_90d),
                "total_duration_last_90d": float(total_duration_last_90d),
                "tx_count_last_90d": int(tx_count_last_90d),
                "tx_volume_last_90d": float(tx_volume_last_90d),
                "avg_tx_size_last_90d": float(avg_tx_size),
                "recency_days": int(recency_days),
                "segment": segment,
                "acquisition_channel": acquisition_channel
            }

            try:
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Inference successful!")
                    
                    # Exact key extraction matching ScoringResponse schema
                    propensity = result.get("propensity_score", 0.0)
                    category = result.get("risk_category", "N/A")
                    decile = result.get("risk_decile", 0)
                    action = result.get("recommended_action", "No action specified")

                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Predicted Churn Probability", f"{propensity * 100:.2f}%")
                    res_col2.metric("Risk Category", category)
                    res_col3.metric("Risk Decile", f"Decile {decile}")
                    
                    st.info(f"💡 **Recommended Action:** {action}")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as err:
                st.error(f"Failed to connect to FastAPI endpoint: {err}")

    else:
        st.subheader("Batch Customer Payload")
        uploaded_file = st.file_uploader("Upload CSV File for Batch Scoring", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Sample Preview:", batch_df.head())
            
            if st.button("Send Batch Request"):
                records = batch_df.to_dict(orient="records")
                payload = {"customers": records}
                
                try:
                    response = requests.post(f"{API_BASE_URL}/predict/batch", json=payload, timeout=15)
                    if response.status_code == 200:
                        batch_result = response.json()
                        st.success(f"Successfully processed {batch_result.get('total_processed')} customer records!")
                        preds_df = pd.DataFrame(batch_result.get("predictions", []))
                        st.dataframe(preds_df)
                    else:
                        st.error(f"Batch API Error ({response.status_code}): {response.text}")
                except Exception as err:
                    st.error(f"Failed to connect to FastAPI endpoint: {err}")

# TAB 4: EXPERIMENTATION & POWER CALCULATOR
with tabs[3]:
    st.header("A/B Test Power Calculator")
    c1, c2 = st.columns(2)
    mde = c1.slider("Minimum Detectable Effect (MDE)", 0.01, 0.10, 0.03)
    baseline = c2.slider("Baseline Retention", 0.50, 0.90, 0.65)
    
    analysis = NormalIndPower()
    eff_size = mde / np.sqrt(baseline * (1 - baseline))
    req_n = analysis.solve_power(effect_size=eff_size, power=0.80, alpha=0.05)
    st.info(f"Required Sample Size per Group: **{int(np.ceil(req_n)):,}**")

# TAB 5: PSM & ECONOMICS
with tabs[4]:
    st.header("Causal Inference: Propensity Score Matching")
    if not balance_df.empty:
        love_df = balance_df.melt(id_vars=['covariate'], value_vars=['smd_unmatched', 'smd_matched'], var_name='sample', value_name='smd')
        fig_love = px.bar(love_df, x='smd', y='covariate', color='sample', barmode='group', orientation='h', title="Standardized Mean Differences (Pre vs Post Match)")
        fig_love.add_vline(x=0.1, line_dash="dash", line_color="red")
        st.plotly_chart(fig_love, width="stretch")
    else:
        st.info("PSM balance diagnostics data not found.")