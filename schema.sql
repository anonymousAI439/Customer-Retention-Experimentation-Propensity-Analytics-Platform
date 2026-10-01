DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS user_activity;
DROP TABLE IF EXISTS experiment_exposure;
DROP TABLE IF EXISTS marketing_campaigns;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    signup_date DATE NOT NULL,
    segment TEXT NOT NULL CHECK (segment IN ('Retail', 'Wealth', 'SMB')),
    acquisition_channel TEXT NOT NULL,
    initial_credit_score INTEGER NOT NULL
);

CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    merchant_category TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE user_activity (
    activity_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    activity_date TIMESTAMP NOT NULL,
    session_duration_mins REAL NOT NULL,
    actions_count INTEGER NOT NULL,
    device_type TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE experiment_exposure (
    customer_id TEXT PRIMARY KEY,
    variant TEXT NOT NULL CHECK (variant IN ('control', 'treatment')),
    exposure_date DATE NOT NULL,
    pre_period_avg_spend DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE marketing_campaigns (
    customer_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    treatment_group INTEGER NOT NULL CHECK (treatment_group IN (0, 1)),
    pre_policy_activity INTEGER NOT NULL,
    post_policy_spend DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX idx_tx_cust_date ON transactions(customer_id, transaction_date);
CREATE INDEX idx_act_cust_date ON user_activity(customer_id, activity_date);