# 🏦 Customer Retention, Experimentation & Propensity Analytics Platform

An end-to-end customer analytics platform combining **churn prediction, A/B testing, causal inference, and unit economics** to identify at-risk customers, measure intervention impact, and support value-based retention decisions.

## 🚀 Key Capabilities

- Leakage-safe churn prediction using point-in-time behavioral features
- Customer risk segmentation and decile lift analysis
- A/B testing with statistical significance and power analysis
- Causal inference using Propensity Score Matching (PSM)
- Treatment-effect estimation using ATT and DiD
- Value-based retention targeting using CLV and campaign cost
- Production-style REST API with FastAPI
- Interactive Streamlit dashboard
- Automated testing with pytest
- Reproducible SQL/SQLite analytics pipeline

## 🏗️ Architecture

    Customer Activity
           │
           ▼
       SQLite / SQL
           │
           ▼
    Point-in-Time Features
           │
           ▼
    Churn Propensity Model
           │
       ┌───┴───────────────┐
       ▼                   ▼
    Risk Deciles       A/B Testing
                           │
                           ▼
                     PSM / DiD
                           │
                           ▼
                     Causal Effect
                           │
                           ▼
                  CLV + Campaign Cost
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
               FastAPI          Streamlit
                 API             Dashboard

## 📊 Analytical Methodology

### Churn Prediction

Uses a 90-day observation window and future outcome window to prevent temporal leakage.

    Observation Window: T-90 days → T
    Outcome Window:     T → T+90 days

Behavioral features include:

- Recency
- Session frequency
- Transaction count
- Transaction volume
- Average transaction size
- Account age
- Customer activity duration

Model:

- Regularized Logistic Regression
- Churn propensity scoring
- Risk deciles
- Decile lift analysis

### A/B Testing

Evaluates campaign effectiveness using:

- Treatment vs. control conversion rates
- Absolute treatment effect
- Two-sample proportion Z-test
- 95% confidence intervals
- Statistical power
- Minimum detectable effect
- Sample-size analysis

Configuration:

    Significance level (α) = 0.05
    Statistical power      = 0.80
    Confidence level       = 95%

### Causal Inference

Uses Propensity Score Matching to estimate treatment effects from observational data.

    Customer Features
           ↓
    Treatment Propensity
           ↓
    Logit Transformation
           ↓
    1:1 Nearest-Neighbor Matching
           ↓
    Caliper Matching
           ↓
    Balance Diagnostics
           ↓
    ATT / Treatment Effect

Matching configuration:

    Caliper = 0.25 × σ(logit propensity)

Covariate balance is evaluated using Standardized Mean Difference (SMD).

### Unit Economics

Converts analytical results into expected business value.

    Expected Value
    = P(Churn)
    × Intervention Success Rate
    × CLV
    − Campaign Cost

Example configuration:

    Intervention success rate = 22%
    Campaign cost             = $12

This allows retention targeting based on both **customer risk and expected economic value**.

## 🗂️ Project Structure

    customer-retention-experimentation/
    │
    ├── .gitignore
    ├── Makefile
    ├── README.md
    ├── config.yaml
    ├── requirements.txt
    ├── schema.sql
    │
    ├── data/
    │   └── retention_analytics.db
    │
    ├── api/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── schemas.py
    │   └── service.py
    │
    ├── src/
    │   ├── __init__.py
    │   ├── causal_psm.py
    │   ├── data_generator.py
    │   └── pipeline.py
    │
    ├── tests/
    │   ├── __init__.py
    │   ├── test_api.py
    │   └── test_pipeline.py
    │
    ├── app.py
    └── run_pipeline.py

## ⚡ Quickstart

### 1. Clone

    git clone <your-repository-url>
    cd customer-retention-experimentation

### 2. Create Environment

    python -m venv venv

### 3. Activate

Windows:

    venv\Scripts\activate

Linux/macOS:

    source venv/bin/activate

### 4. Install Dependencies

    pip install -r requirements.txt

### 5. Run Pipeline

    python run_pipeline.py

### 6. Run Tests

    pytest tests/

### 7. Launch Dashboard

    streamlit run app.py

### 8. Start API

    uvicorn api.app:app --reload --port 8000

## 🔌 API

### Endpoints

    GET  /health
    POST /predict
    POST /predict/batch

### Example Request

    {
      "customer_id": "CUST_001",
      "initial_credit_score": 720,
      "account_age_days": 850,
      "sessions_last_90d": 18,
      "total_duration_last_90d": 420,
      "tx_count_last_90d": 25,
      "tx_volume_last_90d": 8500,
      "avg_tx_size_last_90d": 340,
      "recency_days": 7
    }

### Example Response

    {
      "customer_id": "CUST_001",
      "propensity_score": 0.73,
      "risk_decile": 9,
      "risk_category": "High",
      "recommended_action": "Retention Campaign"
    }

Swagger documentation:

    http://localhost:8000/docs

## 🧪 Testing

The project includes tests for:

- Temporal leakage prevention
- Point-in-time feature generation
- Model prediction
- API validation
- Single-customer prediction
- Batch prediction
- Model artifacts

Run:

    pytest tests/

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Database | SQLite / SQL |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Statistics | SciPy |
| API | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Testing | Pytest |
| Model Persistence | Joblib |
| Configuration | YAML |
| Automation | Makefile |

## 🔑 Key Design Principles

**Temporal Integrity**  
Features are generated only from information available before the prediction cutoff.

**Prediction ≠ Causality**  
A high churn probability does not imply that an intervention will cause retention.

**Statistical Effect ≠ Business Value**  
A statistically significant campaign does not automatically mean it is economically worthwhile.

**Risk ≠ Actionability**  
Customers are prioritized using both predicted risk and expected intervention value.

## 🔮 Future Extensions

- Uplift modeling
- Causal forests
- Doubly robust estimation
- Inverse Probability Weighting
- SHAP explainability
- Probability calibration
- Model monitoring and drift detection
- MLflow experiment tracking
- Docker deployment
- CI/CD
- Cloud deployment
- Automated retraining
- Feature store integration

## Data (retention_analytics.db) - https://drive.google.com/file/d/1OUsHCPamOfvMsPBh3-lc9JdW2vmPEc4Y/view?usp=drive_link
## 👨‍💻 Author

**Shanmukha Sikkireddy**
