# Customer Retention, Experimentation & Propensity Analytics Platform

Enterprise customer analytics platform featuring temporal leakage-safe feature engineering, Propensity Score Matching (PSM), A/B testing, unit economics modeling, and REST API deployment.

## Execution Guide

### 1. Set Up Environment
```bash
pip install -r requirements.txt

### 2.Execute Data Generation & Model Pipeline
```bash
python run_pipeline.py

### 3. Run Automated Integration Tests
```bash
pytest tests/

### 4.Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py

### 5.(In a separate terminal tab) Launch FastAPI Microservice
```bash
uvicorn api.app:app --reload --port 8000

Note:

Open Swagger documentation: 
```bash
http://localhost:8000/docs