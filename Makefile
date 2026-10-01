.PHONY: all setup data train test dashboard api clean

all: setup train test

setup:
	pip install -r requirements.txt

train:
	python run_pipeline.py

test:
	pytest tests/

dashboard:
	streamlit run app.py

api:
	uvicorn api.app:app --reload --port 8000

clean:
	rm -rf data/*.db *.joblib *.csv .pytest_cache __pycache__ api/__pycache__ src/__pycache__ tests/__pycache__