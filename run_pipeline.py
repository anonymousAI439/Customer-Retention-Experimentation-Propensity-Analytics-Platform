from src.data_generator import generate_production_data
from src.pipeline import execute_pipeline

if __name__ == "__main__":
    print("Step 1: Generating Synthetic Production Database...")
    generate_production_data()
    print("\nStep 2: Training Pipeline, Model, and PSM Diagnostics...")
    execute_pipeline()