import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODEL_DIR = ROOT_DIR / "models"

# Output directory
OUTPUT_DIR = ROOT_DIR / "outputs"
SHAP_PLOTS_DIR = OUTPUT_DIR / "shap_plots"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, SHAP_PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration - Working URL
DATASET_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
RAW_DATA_FILE = RAW_DATA_DIR / "loan_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "loan_data_processed.csv"

# Model configuration
MODEL_FILE = MODEL_DIR / "xgboost_model.joblib"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "Loan_Status"

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
