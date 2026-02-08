from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import shap
import logging
from pathlib import Path
import numpy as np
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup monitoring logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

file_handler = logging.FileHandler(LOG_DIR / f"api_{datetime.now().strftime('%Y%m%d')}.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API with XAI",
    description="ML API for loan default prediction with SHAP explainability",
    version="1.0.0"
)


# Metrics tracking
class APIMetrics:
    def __init__(self):
        self.total_requests = 0
        self.total_predictions = 0
        self.total_batch_predictions = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.error_count = 0
        self.total_response_time = 0
        self.start_time = datetime.now()
        
    def log_prediction(self, prediction: str, response_time: float):
        self.total_predictions += 1
        self.total_requests += 1
        self.total_response_time += response_time
        
        if prediction == "Approved":
            self.approved_count += 1
        else:
            self.rejected_count += 1
            
        logger.info(f"Prediction: {prediction} | Response Time: {response_time:.3f}s")
    
    def log_batch_prediction(self, count: int, approved: int, rejected: int, response_time: float):
        self.total_batch_predictions += count
        self.total_requests += 1
        self.approved_count += approved
        self.rejected_count += rejected
        self.total_response_time += response_time
        logger.info(f"Batch: {count} apps | Approved: {approved} | Rejected: {rejected} | Time: {response_time:.3f}s")
    
    def log_error(self, error: str):
        self.error_count += 1
        logger.error(f"Error: {error}")
    
    def get_stats(self):
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        total_decisions = self.approved_count + self.rejected_count
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.total_requests,
            "total_predictions": self.total_predictions,
            "total_batch_predictions": self.total_batch_predictions,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "approval_rate_percent": round(
                self.approved_count / total_decisions * 100, 2
            ) if total_decisions > 0 else 0,
            "error_count": self.error_count,
            "avg_response_time_seconds": round(avg_response_time, 3),
            "requests_per_minute": round(self.total_requests / (uptime / 60), 2) if uptime > 0 else 0
        }


metrics = APIMetrics()


# Load model and preprocessor
MODEL_PATH = Path("models/xgboost_model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor_dict = joblib.load(PREPROCESSOR_PATH)
    
    # Extract preprocessor from dictionary
    if isinstance(preprocessor_dict, dict):
        preprocessor = preprocessor_dict['preprocessor']
        feature_names_list = preprocessor_dict['feature_names']
    else:
        preprocessor = preprocessor_dict
        feature_names_list = None
    
    # Initialize SHAP explainer
    sample_data = pd.DataFrame({
        'Gender': ['Male'],
        'Married': ['Yes'],
        'Dependents': ['0'],
        'Education': ['Graduate'],
        'Self_Employed': ['No'],
        'ApplicantIncome': [5000],
        'CoapplicantIncome': [0],
        'LoanAmount': [100],
        'Loan_Amount_Term': [360],
        'Credit_History': [1.0],
        'Property_Area': ['Urban']
    })
    sample_processed = preprocessor.transform(sample_data)
    explainer = shap.TreeExplainer(model)
    
    logger.info("✅ Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None
    preprocessor = None
    explainer = None
    feature_names_list = None


# Request models
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


class BatchPredictionRequest(BaseModel):
    applications: List[LoanApplication]


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Loan Default Prediction API with XAI is running", 
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/batch_predict", "/metrics", "/model_info"]
    }


@app.get("/health")
async def health_check():
    """Check if API and model are healthy"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "uptime_seconds": (datetime.now() - metrics.start_time).total_seconds()
    }


@app.post("/predict")
async def predict(application: LoanApplication):
    """Predict loan default for a single application with SHAP explanations"""
    start_time = time.time()
    
    if model is None or preprocessor is None:
        metrics.log_error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([application.dict()])
        
        # Preprocess
        processed = preprocessor.transform(input_df)
        
        # Predict
        prediction = int(model.predict(processed)[0])
        probability = float(model.predict_proba(processed)[0][1])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(processed)
        
        # Use stored feature names
        if feature_names_list:
            feature_names = feature_names_list
        else:
            feature_names = list(preprocessor.get_feature_names_out())
        
        # Get SHAP contributions
        if len(shap_values.shape) == 3:
            shap_contribution = {k: float(v) for k, v in zip(feature_names, shap_values[0][0])}
        else:
            shap_contribution = {k: float(v) for k, v in zip(feature_names, shap_values[0])}
        
        # Top factors
        sorted_features = sorted(shap_contribution.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)[:5]
        
        top_factors = [{
            'feature': str(feat),
            'contribution': round(float(contrib), 4),
            'impact': 'Positive' if contrib > 0 else 'Negative'
        } for feat, contrib in sorted_features]
        
        # Log metrics
        response_time = time.time() - start_time
        prediction_label = "Approved" if prediction == 1 else "Rejected"
        metrics.log_prediction(prediction_label, response_time)
        
        return {
            "prediction": prediction_label,
            "probability": round(probability, 4),
            "shap_contributions": shap_contribution,
            "top_factors": top_factors,
            "response_time_seconds": round(response_time, 3)
        }
    
    except Exception as e:
        response_time = time.time() - start_time
        metrics.log_error(str(e))
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Predict loan default for multiple applications"""
    start_time = time.time()
    
    if model is None or preprocessor is None:
        metrics.log_error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for idx, app in enumerate(request.applications):
            # Convert to DataFrame
            input_df = pd.DataFrame([app.dict()])
            
            # Preprocess
            processed = preprocessor.transform(input_df)
            
            # Predict
            prediction = int(model.predict(processed)[0])
            probability = float(model.predict_proba(processed)[0][1])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(processed)
            
            # Use stored feature names
            if feature_names_list:
                feature_names = feature_names_list
            else:
                feature_names = list(preprocessor.get_feature_names_out())
            
            # Get SHAP contributions
            if len(shap_values.shape) == 3:
                shap_contribution = {k: float(v) for k, v in zip(feature_names, shap_values[0][0])}
            else:
                shap_contribution = {k: float(v) for k, v in zip(feature_names, shap_values[0])}
            
            # Top factors
            sorted_features = sorted(shap_contribution.items(), 
                                   key=lambda x: abs(x[1]), 
                                   reverse=True)[:5]
            
            top_factors = [{
                'feature': str(feat),
                'contribution': round(float(contrib), 4),
                'impact': 'Positive' if contrib > 0 else 'Negative'
            } for feat, contrib in sorted_features]
            
            results.append({
                "application_id": idx + 1,
                "prediction": "Approved" if prediction == 1 else "Rejected",
                "probability": round(probability, 4),
                "top_factors": top_factors
            })
        
        # Calculate summary
        approved = sum(1 for r in results if r["prediction"] == "Approved")
        rejected = sum(1 for r in results if r["prediction"] == "Rejected")
        
        # Log metrics
        response_time = time.time() - start_time
        metrics.log_batch_prediction(len(results), approved, rejected, response_time)
        
        return {
            "predictions": results, 
            "total_processed": len(results),
            "approved": approved,
            "rejected": rejected,
            "approval_rate_percent": round(approved / len(results) * 100, 2),
            "response_time_seconds": round(response_time, 3)
        }
    
    except Exception as e:
        response_time = time.time() - start_time
        metrics.log_error(str(e))
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics and statistics"""
    return metrics.get_stats()


@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if feature_names_list:
        features = feature_names_list
    else:
        features = list(preprocessor.get_feature_names_out()) if preprocessor else []
    
    return {
        "model_type": type(model).__name__,
        "n_features": len(features),
        "feature_names": features
    }
