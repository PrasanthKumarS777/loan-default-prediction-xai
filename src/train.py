import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.xgboost
from src.config import XGBOOST_PARAMS, MODEL_FILE, RANDOM_STATE
from src.logger import setup_logger
from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data

logger = setup_logger(__name__)


class LoanDefaultModel:
    """XGBoost model for loan default prediction"""
    
    def __init__(self, params=None):
        """
        Initialize model
        
        Args:
            params: XGBoost hyperparameters
        """
        self.params = params if params else XGBOOST_PARAMS
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
        """
        logger.info("Starting model training...")
        
        self.feature_names = feature_names
        
        # Create XGBoost classifier
        self.model = xgb.XGBClassifier(**self.params)
        
        # Train model with evaluation set
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        logger.info("Model training completed")
        
        # Get training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath=MODEL_FILE):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath=MODEL_FILE):
        """Load model from disk"""
        model_obj = LoanDefaultModel()
        model_obj.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model_obj


def train_model_pipeline():
    """
    Complete training pipeline
    
    Returns:
        Trained model and test data
    """
    # Start MLflow run
    mlflow.set_experiment("loan-default-prediction")
    
    with mlflow.start_run():
        # Load and preprocess data
        logger.info("Loading data...")
        df = ingest_data()
        
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
        
        # Initialize and train model
        logger.info("Training model...")
        loan_model = LoanDefaultModel()
        loan_model.train(X_train, y_train, X_test, y_test, feature_names)
        
        # Log parameters to MLflow
        mlflow.log_params(XGBOOST_PARAMS)
        
        # Calculate metrics
        y_pred = loan_model.predict(X_test)
        y_pred_proba = loan_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        logger.info("\n" + "="*50)
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info("="*50)
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"ROC-AUC:   {roc_auc:.4f}")
        logger.info("="*50 + "\n")
        
        # Log model to MLflow
        mlflow.xgboost.log_model(loan_model.model, "model")
        
        # Save model locally
        loan_model.save_model()
        
        return loan_model, X_test, y_test, feature_names


if __name__ == "__main__":
    model, X_test, y_test, feature_names = train_model_pipeline()
    print("\n✅ Model training completed successfully!")
    print(f"Model saved to: {MODEL_FILE}")
