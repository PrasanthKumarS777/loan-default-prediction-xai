import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import SHAP_PLOTS_DIR
from src.logger import setup_logger

logger = setup_logger(__name__)


class ModelExplainer:
    """SHAP-based model explainability"""
    
    def __init__(self, model, X_train, feature_names):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            X_train: Training data for SHAP background
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
        # Create SHAP explainer
        logger.info("Creating SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for training data
        logger.info("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X_train)
        self.X_train = X_train
        
        logger.info("SHAP explainer initialized successfully")
    
    def plot_summary(self, save_path=None):
        """
        Create SHAP summary plot showing feature importance
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, 
            self.X_train,
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Summary plot saved to {save_path}")
        else:
            save_path = SHAP_PLOTS_DIR / "shap_summary.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, save_path=None):
        """
        Create bar plot of mean absolute SHAP values
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X_train,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            save_path = SHAP_PLOTS_DIR / "feature_importance.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def explain_prediction(self, X_sample, sample_index=0, save_path=None):
        """
        Create waterfall plot for individual prediction
        
        Args:
            X_sample: Single sample or array of samples
            sample_index: Index of sample to explain
            save_path: Path to save the plot
        """
        # Calculate SHAP values for the sample
        shap_values_sample = self.explainer.shap_values(X_sample)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
            shap_values_sample = shap_values_sample.reshape(1, -1)
        
        # Use shap.plots.waterfall for newer versions
        try:
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_sample[sample_index],
                    base_values=self.explainer.expected_value,
                    data=X_sample[sample_index],
                    feature_names=self.feature_names
                ),
                show=False
            )
        except:
            # Fallback for older SHAP versions
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_sample[sample_index],
                    base_values=self.explainer.expected_value,
                    data=X_sample[sample_index],
                    feature_names=self.feature_names
                ),
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Waterfall plot saved to {save_path}")
        else:
            save_path = SHAP_PLOTS_DIR / f"waterfall_sample_{sample_index}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.close()
        
        return shap_values_sample[sample_index]
    
    def get_feature_contributions(self, X_sample):
        """
        Get SHAP values as feature contributions for API response
        
        Args:
            X_sample: Single sample
            
        Returns:
            Dictionary of feature contributions
        """
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        shap_values_sample = self.explainer.shap_values(X_sample)
        
        # Create dictionary of feature: shap_value
        contributions = {}
        for i, feature in enumerate(self.feature_names):
            contributions[feature] = float(shap_values_sample[0][i])
        
        # Sort by absolute value
        contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        return contributions


if __name__ == "__main__":
    from src.train import LoanDefaultModel
    from src.data_ingestion import ingest_data
    from src.preprocessing import preprocess_data, DataPreprocessor
    from src.config import MODEL_FILE, PREPROCESSOR_FILE
    
    # Load data
    df = ingest_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    
    # Load trained model
    model = LoanDefaultModel.load_model(MODEL_FILE)
    
    # Create explainer
    explainer = ModelExplainer(model.model, X_train, feature_names)
    
    # Generate plots
    print("Generating SHAP visualizations...")
    explainer.plot_summary()
    explainer.plot_feature_importance()
    explainer.explain_prediction(X_test, sample_index=0)
    explainer.explain_prediction(X_test, sample_index=1)
    
    print(f"\n✅ SHAP plots saved to: {SHAP_PLOTS_DIR}")
    
    # Show feature contributions for a sample
    contributions = explainer.get_feature_contributions(X_test[0])
    print("\nTop 5 Feature Contributions for Sample 0:")
    for feature, value in list(contributions.items())[:5]:
        print(f"  {feature}: {value:.4f}")
