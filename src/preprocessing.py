import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.config import (
    PROCESSED_DATA_FILE, 
    TEST_SIZE, 
    RANDOM_STATE, 
    TARGET_COLUMN,
    PREPROCESSOR_FILE
)
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.label_encoders = {}
        self.preprocessor = None
        self.feature_names = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataset
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Remove Loan_ID if exists
        if 'Loan_ID' in df_clean.columns:
            df_clean = df_clean.drop('Loan_ID', axis=1)
        
        # Handle target variable - convert to binary
        if TARGET_COLUMN in df_clean.columns:
            # Assuming 'Y' = 1 (approved), 'N' = 0 (rejected)
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({'Y': 1, 'N': 0})
        
        logger.info(f"Data after cleaning shape: {df_clean.shape}")
        logger.info(f"Missing values:\n{df_clean.isnull().sum()}")
        
        return df_clean
    
    def get_feature_types(self, df: pd.DataFrame) -> tuple:
        """
        Identify numerical and categorical features
        
        Args:
            df: Dataframe
            
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        # Exclude target column
        features = [col for col in df.columns if col != TARGET_COLUMN]
        
        numerical_features = df[features].select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        categorical_features = df[features].select_dtypes(
            include=['object']
        ).columns.tolist()
        
        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        return numerical_features, categorical_features
    
    def create_preprocessor(self, numerical_features: list, categorical_features: list):
        """
        Create sklearn preprocessing pipeline
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
        """
        from sklearn.preprocessing import OneHotEncoder
        
        # Numerical pipeline: impute then scale
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute then one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        logger.info("Preprocessor pipeline created")
    
    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit preprocessor and transform data
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Tuple of (X_transformed, y)
        """
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
        
        # Get feature types
        numerical_features, categorical_features = self.get_feature_types(df)
        
        # Create and fit preprocessor
        self.create_preprocessor(numerical_features, categorical_features)
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Store feature names for later use
        self.feature_names = self._get_feature_names(numerical_features, categorical_features)
        
        logger.info(f"Data transformed. Shape: {X_transformed.shape}")
        
        return X_transformed, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: New dataframe
            
        Returns:
            Transformed array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        X = df.drop(TARGET_COLUMN, axis=1) if TARGET_COLUMN in df.columns else df
        return self.preprocessor.transform(X)
    
    def _get_feature_names(self, numerical_features: list, categorical_features: list) -> list:
        """Get all feature names after one-hot encoding"""
        feature_names = numerical_features.copy()
        
        # Get categorical feature names after one-hot encoding
        if categorical_features:
            cat_encoder = self.preprocessor.named_transformers_['cat']['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
        
        return feature_names
    
    def save_preprocessor(self, filepath=PREPROCESSOR_FILE):
        """Save preprocessor to disk"""
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load_preprocessor(filepath=PREPROCESSOR_FILE):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        preprocessor_obj = DataPreprocessor()
        preprocessor_obj.preprocessor = data['preprocessor']
        preprocessor_obj.feature_names = data['feature_names']
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor_obj


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Main preprocessing function
    
    Args:
        df: Raw dataframe
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Transform data
    X_transformed, y = preprocessor.fit_transform(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    logger.info(f"Train set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Target distribution - Train: {np.bincount(y_train)}")
    logger.info(f"Target distribution - Test: {np.bincount(y_test)}")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    return X_train, X_test, y_train, y_test, preprocessor, preprocessor.feature_names


if __name__ == "__main__":
    from src.data_ingestion import ingest_data
    
    # Load data
    df = ingest_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    
    print(f"\nPreprocessing completed!")
    print(f"Feature names: {feature_names}")
