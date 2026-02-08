import pandas as pd
import requests
from pathlib import Path
from src.config import DATASET_URL, RAW_DATA_FILE
from src.logger import setup_logger

logger = setup_logger(__name__)


def download_data(url: str = DATASET_URL, save_path: Path = RAW_DATA_FILE) -> None:
    """
    Download dataset from URL and save to local path
    
    Args:
        url: Dataset URL
        save_path: Path to save the downloaded file
    """
    try:
        logger.info(f"Downloading data from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Data downloaded successfully to {save_path}")
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def load_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame containing the data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def ingest_data() -> pd.DataFrame:
    """
    Main function to ingest data (download if not exists, then load)
    
    Returns:
        DataFrame containing the raw data
    """
    if not RAW_DATA_FILE.exists():
        logger.info("Data file not found. Downloading...")
        download_data()
    else:
        logger.info("Data file already exists. Skipping download.")
    
    return load_data()


if __name__ == "__main__":
    df = ingest_data()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
