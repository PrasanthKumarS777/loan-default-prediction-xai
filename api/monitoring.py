import time
import logging
from functools import wraps
from datetime import datetime
import json
from pathlib import Path

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure file handler
file_handler = logging.FileHandler(LOG_DIR / f"api_{datetime.now().strftime('%Y%m%d')}.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))

# Get logger
logger = logging.getLogger("api_monitoring")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class APIMetrics:
    """Track API metrics"""
    def __init__(self):
        self.total_requests = 0
        self.total_predictions = 0
        self.total_batch_predictions = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.error_count = 0
        self.total_response_time = 0
        
    def log_prediction(self, prediction: str, response_time: float):
        self.total_predictions += 1
        self.total_requests += 1
        self.total_response_time += response_time
        
        if prediction == "Approved":
            self.approved_count += 1
        else:
            self.rejected_count += 1
            
        logger.info(f"Prediction: {prediction} | Response Time: {response_time:.3f}s")
    
    def log_batch_prediction(self, count: int, response_time: float):
        self.total_batch_predictions += count
        self.total_requests += 1
        self.total_response_time += response_time
        logger.info(f"Batch Prediction: {count} applications | Response Time: {response_time:.3f}s")
    
    def log_error(self, error: str):
        self.error_count += 1
        logger.error(f"Error occurred: {error}")
    
    def get_stats(self):
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_predictions": self.total_predictions,
            "total_batch_predictions": self.total_batch_predictions,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "approval_rate": round(
                self.approved_count / self.total_predictions * 100, 2
            ) if self.total_predictions > 0 else 0,
            "error_count": self.error_count,
            "avg_response_time": round(avg_response_time, 3)
        }


# Global metrics instance
metrics = APIMetrics()


def track_time(func):
    """Decorator to track execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            response_time = time.time() - start_time
            return result, response_time
        except Exception as e:
            response_time = time.time() - start_time
            metrics.log_error(str(e))
            raise
    return wrapper
