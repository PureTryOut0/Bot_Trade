import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
from config import Config

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger('crypto_analyzer')
    logger.setLevel(Config.LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        f'logs/crypto_analyzer_{datetime.now().strftime("%Y%m%d")}.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """Get a logger with the given name"""
    return logging.getLogger(f'crypto_analyzer.{name}')

def log_request(request):
    """Log incoming request details"""
    logger = get_logger('api')
    logger.info(f"Request received - Method: {request.method}, Path: {request.path}, IP: {request.remote_addr}")

def log_response(response, request_time):
    """Log response details"""
    logger = get_logger('api')
    logger.info(f"Response sent - Status: {response.status_code}, Time: {request_time:.2f}s")

def log_error(error, request=None):
    """Log error with request context"""
    logger = get_logger('error')
    error_msg = str(error)
    if request:
        logger.error(f"Error in request - Method: {request.method}, Path: {request.path}, Error: {error_msg}")
    else:
        logger.error(f"Error: {error_msg}")
