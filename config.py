import os
from dotenv import load_dotenv
from datetime import timedelta
from functools import lru_cache

load_dotenv()

class Config:
    # API Rate Limiting
    RATE_LIMIT_WINDOW = timedelta(minutes=1)
    RATE_LIMIT_MAX_REQUESTS = 100
    
    # Cache Settings
    CACHE_MAX_SIZE = 1000
    CACHE_TTL = timedelta(minutes=5)
    
    # Logging Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Database Settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///crypto_analyzer.db')
    
    # API Settings
    API_KEY = os.getenv('API_KEY', 'your_api_key_here')
    API_KEY_HEADER = 'X-API-Key'
    
    # External Services
    COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'
    COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
    
    # Model Settings
    MODEL_NAME = os.getenv('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
    
    # Data Collection Settings
    REDDIT_LIMIT = int(os.getenv('REDDIT_LIMIT', '10'))
    TWITTER_LIMIT = int(os.getenv('TWITTER_LIMIT', '50'))
    NEWS_LIMIT = int(os.getenv('NEWS_LIMIT', '20'))
    
    @classmethod
    @lru_cache()
    def get_config(cls):
        return cls()
