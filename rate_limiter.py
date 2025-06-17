from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
from flask import request, jsonify
from config import Config

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.window = Config.RATE_LIMIT_WINDOW
        self.max_requests = Config.RATE_LIMIT_MAX_REQUESTS
        
    def is_allowed(self, client_id):
        """Check if client is allowed to make request"""
        now = datetime.now()
        
        # Remove expired requests
        self.requests[client_id] = [t for t in self.requests[client_id] 
                                  if now - t < self.window]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
            
        self.requests[client_id].append(now)
        return True

def rate_limit():
    """Decorator to apply rate limiting to API endpoints"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            limiter = RateLimiter()
            client_id = request.remote_addr
            
            if not limiter.is_allowed(client_id):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'You have exceeded the rate limit of {Config.RATE_LIMIT_MAX_REQUESTS} requests per {Config.RATE_LIMIT_WINDOW.total_seconds()} seconds'
                }), 429
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def get_limiter():
    """Get the rate limiter instance"""
    return RateLimiter()
