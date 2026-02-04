"""Rate limiting dependencies for API endpoints."""

from typing import Dict
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import time
import asyncio
from collections import defaultdict

# Simple in-memory rate limiter (use Redis in production)
rate_limit_storage: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "reset_time": 0})


class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def __call__(self, request: Request) -> None:
        """Check rate limit for the request."""
        client_ip = request.client.host
        current_time = time.time()
        
        # Get or create rate limit data for this IP
        rate_data = rate_limit_storage[client_ip]
        
        # Reset if window has passed
        if current_time >= rate_data["reset_time"]:
            rate_data["count"] = 0
            rate_data["reset_time"] = current_time + self.window_seconds
        
        # Check if limit exceeded
        if rate_data["count"] >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds} seconds."
            )
        
        # Increment counter
        rate_data["count"] += 1


# Import API config for rate limiting settings
from server.api.config import api_config

# Rate limiters for different endpoints
query_rate_limiter = RateLimiter(
    max_requests=api_config.query_rate_limit, 
    window_seconds=api_config.rate_limit_window
)
health_rate_limiter = RateLimiter(
    max_requests=api_config.health_rate_limit, 
    window_seconds=api_config.rate_limit_window
)
