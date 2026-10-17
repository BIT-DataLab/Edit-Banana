"""
Rate limiting middleware for Edit Banana API.

Provides IP-based rate limiting for preview generation.
"""

import time
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for IP-based rate limiting.

    Tracks request counts per IP within a time window.
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exclude_paths = exclude_paths or []
        # Simple in-memory store: {ip: [(timestamp, count), ...]}
        self._requests: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for excluded paths
        path = request.url.path
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check rate limit
        if self._is_rate_limited(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.max_requests} per {self.window_seconds}s",
                    "limit": self.max_requests,
                    "window_seconds": self.window_seconds,
                },
            )

        # Record request
        self._record_request(client_ip)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if IP has exceeded rate limit."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Get recent requests for this IP
        requests = self._requests.get(client_ip, [])
        recent_count = sum(
            1 for ts, _ in requests if ts > cutoff
        )

        return recent_count >= self.max_requests

    def _record_request(self, client_ip: str):
        """Record a request timestamp."""
        now = time.time()

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        self._requests[client_ip].append((now, 1))

        # Cleanup old entries periodically
        if len(self._requests[client_ip]) % 100 == 0:
            self._cleanup_old_requests(client_ip)

    def _get_remaining_requests(self, client_ip: str) -> int:
        """Calculate remaining requests for IP."""
        now = time.time()
        cutoff = now - self.window_seconds

        requests = self._requests.get(client_ip, [])
        recent_count = sum(
            1 for ts, _ in requests if ts > cutoff
        )

        return max(0, self.max_requests - recent_count)

    def _cleanup_old_requests(self, client_ip: str):
        """Remove expired request records."""
        now = time.time()
        cutoff = now - self.window_seconds

        self._requests[client_ip] = [
            (ts, count) for ts, count in self._requests.get(client_ip, [])
            if ts > cutoff
        ]


class PreviewRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Specialized middleware for preview endpoint rate limiting.

    Stricter limits for preview generation (3 per hour).
    """

    def __init__(self, app):
        super().__init__(app)
        # Preview-specific limits
        self.preview_limit = 3
        self.preview_window = 3600  # 1 hour
        self._preview_requests: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next):
        """Process request with preview-specific rate limiting."""
        path = request.url.path

        # Only apply to preview creation endpoint
        if not self._is_preview_endpoint(path, request.method):
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check preview rate limit
        if self._is_rate_limited(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Preview rate limit exceeded",
                    "message": f"Maximum {self.preview_limit} previews per hour. Please try again later.",
                    "limit": self.preview_limit,
                    "window_seconds": self.preview_window,
                },
            )

        # Record request
        self._record_request(client_ip)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-Preview-RateLimit-Limit"] = str(self.preview_limit)
        response.headers["X-Preview-RateLimit-Remaining"] = str(remaining)
        response.headers["X-Preview-RateLimit-Window"] = str(self.preview_window)

        return response

    def _is_preview_endpoint(self, path: str, method: str) -> bool:
        """Check if this is a preview creation endpoint."""
        return method == "POST" and "/preview" in path

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if IP has exceeded preview rate limit."""
        now = time.time()
        cutoff = now - self.preview_window

        requests = self._preview_requests.get(client_ip, [])
        recent_count = sum(
            1 for ts, _ in requests if ts > cutoff
        )

        return recent_count >= self.preview_limit

    def _record_request(self, client_ip: str):
        """Record a preview request timestamp."""
        now = time.time()

        if client_ip not in self._preview_requests:
            self._preview_requests[client_ip] = []

        self._preview_requests[client_ip].append((now, 1))

    def _get_remaining_requests(self, client_ip: str) -> int:
        """Calculate remaining preview requests for IP."""
        now = time.time()
        cutoff = now - self.preview_window

        requests = self._preview_requests.get(client_ip, [])
        recent_count = sum(
            1 for ts, _ in requests if ts > cutoff
        )

        return max(0, self.preview_limit - recent_count)
