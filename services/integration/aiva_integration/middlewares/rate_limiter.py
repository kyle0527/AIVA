

import asyncio
from collections import defaultdict
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class TokenBucket:
    """Token bucket 速率限制算法實現"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        # tokens may be fractional during refill calculations
        self.tokens: float = float(capacity)
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        嘗試消費 tokens

        Args:
            tokens: 要消費的 token 數量

        Returns:
            是否成功消費
        """
        async with self.lock:
            now = time.time()
            # 根據時間差補充 tokens
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """速率限制中間件"""

    def __init__(self, app, rps: float = 20.0, burst: int = 60):
        super().__init__(app)
        self.rps = rps
        self.burst = burst
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(burst, rps)
        )
        self.cleanup_interval = 300  # 5分鐘清理一次
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next) -> Response:
        # 獲取客戶端 IP
        client_ip = self._get_client_ip(request)

        # 清理過期的 buckets
        await self._cleanup_buckets()

        # 檢查速率限制
        bucket = self.buckets[client_ip]
        if not await bucket.consume():
            # 速率限制觸發
            response = Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Retry-After": str(int(1 / self.rps) + 1),
                    "X-RateLimit-Limit": str(int(self.rps)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + 1 / self.rps)),
                },
            )
            return response

        # 處理請求
        response = await call_next(request)

        # 添加速率限制 headers
        remaining_tokens = int(bucket.tokens)
        response.headers["X-RateLimit-Limit"] = str(int(self.rps))
        response.headers["X-RateLimit-Remaining"] = str(remaining_tokens)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """獲取客戶端 IP 地址"""
        # 檢查代理 headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # 回退到直接連接的 IP
        return request.client.host if request.client else "unknown"

    async def _cleanup_buckets(self) -> None:
        """清理不活躍的 token buckets"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        # 清理超過 10 分鐘沒有活動的 buckets
        cutoff_time = now - 600
        to_remove = [
            ip
            for ip, bucket in self.buckets.items()
            if bucket.last_refill < cutoff_time
        ]

        for ip in to_remove:
            del self.buckets[ip]

        self.last_cleanup = now
