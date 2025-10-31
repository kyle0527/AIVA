"""
AIVA Security Middleware
AIVA 安全中間件

為 HTTP 請求提供統一的安全處理，包括認證、授權、速率限制、CORS 等。
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .security import (
    AuthenticationType,
    SecurityManager,
    get_security_manager,
)


@dataclass
class RateLimitRule:
    """速率限制規則"""

    max_requests: int
    time_window: int  # 秒
    identifier_key: str = "ip_address"  # ip_address, user_id, api_key

    def get_identifier(self, request_context: dict[str, Any]) -> str:
        """獲取限制標識符"""
        return request_context.get(self.identifier_key, "unknown")


@dataclass
class RequestInfo:
    """請求信息"""

    timestamp: float
    ip_address: str | None = None
    user_agent: str | None = None
    path: str | None = None
    method: str | None = None


class RateLimiter:
    """速率限制器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._request_history: dict[str, deque] = defaultdict(lambda: deque())
        self._rules: list[RateLimitRule] = []

        # 默認規則
        self.add_rule(RateLimitRule(max_requests=100, time_window=60))  # 每分鐘100次
        self.add_rule(
            RateLimitRule(max_requests=1000, time_window=3600)
        )  # 每小時1000次

    def add_rule(self, rule: RateLimitRule):
        """添加速率限制規則"""
        self._rules.append(rule)
        self.logger.info(
            f"添加速率限制規則: {rule.max_requests}次/{rule.time_window}秒"
        )

    def is_allowed(
        self, request_context: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """檢查請求是否被允許"""
        current_time = time.time()

        for rule in self._rules:
            identifier = rule.get_identifier(request_context)
            key = f"{rule.identifier_key}:{identifier}:{rule.time_window}"

            # 清理過期記錄
            request_times = self._request_history[key]
            cutoff_time = current_time - rule.time_window

            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()

            # 檢查是否超過限制
            if len(request_times) >= rule.max_requests:
                self.logger.warning(
                    f"速率限制觸發: {identifier} ({len(request_times)}/{rule.max_requests})"
                )
                return False, {
                    "rule": rule,
                    "current_requests": len(request_times),
                    "reset_time": request_times[0] + rule.time_window,
                }

            # 記錄當前請求
            request_times.append(current_time)

        return True, {}


class CORSHandler:
    """CORS處理器"""

    def __init__(self):
        self.allowed_origins = {"*"}  # 默認允許所有來源
        self.allowed_methods = {"GET", "POST", "PUT", "DELETE", "OPTIONS"}
        self.allowed_headers = {
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Trace-ID",
        }
        self.allow_credentials = False
        self.max_age = 86400  # 24小時

    def configure(
        self,
        allowed_origins: list[str] = None,
        allowed_methods: list[str] = None,
        allowed_headers: list[str] = None,
        allow_credentials: bool = False,
        max_age: int = 86400,
    ):
        """配置CORS設置"""
        if allowed_origins:
            self.allowed_origins = set(allowed_origins)
        if allowed_methods:
            self.allowed_methods = set(allowed_methods)
        if allowed_headers:
            self.allowed_headers = set(allowed_headers)
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    def get_cors_headers(
        self, origin: str = None, method: str = None
    ) -> dict[str, str]:
        """獲取CORS響應頭"""
        headers = {}

        # Access-Control-Allow-Origin
        if "*" in self.allowed_origins:
            headers["Access-Control-Allow-Origin"] = "*"
        elif origin and origin in self.allowed_origins:
            headers["Access-Control-Allow-Origin"] = origin

        # Access-Control-Allow-Methods
        headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)

        # Access-Control-Allow-Headers
        headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)

        # Access-Control-Allow-Credentials
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"

        # Access-Control-Max-Age
        headers["Access-Control-Max-Age"] = str(self.max_age)

        return headers

    def is_cors_request(self, headers: dict[str, str]) -> bool:
        """檢查是否為CORS請求"""
        return "Origin" in headers

    def is_preflight_request(self, method: str, headers: dict[str, str]) -> bool:
        """檢查是否為預檢請求"""
        return (
            method.upper() == "OPTIONS"
            and "Origin" in headers
            and "Access-Control-Request-Method" in headers
        )


class SecurityHeaders:
    """安全頭處理器"""

    @staticmethod
    def get_security_headers() -> dict[str, str]:
        """獲取安全響應頭"""
        return {
            # 防止XSS攻擊
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            # HTTPS相關
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            # 內容安全策略
            "Content-Security-Policy": "default-src 'self'",
            # 隱私相關
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # 權限策略
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }


class SecurityMiddleware:
    """安全中間件"""

    def __init__(self, security_manager: SecurityManager = None):
        self.security_manager = security_manager or get_security_manager()
        self.rate_limiter = RateLimiter()
        self.cors_handler = CORSHandler()
        self.security_headers = SecurityHeaders()
        self.logger = logging.getLogger(self.__class__.__name__)

        # 白名單路徑（不需要認證）
        self.whitelist_paths = {"/health", "/metrics", "/docs", "/openapi.json"}

        # 認證配置
        self.default_auth_type = AuthenticationType.JWT
        self.auth_header_name = "Authorization"
        self.api_key_header_name = "X-API-Key"

    def configure_cors(self, **kwargs):
        """配置CORS設置"""
        self.cors_handler.configure(**kwargs)

    def configure_rate_limiting(self, rules: list[RateLimitRule]):
        """配置速率限制"""
        self.rate_limiter._rules = rules

    def add_whitelist_path(self, path: str):
        """添加白名單路徑"""
        self.whitelist_paths.add(path)

    async def process_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes = None,
        query_params: dict[str, str] = None,
        client_ip: str = None,
    ) -> dict[str, Any]:
        """處理HTTP請求的安全檢查"""

        request_context = {
            "method": method,
            "path": path,
            "ip_address": client_ip,
            "user_agent": headers.get("User-Agent"),
            "timestamp": time.time(),
        }

        result = {
            "allowed": True,
            "status_code": 200,
            "headers": {},
            "body": None,
            "credentials": None,
            "error": None,
        }

        try:
            # 1. CORS處理
            if self.cors_handler.is_cors_request(headers):
                cors_headers = self.cors_handler.get_cors_headers(
                    headers.get("Origin"), method
                )
                result["headers"].update(cors_headers)

                # 預檢請求直接返回
                if self.cors_handler.is_preflight_request(method, headers):
                    result["status_code"] = 204
                    return result

            # 2. 速率限制檢查
            allowed, rate_limit_info = self.rate_limiter.is_allowed(request_context)
            if not allowed:
                result["allowed"] = False
                result["status_code"] = 429
                result["error"] = "Rate limit exceeded"
                result["headers"]["Retry-After"] = str(
                    int(
                        rate_limit_info.get("reset_time", time.time() + 60)
                        - time.time()
                    )
                )
                return result

            # 3. 安全頭
            result["headers"].update(self.security_headers.get_security_headers())

            # 4. 認證檢查（白名單路徑跳過）
            if path not in self.whitelist_paths:
                credentials = await self._authenticate_request(headers, request_context)
                if not credentials:
                    result["allowed"] = False
                    result["status_code"] = 401
                    result["error"] = "Authentication required"
                    return result

                result["credentials"] = credentials

            return result

        except Exception as e:
            self.logger.error(f"安全中間件處理錯誤: {e}")
            result["allowed"] = False
            result["status_code"] = 500
            result["error"] = "Internal security error"
            return result

    async def _authenticate_request(
        self, headers: dict[str, str], context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """認證請求"""

        # 嘗試JWT認證
        auth_header = headers.get(self.auth_header_name, "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # 移除 "Bearer " 前綴

            credentials = await self.security_manager.authenticate_request(
                AuthenticationType.JWT, {"token": token}, context
            )

            if credentials:
                return credentials.to_dict()

        # 嘗試API Key認證
        api_key = headers.get(self.api_key_header_name)
        if api_key:
            credentials = await self.security_manager.authenticate_request(
                AuthenticationType.API_KEY, {"api_key": api_key}, context
            )

            if credentials:
                return credentials.to_dict()

        return None

    async def authorize_request(
        self,
        credentials: dict[str, Any],
        resource: str,
        action: str,
        context: dict[str, Any] = None,
    ) -> bool:
        """授權請求"""
        if not credentials:
            return False

        subject_id = credentials.get("subject")
        if not subject_id:
            return False

        return await self.security_manager.authorize_request(
            subject_id, resource, action, context or {}
        )

    def create_error_response(
        self, status_code: int, error_message: str
    ) -> dict[str, Any]:
        """創建錯誤響應"""
        return {
            "status_code": status_code,
            "headers": {
                "Content-Type": "application/json",
                **self.security_headers.get_security_headers(),
            },
            "body": {
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }


class SecurityValidator:
    """安全驗證器"""

    @staticmethod
    def validate_input(
        data: Any, max_length: int = 1000, allowed_chars: str = None
    ) -> bool:
        """驗證輸入數據"""
        if isinstance(data, str):
            # 長度檢查
            if len(data) > max_length:
                return False

            # 字符檢查
            if allowed_chars:
                if not all(c in allowed_chars for c in data):
                    return False

            # 常見惡意模式檢查
            malicious_patterns = [
                "<script",
                "javascript:",
                "vbscript:",
                "onload=",
                "onerror=",
                "eval(",
                "alert(",
                "prompt(",
                "confirm(",
                "document.cookie",
                "window.location",
                "../",
                "..\\",
                "union select",
                "drop table",
                "truncate",
                "delete from",
            ]

            data_lower = data.lower()
            for pattern in malicious_patterns:
                if pattern in data_lower:
                    return False

        return True

    @staticmethod
    def sanitize_html(text: str) -> str:
        """清理HTML內容"""
        if not isinstance(text, str):
            return str(text)

        # 簡單的HTML轉義
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;",
        }

        for char, escape in replacements.items():
            text = text.replace(char, escape)

        return text

    @staticmethod
    def validate_json_structure(
        data: dict[str, Any], required_fields: list[str] = None, max_depth: int = 5
    ) -> bool:
        """驗證JSON結構"""
        if not isinstance(data, dict):
            return False

        # 檢查必需字段
        if required_fields:
            for field in required_fields:
                if field not in data:
                    return False

        # 檢查嵌套深度
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                return False

            if isinstance(obj, dict):
                for value in obj.values():
                    if not check_depth(value, current_depth + 1):
                        return False
            elif isinstance(obj, list):
                for item in obj:
                    if not check_depth(item, current_depth + 1):
                        return False

            return True

        return check_depth(data)


# 便捷函數和裝飾器
def create_security_middleware(
    security_manager: SecurityManager = None,
) -> SecurityMiddleware:
    """創建安全中間件"""
    return SecurityMiddleware(security_manager)


def secure_api_endpoint(resource: str, action: str, auth_required: bool = True):
    """安全API端點裝飾器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 獲取請求上下文
            request_context = kwargs.get("request_context", {})

            if auth_required:
                # 檢查認證
                credentials = request_context.get("credentials")
                if not credentials:
                    raise PermissionError("Authentication required")

                # 檢查授權
                security_manager = get_security_manager()
                authorized = await security_manager.authorize_request(
                    credentials["subject"], resource, action, request_context
                )

                if not authorized:
                    raise PermissionError(f"Access denied to {resource}:{action}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def validate_request_data(max_length: int = 1000, required_fields: list[str] = None):
    """請求數據驗證裝飾器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 獲取請求數據
            request_data = kwargs.get("request_data", {})

            # 驗證JSON結構
            if not SecurityValidator.validate_json_structure(
                request_data, required_fields
            ):
                raise ValueError("Invalid request data structure")

            # 驗證字段內容
            for key, value in request_data.items():
                if isinstance(value, str):
                    if not SecurityValidator.validate_input(value, max_length):
                        raise ValueError(f"Invalid input for field: {key}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
