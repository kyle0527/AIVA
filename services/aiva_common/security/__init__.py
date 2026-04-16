"""
AIVA Security 模組
安全功能：認證、授權、加密、中間件
"""

from .security import (
    AuthenticationType,
    AuthorizationType,
    CryptographyService,
    EncryptionType,
    SecurityAuditService,
    SecurityCredentials,
    SecurityEventType,
    SecurityLevel,
    SecurityManager,
    SecurityPermission,
    SecurityRole,
    SecuritySubject,
    TokenService,
    create_security_manager,
    get_security_manager,
)
from .security_config import (
    SECURITY_CONFIG_EXAMPLE,
    get_security_config,
    validate_security_config,
)
from .security_middleware import (
    CORSHandler,
    RateLimiter,
    RateLimitRule,
    RequestInfo,
    SecurityHeaders,
    SecurityMiddleware,
)

__all__ = [
    # Security Core
    "AuthenticationType",
    "AuthorizationType",
    "CryptographyService",
    "EncryptionType",
    "SecurityAuditService",
    "SecurityCredentials",
    "SecurityEventType",
    "SecurityLevel",
    "SecurityManager",
    "SecurityPermission",
    "SecurityRole",
    "SecuritySubject",
    "TokenService",
    "get_security_manager",
    "create_security_manager",
    # Security Config
    "SECURITY_CONFIG_EXAMPLE",
    "get_security_config",
    "validate_security_config",
    # Security Middleware
    "CORSHandler",
    "RateLimiter",
    "RateLimitRule",
    "RequestInfo",
    "SecurityHeaders",
    "SecurityMiddleware",
]
