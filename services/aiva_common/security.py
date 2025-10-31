"""
AIVA Security Framework
AIVA 安全認證框架

實施 TODO 項目 13: 設計安全認證框架
- 服務間身份認證和授權
- 端到端加密通訊
- 訪問控制和權限管理
- 安全令牌管理和驗證
- 密鑰管理和輪換

特性：
1. 多層安全架構：認證、授權、加密
2. 跨語言安全通訊：統一安全協議
3. 細粒度權限控制：基於角色和資源的訪問控制
4. 安全審計：完整的安全事件日誌
5. 密鑰管理：自動密鑰生成、輪換和撤銷
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
)

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class AuthenticationType(Enum):
    """認證類型"""

    TOKEN = "token"  # 令牌認證
    CERTIFICATE = "certificate"  # 證書認證
    API_KEY = "api_key"  # API密鑰認證
    MUTUAL_TLS = "mutual_tls"  # 雙向TLS認證
    JWT = "jwt"  # JWT令牌認證


class AuthorizationType(Enum):
    """授權類型"""

    RBAC = "rbac"  # 基於角色的訪問控制
    ABAC = "abac"  # 基於屬性的訪問控制
    ACL = "acl"  # 訪問控制列表
    POLICY = "policy"  # 基於策略的授權


class EncryptionType(Enum):
    """加密類型"""

    AES_256_GCM = "aes_256_gcm"  # AES-256-GCM對稱加密
    RSA_2048 = "rsa_2048"  # RSA-2048非對稱加密
    RSA_4096 = "rsa_4096"  # RSA-4096非對稱加密
    CHACHA20_POLY1305 = "chacha20_poly1305"  # ChaCha20-Poly1305


class SecurityLevel(Enum):
    """安全級別"""

    LOW = "low"  # 低安全級別
    MEDIUM = "medium"  # 中等安全級別
    HIGH = "high"  # 高安全級別
    CRITICAL = "critical"  # 關鍵安全級別


class SecurityEventType(Enum):
    """安全事件類型"""

    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_SUCCESS = "authz_success"
    AUTHORIZATION_FAILURE = "authz_failure"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REVOKED = "token_revoked"
    ENCRYPTION_ERROR = "encryption_error"
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


@dataclass
class SecurityCredentials:
    """安全憑據"""

    credential_id: str
    credential_type: AuthenticationType
    credential_data: dict[str, Any]
    expires_at: datetime | None = None
    issued_at: datetime = field(default_factory=datetime.utcnow)
    issuer: str | None = None
    subject: str | None = None
    scopes: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """檢查憑據是否過期"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "credential_id": self.credential_id,
            "credential_type": self.credential_type.value,
            "credential_data": self.credential_data,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "issued_at": self.issued_at.isoformat(),
            "issuer": self.issuer,
            "subject": self.subject,
            "scopes": list(self.scopes),
            "metadata": self.metadata,
        }


@dataclass
class SecurityPermission:
    """安全權限"""

    permission_id: str
    resource: str
    action: str
    conditions: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime | None = None
    granted_at: datetime = field(default_factory=datetime.utcnow)
    granted_by: str | None = None

    def is_expired(self) -> bool:
        """檢查權限是否過期"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def matches(
        self, resource: str, action: str, context: dict[str, Any] = None
    ) -> bool:
        """檢查權限是否匹配"""
        # 檢查資源匹配
        if not self._match_resource(resource):
            return False

        # 檢查動作匹配
        if not self._match_action(action):
            return False

        # 檢查條件匹配
        if not self._match_conditions(context or {}):
            return False

        return True

    def _match_resource(self, resource: str) -> bool:
        """匹配資源"""
        # 支持通配符匹配
        if self.resource == "*":
            return True
        if self.resource.endswith("*"):
            prefix = self.resource[:-1]
            return resource.startswith(prefix)
        return self.resource == resource

    def _match_action(self, action: str) -> bool:
        """匹配動作"""
        if self.action == "*":
            return True
        if self.action.endswith("*"):
            prefix = self.action[:-1]
            return action.startswith(prefix)
        return self.action == action

    def _match_conditions(self, context: dict[str, Any]) -> bool:
        """匹配條件"""
        for key, expected_value in self.conditions.items():
            actual_value = context.get(key)
            if actual_value != expected_value:
                return False
        return True


@dataclass
class SecurityRole:
    """安全角色"""

    role_id: str
    role_name: str
    permissions: set[str] = field(default_factory=set)
    parent_roles: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str | None = None

    def has_permission(self, permission_id: str) -> bool:
        """檢查是否有權限"""
        return permission_id in self.permissions


@dataclass
class SecuritySubject:
    """安全主體（用戶或服務）"""

    subject_id: str
    subject_type: str  # user, service, system
    roles: set[str] = field(default_factory=set)
    direct_permissions: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime | None = None
    is_active: bool = True

    def has_role(self, role_id: str) -> bool:
        """檢查是否有角色"""
        return role_id in self.roles

    def has_direct_permission(self, permission_id: str) -> bool:
        """檢查是否有直接權限"""
        return permission_id in self.direct_permissions


@dataclass
class SecurityEvent:
    """安全事件"""

    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    subject_id: str | None = None
    resource: str | None = None
    action: str | None = None
    result: str = "unknown"  # success, failure, error
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "subject_id": self.subject_id,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "trace_id": self.trace_id,
        }


class CryptographyService:
    """加密服務"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._key_cache: dict[str, Any] = {}

    def generate_symmetric_key(self, key_size: int = 32) -> bytes:
        """生成對稱加密密鑰"""
        return secrets.token_bytes(key_size)

    def generate_rsa_key_pair(self, key_size: int = 2048) -> tuple[bytes, bytes]:
        """生成RSA密鑰對"""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size, backend=default_backend()
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    def encrypt_aes_gcm(
        self, data: bytes, key: bytes, associated_data: bytes | None = None
    ) -> tuple[bytes, bytes, bytes]:
        """AES-GCM加密"""
        iv = secrets.token_bytes(12)  # 96位IV

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())

        encryptor = cipher.encryptor()
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)

        ciphertext = encryptor.update(data) + encryptor.finalize()
        tag = encryptor.tag

        return ciphertext, iv, tag

    def decrypt_aes_gcm(
        self,
        ciphertext: bytes,
        key: bytes,
        iv: bytes,
        tag: bytes,
        associated_data: bytes | None = None,
    ) -> bytes:
        """AES-GCM解密"""
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
        )

        decryptor = cipher.decryptor()
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext

    def encrypt_rsa(self, data: bytes, public_key_pem: bytes) -> bytes:
        """RSA加密"""
        public_key = serialization.load_pem_public_key(
            public_key_pem, backend=default_backend()
        )

        ciphertext = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        return ciphertext

    def decrypt_rsa(self, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """RSA解密"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=default_backend()
        )

        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        return plaintext

    def sign_data(self, data: bytes, private_key_pem: bytes) -> bytes:
        """數據簽名"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=default_backend()
        )

        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        return signature

    def verify_signature(
        self, data: bytes, signature: bytes, public_key_pem: bytes
    ) -> bool:
        """驗證簽名"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=default_backend()
            )

            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception as e:
            self.logger.warning(f"簽名驗證失敗: {e}")
            return False

    def derive_key(self, password: str, salt: bytes, iterations: int = 100000) -> bytes:
        """密鑰派生"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend(),
        )

        return kdf.derive(password.encode())

    def generate_hmac(self, data: bytes, key: bytes) -> bytes:
        """生成HMAC"""
        return hmac.new(key, data, hashlib.sha256).digest()

    def verify_hmac(self, data: bytes, key: bytes, expected_hmac: bytes) -> bool:
        """驗證HMAC"""
        try:
            computed_hmac = self.generate_hmac(data, key)
            return hmac.compare_digest(computed_hmac, expected_hmac)
        except Exception as e:
            self.logger.warning(f"HMAC驗證失敗: {e}")
            return False


class TokenService:
    """令牌服務"""

    def __init__(self, secret_key: str, issuer: str = "AIVA"):
        self.secret_key = secret_key
        self.issuer = issuer
        self.logger = logging.getLogger(self.__class__.__name__)
        self._revoked_tokens: set[str] = set()

    def generate_jwt_token(
        self,
        subject: str,
        scopes: list[str] = None,
        expires_in: int = 3600,
        additional_claims: dict[str, Any] = None,
    ) -> str:
        """生成JWT令牌"""
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=expires_in)

        payload = {
            "iss": self.issuer,
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": str(uuid.uuid4()),  # JWT ID
            "scopes": scopes or [],
        }

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")

        self.logger.info(f"JWT令牌已生成 - 主體: {subject}, 範圍: {scopes}")
        return token

    def verify_jwt_token(self, token: str) -> dict[str, Any] | None:
        """驗證JWT令牌"""
        try:
            # 檢查令牌是否被撤銷
            decoded = jwt.decode(token, options={"verify_signature": False})
            jti = decoded.get("jti")
            if jti and jti in self._revoked_tokens:
                self.logger.warning(f"使用已撤銷的令牌: {jti}")
                return None

            # 驗證令牌
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            self.logger.debug(f"JWT令牌驗證成功 - 主體: {payload.get('sub')}")
            return payload

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT令牌已過期")
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"JWT令牌無效: {e}")
        except Exception as e:
            self.logger.error(f"JWT令牌驗證錯誤: {e}")

        return None

    def revoke_token(self, token: str):
        """撤銷令牌"""
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            jti = decoded.get("jti")
            if jti:
                self._revoked_tokens.add(jti)
                self.logger.info(f"令牌已撤銷: {jti}")
        except Exception as e:
            self.logger.error(f"撤銷令牌失敗: {e}")

    def generate_api_key(self, prefix: str = "aiva") -> str:
        """生成API密鑰"""
        key_id = secrets.token_hex(8)
        secret = secrets.token_urlsafe(32)
        return f"{prefix}_{key_id}_{secret}"

    def validate_api_key_format(
        self, api_key: str, expected_prefix: str = "aiva"
    ) -> bool:
        """驗證API密鑰格式"""
        parts = api_key.split("_")
        if len(parts) != 3:
            return False

        prefix, key_id, secret = parts
        if prefix != expected_prefix:
            return False

        if len(key_id) != 16:  # 8字節的十六進制
            return False

        if len(secret) < 20:  # URL安全的base64編碼應該至少20字符
            return False

        return True


class AuthenticationService:
    """認證服務"""

    def __init__(
        self, token_service: TokenService, crypto_service: CryptographyService
    ):
        self.token_service = token_service
        self.crypto_service = crypto_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self._registered_credentials: dict[str, SecurityCredentials] = {}
        self._authentication_cache: dict[str, tuple[SecurityCredentials, datetime]] = {}
        self.cache_ttl = 300  # 5分鐘緩存

    def register_credentials(self, credentials: SecurityCredentials):
        """註冊憑據"""
        self._registered_credentials[credentials.credential_id] = credentials
        self.logger.info(f"憑據已註冊: {credentials.credential_id}")

    def unregister_credentials(self, credential_id: str):
        """註銷憑據"""
        if credential_id in self._registered_credentials:
            del self._registered_credentials[credential_id]
            self.logger.info(f"憑據已註銷: {credential_id}")

    async def authenticate(
        self,
        auth_type: AuthenticationType,
        auth_data: dict[str, Any],
        context: dict[str, Any] = None,
    ) -> SecurityCredentials | None:
        """認證請求"""
        try:
            # 檢查緩存
            cache_key = self._generate_cache_key(auth_type, auth_data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.logger.debug(f"使用緩存的認證結果: {auth_type.value}")
                return cached_result

            # 執行認證
            credentials = None
            if auth_type == AuthenticationType.JWT:
                credentials = await self._authenticate_jwt(auth_data)
            elif auth_type == AuthenticationType.API_KEY:
                credentials = await self._authenticate_api_key(auth_data)
            elif auth_type == AuthenticationType.TOKEN:
                credentials = await self._authenticate_token(auth_data)
            elif auth_type == AuthenticationType.CERTIFICATE:
                credentials = await self._authenticate_certificate(auth_data)
            else:
                self.logger.warning(f"不支持的認證類型: {auth_type}")
                return None

            # 緩存結果
            if credentials:
                self._cache_result(cache_key, credentials)
                self.logger.info(f"認證成功: {credentials.subject} ({auth_type.value})")
            else:
                self.logger.warning(f"認證失敗: {auth_type.value}")

            return credentials

        except Exception as e:
            self.logger.error(f"認證過程出錯: {e}")
            return None

    async def _authenticate_jwt(
        self, auth_data: dict[str, Any]
    ) -> SecurityCredentials | None:
        """JWT認證"""
        token = auth_data.get("token")
        if not token:
            return None

        payload = self.token_service.verify_jwt_token(token)
        if not payload:
            return None

        credentials = SecurityCredentials(
            credential_id=payload.get("jti", str(uuid.uuid4())),
            credential_type=AuthenticationType.JWT,
            credential_data={"token": token, "payload": payload},
            subject=payload.get("sub"),
            issuer=payload.get("iss"),
            expires_at=datetime.fromtimestamp(payload.get("exp", time.time())),
            scopes=set(payload.get("scopes", [])),
        )

        return credentials

    async def _authenticate_api_key(
        self, auth_data: dict[str, Any]
    ) -> SecurityCredentials | None:
        """API密鑰認證"""
        api_key = auth_data.get("api_key")
        if not api_key or not self.token_service.validate_api_key_format(api_key):
            return None

        # 在實際應用中，這裡應該從數據庫或配置中驗證API密鑰
        # 這裡簡化為檢查是否在註冊的憑據中
        for credentials in self._registered_credentials.values():
            if (
                credentials.credential_type == AuthenticationType.API_KEY
                and credentials.credential_data.get("api_key") == api_key
            ):
                if not credentials.is_expired():
                    return credentials

        return None

    async def _authenticate_token(
        self, auth_data: dict[str, Any]
    ) -> SecurityCredentials | None:
        """令牌認證"""
        token = auth_data.get("token")
        if not token:
            return None

        # 查找匹配的憑據
        for credentials in self._registered_credentials.values():
            if (
                credentials.credential_type == AuthenticationType.TOKEN
                and credentials.credential_data.get("token") == token
            ):
                if not credentials.is_expired():
                    return credentials

        return None

    async def _authenticate_certificate(
        self, auth_data: dict[str, Any]
    ) -> SecurityCredentials | None:
        """證書認證"""
        certificate = auth_data.get("certificate")
        signature = auth_data.get("signature")
        challenge = auth_data.get("challenge")

        if not all([certificate, signature, challenge]):
            return None

        # 驗證證書簽名
        try:
            challenge_bytes = (
                challenge.encode() if isinstance(challenge, str) else challenge
            )
            signature_bytes = base64.b64decode(signature)

            if self.crypto_service.verify_signature(
                challenge_bytes, signature_bytes, certificate
            ):
                # 創建憑據
                credentials = SecurityCredentials(
                    credential_id=str(uuid.uuid4()),
                    credential_type=AuthenticationType.CERTIFICATE,
                    credential_data={
                        "certificate": certificate,
                        "verified_at": datetime.utcnow().isoformat(),
                    },
                )
                return credentials
        except Exception as e:
            self.logger.error(f"證書認證錯誤: {e}")

        return None

    def _generate_cache_key(
        self, auth_type: AuthenticationType, auth_data: dict[str, Any]
    ) -> str:
        """生成緩存鍵"""
        # 創建一個基於認證數據的哈希鍵
        data_str = json.dumps(auth_data, sort_keys=True)
        hash_obj = hashlib.sha256(f"{auth_type.value}:{data_str}".encode())
        return hash_obj.hexdigest()

    def _get_from_cache(self, cache_key: str) -> SecurityCredentials | None:
        """從緩存獲取結果"""
        if cache_key in self._authentication_cache:
            credentials, cached_at = self._authentication_cache[cache_key]
            if datetime.utcnow() - cached_at < timedelta(seconds=self.cache_ttl):
                if not credentials.is_expired():
                    return credentials
            else:
                # 緩存過期，刪除
                del self._authentication_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, credentials: SecurityCredentials):
        """緩存認證結果"""
        self._authentication_cache[cache_key] = (credentials, datetime.utcnow())

        # 清理過期緩存
        self._cleanup_cache()

    def _cleanup_cache(self):
        """清理過期緩存"""
        current_time = datetime.utcnow()
        expired_keys = []

        for key, (credentials, cached_at) in self._authentication_cache.items():
            if (
                current_time - cached_at > timedelta(seconds=self.cache_ttl)
                or credentials.is_expired()
            ):
                expired_keys.append(key)

        for key in expired_keys:
            del self._authentication_cache[key]


class AuthorizationService:
    """授權服務"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._subjects: dict[str, SecuritySubject] = {}
        self._roles: dict[str, SecurityRole] = {}
        self._permissions: dict[str, SecurityPermission] = {}
        self._authorization_cache: dict[str, tuple[bool, datetime]] = {}
        self.cache_ttl = 180  # 3分鐘緩存

    def register_subject(self, subject: SecuritySubject):
        """註冊安全主體"""
        self._subjects[subject.subject_id] = subject
        self.logger.info(f"安全主體已註冊: {subject.subject_id}")

    def register_role(self, role: SecurityRole):
        """註冊角色"""
        self._roles[role.role_id] = role
        self.logger.info(f"角色已註冊: {role.role_id}")

    def register_permission(self, permission: SecurityPermission):
        """註冊權限"""
        self._permissions[permission.permission_id] = permission
        self.logger.info(f"權限已註冊: {permission.permission_id}")

    def assign_role_to_subject(self, subject_id: str, role_id: str):
        """給主體分配角色"""
        if subject_id in self._subjects and role_id in self._roles:
            self._subjects[subject_id].roles.add(role_id)
            self.logger.info(f"角色 {role_id} 已分配給主體 {subject_id}")
            self._clear_authorization_cache()

    def grant_permission_to_role(self, role_id: str, permission_id: str):
        """給角色授予權限"""
        if role_id in self._roles and permission_id in self._permissions:
            self._roles[role_id].permissions.add(permission_id)
            self.logger.info(f"權限 {permission_id} 已授予角色 {role_id}")
            self._clear_authorization_cache()

    def grant_direct_permission(self, subject_id: str, permission_id: str):
        """給主體授予直接權限"""
        if subject_id in self._subjects and permission_id in self._permissions:
            self._subjects[subject_id].direct_permissions.add(permission_id)
            self.logger.info(f"直接權限 {permission_id} 已授予主體 {subject_id}")
            self._clear_authorization_cache()

    async def authorize(
        self,
        subject_id: str,
        resource: str,
        action: str,
        context: dict[str, Any] = None,
    ) -> bool:
        """授權檢查"""
        try:
            # 檢查緩存
            cache_key = self._generate_cache_key(subject_id, resource, action, context)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.logger.debug(f"使用緩存的授權結果: {subject_id}")
                return cached_result

            # 獲取主體
            subject = self._subjects.get(subject_id)
            if not subject or not subject.is_active:
                self.logger.warning(f"主體不存在或不活躍: {subject_id}")
                self._cache_result(cache_key, False)
                return False

            # 檢查直接權限
            if await self._check_direct_permissions(subject, resource, action, context):
                self.logger.debug(f"通過直接權限授權: {subject_id}")
                self._cache_result(cache_key, True)
                return True

            # 檢查角色權限
            if await self._check_role_permissions(subject, resource, action, context):
                self.logger.debug(f"通過角色權限授權: {subject_id}")
                self._cache_result(cache_key, True)
                return True

            self.logger.warning(f"授權失敗: {subject_id} -> {resource}:{action}")
            self._cache_result(cache_key, False)
            return False

        except Exception as e:
            self.logger.error(f"授權過程出錯: {e}")
            return False

    async def _check_direct_permissions(
        self,
        subject: SecuritySubject,
        resource: str,
        action: str,
        context: dict[str, Any] = None,
    ) -> bool:
        """檢查直接權限"""
        for permission_id in subject.direct_permissions:
            permission = self._permissions.get(permission_id)
            if permission and not permission.is_expired():
                if permission.matches(resource, action, context):
                    return True
        return False

    async def _check_role_permissions(
        self,
        subject: SecuritySubject,
        resource: str,
        action: str,
        context: dict[str, Any] = None,
    ) -> bool:
        """檢查角色權限"""
        # 收集所有角色（包括繼承的角色）
        all_roles = set()
        self._collect_inherited_roles(subject.roles, all_roles)

        # 檢查所有角色的權限
        for role_id in all_roles:
            role = self._roles.get(role_id)
            if not role:
                continue

            for permission_id in role.permissions:
                permission = self._permissions.get(permission_id)
                if permission and not permission.is_expired():
                    if permission.matches(resource, action, context):
                        return True

        return False

    def _collect_inherited_roles(self, role_ids: set[str], collected: set[str]):
        """收集繼承的角色"""
        for role_id in role_ids:
            if role_id in collected:
                continue

            collected.add(role_id)
            role = self._roles.get(role_id)
            if role and role.parent_roles:
                self._collect_inherited_roles(role.parent_roles, collected)

    def _generate_cache_key(
        self,
        subject_id: str,
        resource: str,
        action: str,
        context: dict[str, Any] = None,
    ) -> str:
        """生成緩存鍵"""
        context_str = json.dumps(context or {}, sort_keys=True)
        data = f"{subject_id}:{resource}:{action}:{context_str}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> bool | None:
        """從緩存獲取結果"""
        if cache_key in self._authorization_cache:
            result, cached_at = self._authorization_cache[cache_key]
            if datetime.utcnow() - cached_at < timedelta(seconds=self.cache_ttl):
                return result
            else:
                del self._authorization_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: bool):
        """緩存授權結果"""
        self._authorization_cache[cache_key] = (result, datetime.utcnow())
        self._cleanup_cache()

    def _clear_authorization_cache(self):
        """清空授權緩存"""
        self._authorization_cache.clear()

    def _cleanup_cache(self):
        """清理過期緩存"""
        current_time = datetime.utcnow()
        expired_keys = [
            key
            for key, (_, cached_at) in self._authorization_cache.items()
            if current_time - cached_at > timedelta(seconds=self.cache_ttl)
        ]

        for key in expired_keys:
            del self._authorization_cache[key]


class SecurityAuditService:
    """安全審計服務"""

    def __init__(self, max_events: int = 10000):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_events = max_events
        self._security_events: list[SecurityEvent] = []

    def log_security_event(self, event: SecurityEvent):
        """記錄安全事件"""
        self._security_events.append(event)

        # 保持事件數量在限制內
        if len(self._security_events) > self.max_events:
            self._security_events = self._security_events[-self.max_events :]

        # 根據事件類型和結果選擇日誌級別
        if event.result == "failure" or event.event_type in [
            SecurityEventType.AUTHENTICATION_FAILURE,
            SecurityEventType.AUTHORIZATION_FAILURE,
            SecurityEventType.SECURITY_VIOLATION,
        ]:
            self.logger.warning(f"安全事件: {event.event_type.value} - {event.result}")
        else:
            self.logger.info(f"安全事件: {event.event_type.value} - {event.result}")

    def query_security_events(
        self,
        event_type: SecurityEventType = None,
        subject_id: str = None,
        result: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
    ) -> list[SecurityEvent]:
        """查詢安全事件"""
        filtered_events = []

        for event in reversed(self._security_events):  # 最新的在前
            # 應用過濾條件
            if event_type and event.event_type != event_type:
                continue
            if subject_id and event.subject_id != subject_id:
                continue
            if result and event.result != result:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            filtered_events.append(event)

            if len(filtered_events) >= limit:
                break

        return filtered_events

    def get_security_statistics(self, hours: int = 24) -> dict[str, Any]:
        """獲取安全統計"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in self._security_events if e.timestamp > cutoff_time]

        # 統計事件類型
        event_type_counts = {}
        result_counts = {}
        subject_counts = {}

        for event in recent_events:
            # 事件類型統計
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

            # 結果統計
            result_counts[event.result] = result_counts.get(event.result, 0) + 1

            # 主體統計
            if event.subject_id:
                subject = event.subject_id
                subject_counts[subject] = subject_counts.get(subject, 0) + 1

        return {
            "time_range_hours": hours,
            "total_events": len(recent_events),
            "event_type_distribution": event_type_counts,
            "result_distribution": result_counts,
            "top_subjects": dict(
                sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "failure_rate": result_counts.get("failure", 0)
            / max(len(recent_events), 1),
        }


class SecurityManager:
    """安全管理器 - 統一的安全服務入口"""

    def __init__(self, secret_key: str = None, issuer: str = "AIVA"):
        # 初始化密鑰
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)

        # 初始化服務
        self.crypto_service = CryptographyService()
        self.token_service = TokenService(secret_key, issuer)
        self.auth_service = AuthenticationService(
            self.token_service, self.crypto_service
        )
        self.authz_service = AuthorizationService()
        self.audit_service = SecurityAuditService()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_initialized = False

    async def initialize(self):
        """初始化安全管理器"""
        if self.is_initialized:
            return

        try:
            # 創建默認角色和權限
            await self._create_default_roles_and_permissions()

            self.is_initialized = True
            self.logger.info("安全管理器初始化完成")

        except Exception as e:
            self.logger.error(f"安全管理器初始化失敗: {e}")
            raise

    async def _create_default_roles_and_permissions(self):
        """創建默認角色和權限"""
        # 系統權限
        system_perms = [
            SecurityPermission("system.read", "system", "read"),
            SecurityPermission("system.write", "system", "write"),
            SecurityPermission("system.admin", "system", "*"),
        ]

        # 服務權限
        service_perms = [
            SecurityPermission("service.read", "service", "read"),
            SecurityPermission("service.write", "service", "write"),
            SecurityPermission("service.execute", "service", "execute"),
        ]

        # 數據權限
        data_perms = [
            SecurityPermission("data.read", "data", "read"),
            SecurityPermission("data.write", "data", "write"),
            SecurityPermission("data.delete", "data", "delete"),
        ]

        # 註冊權限
        for perm in system_perms + service_perms + data_perms:
            self.authz_service.register_permission(perm)

        # 創建默認角色
        admin_role = SecurityRole(
            "admin",
            "系統管理員",
            permissions={"system.admin", "service.write", "data.write"},
            description="完整的系統管理權限",
        )

        service_role = SecurityRole(
            "service",
            "服務用戶",
            permissions={"service.read", "service.execute", "data.read"},
            description="服務調用權限",
        )

        readonly_role = SecurityRole(
            "readonly",
            "只讀用戶",
            permissions={"system.read", "service.read", "data.read"},
            description="只讀權限",
        )

        # 註冊角色
        for role in [admin_role, service_role, readonly_role]:
            self.authz_service.register_role(role)

    async def authenticate_request(
        self,
        auth_type: AuthenticationType,
        auth_data: dict[str, Any],
        context: dict[str, Any] = None,
    ) -> SecurityCredentials | None:
        """認證請求"""
        try:
            credentials = await self.auth_service.authenticate(
                auth_type, auth_data, context
            )

            # 記錄認證事件
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=(
                    SecurityEventType.AUTHENTICATION_SUCCESS
                    if credentials
                    else SecurityEventType.AUTHENTICATION_FAILURE
                ),
                timestamp=datetime.utcnow(),
                subject_id=credentials.subject if credentials else None,
                result="success" if credentials else "failure",
                details={"auth_type": auth_type.value, "context": context or {}},
                ip_address=context.get("ip_address") if context else None,
                user_agent=context.get("user_agent") if context else None,
                trace_id=context.get("trace_id") if context else None,
            )

            self.audit_service.log_security_event(event)
            return credentials

        except Exception as e:
            self.logger.error(f"認證請求錯誤: {e}")
            return None

    async def authorize_request(
        self,
        subject_id: str,
        resource: str,
        action: str,
        context: dict[str, Any] = None,
    ) -> bool:
        """授權請求"""
        try:
            authorized = await self.authz_service.authorize(
                subject_id, resource, action, context
            )

            # 記錄授權事件
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=(
                    SecurityEventType.AUTHORIZATION_SUCCESS
                    if authorized
                    else SecurityEventType.AUTHORIZATION_FAILURE
                ),
                timestamp=datetime.utcnow(),
                subject_id=subject_id,
                resource=resource,
                action=action,
                result="success" if authorized else "failure",
                details={"context": context or {}},
                trace_id=context.get("trace_id") if context else None,
            )

            self.audit_service.log_security_event(event)
            return authorized

        except Exception as e:
            self.logger.error(f"授權請求錯誤: {e}")
            return False

    def create_service_credentials(
        self, service_id: str, scopes: list[str] = None, expires_in: int = 86400
    ) -> dict[str, str]:
        """為服務創建憑據"""
        try:
            # 生成JWT令牌
            jwt_token = self.token_service.generate_jwt_token(
                subject=service_id,
                scopes=scopes or ["service.read", "service.execute"],
                expires_in=expires_in,
            )

            # 生成API密鑰
            api_key = self.token_service.generate_api_key("svc")

            # 註冊憑據
            jwt_credentials = SecurityCredentials(
                credential_id=str(uuid.uuid4()),
                credential_type=AuthenticationType.JWT,
                credential_data={"token": jwt_token},
                subject=service_id,
                expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
                scopes=set(scopes or []),
            )

            api_credentials = SecurityCredentials(
                credential_id=str(uuid.uuid4()),
                credential_type=AuthenticationType.API_KEY,
                credential_data={"api_key": api_key},
                subject=service_id,
                scopes=set(scopes or []),
            )

            self.auth_service.register_credentials(jwt_credentials)
            self.auth_service.register_credentials(api_credentials)

            # 記錄令牌頒發事件
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=SecurityEventType.TOKEN_ISSUED,
                timestamp=datetime.utcnow(),
                subject_id=service_id,
                result="success",
                details={"token_types": ["jwt", "api_key"], "scopes": scopes or []},
            )

            self.audit_service.log_security_event(event)

            return {
                "jwt_token": jwt_token,
                "api_key": api_key,
                "expires_in": expires_in,
            }

        except Exception as e:
            self.logger.error(f"創建服務憑據錯誤: {e}")
            return {}

    def register_service(
        self,
        service_id: str,
        roles: list[str] = None,
        direct_permissions: list[str] = None,
    ):
        """註冊服務"""
        subject = SecuritySubject(
            subject_id=service_id,
            subject_type="service",
            roles=set(roles or ["service"]),
            direct_permissions=set(direct_permissions or []),
        )

        self.authz_service.register_subject(subject)
        self.logger.info(f"服務已註冊: {service_id}")

    def get_security_status(self) -> dict[str, Any]:
        """獲取安全狀態"""
        return {
            "is_initialized": self.is_initialized,
            "registered_subjects": len(self.authz_service._subjects),
            "registered_roles": len(self.authz_service._roles),
            "registered_permissions": len(self.authz_service._permissions),
            "registered_credentials": len(self.auth_service._registered_credentials),
            "security_statistics": self.audit_service.get_security_statistics(),
            "timestamp": datetime.utcnow().isoformat(),
        }


# 全局安全管理器實例
_security_manager: SecurityManager | None = None


def get_security_manager() -> SecurityManager:
    """獲取安全管理器實例"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def create_security_manager(
    secret_key: str = None, issuer: str = "AIVA"
) -> SecurityManager:
    """創建新的安全管理器實例"""
    return SecurityManager(secret_key, issuer)


# 裝飾器和中間件
def require_authentication(auth_type: AuthenticationType = AuthenticationType.JWT):
    """需要認證的裝飾器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            security_manager = get_security_manager()

            # 從請求上下文提取認證信息
            # 這裡簡化處理，實際應用中需要從HTTP頭或其他位置提取
            auth_data = kwargs.get("auth_data", {})
            context = kwargs.get("context", {})

            credentials = await security_manager.authenticate_request(
                auth_type, auth_data, context
            )
            if not credentials:
                raise PermissionError("認證失敗")

            # 將憑據添加到kwargs中
            kwargs["credentials"] = credentials
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_authorization(resource: str, action: str):
    """需要授權的裝飾器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            credentials = kwargs.get("credentials")
            if not credentials:
                raise PermissionError("缺少認證憑據")

            security_manager = get_security_manager()
            context = kwargs.get("context", {})

            authorized = await security_manager.authorize_request(
                credentials.subject, resource, action, context
            )

            if not authorized:
                raise PermissionError(f"無權限執行 {action} 操作於資源 {resource}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def secure_endpoint(
    auth_type: AuthenticationType = AuthenticationType.JWT,
    resource: str = None,
    action: str = None,
):
    """安全端點裝飾器，結合認證和授權"""

    def decorator(func):
        @require_authentication(auth_type)
        async def wrapper(*args, **kwargs):
            if resource and action:
                # 應用授權檢查
                credentials = kwargs.get("credentials")
                security_manager = get_security_manager()
                context = kwargs.get("context", {})

                authorized = await security_manager.authorize_request(
                    credentials.subject, resource, action, context
                )

                if not authorized:
                    raise PermissionError(f"無權限執行 {action} 操作於資源 {resource}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
