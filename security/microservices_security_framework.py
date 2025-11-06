"""
AIVA 微服務安全最佳實踐框架
基於 Microsoft Azure Well-Architected Framework 和 Microsoft Learn 指導原則

實施功能：
1. mTLS (Mutual TLS) 服務間加密
2. API Gateway 安全策略
3. 服務網格安全控制
4. Zero Trust 架構
5. 分散式身份驗證
"""

import asyncio
import ssl
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from pathlib import Path
import aiohttp
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import uuid

# AIVA 內部導入
from services.aiva_common.security import (
    SecurityManager,
    AuthenticationType,
    get_security_manager
)
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

@dataclass
class ServiceIdentity:
    """服務身份識別"""
    service_name: str
    service_id: str
    certificate_path: str
    private_key_path: str
    trusted_ca_path: str
    permissions: List[str] = field(default_factory=list)

@dataclass
class MTLSConfig:
    """mTLS 配置"""
    enabled: bool = True
    ca_cert_path: str = ""
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    check_hostname: bool = True
    protocol: ssl.Protocol = ssl.PROTOCOL_TLS_CLIENT


class CertificateManager:
    """證書管理器 - 實施 mTLS 最佳實踐"""
    
    def __init__(self, ca_cert_path: str, ca_key_path: str):
        self.ca_cert_path = Path(ca_cert_path)
        self.ca_key_path = Path(ca_key_path)
        self.logger = get_logger(self.__class__.__name__)
        
    async def generate_service_certificate(self, service_identity: ServiceIdentity) -> tuple[str, str]:
        """為服務生成證書"""
        try:
            # 生成私鑰
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
            )
            
            # 讀取 CA 證書和私鑰
            with open(self.ca_cert_path, 'rb') as f:
                ca_cert = x509.load_pem_x509_certificate(f.read())
            
            with open(self.ca_key_path, 'rb') as f:
                ca_private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            # 創建證書主題
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "TW"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Taiwan"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Taipei"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIVA Security Platform"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Microservices"),
                x509.NameAttribute(NameOID.COMMON_NAME, service_identity.service_name),
            ])
            
            # 創建證書
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                ca_cert.subject
            ).public_key(
                private_key.public_key()
            ).serial_number(
                int(uuid.uuid4())
            ).not_valid_before(
                datetime.datetime.now(datetime.timezone.utc)
            ).not_valid_after(
                datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(service_identity.service_name),
                    x509.DNSName(f"{service_identity.service_name}.aiva.local"),
                    x509.IPAddress("127.0.0.1"),
                ]),
                critical=False,
            ).add_extension(
                x509.KeyUsage(
                    key_encipherment=True,
                    digital_signature=True,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                ]),
                critical=True,
            ).sign(ca_private_key, hashes.SHA256())
            
            # 儲存證書和私鑰
            cert_path = Path(service_identity.certificate_path)
            key_path = Path(service_identity.private_key_path)
            
            cert_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 寫入證書
            with open(cert_path, 'wb') as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            # 寫入私鑰
            with open(key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            self.logger.info(f"為服務 {service_identity.service_name} 生成證書成功")
            return str(cert_path), str(key_path)
            
        except Exception as e:
            self.logger.error(f"生成服務證書失敗: {e}")
            raise


class ServiceMeshSecurityManager:
    """服務網格安全管理器"""
    
    def __init__(self, security_manager: SecurityManager = None):
        self.security_manager = security_manager or get_security_manager()
        self.logger = get_logger(self.__class__.__name__)
        self.service_registry: Dict[str, ServiceIdentity] = {}
        self.trusted_services: Dict[str, List[str]] = {}
        
    def register_service(self, service_identity: ServiceIdentity):
        """註冊服務"""
        self.service_registry[service_identity.service_id] = service_identity
        self.logger.info(f"註冊服務: {service_identity.service_name}")
        
    def establish_trust_relationship(self, service_a: str, service_b: str):
        """建立服務間信任關係"""
        if service_a not in self.trusted_services:
            self.trusted_services[service_a] = []
        if service_b not in self.trusted_services:
            self.trusted_services[service_b] = []
            
        self.trusted_services[service_a].append(service_b)
        self.trusted_services[service_b].append(service_a)
        
        self.logger.info(f"建立信任關係: {service_a} ↔ {service_b}")
        
    async def validate_service_identity(self, service_id: str, certificate_data: bytes) -> bool:
        """驗證服務身份"""
        try:
            # 載入證書
            cert = x509.load_pem_x509_certificate(certificate_data)
            
            # 檢查證書是否有效
            now = datetime.datetime.now(datetime.timezone.utc)
            if now < cert.not_valid_before or now > cert.not_valid_after:
                self.logger.warning(f"服務證書已過期或尚未生效: {service_id}")
                return False
                
            # 檢查是否為註冊服務
            if service_id not in self.service_registry:
                self.logger.warning(f"未註冊的服務: {service_id}")
                return False
                
            # 檢查證書主題
            service_identity = self.service_registry[service_id]
            expected_cn = service_identity.service_name
            
            for attribute in cert.subject:
                if attribute.oid == NameOID.COMMON_NAME:
                    if attribute.value != expected_cn:
                        self.logger.warning(f"證書主題不符: 期望 {expected_cn}, 實際 {attribute.value}")
                        return False
                        
            self.logger.info(f"服務身份驗證成功: {service_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"服務身份驗證失敗: {e}")
            return False
            
    async def authorize_service_call(self, caller_service: str, target_service: str, action: str) -> bool:
        """授權服務調用"""
        try:
            # 檢查信任關係
            if caller_service not in self.trusted_services:
                self.logger.warning(f"調用方服務未建立信任關係: {caller_service}")
                return False
                
            if target_service not in self.trusted_services[caller_service]:
                self.logger.warning(f"服務間無信任關係: {caller_service} → {target_service}")
                return False
                
            # 檢查權限
            caller_identity = self.service_registry.get(caller_service)
            if not caller_identity:
                self.logger.warning(f"找不到調用方服務身份: {caller_service}")
                return False
                
            # 簡化權限檢查 - 可以根據需要擴展
            if action not in caller_identity.permissions and "*" not in caller_identity.permissions:
                self.logger.warning(f"服務 {caller_service} 無權限執行 {action}")
                return False
                
            self.logger.info(f"授權服務調用: {caller_service} → {target_service}:{action}")
            return True
            
        except Exception as e:
            self.logger.error(f"服務調用授權失敗: {e}")
            return False


class ZeroTrustPolicyEngine:
    """Zero Trust 策略引擎"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.load_default_policies()
        
    def load_default_policies(self):
        """載入預設的 Zero Trust 策略"""
        self.policies = {
            "default_deny": {
                "description": "預設拒絕所有請求",
                "effect": "deny",
                "conditions": {"default": True}
            },
            "service_authentication_required": {
                "description": "所有服務間通訊需要身份驗證",
                "effect": "require_auth",
                "conditions": {"service_to_service": True}
            },
            "encrypted_communication": {
                "description": "所有通訊必須加密",
                "effect": "require_encryption",
                "conditions": {"all_traffic": True}
            },
            "audit_all_requests": {
                "description": "審計所有請求",
                "effect": "audit",
                "conditions": {"all_requests": True}
            }
        }
        
    async def evaluate_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """評估請求是否符合 Zero Trust 策略"""
        results = {
            "allowed": False,
            "policies_applied": [],
            "violations": [],
            "required_actions": []
        }
        
        for policy_name, policy in self.policies.items():
            if await self._matches_conditions(request_context, policy["conditions"]):
                results["policies_applied"].append(policy_name)
                
                if policy["effect"] == "deny":
                    results["violations"].append(f"違反策略: {policy['description']}")
                elif policy["effect"] == "require_auth":
                    if not request_context.get("authenticated"):
                        results["violations"].append("需要身份驗證")
                        results["required_actions"].append("authenticate")
                elif policy["effect"] == "require_encryption":
                    if not request_context.get("encrypted"):
                        results["violations"].append("需要加密通訊")
                        results["required_actions"].append("encrypt")
                elif policy["effect"] == "audit":
                    results["required_actions"].append("audit")
                    
        # 如果沒有違反策略且滿足所有要求，則允許請求
        if not results["violations"]:
            results["allowed"] = True
            
        return results
        
    async def _matches_conditions(self, context: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """檢查請求是否符合策略條件"""
        for key, expected_value in conditions.items():
            if key == "default":
                return expected_value
            elif key == "service_to_service":
                return context.get("source_service") and context.get("target_service")
            elif key == "all_traffic":
                return True
            elif key == "all_requests":
                return True
            elif key in context:
                if context[key] != expected_value:
                    return False
                    
        return True


class SecureMicroservicesFramework:
    """安全微服務框架 - 主要協調器"""
    
    def __init__(self, ca_cert_path: str, ca_key_path: str):
        self.cert_manager = CertificateManager(ca_cert_path, ca_key_path)
        self.service_mesh = ServiceMeshSecurityManager()
        self.zero_trust = ZeroTrustPolicyEngine()
        self.logger = get_logger(self.__class__.__name__)
        
    async def setup_service_security(self, service_name: str, permissions: List[str] = None) -> ServiceIdentity:
        """為服務設置安全配置"""
        service_id = f"aiva-{service_name}-{uuid.uuid4().hex[:8]}"
        
        # 創建服務身份
        service_identity = ServiceIdentity(
            service_name=service_name,
            service_id=service_id,
            certificate_path=f"./security/certs/{service_name}.crt",
            private_key_path=f"./security/certs/{service_name}.key",
            trusted_ca_path="./security/certs/ca.crt",
            permissions=permissions or ["read", "write"]
        )
        
        # 生成證書
        await self.cert_manager.generate_service_certificate(service_identity)
        
        # 註冊服務
        self.service_mesh.register_service(service_identity)
        
        self.logger.info(f"服務安全設置完成: {service_name}")
        return service_identity
        
    async def create_secure_http_client(self, service_identity: ServiceIdentity) -> aiohttp.ClientSession:
        """創建安全的 HTTP 客戶端"""
        # 設置 mTLS
        ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ssl_context.load_cert_chain(
            service_identity.certificate_path, 
            service_identity.private_key_path
        )
        ssl_context.load_verify_locations(service_identity.trusted_ca_path)
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # 創建連接器
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        # 創建會話
        session = aiohttp.ClientSession(connector=connector)
        
        return session
        
    async def validate_service_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """驗證服務請求 - 整合所有安全檢查"""
        validation_result = {
            "valid": False,
            "security_checks": {},
            "recommendations": []
        }
        
        # 1. Zero Trust 策略評估
        zt_result = await self.zero_trust.evaluate_request(request_context)
        validation_result["security_checks"]["zero_trust"] = zt_result
        
        # 2. 服務身份驗證
        caller_service = request_context.get("caller_service")
        if caller_service:
            cert_data = request_context.get("client_certificate")
            if cert_data:
                identity_valid = await self.service_mesh.validate_service_identity(
                    caller_service, cert_data
                )
                validation_result["security_checks"]["identity"] = identity_valid
            else:
                validation_result["security_checks"]["identity"] = False
                validation_result["recommendations"].append("提供客戶端證書")
        
        # 3. 服務授權
        target_service = request_context.get("target_service")
        action = request_context.get("action", "access")
        if caller_service and target_service:
            authorized = await self.service_mesh.authorize_service_call(
                caller_service, target_service, action
            )
            validation_result["security_checks"]["authorization"] = authorized
        
        # 綜合判斷
        all_checks_passed = all(
            result for result in validation_result["security_checks"].values()
            if isinstance(result, bool)
        )
        
        zt_allowed = zt_result.get("allowed", False)
        validation_result["valid"] = all_checks_passed and zt_allowed
        
        return validation_result


# 使用範例和工廠函數
async def create_aiva_microservices_security() -> SecureMicroservicesFramework:
    """創建 AIVA 微服務安全框架"""
    
    # 設置證書目錄
    cert_dir = Path("./security/certs")
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    ca_cert_path = cert_dir / "ca.crt"
    ca_key_path = cert_dir / "ca.key"
    
    # 如果 CA 證書不存在，創建自簽名 CA
    if not ca_cert_path.exists():
        await _create_ca_certificate(ca_cert_path, ca_key_path)
    
    return SecureMicroservicesFramework(str(ca_cert_path), str(ca_key_path))


async def _create_ca_certificate(ca_cert_path: Path, ca_key_path: Path):
    """創建 CA 證書"""
    # 生成 CA 私鑰
    ca_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )
    
    # 創建 CA 證書
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "TW"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Taiwan"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Taipei"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIVA Security Platform"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Certificate Authority"),
        x509.NameAttribute(NameOID.COMMON_NAME, "AIVA Internal CA"),
    ])
    
    ca_cert = x509.CertificateBuilder().subject_name(
        ca_subject
    ).issuer_name(
        ca_subject
    ).public_key(
        ca_private_key.public_key()
    ).serial_number(
        1
    ).not_valid_before(
        datetime.datetime.now(datetime.timezone.utc)
    ).not_valid_after(
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650)  # 10年
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    ).add_extension(
        x509.KeyUsage(
            key_cert_sign=True,
            crl_sign=True,
            digital_signature=False,
            key_encipherment=False,
            key_agreement=False,
            content_commitment=False,
            data_encipherment=False,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=True,
    ).sign(ca_private_key, hashes.SHA256())
    
    # 儲存 CA 證書和私鑰
    with open(ca_cert_path, 'wb') as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
    
    with open(ca_key_path, 'wb') as f:
        f.write(ca_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))


# 測試和驗證函數
async def test_microservices_security():
    """測試微服務安全框架"""
    print("🔐 測試 AIVA 微服務安全框架")
    
    # 創建安全框架
    security_framework = await create_aiva_microservices_security()
    
    # 設置兩個服務
    core_service = await security_framework.setup_service_security("core", ["read", "write", "admin"])
    features_service = await security_framework.setup_service_security("features", ["read", "execute"])
    
    # 建立信任關係
    security_framework.service_mesh.establish_trust_relationship(
        core_service.service_id, 
        features_service.service_id
    )
    
    # 測試請求驗證
    test_request = {
        "caller_service": core_service.service_id,
        "target_service": features_service.service_id,
        "action": "execute",
        "authenticated": True,
        "encrypted": True,
        "source_service": "core",
        "target_service": "features"
    }
    
    validation_result = await security_framework.validate_service_request(test_request)
    
    print(f"📊 驗證結果: {validation_result}")
    print("✅ 微服務安全框架測試完成")
    
    return validation_result


if __name__ == "__main__":
    asyncio.run(test_microservices_security())