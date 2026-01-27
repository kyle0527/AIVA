"""
Web 架構安全漏洞檢測模組

基於 Web架構安全漏洞檢測指南.md 的專業知識，
提供針對現代 Web 架構的安全檢測能力。

支援的架構類型:
    - GraphQL API
    - JWT 認證系統
    - WebSocket 實時通訊
    - REST API
    - gRPC
    - Server-Sent Events (SSE)

漏洞類型:
    - GraphQL Introspection 暴露
    - JWT None Algorithm 攻擊
    - Mass Assignment
    - BOLA (Broken Object Level Authorization)
    - WebSocket Hijacking
    - API Rate Limit Bypass
"""

import re
import json
import base64
from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto

from .base import (
    DetectionResult,
    ConfidenceLevel,
    VulnerabilityType,
    AttackContext,
    ResponseAnalysis,
)


class ArchitectureType(Enum):
    """Web 架構類型"""
    REST_API = auto()
    GRAPHQL = auto()
    WEBSOCKET = auto()
    GRPC = auto()
    SSE = auto()  # Server-Sent Events
    SOAP = auto()


class AuthScheme(Enum):
    """認證方案"""
    JWT = auto()
    OAUTH2 = auto()
    SESSION = auto()
    BASIC = auto()
    API_KEY = auto()
    NONE = auto()


@dataclass
class ArchitectureFingerprint:
    """架構指紋"""
    arch_type: ArchitectureType
    confidence: ConfidenceLevel
    indicators: list[str]
    endpoints: list[str] = field(default_factory=list)
    auth_scheme: AuthScheme = AuthScheme.NONE
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "arch_type": self.arch_type.name,
            "confidence": self.confidence.name,
            "indicators": self.indicators,
            "endpoints": self.endpoints,
            "auth_scheme": self.auth_scheme.name,
            "metadata": self.metadata,
        }


@dataclass
class JWTAnalysis:
    """JWT 分析結果"""
    is_vulnerable: bool
    algorithms: list[str]
    issues: list[str]
    header: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    exploitable_attacks: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_vulnerable": self.is_vulnerable,
            "algorithms": self.algorithms,
            "issues": self.issues,
            "header": self.header,
            "payload": self.payload,
            "exploitable_attacks": self.exploitable_attacks,
        }


@dataclass
class GraphQLSchema:
    """GraphQL Schema 分析"""
    has_introspection: bool
    types: list[str]
    mutations: list[str]
    queries: list[str]
    sensitive_fields: list[str] = field(default_factory=list)
    security_issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "has_introspection": self.has_introspection,
            "types": self.types,
            "mutations": self.mutations,
            "queries": self.queries,
            "sensitive_fields": self.sensitive_fields,
            "security_issues": self.security_issues,
        }


class WebArchitectureAnalyzer:
    """
    Web 架構安全分析器
    
    提供針對現代 Web 架構的安全漏洞檢測能力，
    AI 決策系統可調用此分析器進行深度架構分析。
    
    使用範例:
        # 檢測 GraphQL introspection
        result = WebArchitectureAnalyzer.detect_graphql_introspection(
            endpoint="https://api.example.com/graphql"
        )
        
        # 分析 JWT token
        jwt_analysis = WebArchitectureAnalyzer.analyze_jwt(
            token="eyJhbGciOiJub25lIn0..."
        )
        
        # 檢測 BOLA 漏洞
        bola_result = WebArchitectureAnalyzer.check_bola(
            base_url="https://api.example.com",
            resource_type="users",
            user1_token="token1",
            user2_token="token2"
        )
    """
    
    # ==================== GraphQL 檢測 ====================
    
    GRAPHQL_INTROSPECTION_QUERY = """
    query IntrospectionQuery {
      __schema {
        queryType { name }
        mutationType { name }
        subscriptionType { name }
        types {
          ...FullType
        }
        directives {
          name
          description
          locations
          args {
            ...InputValue
          }
        }
      }
    }
    
    fragment FullType on __Type {
      kind
      name
      description
      fields(includeDeprecated: true) {
        name
        description
        args {
          ...InputValue
        }
        type {
          ...TypeRef
        }
        isDeprecated
        deprecationReason
      }
      inputFields {
        ...InputValue
      }
      interfaces {
        ...TypeRef
      }
      enumValues(includeDeprecated: true) {
        name
        description
        isDeprecated
        deprecationReason
      }
      possibleTypes {
        ...TypeRef
      }
    }
    
    fragment InputValue on __InputValue {
      name
      description
      type { ...TypeRef }
      defaultValue
    }
    
    fragment TypeRef on __Type {
      kind
      name
      ofType {
        kind
        name
        ofType {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    
    SENSITIVE_FIELD_PATTERNS = [
        r"password", r"secret", r"token", r"api[_-]?key",
        r"private[_-]?key", r"ssn", r"credit[_-]?card",
        r"cvv", r"pin", r"salt", r"hash",
    ]
    
    @classmethod
    def detect_graphql_introspection(
        cls,
        endpoint: str,
        response_data: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """
        檢測 GraphQL introspection 是否啟用
        
        Args:
            endpoint: GraphQL 端點
            response_data: 如果已發送 introspection query，提供響應數據
            
        Returns:
            檢測結果
        """
        if response_data is None:
            # 提供檢測建議
            return DetectionResult(
                detected=False,
                vulnerability_type=VulnerabilityType.INFO_DISCLOSURE,
                confidence=ConfidenceLevel.UNCERTAIN,
                confidence_score=0.0,
                evidence=[
                    "需要發送 introspection query 進行檢測"
                ],
                recommendations=[
                    f"發送 POST 請求到 {endpoint}",
                    "使用標準 introspection query",
                    "Content-Type: application/json",
                ],
                technical_details={
                    "introspection_query": cls.GRAPHQL_INTROSPECTION_QUERY,
                },
            )
        
        # 分析 introspection 響應
        schema_data = response_data.get("data", {}).get("__schema", {})
        
        if not schema_data:
            return DetectionResult(
                detected=False,
                vulnerability_type=VulnerabilityType.INFO_DISCLOSURE,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                evidence=["Introspection query 被拒絕或禁用"],
                recommendations=["這是正確的安全配置"],
            )
        
        # Introspection 啟用，分析 schema
        types = schema_data.get("types", [])
        type_names = [t.get("name", "") for t in types if t.get("name")]
        
        # 檢測敏感字段
        sensitive_fields: list[str] = []
        for type_obj in types:
            fields = type_obj.get("fields", [])
            if fields:
                for field in fields:
                    field_name = field.get("name", "")
                    for pattern in cls.SENSITIVE_FIELD_PATTERNS:
                        if re.search(pattern, field_name, re.IGNORECASE):
                            sensitive_fields.append(
                                f"{type_obj.get('name', 'Unknown')}.{field_name}"
                            )
        
        queries = schema_data.get("queryType", {}).get("name", "")
        mutations = schema_data.get("mutationType", {}).get("name", "")
        
        evidence = [
            f"Introspection 已啟用",
            f"發現 {len(type_names)} 個類型",
            f"Query 類型: {queries}" if queries else "無 Query",
            f"Mutation 類型: {mutations}" if mutations else "無 Mutation",
        ]
        
        if sensitive_fields:
            evidence.append(f"發現 {len(sensitive_fields)} 個敏感字段")
        
        return DetectionResult(
            detected=True,
            vulnerability_type=VulnerabilityType.INFO_DISCLOSURE,
            confidence=ConfidenceLevel.ABSOLUTE,
            confidence_score=1.0,
            evidence=evidence,
            recommendations=[
                "生產環境應禁用 introspection",
                "使用 persisted queries 替代",
                "實施查詢複雜度限制",
                "對敏感字段實施字段級授權",
            ],
            technical_details={
                "types": type_names[:20],  # 限制數量
                "sensitive_fields": sensitive_fields,
                "queries": queries,
                "mutations": mutations,
            },
        )
    
    @classmethod
    def parse_graphql_schema(
        cls,
        schema_response: dict[str, Any],
    ) -> GraphQLSchema:
        """
        解析 GraphQL schema
        
        Returns:
            Schema 分析結果
        """
        schema_data = schema_response.get("data", {}).get("__schema", {})
        
        if not schema_data:
            return GraphQLSchema(
                has_introspection=False,
                types=[],
                mutations=[],
                queries=[],
            )
        
        types = [t.get("name", "") for t in schema_data.get("types", [])]
        
        # 提取 mutations
        mutations: list[str] = []
        mutation_type = schema_data.get("mutationType", {})
        if mutation_type:
            # 需要在 types 中找到 Mutation 類型的詳細定義
            for type_obj in schema_data.get("types", []):
                if type_obj.get("name") == mutation_type.get("name"):
                    fields = type_obj.get("fields", [])
                    mutations = [f.get("name", "") for f in fields]
                    break
        
        # 提取 queries
        queries: list[str] = []
        query_type = schema_data.get("queryType", {})
        if query_type:
            for type_obj in schema_data.get("types", []):
                if type_obj.get("name") == query_type.get("name"):
                    fields = type_obj.get("fields", [])
                    queries = [f.get("name", "") for f in fields]
                    break
        
        # 檢測敏感字段
        sensitive_fields: list[str] = []
        for type_obj in schema_data.get("types", []):
            fields = type_obj.get("fields", [])
            if fields:
                for field in fields:
                    field_name = field.get("name", "")
                    for pattern in cls.SENSITIVE_FIELD_PATTERNS:
                        if re.search(pattern, field_name, re.IGNORECASE):
                            sensitive_fields.append(
                                f"{type_obj.get('name')}.{field_name}"
                            )
        
        security_issues: list[str] = []
        if sensitive_fields:
            security_issues.append(f"發現 {len(sensitive_fields)} 個敏感字段暴露")
        if len(queries) > 100:
            security_issues.append("Query 數量過多，可能暴露過多業務邏輯")
        if "Admin" in types or "Internal" in types:
            security_issues.append("發現管理/內部相關類型")
        
        return GraphQLSchema(
            has_introspection=True,
            types=types,
            mutations=mutations,
            queries=queries,
            sensitive_fields=sensitive_fields,
            security_issues=security_issues,
        )
    
    # ==================== JWT 分析 ====================
    
    @classmethod
    def analyze_jwt(
        cls,
        token: str,
    ) -> JWTAnalysis:
        """
        分析 JWT token 安全性
        
        檢測:
            - None algorithm 攻擊
            - 弱簽名算法
            - 敏感信息洩露
            - 過期時間配置
            
        Args:
            token: JWT token
            
        Returns:
            JWT 分析結果
        """
        issues: list[str] = []
        exploitable_attacks: list[str] = []
        
        # 解析 JWT
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return JWTAnalysis(
                    is_vulnerable=False,
                    algorithms=[],
                    issues=["無效的 JWT 格式"],
                )
            
            # 解碼 header
            header_b64 = parts[0]
            # 添加填充
            header_b64 += "=" * (4 - len(header_b64) % 4)
            header_json = base64.urlsafe_b64decode(header_b64).decode()
            header = json.loads(header_json)
            
            # 解碼 payload
            payload_b64 = parts[1]
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64).decode()
            payload = json.loads(payload_json)
            
        except Exception as e:
            return JWTAnalysis(
                is_vulnerable=False,
                algorithms=[],
                issues=[f"JWT 解析失敗: {str(e)}"],
            )
        
        # 檢查算法
        alg = header.get("alg", "").upper()
        algorithms = [alg]
        
        # None algorithm 檢測
        if alg == "NONE" or alg == "":
            issues.append("使用 'none' 算法，token 未簽名")
            exploitable_attacks.append("none_algorithm_bypass")
        
        # 弱算法檢測
        weak_algs = ["HS256", "HS384", "HS512"]  # HMAC 可能被暴力破解
        if alg in weak_algs:
            issues.append(f"使用對稱算法 {alg}，容易被暴力破解")
            exploitable_attacks.append("hmac_bruteforce")
        
        # RS256/HS256 混淆攻擊
        if alg.startswith("RS"):
            issues.append("使用非對稱算法，注意 algorithm confusion 攻擊")
            exploitable_attacks.append("algorithm_confusion")
        
        # 檢查 payload 敏感信息
        sensitive_keys = ["password", "secret", "api_key", "private_key", "ssn"]
        for key in sensitive_keys:
            if key in payload:
                issues.append(f"Payload 包含敏感信息: {key}")
        
        # 檢查過期時間
        exp = payload.get("exp")
        iat = payload.get("iat")
        if not exp:
            issues.append("缺少過期時間 (exp)")
            exploitable_attacks.append("token_never_expires")
        elif exp and iat:
            lifetime = exp - iat
            if lifetime > 86400 * 30:  # 30 天
                issues.append(f"Token 有效期過長: {lifetime // 86400} 天")
        
        # 檢查 kid (Key ID) header injection
        if "kid" in header:
            kid = header["kid"]
            if "../" in kid or "%" in kid:
                issues.append("kid header 可能存在路徑遍歷漏洞")
                exploitable_attacks.append("kid_path_traversal")
        
        # 檢查 jku/jwk headers
        if "jku" in header:
            issues.append("使用 jku header，可能被篡改為惡意 URL")
            exploitable_attacks.append("jku_injection")
        
        if "jwk" in header:
            issues.append("JWT header 中嵌入公鑰，可被替換")
            exploitable_attacks.append("jwk_injection")
        
        is_vulnerable = len(exploitable_attacks) > 0
        
        return JWTAnalysis(
            is_vulnerable=is_vulnerable,
            algorithms=algorithms,
            issues=issues,
            header=header,
            payload=payload,
            exploitable_attacks=exploitable_attacks,
        )
    
    @classmethod
    def generate_jwt_attack_payloads(
        cls,
        original_token: str,
        attack_type: str = "none_algorithm",
    ) -> list[dict[str, str]]:
        """
        生成 JWT 攻擊 payload
        
        Args:
            original_token: 原始 JWT token
            attack_type: 攻擊類型
                - none_algorithm: None 算法繞過
                - algorithm_confusion: RS256 -> HS256 混淆
                - kid_sqli: kid header SQL 注入
                
        Returns:
            攻擊 payload 列表
        """
        payloads: list[dict[str, str]] = []
        
        try:
            parts = original_token.split(".")
            if len(parts) != 3:
                return []
            
            # 解碼 header 和 payload
            header_b64 = parts[0]
            header_b64 += "=" * (4 - len(header_b64) % 4)
            header = json.loads(base64.urlsafe_b64decode(header_b64))
            
            payload_b64 = parts[1]
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            
        except Exception:
            return []
        
        if attack_type == "none_algorithm":
            # 修改 alg 為 none
            header["alg"] = "none"
            new_header = base64.urlsafe_b64encode(
                json.dumps(header).encode()
            ).decode().rstrip("=")
            new_payload = base64.urlsafe_b64encode(
                json.dumps(payload).encode()
            ).decode().rstrip("=")
            
            # None 算法不需要簽名
            payloads.append({
                "description": "None algorithm bypass (無簽名)",
                "token": f"{new_header}.{new_payload}.",
            })
            payloads.append({
                "description": "None algorithm bypass (空簽名)",
                "token": f"{new_header}.{new_payload}",
            })
            
        elif attack_type == "algorithm_confusion":
            # RS256 -> HS256 混淆
            if header.get("alg", "").startswith("RS"):
                header["alg"] = "HS256"
                new_header = base64.urlsafe_b64encode(
                    json.dumps(header).encode()
                ).decode().rstrip("=")
                payloads.append({
                    "description": "Algorithm confusion (RS256 -> HS256)",
                    "token": f"{new_header}.{parts[1]}.SIGNATURE_HERE",
                    "note": "需要用公鑰作為 HMAC secret 重新簽名",
                })
        
        elif attack_type == "kid_sqli":
            # kid header SQL 注入
            sqli_payloads = [
                "../../dev/null",
                "../../../etc/passwd",
                "key' UNION SELECT 'secret",
                "key'; DROP TABLE keys--",
            ]
            for kid_payload in sqli_payloads:
                header["kid"] = kid_payload
                new_header = base64.urlsafe_b64encode(
                    json.dumps(header).encode()
                ).decode().rstrip("=")
                payloads.append({
                    "description": f"kid header injection: {kid_payload}",
                    "token": f"{new_header}.{parts[1]}.{parts[2]}",
                })
        
        return payloads
    
    # ==================== BOLA/IDOR 檢測 ====================
    
    @classmethod
    def check_bola(
        cls,
        ctx: AttackContext,
        user1_response: ResponseAnalysis,
        user2_response: ResponseAnalysis,
        resource_id_accessed: str,
    ) -> DetectionResult:
        """
        檢測 BOLA (Broken Object Level Authorization)
        
        BOLA 也稱為 IDOR (Insecure Direct Object Reference)
        
        測試邏輯:
            1. User1 訪問 /api/resource/{user2_resource_id}
            2. 如果成功且返回 User2 的數據 -> BOLA 漏洞
            
        Args:
            ctx: 攻擊上下文
            user1_response: User1 的響應
            user2_response: User2 訪問自己資源的響應 (用於對比)
            resource_id_accessed: User1 嘗試訪問的資源 ID (屬於 User2)
            
        Returns:
            檢測結果
        """
        evidence: list[str] = []
        
        # 檢查 status code
        if user1_response.status_code == 403 or user1_response.status_code == 401:
            return DetectionResult(
                detected=False,
                vulnerability_type=VulnerabilityType.BOLA,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                evidence=[
                    "訪問被拒絕 (403/401)",
                    "存在對象級授權檢查",
                ],
                recommendations=["這是正確的授權實現"],
            )
        
        if user1_response.status_code != 200:
            return DetectionResult(
                detected=False,
                vulnerability_type=VulnerabilityType.BOLA,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.5,
                evidence=[f"響應狀態碼: {user1_response.status_code}"],
            )
        
        # 成功訪問 (200)
        evidence.append(f"User1 成功訪問 User2 的資源 ID: {resource_id_accessed}")
        
        # 對比響應內容相似度
        similarity = cls._calculate_response_similarity(
            user1_response.body,
            user2_response.body,
        )
        
        evidence.append(f"響應相似度: {similarity:.1%}")
        
        # 檢查是否包含 User2 的敏感數據
        user2_indicators = ["user2", "email", "phone", "address"]
        found_indicators = []
        for indicator in user2_indicators:
            if indicator in user1_response.body.lower():
                found_indicators.append(indicator)
        
        if found_indicators:
            evidence.append(f"響應包含 User2 敏感字段: {found_indicators}")
        
        # 判斷
        if similarity > 0.8 or found_indicators:
            return DetectionResult(
                detected=True,
                vulnerability_type=VulnerabilityType.BOLA,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.95,
                evidence=evidence,
                recommendations=[
                    "實施對象級授權檢查",
                    "驗證當前用戶是否有權訪問該資源",
                    "不要僅依賴 URL 中的資源 ID",
                    "使用 UUID 而非自增 ID",
                ],
                technical_details={
                    "similarity": similarity,
                    "resource_id": resource_id_accessed,
                },
            )
        
        return DetectionResult(
            detected=False,
            vulnerability_type=VulnerabilityType.BOLA,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            evidence=evidence,
        )
    
    @classmethod
    def _calculate_response_similarity(
        cls,
        response1: str,
        response2: str,
    ) -> float:
        """計算兩個響應的相似度"""
        if not response1 or not response2:
            return 0.0
        
        # 簡單的字符集相似度
        set1 = set(response1.split())
        set2 = set(response2.split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    # ==================== Mass Assignment 檢測 ====================
    
    @classmethod
    def check_mass_assignment(
        cls,
        endpoint: str,
        baseline_params: dict[str, Any],
        test_params: dict[str, Any],
        baseline_response: ResponseAnalysis,
        test_response: ResponseAnalysis,
    ) -> DetectionResult:
        """
        檢測 Mass Assignment 漏洞
        
        測試邏輯:
            1. 正常請求: {"username": "test", "email": "test@example.com"}
            2. 測試請求: {"username": "test", "email": "test@example.com", "role": "admin", "is_admin": true}
            3. 如果 role/is_admin 被成功設置 -> Mass Assignment
            
        Args:
            endpoint: API 端點
            baseline_params: 基準參數
            test_params: 測試參數 (包含額外字段)
            baseline_response: 基準響應
            test_response: 測試響應
            
        Returns:
            檢測結果
        """
        evidence: list[str] = []
        
        # 找出額外的字段
        extra_fields = set(test_params.keys()) - set(baseline_params.keys())
        evidence.append(f"測試了 {len(extra_fields)} 個額外字段: {list(extra_fields)}")
        
        # 檢查 status code
        if test_response.status_code >= 400:
            return DetectionResult(
                detected=False,
                vulnerability_type=VulnerabilityType.MASS_ASSIGNMENT,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.6,
                evidence=[
                    f"額外字段被拒絕 ({test_response.status_code})",
                    "可能存在輸入驗證",
                ],
            )
        
        # 檢查響應中是否反映了額外字段
        try:
            test_body = json.loads(test_response.body)
            assigned_fields: list[str] = []
            
            for field in extra_fields:
                if field in test_body:
                    # 檢查值是否與我們提交的相同
                    if test_body[field] == test_params[field]:
                        assigned_fields.append(field)
                        evidence.append(f"字段 '{field}' 被成功賦值")
            
            if assigned_fields:
                # 檢查是否包含特權字段
                privileged_fields = {"role", "is_admin", "admin", "permissions", "level"}
                dangerous = [f for f in assigned_fields if f in privileged_fields]
                
                confidence = ConfidenceLevel.HIGH if dangerous else ConfidenceLevel.MEDIUM
                
                return DetectionResult(
                    detected=True,
                    vulnerability_type=VulnerabilityType.MASS_ASSIGNMENT,
                    confidence=confidence,
                    confidence_score=0.9 if dangerous else 0.7,
                    evidence=evidence,
                    recommendations=[
                        "使用白名單機制限制可賦值字段",
                        "實施 DTO (Data Transfer Object) 模式",
                        "避免直接將用戶輸入綁定到模型",
                        "對敏感字段實施額外驗證",
                    ],
                    technical_details={
                        "assigned_fields": assigned_fields,
                        "dangerous_fields": dangerous,
                    },
                )
        except json.JSONDecodeError:
            pass
        
        return DetectionResult(
            detected=False,
            vulnerability_type=VulnerabilityType.MASS_ASSIGNMENT,
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
            evidence=evidence,
        )
    
    # ==================== WebSocket 安全檢測 ====================
    
    @classmethod
    def check_websocket_security(
        cls,
        ws_url: str,
        handshake_response: dict[str, Any],
        initial_message: str | None = None,
    ) -> DetectionResult:
        """
        檢測 WebSocket 安全配置
        
        檢測項:
            - Origin 驗證
            - 認證機制
            - CSRF token
            
        Args:
            ws_url: WebSocket URL
            handshake_response: 握手響應
            initial_message: 初始消息
            
        Returns:
            檢測結果
        """
        evidence: list[str] = []
        issues: list[str] = []
        
        headers = handshake_response.get("headers", {})
        status_code = handshake_response.get("status_code", 0)
        
        # 檢查握手是否成功
        if status_code == 101:
            evidence.append("WebSocket 握手成功 (101 Switching Protocols)")
        else:
            return DetectionResult(
                detected=False,
                vulnerability_type=VulnerabilityType.WEBSOCKET_HIJACK,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.2,
                evidence=[f"握手失敗: {status_code}"],
            )
        
        # 檢查 Origin 驗證
        if "origin" not in [h.lower() for h in headers.keys()]:
            issues.append("未檢查 Origin header")
            evidence.append("可能存在 CSRF 漏洞")
        
        # 檢查認證
        auth_headers = ["authorization", "cookie", "sec-websocket-protocol"]
        has_auth = any(h.lower() in [ah.lower() for ah in auth_headers] for h in headers.keys())
        
        if not has_auth:
            issues.append("未發現認證機制")
            evidence.append("WebSocket 連接未受保護")
        
        # 檢查協議升級頭
        upgrade = headers.get("Upgrade", "").lower()
        connection = headers.get("Connection", "").lower()
        
        if "websocket" not in upgrade or "upgrade" not in connection:
            issues.append("協議升級 headers 異常")
        
        is_vulnerable = len(issues) > 0
        
        return DetectionResult(
            detected=is_vulnerable,
            vulnerability_type=VulnerabilityType.WEBSOCKET_HIJACK,
            confidence=ConfidenceLevel.MEDIUM if is_vulnerable else ConfidenceLevel.LOW,
            confidence_score=0.7 if is_vulnerable else 0.3,
            evidence=evidence,
            recommendations=[
                "實施 Origin 白名單驗證",
                "在握手階段進行認證",
                "使用 CSRF token",
                "僅允許 wss:// (WebSocket over TLS)",
            ] if is_vulnerable else [],
            technical_details={
                "issues": issues,
                "ws_url": ws_url,
            },
        )
    
    # ==================== 架構指紋識別 ====================
    
    @classmethod
    def identify_architecture(
        cls,
        response_headers: dict[str, str],
        response_body: str,
        endpoints: list[str] | None = None,
    ) -> ArchitectureFingerprint:
        """
        識別 Web 架構類型
        
        Args:
            response_headers: 響應 headers
            response_body: 響應 body
            endpoints: 已知端點列表
            
        Returns:
            架構指紋
        """
        endpoints = endpoints or []
        indicators: list[str] = []
        metadata: dict[str, Any] = {}
        
        # GraphQL 檢測
        if "/graphql" in response_body.lower() or any("/graphql" in e for e in endpoints):
            indicators.append("GraphQL endpoint detected")
            if "application/graphql" in response_headers.get("content-type", ""):
                indicators.append("GraphQL content-type header")
            
            return ArchitectureFingerprint(
                arch_type=ArchitectureType.GRAPHQL,
                confidence=ConfidenceLevel.HIGH,
                indicators=indicators,
                endpoints=[e for e in endpoints if "graphql" in e.lower()],
                metadata=metadata,
            )
        
        # gRPC 檢測
        if "application/grpc" in response_headers.get("content-type", ""):
            indicators.append("gRPC content-type")
            return ArchitectureFingerprint(
                arch_type=ArchitectureType.GRPC,
                confidence=ConfidenceLevel.ABSOLUTE,
                indicators=indicators,
                endpoints=endpoints,
                metadata=metadata,
            )
        
        # WebSocket 檢測
        if "upgrade" in response_headers.get("connection", "").lower():
            if "websocket" in response_headers.get("upgrade", "").lower():
                indicators.append("WebSocket upgrade header")
                return ArchitectureFingerprint(
                    arch_type=ArchitectureType.WEBSOCKET,
                    confidence=ConfidenceLevel.ABSOLUTE,
                    indicators=indicators,
                    endpoints=endpoints,
                    metadata=metadata,
                )
        
        # SSE 檢測
        if "text/event-stream" in response_headers.get("content-type", ""):
            indicators.append("SSE content-type")
            return ArchitectureFingerprint(
                arch_type=ArchitectureType.SSE,
                confidence=ConfidenceLevel.ABSOLUTE,
                indicators=indicators,
                endpoints=endpoints,
                metadata=metadata,
            )
        
        # SOAP 檢測
        if "xml" in response_headers.get("content-type", ""):
            if "<soap:" in response_body.lower() or "<wsdl:" in response_body.lower():
                indicators.append("SOAP/WSDL detected in body")
                return ArchitectureFingerprint(
                    arch_type=ArchitectureType.SOAP,
                    confidence=ConfidenceLevel.HIGH,
                    indicators=indicators,
                    endpoints=endpoints,
                    metadata=metadata,
                )
        
        # 默認 REST API
        indicators.append("Standard REST API")
        return ArchitectureFingerprint(
            arch_type=ArchitectureType.REST_API,
            confidence=ConfidenceLevel.MEDIUM,
            indicators=indicators,
            endpoints=endpoints,
            metadata=metadata,
        )
