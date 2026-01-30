# Embedded Knowledge 使用指南

## 概述

`embedded_knowledge` 模組提供了 AIVA 攻擊決策系統所需的內建專業安全知識，無需 RAG 搜索即可直接調用。

## 模組結構

```
embedded_knowledge/
├── base.py                      # 基礎類型定義
├── vulnerability_detection.py   # 漏洞檢測 (SQLi/XSS/SSRF/IDOR)
├── cve_identification.py        # 高危 CVE 識別
├── waf_bypass.py               # WAF 繞過技術
└── web_architecture.py         # 現代 Web 架構安全
```

## 快速開始

### 1. SQLi 檢測

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    VulnerabilityDetector,
    AttackContext,
    DatabaseType,
)

# 創建攻擊上下文
ctx = AttackContext(
    target_url="https://example.com/login",
    injection_point="username",
    database=DatabaseType.MYSQL,
)

# 檢測 SQL 注入
result = VulnerabilityDetector.check_sqli(
    response_body="You have an error in your SQL syntax",
    response_time=0.15,
    context=ctx,
)

# AI 決策判斷
if result.should_exploit():
    print(f"檢測到 {result.vulnerability_type.value}")
    print(f"置信度: {result.confidence_score:.2%}")
    print(f"建議動作: {result.recommendations}")
```

### 2. CVE 識別

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import CVEIdentifier

# 目標指紋
fingerprint = {
    "http_headers": {
        "Server": "Apache/2.4.48",
        "X-Powered-By": "Spring Boot 2.5.0",
    },
    "response_body": "<title>Spring Framework</title>",
    "technologies": ["java", "spring"],
}

# 識別已知 CVE
matches = CVEIdentifier.identify(fingerprint)

for match in matches:
    if match.is_exploitable():
        print(f"發現高危 CVE: {match.cve_id}")
        print(f"CVSS: {match.cvss_score}")
        print(f"信號層級: {match.highest_signal_tier.name}")
        
        # 獲取利用 payload
        payloads = CVEIdentifier.get_exploit_payloads(match.cve_id)
        print(f"可用 Payload: {len(payloads)} 個")
```

### 3. WAF 檢測與繞過

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    WAFBypassEngine,
    WAFVendor,
)

# 檢測 WAF
is_waf, vendor, indicators = WAFBypassEngine.detect_waf(
    response_body="Attention Required! Cloudflare Ray ID: 123456",
    response_headers={"cf-ray": "123456"},
    status_code=403,
)

if is_waf:
    print(f"檢測到 WAF: {vendor.name}")
    
    # 獲取繞過技術
    techniques = WAFBypassEngine.get_bypass_techniques(
        waf_vendor=vendor,
        attack_type="sqli",
        min_success_rate=0.5,
    )
    
    for tech in techniques:
        print(f"\n技術: {tech.name}")
        print(f"成功率: {tech.success_rate:.0%}")
        print(f"描述: {tech.description}")
        
        # 使用繞過 headers
        if tech.headers:
            print(f"特殊 Headers: {tech.headers}")

# Payload 變形
original_payload = "' OR 1=1--"
variants = WAFBypassEngine.mutate_payload(
    original_payload,
    mutation_types=["double_url", "comment_inject", "hex_encode"]
)

for variant in variants:
    print(f"{variant['type']}: {variant['payload']}")
```

### 4. GraphQL 安全檢測

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    WebArchitectureAnalyzer,
)

# 檢測 GraphQL introspection
result = WebArchitectureAnalyzer.detect_graphql_introspection(
    endpoint="https://api.example.com/graphql",
    response_data={
        "data": {
            "__schema": {
                "types": [{"name": "User"}, {"name": "Admin"}],
                "queryType": {"name": "Query"},
            }
        }
    }
)

if result.detected:
    print("GraphQL Introspection 已啟用")
    print(f"置信度: {result.confidence.name}")
    
    # 解析 Schema
    schema = WebArchitectureAnalyzer.parse_graphql_schema(response_data)
    print(f"發現 {len(schema.types)} 個類型")
    print(f"敏感字段: {schema.sensitive_fields}")
```

### 5. JWT 安全分析

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    WebArchitectureAnalyzer,
)

# 分析 JWT token
jwt_token = "eyJhbGciOiJub25lIn0.eyJ1c2VyIjoiYWRtaW4ifQ."

analysis = WebArchitectureAnalyzer.analyze_jwt(jwt_token)

if analysis.is_vulnerable:
    print("JWT Token 存在安全問題:")
    for issue in analysis.issues:
        print(f"  - {issue}")
    
    # 可利用的攻擊
    print(f"\n可利用攻擊: {analysis.exploitable_attacks}")
    
    # 生成攻擊 payload
    payloads = WebArchitectureAnalyzer.generate_jwt_attack_payloads(
        jwt_token,
        attack_type="none_algorithm"
    )
    
    for payload in payloads:
        print(f"\n{payload['description']}")
        print(f"Token: {payload['token']}")
```

### 6. BOLA/IDOR 檢測

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    WebArchitectureAnalyzer,
    AttackContext,
    ResponseAnalysis,
)

# 模擬兩個用戶的響應
user1_response = ResponseAnalysis(
    status_code=200,
    body='{"id": 2, "name": "User2", "email": "user2@example.com"}',
    headers={},
)

user2_response = ResponseAnalysis(
    status_code=200,
    body='{"id": 2, "name": "User2", "email": "user2@example.com"}',
    headers={},
)

ctx = AttackContext(
    target_url="https://api.example.com/users/2",
    injection_point="path",
)

# 檢測 BOLA
result = WebArchitectureAnalyzer.check_bola(
    ctx=ctx,
    user1_response=user1_response,
    user2_response=user2_response,
    resource_id_accessed="2",
)

if result.detected:
    print("檢測到 BOLA 漏洞")
    print(f"證據: {result.evidence}")
    print(f"建議: {result.recommendations}")
```

## 與決策系統整合

### 在 EnhancedDecisionAgent 中使用

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    VulnerabilityDetector,
    CVEIdentifier,
    WAFBypassEngine,
    KnowledgeRegistry,
)

class EnhancedDecisionAgent:
    def __init__(self):
        self.vulnerability_detector = VulnerabilityDetector
        self.cve_identifier = CVEIdentifier
        self.waf_bypass_engine = WAFBypassEngine
    
    def analyze_attack_result(self, attack_result: dict) -> dict:
        """分析攻擊結果，決定下一步行動"""
        
        # 1. 檢測漏洞類型
        detection = self.vulnerability_detector.check_sqli(
            response_body=attack_result["response"],
            response_time=attack_result["time"],
        )
        
        if not detection.detected:
            return {"action": "try_different_payload"}
        
        # 2. 檢測 WAF
        is_waf, vendor, _ = self.waf_bypass_engine.detect_waf(
            response_body=attack_result["response"],
            response_headers=attack_result["headers"],
            status_code=attack_result["status"],
        )
        
        # 3. 如果有 WAF，獲取繞過技術
        if is_waf:
            bypass_techniques = self.waf_bypass_engine.get_bypass_techniques(
                waf_vendor=vendor,
                attack_type="sqli",
            )
            return {
                "action": "apply_waf_bypass",
                "techniques": [t.to_dict() for t in bypass_techniques],
            }
        
        # 4. 直接利用
        return {
            "action": "exploit",
            "confidence": detection.confidence_score,
        }
```

## 擴展知識庫

### 註冊自定義 SQLi 指紋

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    KnowledgeRegistry,
    DatabaseType,
)

# 註冊新的 SQLi 錯誤指紋
KnowledgeRegistry.register_sqli_fingerprint(
    database=DatabaseType.MYSQL,
    pattern=r"custom error pattern",
)

# 註冊新的 XSS payload
KnowledgeRegistry.register_xss_payload(
    payload="<custom>alert(1)</custom>",
)
```

### 註冊自定義 CVE

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    CVEIdentifier,
    CVESignature,
    SignalTier,
)

# 定義新的 CVE 簽名
new_cve = CVESignature(
    cve_id="CVE-2024-XXXXX",
    description="Custom vulnerability",
    cvss_score=9.5,
    tier3_triggers=["custom", "framework"],
    tier2_payloads=["custom_payload"],
    tier1_indicators=["critical_error"],
)

# 註冊
CVEIdentifier.register_cve(new_cve)
```

### 註冊自定義 WAF 繞過技術

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    WAFBypassEngine,
    BypassTechnique,
    BypassCategory,
    WAFVendor,
)

custom_technique = BypassTechnique(
    name="Custom Encoding",
    category=BypassCategory.ENCODING,
    target_waf=[WAFVendor.CLOUDFLARE],
    description="Custom encoding technique",
    payloads=["custom_encoded_payload"],
    success_rate=0.6,
)

WAFBypassEngine.register_technique(custom_technique)
```

## 數據結構說明

### DetectionResult

所有檢測方法返回的標準結果格式：

```python
@dataclass
class DetectionResult:
    detected: bool                      # 是否檢測到漏洞
    vulnerability_type: VulnerabilityType  # 漏洞類型
    confidence: ConfidenceLevel         # 置信度等級
    confidence_score: float             # 0.0-1.0
    evidence: list[str]                 # 證據列表
    indicators: list[str]               # 匹配的指標
    false_positive_risk: float          # 誤報風險
    recommendations: list[str]          # 建議動作
    technical_details: dict[str, Any]   # 技術細節
    raw_data: dict[str, Any]           # 原始數據
    
    def should_exploit(self, risk_threshold: float = 0.7) -> bool:
        """判斷是否應該進行利用"""
        return (
            self.detected 
            and self.confidence_score >= risk_threshold
            and self.false_positive_risk < 0.5
        )
```

### ConfidenceLevel

```python
class ConfidenceLevel(Enum):
    ABSOLUTE = auto()   # 100% 確定
    HIGH = auto()       # >85%
    MEDIUM = auto()     # 50-85%
    LOW = auto()        # 20-50%
    UNCERTAIN = auto()  # <20%
```

## 性能考慮

- **零延遲**: 所有知識直接編碼在代碼中，無需數據庫或 RAG 查詢
- **內存開銷**: 約 5-10MB (指紋庫、CVE 數據庫等)
- **並發安全**: 所有方法都是 `@classmethod`，無共享狀態

## 最佳實踐

1. **始終檢查 `should_exploit()`**: 避免盲目利用低置信度結果
2. **使用 `to_dict()` 序列化**: 便於日誌記錄和 AI 分析
3. **結合多個檢測器**: 交叉驗證提高準確性
4. **動態擴展知識庫**: 使用 `KnowledgeRegistry` 添加新知識
5. **關注 `false_positive_risk`**: 特別是在有 WAF 的環境中

## 未來擴展方向

- [ ] NoSQL 注入檢測 (MongoDB, Redis)
- [ ] LDAP 注入檢測
- [ ] OAuth 2.0 漏洞檢測
- [ ] gRPC 安全分析
- [ ] 更多 CVE 簽名庫
- [ ] 機器學習模型集成
