"""
整合文檔 - AIVA 模組整合指南
說明新建模組和擴展功能的整合方法
"""

# AIVA 模組整合指南

## 1. 概述

本文檔說明如何整合新建的模組和功能到 AIVA 系統中。

## 2. 新建模組

### 2.1 BizLogic 模組

**位置**: `services/bizlogic/`

**功能**: 業務邏輯漏洞測試

**組件**:

- `price_manipulation_tester.py` - 價格操縱測試
- `workflow_bypass_tester.py` - 工作流繞過測試
- `race_condition_tester.py` - 競態條件測試
- `worker.py` - 消息隊列工作者

**整合步驟**:

1. **更新 Topic 枚舉** (`services/aiva_common/enums.py`):

```python
class Topic(str, Enum):
    # ... existing topics ...
    TASK_BIZLOGIC = "tasks.bizlogic"
    RESULTS_BIZLOGIC = "results.bizlogic"
```

2.**更新 Vulnerability 枚舉**:

```python
class Vulnerability(str, Enum):
    # ... existing vulnerabilities ...
    PRICE_MANIPULATION = "price_manipulation"
    WORKFLOW_BYPASS = "workflow_bypass"
    RACE_CONDITION = "race_condition"
```

3.**更新 FindingPayload Schema** (`services/aiva_common/schemas.py`):

```python
@dataclass
class FindingPayload:
    finding_id: str
    vulnerability_type: Vulnerability
    title: str
    description: str
    severity: str
    target: FindingTarget
    evidence: dict[str, Any]
    remediation: str | None = None
    cve_id: str | None = None
    metadata: dict[str, Any] | None = None
```

4.**啟動 BizLogic Worker**:

```bash
# 在 start_all.ps1 中添加
Start-Job -Name "bizlogic" -ScriptBlock {
    cd $using:PROJECT_ROOT\services\bizlogic
    python -m aiva_bizlogic.worker
}
```

### 2.2 Risk Assessment Engine

**位置**: `services/core/aiva_core/analysis/risk_assessment_engine.py`

**功能**: 綜合風險評估和 CVSS 評分

**整合步驟**:

1. **在 Core 模組中導入**:

```python
from services.core.aiva_core.analysis.risk_assessment_engine import RiskAssessmentEngine
```

2.**在掃描結果處理中使用**:

```python
risk_engine = RiskAssessmentEngine()
for finding in findings:
    assessment = await risk_engine.assess_risk(finding, scan_context)
    finding.risk_assessment = assessment
```

## 3. 功能擴展

### 3.1 ScanContext 擴展

**位置**: `services/scan/aiva_scan/scan_context.py`

**新增功能**:

- `add_sensitive_match()` - 記錄敏感資料發現
- `add_js_analysis_result()` - 記錄 JS 分析結果

**使用方法**:

```python
from services.aiva_common.schemas import SensitiveMatch, JavaScriptAnalysisResult

# 在掃描過程中
scan_context.add_sensitive_match(sensitive_match)
scan_context.add_js_analysis_result(js_result)
```

**需要更新 Schema**:

```python
@dataclass
class ScanContext:
    # ... existing fields ...
    sensitive_matches: list[SensitiveMatch] = field(default_factory=list)
    js_analysis_results: list[JavaScriptAnalysisResult] = field(default_factory=list)
```

### 3.2 動態引擎 - AJAX/API 處理

**位置**: `services/scan/aiva_scan/dynamic_engine/ajax_api_handler.py`

**功能**: 識別和測試 AJAX 請求和 API 端點

**整合步驟**:

1.**在 ScanOrchestrator 中初始化**:

```python
from services.scan.aiva_scan.dynamic_engine import AjaxApiHandler

class ScanOrchestrator:
    def __init__(self):
        # ... existing initialization ...
        self.ajax_handler = AjaxApiHandler()
```

2.**在 JavaScript 處理中使用**:

```python
# 當發現 JavaScript 文件時
async def process_javascript(self, js_url: str, js_content: str):
    # 提取 API 端點
    api_assets = await self.ajax_handler.analyze_javascript_for_ajax(
        js_content, js_url
    )
    for asset in api_assets:
        self.scan_context.add_asset(asset)
    
    # 測試端點
    for asset in api_assets:
        test_result = await self.ajax_handler.test_ajax_endpoint(
            asset.value, asset.metadata.get("method", "GET")
        )
        # 處理測試結果...
```

### 3.3 敏感資料掃描器

**位置**: `services/scan/aiva_scan/sensitive_data_scanner.py`

**功能**: 檢測響應中的敏感資訊

**整合步驟**:

1. **在 ScanOrchestrator 中初始化**:

```python
from services.scan.aiva_scan.sensitive_data_scanner import SensitiveDataScanner

class ScanOrchestrator:
    def __init__(self):
        # ... existing initialization ...
        self.sensitive_scanner = SensitiveDataScanner()
```

2.**在響應處理中使用**:

```python
# 當收到 HTTP 響應時
async def process_response(self, url: str, response):
    # 掃描內容
    sensitive_matches = self.sensitive_scanner.scan_content(
        response.text, url, "html"
    )
    for match in sensitive_matches:
        self.scan_context.add_sensitive_match(match)
    
    # 掃描標頭
    header_matches = self.sensitive_scanner.scan_headers(
        dict(response.headers), url
    )
    for match in header_matches:
        self.scan_context.add_sensitive_match(match)
```

### 3.4 JavaScript 分析器

**位置**: `services/scan/aiva_scan/javascript_analyzer.py`

**功能**: 靜態分析 JavaScript 代碼安全性

**整合步驟**:

1. **在 ScanOrchestrator 中初始化**:

```python
from services.scan.aiva_scan.javascript_analyzer import JavaScriptAnalyzer

class ScanOrchestrator:
    def __init__(self):
        # ... existing initialization ...
        self.js_analyzer = JavaScriptAnalyzer()
```

2.**在 JavaScript 處理中使用**:

```python
# 當發現 JavaScript 文件時
async def process_javascript(self, js_url: str, js_content: str):
    # 分析安全性
    analysis_result = await self.js_analyzer.analyze_javascript(
        js_content, js_url
    )
    self.scan_context.add_js_analysis_result(analysis_result)
    
    # 如果發現危險函數，創建 Finding
    if analysis_result.dangerous_functions:
        finding = self._create_js_security_finding(analysis_result)
        # 發送到結果隊列...
```

### 3.5 Worker 重構

**位置**: `services/scan/aiva_scan/worker_refactored.py`

**功能**: 使用 ScanOrchestrator 的統一 Worker

**整合步驟**:

1. **替換舊的 worker.py**:

```bash
mv services/scan/aiva_scan/worker.py services/scan/aiva_scan/worker_old_backup.py
mv services/scan/aiva_scan/worker_refactored.py services/scan/aiva_scan/worker.py
```

2.**更新 ScanOrchestrator 構造函數** (如果需要):

```python
# 確保 ScanOrchestrator.__init__ 接受 ScanStartPayload
class ScanOrchestrator:
    def __init__(self, request: ScanStartPayload):
        self.request = request
        # ... initialization ...
```

3.**確保 execute_scan 方法簽名正確**:

```python
async def execute_scan(self) -> ScanContext:
    # ... implementation ...
    return self.scan_context
```

## 4. 配置更新

### 4.1 消息隊列配置

確保 RabbitMQ 或使用的消息隊列已配置新的 Topics:

- `tasks.bizlogic`
- `results.bizlogic`

### 4.2 環境變量

添加必要的環境變量（如果需要）:

```bash
# .env 文件
THREAT_INTEL_ENABLED=true
VIRUSTOTAL_API_KEY=your_key_here
ABUSEIPDB_API_KEY=your_key_here
SHODAN_API_KEY=your_key_here
```

## 5. 測試

### 5.1 單元測試

為每個新模組創建測試:

```bash
pytest services/bizlogic/tests/
pytest services/core/tests/test_risk_assessment.py
pytest services/scan/tests/test_sensitive_scanner.py
pytest services/scan/tests/test_js_analyzer.py
pytest services/scan/tests/test_ajax_handler.py
```

### 5.2 集成測試

測試完整的掃描流程:

```bash
# 啟動所有服務
./start_all.ps1

# 運行集成測試
pytest tests/integration/test_full_scan_with_new_features.py
```

### 5.3 端到端測試

使用真實目標進行完整測試:

```bash
python -m services.scan.aiva_scan.cli --target https://example.com --enable-bizlogic
```

## 6. 部署

### 6.1 Docker 配置更新

更新 `docker-compose.yml` 添加新服務:

```yaml
services:
  bizlogic:
    build: ./services/bizlogic
    environment:
      - RABBITMQ_URL=${RABBITMQ_URL}
    depends_on:
      - rabbitmq
```

### 6.2 Kubernetes 配置

創建新的 Deployment 和 Service:

```yaml
# k8s/bizlogic-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiva-bizlogic
spec:
  replicas: 2
  # ... rest of config ...
```

## 7. 監控

### 7.1 日誌

確保新模組使用統一的日誌格式:

```python
from services.aiva_common.utils import get_logger
logger = get_logger(__name__)
```

### 7.2 指標

添加 Prometheus 指標:

```python
from prometheus_client import Counter, Histogram

bizlogic_tests_total = Counter(
    'aiva_bizlogic_tests_total',
    'Total number of bizlogic tests executed'
)
```

## 8. 文檔

更新用戶文檔:

- README.md - 添加新功能說明
- API 文檔 - 添加新的 API 端點
- 配置指南 - 添加新的配置選項

## 9. 已知問題和 TODO

### 9.1 需要修復的類型錯誤

1. **FindingPayload 參數不匹配**:
   - 需要更新 `FindingPayload` 的 `__init__` 以接受 `title`, `description` 等參數
   - 或修改創建 `FindingPayload` 的代碼以使用正確的參數

2. **Vulnerability 枚舉缺少 BizLogic 類型**:
   - 在 `enums.py` 中添加相關枚舉值

3. **ScanContext 缺少類型導入**:
   - 在 `scan_context.py` 中導入 `SensitiveMatch` 和 `JavaScriptAnalysisResult`

4. **ScanOrchestrator 構造函數簽名**:
   - 確認 `ScanOrchestrator` 是否需要在 `__init__` 中接受參數

### 9.2 後續改進

1. 添加更多業務邏輯測試類型
2. 改進風險評估算法
3. 增強 JavaScript 分析能力
4. 添加機器學習輔助檢測
5. 實現漏洞去重和聚合

## 10. 聯繫與支持

如有問題，請聯繫開發團隊或查閱內部文檔。
