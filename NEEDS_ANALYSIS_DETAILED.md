# AIVA 系統深度分析報告 - 需新增功能清單

**文件版本**: 1.0  
**建立日期**: 2025-10-13  
**分析範圍**: 掃描編排器重構 + 模組整合需求  
**完成狀態**: ✅ 已清理重複檔案 (commit 82c9e7a)

---

## 📋 執行摘要

### ✅ 已完成

- **清理重複檔案**: 刪除 `scan_orchestrator_new.py` 和 `scan_orchestrator_old.py`
- **保留統一版本**: `scan_orchestrator.py` (373 行,功能完整)

### 🔍 核心發現

#### 1. Worker.py 需要重構

**問題**: `worker.py` 的 `_perform_scan` 方法直接實作掃描邏輯,**完全未使用** `ScanOrchestrator`

**現況**:

```python
# services/scan/aiva_scan/worker.py (第 54-96 行)
async def _perform_scan(req: ScanStartPayload) -> ScanCompletedPayload:
    # ❌ 重複實作所有初始化邏輯
    auth = AuthenticationManager(req.authentication)
    headers = HeaderConfiguration(req.custom_headers)
    urlq = UrlQueueManager([str(t) for t in req.targets])
    http = HiHttpClient(auth, headers)
    static = StaticContentParser()
    fingerprint_collector = FingerprintCollector()
    
    # ❌ 直接寫爬蟲邏輯 (與 ScanOrchestrator 重複)
    while urlq.has_next():
        url = urlq.next()
        r = await http.get(url)
        # ... (40+ 行掃描邏輯)
```

**影響**:

- 代碼重複,維護困難
- 無法享受 ScanOrchestrator 的完整功能 (動態掃描、策略控制等)
- worker.py 高達 106 行,大部分是應該被封裝的邏輯

---

#### 2. PostEx 模組是「後滲透測試」,非業務邏輯測試

**發現**: `services/postex/` 包含以下模組:

- `data_exfiltration_tester.py` - 數據外洩測試器
- `lateral_movement.py` - 橫向移動測試
- `persistence_checker.py` - 持久化檢測
- `privilege_escalator.py` - 權限提升測試

**結論**: PostEx 是專注於「已入侵後的攻擊鏈」,與建議的 BizLogic 模組 (業務邏輯漏洞) **不重疊**。

---

#### 3. Core 模組已有 StrategyGenerator,但未與配置中心整合

**發現**: `services/core/aiva_core/analysis/strategy_generator.py` 存在,但:

- ✅ 有 `RuleBasedStrategyGenerator` 類別
- ❌ `ScanOrchestrator` 未呼叫 `strategy_controller.apply_to_config()`
- ❌ 策略參數無法動態應用到配置中心

---

#### 4. ScanOrchestrator 動態引擎已部分實作,但處理不完整

**已實作**:

- ✅ 瀏覽器池管理 (`HeadlessBrowserPool`)

- ✅ 動態內容提取器 (`DynamicContentExtractor`)
- ✅ 配置支援 AJAX 和 API 呼叫提取

**缺失**:

- ❌ 提取的 AJAX/API 資料未被處理 (未加入 url_queue)
- ❌ JavaScript 變數未進行敏感資訊分析
- ❌ ScanContext 未記錄這些發現

---

## 🎯 需新增/改進的功能清單

### P0 - 關鍵級 (必須完成)

#### 1. 重構 worker.py 使用 ScanOrchestrator

**現況**:

```python
# ❌ 當前實作 (106 行)
async def _perform_scan(req: ScanStartPayload) -> ScanCompletedPayload:
    # 50+ 行初始化與掃描邏輯
    ...
```

**目標**:

```python
# ✅ 重構後 (<20 行)
async def _perform_scan(req: ScanStartPayload) -> ScanCompletedPayload:
    from .scan_orchestrator import ScanOrchestrator
    
    orchestrator = ScanOrchestrator()
    result = await orchestrator.execute_scan(req)
    return result
```

**效益**:

- worker.py 從 106 行減少到 ~30 行
- 自動支援動態掃描、策略控制等完整功能
- 統一維護點 (所有掃描邏輯在 ScanOrchestrator)

**預估工時**: 4 小時 (含測試)

---

#### 2. 新增 StrategyController 與 ConfigControlCenter 整合

**位置**: `services/scan/aiva_scan/scan_orchestrator.py` 第 ~85 行

**改進**:

```python
async def execute_scan(self, request: ScanStartPayload):
    # ... 現有代碼 ...
    
    strategy_controller = StrategyController(request.strategy)
    strategy_params = strategy_controller.get_parameters()
    
    # ✨ 新增: 動態應用策略到配置中心
    if hasattr(strategy_controller, 'apply_to_config'):
        from .config_control_center import ConfigControlCenter
        config_center = ConfigControlCenter.get_instance()
        strategy_controller.apply_to_config(config_center)
        logger.info("Strategy applied to ConfigControlCenter")
    
    # 後續組件會自動使用更新後的配置
    ...
```

**前提**: 需確認 `ConfigControlCenter` 和 `StrategyController.apply_to_config` 方法是否存在

**預估工時**: 2 小時 (含驗證)

---

### P1 - 高優先級 (2 週內)

#### 3. 擴充動態引擎處理 AJAX/API 類型

**問題**: 動態內容提取器已配置提取 AJAX 和 API,但提取後的資料未被使用

**位置**: `services/scan/aiva_scan/scan_orchestrator.py` 的 `_process_url_dynamic` 方法

**改進**:

```python
async def _process_url_dynamic(
    self, 
    url: str, 
    context: ScanContext,
    url_queue: UrlQueueManager,
    http_client: HiHttpClient,
) -> None:
    """處理動態掃描的 URL"""
    
    browser = await self.browser_pool.acquire()
    try:
        dynamic_contents = await self.dynamic_extractor.extract(browser, url)
        
        for content in dynamic_contents:
            # ✅ 現有處理
            if content.type == "form":
                context.add_form(content)
            elif content.type == "link":
                url_queue.add(content.value)
            
            # ✨ 新增: 處理 AJAX 端點
            elif content.type == "ajax_endpoint":
                url_queue.add(content.value)
                context.add_api_endpoint(Asset(
                    asset_id=new_id("asset"),
                    type="AJAX_ENDPOINT",
                    value=content.value,
                    metadata=content.metadata
                ))
            
            # ✨ 新增: 處理 API 呼叫
            elif content.type == "api_call":
                url_queue.add(content.value)
                context.add_api_endpoint(Asset(
                    asset_id=new_id("asset"),
                    type="API_CALL",
                    value=content.value,
                    metadata=content.metadata
                ))
            
            # ✨ 新增: JavaScript 變數敏感資訊分析
            elif content.type == "javascript_variable":
                sensitive_matches = self.sensitive_detector.detect(content.value)
                if sensitive_matches:
                    context.add_sensitive_findings(sensitive_matches)
                
                # JavaScript 原始碼分析
                js_analysis = self.js_analyzer.analyze(content.value)
                if js_analysis:
                    context.add_js_analysis_result(js_analysis)
    
    finally:
        await self.browser_pool.release(browser)
```

**前提**: 需先完成 P1-4 (ScanContext 新增方法)

**預估工時**: 6 小時

---

#### 4. ScanContext 新增敏感資訊記錄欄位

**位置**: `services/scan/aiva_scan/scan_context.py`

**改進**:

```python
class ScanContext:
    def __init__(self, request: ScanStartPayload):
        # ... 現有欄位 ...
        
        # ✨ 新增: 敏感資訊發現記錄
        self.sensitive_matches: list[SensitiveMatch] = []
        
        # ✨ 新增: JavaScript 分析結果
        self.js_analysis_results: list[JavaScriptAnalysisResult] = []
        
        # ✨ 新增: API 端點記錄 (AJAX/GraphQL/REST)
        self.api_endpoints: list[Asset] = []
    
    def add_sensitive_findings(self, matches: list[SensitiveMatch]) -> None:
        """記錄敏感資訊發現"""
        self.sensitive_matches.extend(matches)
        logger.debug(f"Added {len(matches)} sensitive findings")
    
    def add_js_analysis_result(self, result: JavaScriptAnalysisResult) -> None:
        """記錄 JavaScript 分析結果"""
        self.js_analysis_results.append(result)
    
    def add_api_endpoint(self, endpoint: Asset) -> None:
        """記錄 API 端點"""
        self.api_endpoints.append(endpoint)
    
    def to_summary(self) -> Summary:
        """生成摘要時包含新增資訊"""
        return Summary(
            urls_found=self.urls_found,
            forms_found=self.forms_found,
            apis_found=len(self.api_endpoints),  # ✨ 更新
            sensitive_info_count=len(self.sensitive_matches),  # ✨ 新增
            js_analysis_count=len(self.js_analysis_results),  # ✨ 新增
            scan_duration_seconds=self.scan_duration,
        )
```

**前提**: 需確認 `SensitiveMatch` 和 `JavaScriptAnalysisResult` Schema 是否存在

**預估工時**: 4 小時

---

### P2 - 中優先級 (1 個月內)

#### 5. 整合 ThreatIntel 到 RiskAssessmentEngine

**問題**: `threat_intel` 模組已完整實作,但未與 Core 模組整合

**位置**: `services/core/aiva_core/analysis/risk_assessment_engine.py`

**改進** (需先檢查該檔案是否存在):

```python

# services/core/aiva_core/analysis/risk_assessment_engine.py

from services.threat_intel.intel_aggregator import IntelAggregator

class RiskAssessmentEngine:
    def __init__(self):
        self.intel_aggregator = IntelAggregator()
    
    async def assess_risk(self, finding: FindingPayload) -> float:
        """評估漏洞風險分數 (0-10)"""
        
        # 1. 基礎 CVSS 分數
        base_score = self._calculate_cvss_score(finding)
        
        # 2. ✨ 查詢威脅情報,判斷是否被積極利用
        if finding.cve_id:
            try:
                intel = await self.intel_aggregator.query_cve(finding.cve_id)
                
                if intel.is_actively_exploited:
                    # 被積極利用的漏洞,風險提升 50%
                    base_score *= 1.5
                    logger.warning(
                        f"CVE {finding.cve_id} is actively exploited in the wild!",
                        extra={"intel_source": intel.source}
                    )
                
                # 根據威脅等級調整
                if intel.threat_level == ThreatLevel.CRITICAL:
                    base_score *= 1.3
            
            except Exception as e:
                logger.error(f"Failed to query threat intel: {e}")
        
        # 3. 上限為 10.0
        return min(base_score, 10.0)
```

**前提**: 需確認 `risk_assessment_engine.py` 檔案存在

**預估工時**: 6 小時

---

#### 6. 撰寫已實作模組的整合文檔

**任務**: 更新以下文件:

- `ARCHITECTURE_REPORT.md` - 新增 threat_intel, remediation, authz 模組說明
- `COMPREHENSIVE_ROADMAP.md` - 更新模組狀態為「已完成」

**內容**:

```markdown
## 已實作的進階模組

### 1. Threat Intel (威脅情報整合)
**路徑**: `services/threat_intel/`
**狀態**: ✅ 已完成 (2025 Q3)

**核心功能**:
- 整合 VirusTotal, AbuseIPDB, Shodan 等威脅情報源
- 自動查詢 CVE 是否被積極利用
- MITRE ATT&CK 框架映射
- IOC (Indicator of Compromise) 豐富化

**主要類別**:
- `IntelAggregator`: 情報聚合器 (448 行)
- `IOCEnricher`: IOC 豐富化引擎
- `MITREMapper`: MITRE 映射器

**使用範例**:
\`\`\`python
from services.threat_intel.intel_aggregator import IntelAggregator

aggregator = IntelAggregator(
    vt_api_key="your_key",
    cache_ttl=3600
)

# 查詢 CVE 情報
intel = await aggregator.query_cve("CVE-2021-44228")
if intel.is_actively_exploited:
    print("⚠️ 此漏洞正被積極利用!")
\`\`\`

### 2. Remediation (自動化修復)
**路徑**: `services/remediation/`
**狀態**: ✅ 已完成 (2025 Q3)

**核心功能**:
- 自動生成補丁 (使用 GitPython)
- AI 代碼修復建議
- 配置安全建議
- 修復報告生成

**主要類別**:
- `PatchGenerator`: 補丁生成器 (359 行)
- `CodeFixer`: 代碼修復器
- `ConfigRecommender`: 配置建議器

**使用範例**:
\`\`\`python
from services.remediation.patch_generator import PatchGenerator

generator = PatchGenerator(
    repo_path="/path/to/repo",
    auto_commit=False
)

# 針對 SQL Injection 生成補丁
patch = await generator.generate_sqli_fix(
    file_path="api/users.py",
    vulnerable_line=42
)

# 建立 Pull Request
pr = await generator.create_pull_request(
    branch="fix/sql-injection",
    title="Fix SQL Injection in users API"
)
\`\`\`

### 3. AuthZ (權限映射與測試)
**路徑**: `services/authz/`
**狀態**: ✅ 已完成 (2025 Q3)

**核心功能**:
- 權限矩陣生成與視覺化
- 多角色權限測試
- 權限衝突檢測
- RBAC (Role-Based Access Control) 分析

**主要類別**:
- `AuthZMapper`: 權限映射器 (414 行)
- `PermissionMatrix`: 權限矩陣
- `MatrixVisualizer`: 視覺化工具

**使用範例**:
\`\`\`python
from services.authz.authz_mapper import AuthZMapper
from services.authz.permission_matrix import PermissionMatrix

matrix = PermissionMatrix()
mapper = AuthZMapper(matrix)

# 分配角色
mapper.assign_role_to_user("user123", "admin")
mapper.assign_role_to_user("user456", "user")

# 測試權限
decision = mapper.check_permission(
    user_id="user123",
    resource="/api/admin/users",
    action="DELETE"
)

if decision.allowed:
    print("✅ 允許存取")
else:
    print(f"❌ 拒絕存取: {decision.reason}")
\`\`\`
```

**預估工時**: 4 小時

---

### P3 - 低優先級 (可推遲)

#### 7. 評估是否需要獨立的 BizLogic 模組

**分析**:

- `postex/` 模組專注於「後滲透攻擊鏈」(已入侵後的行為)
- **業務邏輯漏洞測試**是完全不同的領域:
  - 價格操縱
  - 工作流程繞過
  - 優惠券濫用
  - 競爭條件 (Race Condition)
  - 投票/評分系統操縱

**建議**: 新增獨立的 `services/bizlogic/` 模組

**核心類別設計**:

```python
# services/bizlogic/price_manipulation_tester.py
class PriceManipulationTester:
    """價格操縱測試器"""
    
    async def test_cart_race_condition(self, cart_api: str):
        """測試購物車競爭條件"""
        # 並發修改商品數量和價格
        ...
    
    async def test_negative_quantity(self, cart_api: str):
        """測試負數數量漏洞"""
        ...

# services/bizlogic/workflow_bypass_tester.py
class WorkflowBypassTester:
    """工作流程繞過測試器"""
    
    async def test_step_skip(self, workflow_urls: list[str]):
        """測試是否可跳過中間步驟"""
        ...
```

**決策點**: 先檢查 PostEx 模組的完整功能,確認無重疊後再新增

**預估工時**:

- 評估階段: 4 小時
- 如需新增: 20 小時 (基礎框架 + 3-5 個測試器)

---

## 📊 優先級總結

| 任務 | 優先級 | 工時 | 依賴 | 狀態 |
|------|--------|------|------|------|
| 1. 重構 worker.py | P0 | 4h | 無 | 🔴 未開始 |
| 2. StrategyController 整合 | P0 | 2h | 需確認方法存在 | 🔴 未開始 |
| 3. 動態引擎 AJAX/API 處理 | P1 | 6h | 任務 4 | 🔴 未開始 |
| 4. ScanContext 新增欄位 | P1 | 4h | 需確認 Schema | 🔴 未開始 |
| 5. ThreatIntel 整合 | P2 | 6h | 需確認檔案存在 | 🔴 未開始 |
| 6. 撰寫整合文檔 | P2 | 4h | 無 | 🔴 未開始 |
| 7. BizLogic 模組評估 | P3 | 4-24h | 任務 4 (PostEx 檢查) | 🔴 未開始 |

**總計**: 30-50 小時 (根據現有代碼完整度)

---

## 🚀 建議執行順序

### Week 1 (優先完成)

1. **Day 1-2**: 任務 2 - StrategyController 整合 (2h)
   - 先確認 `ConfigControlCenter` 和方法是否存在
   - 如果存在,快速整合

2. **Day 3-4**: 任務 1 - 重構 worker.py (4h)
   - 建立單元測試覆蓋現有功能
   - 重構為呼叫 ScanOrchestrator
   - 驗證測試通過

3. **Day 5**: 任務 4 - ScanContext 新增欄位 (4h)
   - 確認需要的 Schema
   - 實作新方法
   - 更新 to_summary

### Week 2 (功能增強)

4.**Day 1-2**: 任務 3 - 動態引擎擴充 (6h)

- 依賴任務

  - 實作 AJAX/API/JS 處理
- 測試動態掃描流程

5.**Day 3**:

 任務 6 - 撰寫文檔 (4h)

- 更新 ARCHITECTURE_REPORT.md
  - 更新 COMPREHENSIVE_ROADMAP.md
 新增使用範例

### Week 3 (深度整合)

6.**Day 1-2**: 任務 5 - ThreatIntel 整合 (6h)

- 確認 RiskAssessmentEngine 存在
- 實作情報查詢邏輯
- 測試風險評分調整

7.**Day 3**: 任務 7 - BizLogic 評估 (4h)

- 深入檢查 PostEx 模組
- 決策是否新增
  - 如需新增,制定實作計畫

---

## ✅ 前置檢查清單

在開始實作前,需要確認以下檔案/方法是否存在:

- [ ] `services/scan/aiva_scan/config_control_center.py`
- [ ] `StrategyController.apply_to_config()` 方法
- [ ] `services/aiva_common/schemas.py` 中的 `SensitiveMatch` Schema
- [ ] `services/aiva_common/schemas.py` 中的 `JavaScriptAnalysisResult` Schema
- [ ] `services/core/aiva_core/analysis/risk_assessment_engine.py`
- [ ] `services/scan/aiva_scan/scan_context.py` 的完整內容

**建議**: 先執行以下命令進行檢查:

```powershell
# 檢查 ConfigControlCenter
Get-ChildItem -Path "services/scan/aiva_scan" -Filter "config*.py" -Recurse

# 檢查 StrategyController 方法
Select-String -Path "services/scan/aiva_scan/strategy_controller.py" -Pattern "apply_to_config"

# 檢查 Schema 定義
Select-String -Path "services/aiva_common/schemas.py" -Pattern "SensitiveMatch|JavaScriptAnalysisResult"

# 檢查 RiskAssessmentEngine
Test-Path "services/core/aiva_core/analysis/risk_assessment_engine.py"
```

---

## 📝 下一步行動

1. **立即執行**: 前置檢查清單 (預估 30 分鐘)
2. **本週完成**: P0 任務 (預估 6 小時)
3. **下週開始**: P1 任務 (預估 10 小時)
4. **持續更新**: 待辦清單與進度追蹤

---

**文件維護者**: AIVA 技術團隊  
**下次更新**: 完成前置檢查後 (預計 2025-10-14)
