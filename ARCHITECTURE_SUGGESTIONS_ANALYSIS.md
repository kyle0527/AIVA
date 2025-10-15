# AIVA 架構建議分析報告

> **分析日期**: 2025-10-15
> **分析範圍**: Core、Scan、Function、Integration 四大模組
> **程式碼版本**: 基於當前 workspace 實際情況

---

## 📋 執行摘要

本報告針對外部提供的架構建議進行逐項驗證,分析其是否符合 AIVA 平台的實際程式碼情況。總體而言,**該建議準確度約 75%**,多數建議具有參考價值,但部分細節與實際實作有出入。

### 總體評估

| 評估項目 | 準確度 | 備註 |
|---------|--------|------|
| 整體架構評估 | ✅ 90% | 準確描述了四大模組和事件驅動架構 |
| Core Module 建議 | ⚠️ 70% | 部分配置已外部化,部分建議值得採納 |
| Scan Module 建議 | ⚠️ 65% | 動態處理建議有誤解,但整體方向正確 |
| Function Module 建議 | ✅ 85% | 準確識別設計模式,建議具體可行 |
| Integration Module 建議 | ⚠️ 60% | 對實際資料庫實作有誤解 |

---

## 1️⃣ 核心引擎 (Core Module) - 分析結果

### 檔案: `services/core/aiva_core/app.py`

#### ✅ 建議 1.1: 配置外部化

**建議內容**:
> 將組件的啟用/停用和魔術數字（例如 30 秒）移至外部設定檔

**實際情況**: ✅ **部分已實現**

```python
# 實際程式碼已有配置機制
from services.aiva_common.config import get_settings

# 設定檔位置: services/aiva_common/config.py
class Settings(BaseModel):
    rabbitmq_url: str = os.getenv("AIVA_RABBITMQ_URL", "...")
    postgres_dsn: str = os.getenv("AIVA_POSTGRES_DSN", "...")
    req_per_sec_default: int = int(os.getenv("AIVA_RATE_LIMIT_RPS", "25"))
    # ... 更多配置項
```

**需改進之處**:
- ⚠️ **30 秒輪詢間隔** (`asyncio.sleep(30)`) 確實仍為硬編碼
- ⚠️ `strategy_generator` 的啟用/停用邏輯被註解而非配置化

**建議採納度**: 80% - 值得採納

**具體行動**:
```python
# 建議在 Settings 中新增
class Settings(BaseModel):
    # ... 現有欄位
    core_monitor_interval: int = int(os.getenv("AIVA_CORE_MONITOR_INTERVAL", "30"))
    enable_strategy_generator: bool = os.getenv("AIVA_ENABLE_STRATEGY_GEN", "false").lower() == "true"
```

---

#### ⚠️ 建議 1.2: 錯誤處理與重試機制

**建議內容**:
> 引入 tenacity 函式庫實現重試機制,並發送失敗狀態

**實際情況**: ⚠️ **部分正確**

1. **Tenacity 已安裝**: ✅ 在 `pyproject.toml` 中確實有 `tenacity>=8.3.0`
2. **重試機制未實作**: ❌ `process_scan_results()` 中的異常處理僅記錄日誌
3. **失敗狀態發送**: ❌ 未見更新掃描狀態為 `failed` 的邏輯

**程式碼證據**:
```python
# 當前實作 (app.py:264-268)
try:
    # ... 處理邏輯
except Exception as e:
    logger.error(f"[失敗] Error processing scan results: {e}")
    # ❌ 沒有重試
    # ❌ 沒有更新狀態為 failed
```

**建議採納度**: 90% - 強烈建議採納

**具體行動**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _process_scan_with_retry(payload: ScanCompletedPayload):
    """可重試的掃描處理邏輯"""
    await scan_interface.process_scan_data(payload)

async def process_scan_results():
    async for mqmsg in aiterator:
        try:
            await _process_scan_with_retry(payload)
        except Exception as e:
            # 重試失敗後更新狀態
            session_state_manager.update_session_status(
                scan_id, "failed", {"error": str(e)}
            )
            await _send_failure_notification(scan_id, e)
```

---

#### ✅ 建議 1.3: 程式碼可讀性 - 階段邏輯封裝

**建議內容**:
> 將七個階段的邏輯封裝成獨立函式

**實際情況**: ❌ **未實現但強烈建議**

**當前問題**:
- `process_scan_results()` 函式長達 160+ 行
- 七個階段混雜在同一函式中,可讀性差

**建議採納度**: 95% - 極力推薦

**重構範例**:
```python
class ScanResultProcessor:
    """掃描結果處理器 - 七階段處理流程"""

    async def stage_1_ingest_data(self, payload: ScanCompletedPayload) -> None:
        """階段1: 資料接收與預處理"""
        await self.scan_interface.process_scan_data(payload)
        # ... 狀態更新

    async def stage_2_analyze_surface(self, payload: ScanCompletedPayload) -> dict:
        """階段2: 初步攻擊面分析"""
        return self.surface_analyzer.analyze(payload)

    # ... 其他階段

    async def process(self, payload: ScanCompletedPayload) -> None:
        """執行完整的七階段處理"""
        await self.stage_1_ingest_data(payload)
        attack_surface = await self.stage_2_analyze_surface(payload)
        # ...
```

---

## 2️⃣ 掃描引擎 (Scan Module) - 分析結果

### 檔案: `services/scan/aiva_scan/scan_orchestrator.py`

#### ✅ 建議 2.1: 資源清理 - HeadlessBrowserPool shutdown

**建議內容**:
> 確保 shutdown 方法能夠完美處理所有瀏覽器實例的退出

**實際情況**: ✅ **已實現**

**程式碼證據**: `services/scan/aiva_scan/dynamic_engine/headless_browser_pool.py:159-192`

```python
async def shutdown(self) -> None:
    """關閉瀏覽器池"""
    # ✅ 關閉所有頁面
    for page_instance in list(self._pages.values()):
        try:
            await self._close_page(page_instance.page_id)
        except Exception as e:
            logger.error(f"Error closing page {page_instance.page_id}: {e}")

    # ✅ 關閉所有瀏覽器
    for browser_instance in list(self._browsers.values()):
        try:
            await self._close_browser(browser_instance.browser_id)
        except Exception as e:
            logger.error(f"Error closing browser {browser_instance.browser_id}: {e}")

    # ✅ 關閉 Playwright
    if self._playwright:
        try:
            await self._playwright.stop()
        except Exception as e:
            logger.error(f"Error stopping Playwright: {e}")
```

**評估**:
- ✅ 已實作完整的異常處理
- ✅ 使用 `list(self._browsers.values())` 避免迭代時修改字典
- ✅ 分層關閉: Pages → Browsers → Playwright

**建議採納度**: 0% - 無需採納 (已實現)

---

#### ❌ 建議 2.2: 靜態與動態處理的資訊不對稱

**建議內容**:
> 在 `_process_url_dynamic` 中增加對動態渲染後 HTML 的分析步驟

**實際情況**: ❌ **建議有誤**

**實際程式碼**: `scan_orchestrator.py:308-343`

```python
async def _process_url_dynamic(self, url: str, ...) -> None:
    """使用動態引擎處理 URL"""
    async with self.browser_pool.get_page() as page:
        # ✅ 已提取動態內容
        dynamic_contents = await self.dynamic_extractor.extract_from_url(url, page=page)

        for content in dynamic_contents:
            # ✅ 已創建資產
            asset = Asset(...)
            context.add_asset(asset)

            # ✅ 已添加到 URL 隊列
            if content.content_type.value == "link":
                url_queue.add(content.url, parent_url=url, depth=1)

        # ⚠️ 這裡有註解提到可以分析,但未實作
        # rendered_html = await page.content()
```

**誤解之處**:
- 建議者認為動態處理「沒有」進行敏感資訊檢測
- **實際上**: `DynamicContentExtractor` 本身就包含內容分析邏輯
- **設計理念**: 動態引擎專注於「互動式內容提取」,靜態引擎專注於「HTML 結構分析」

**建議採納度**: 30% - 可選擇性採納

**實際建議**:
- 如果需要,可在動態處理中新增「渲染後 HTML 快照分析」
- 但需注意避免與 `DynamicContentExtractor` 的功能重複

---

## 3️⃣ 功能模組 (Function Module - SQLi) - 分析結果

### 檔案: `services/function/function_sqli/aiva_func_sqli/worker.py`

#### ✅ 建議 3.1: 引擎配置的靈活性

**建議內容**:
> 將 `SqliEngineConfig` 與每個掃描任務的策略關聯起來

**實際情況**: ⚠️ **部分正確**

**當前實作**:
```python
@dataclass
class SqliEngineConfig:
    """SQLi 引擎配置"""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    enable_error_detection: bool = True
    enable_boolean_detection: bool = True
    # ... 其他檢測引擎開關
```

**問題**:
- ✅ 配置類別設計良好
- ❌ 配置在服務啟動時固定,無法根據任務動態調整

**建議採納度**: 85% - 建議採納

**改進方案**:
```python
class SqliWorkerService:
    async def process_task(self, task: FunctionTaskPayload, ...) -> SqliContext:
        # 根據任務策略動態創建配置
        config = self._create_config_from_strategy(task.strategy)
        orchestrator = SqliOrchestrator(config)
        # ...

    def _create_config_from_strategy(self, strategy: str) -> SqliEngineConfig:
        """根據策略創建引擎配置"""
        if strategy == "FAST":
            return SqliEngineConfig(
                enable_error_detection=True,
                enable_boolean_detection=False,  # 快速模式禁用
                enable_time_detection=False,
            )
        elif strategy == "DEEP":
            return SqliEngineConfig(enable_all=True)
        # ...
```

---

#### ⚠️ 建議 3.2: 依賴注入容器

**建議內容**:
> 引入 dependency-injector 函式庫管理物件生命週期

**實際情況**: ⚠️ **過度設計**

**當前實作**:
```python
# worker.py:200-210
async def run() -> None:
    broker = await get_broker()
    queue = SqliTaskQueue()
    publisher = SqliResultBinderPublisher(broker)
    service = SqliWorkerService(publisher=publisher)
```

**評估**:
- ✅ 當前的手動依賴注入已經夠清晰
- ⚠️ 專案規模尚未大到需要 DI 容器
- ❌ 引入 DI 容器可能增加複雜度

**建議採納度**: 20% - 不建議採納 (當前階段)

**建議**:
- 保持當前的簡單依賴注入
- 如果未來有 10+ 個 Worker 服務,再考慮 DI 容器

---

## 4️⃣ 整合層 (Integration Module) - 分析結果

### 檔案: `services/integration/aiva_integration/app.py`

#### ❌ 建議 4.1: 資料庫互動 - 統一回傳型別

**建議內容**:
> `db.get_finding` 回傳型別不固定,建議統一回傳 Pydantic 模型

**實際情況**: ❌ **建議基於錯誤理解**

**實際資料庫實作**: `services/integration/aiva_integration/reception/sql_result_database.py:148-162`

```python
# 實際的 get_finding 方法
async def get_finding(self, finding_id: str) -> FindingPayload | None:
    """根據 ID 獲取漏洞發現"""
    record = session.query(FindingRecord).filter_by(finding_id=finding_id).first()

    if record:
        return record.to_finding_payload()  # ✅ 統一回傳 FindingPayload
    return None  # ✅ 找不到時回傳 None
```

**FastAPI 端點**: `app.py:49-58`

```python
@app.get("/findings/{finding_id}")
async def get_finding(finding_id: str) -> dict[str, Any]:
    result = db.get_finding(finding_id)
    if isinstance(result, dict):  # ⚠️ 這個判斷實際上永遠是 False
        return result
    try:
        return result.model_dump()  # ✅ 轉換為 dict
    except Exception:
        return {"error": "not_found", "finding_id": finding_id}
```

**問題分析**:
- ✅ 資料庫層已經統一回傳 `FindingPayload | None`
- ❌ FastAPI 端點的型別判斷邏輯有誤 (防禦性編程,但判斷條件錯誤)
- ⚠️ 應該檢查 `result is None` 而不是 `isinstance(result, dict)`

**建議採納度**: 40% - 部分採納

**改進方案**:
```python
@app.get("/findings/{finding_id}")
async def get_finding(finding_id: str) -> dict[str, Any]:
    result = await db.get_finding(finding_id)  # ✅ 加上 await

    if result is None:  # ✅ 正確判斷
        raise HTTPException(status_code=404, detail="Finding not found")

    return result.model_dump()  # ✅ 回傳 Pydantic 模型的 dict
```

---

#### ✅ 建議 4.2: 擴展性 - 插件化架構

**建議內容**:
> 使用事件驅動架構,讓分析器訂閱 `FINDING_STORED` 事件

**實際情況**: ⚠️ **設計理念不同但值得參考**

**當前架構**: `app.py:25-32`

```python
# 所有分析器在頂層實例化
db = TestResultDatabase()
recv = DataReceptionLayer(db)
corr = VulnerabilityCorrelationAnalyzer()
risk = RiskAssessmentEngine()
comp = CompliancePolicyChecker()
# ...
```

**優點**: 簡單直接,適合當前規模

**缺點**:
- 新增分析器需要修改 `app.py`
- 無法動態啟用/停用分析器

**建議採納度**: 60% - 中等優先級

**改進方案**:
```python
# 事件驅動的分析器架構
class AnalyzerPlugin(Protocol):
    async def on_finding_stored(self, finding: FindingPayload) -> None: ...

class IntegrationEngine:
    def __init__(self):
        self._plugins: list[AnalyzerPlugin] = []

    def register_plugin(self, plugin: AnalyzerPlugin) -> None:
        self._plugins.append(plugin)

    async def emit_finding_stored(self, finding: FindingPayload) -> None:
        await asyncio.gather(*[p.on_finding_stored(finding) for p in self._plugins])

# 使用
engine = IntegrationEngine()
engine.register_plugin(VulnerabilityCorrelationAnalyzer())
engine.register_plugin(RiskAssessmentEngine())
```

---

## 📊 優先級建議

根據實際程式碼分析,建議採納的優先順序:

| 優先級 | 建議項目 | 影響範圍 | 實作難度 | 預期效益 |
|--------|---------|---------|---------|---------|
| 🔴 P0 | Core: 錯誤處理與重試機制 | Core Module | 中 | 高 - 提升可靠性 |
| 🔴 P0 | Core: 階段邏輯封裝 | Core Module | 中 | 高 - 大幅提升可維護性 |
| 🟡 P1 | Core: 配置外部化 (監控間隔) | Core Module | 低 | 中 - 提升靈活性 |
| 🟡 P1 | Function: 引擎配置動態化 | Function Module | 中 | 中 - 支援策略差異化 |
| 🟡 P1 | Integration: API 錯誤處理改進 | Integration Module | 低 | 中 - 改善 API 體驗 |
| 🟢 P2 | Integration: 插件化架構 | Integration Module | 高 | 中 - 提升擴展性 |
| ⚪ P3 | Function: 依賴注入容器 | Function Module | 高 | 低 - 當前規模不需要 |
| ⚪ P3 | Scan: 動態 HTML 分析 | Scan Module | 中 | 低 - 功能重複風險 |

---

## 🎯 結論與行動計畫

### 總體評估
該建議文件展現了對 AIVA 架構的良好理解,但部分細節與實際實作有出入。**建議採納率約 65%**。

### 立即行動 (本週)
1. ✅ **實作重試機制**: 在 Core 模組的 `process_scan_results` 中加入 tenacity
2. ✅ **重構七階段處理**: 建立 `ScanResultProcessor` 類別封裝邏輯
3. ✅ **修正 Integration API**: 改善 `get_finding` 的錯誤處理

### 短期計畫 (2週內)
4. 🔄 **配置外部化**: 將監控間隔和組件開關移至環境變數
5. 🔄 **SQLi 配置動態化**: 支援根據策略調整檢測引擎

### 長期計畫 (1個月內)
6. 🔜 **評估插件架構**: 為 Integration 模組設計插件化分析器系統

### 不建議採納
- ❌ 依賴注入容器: 當前規模不需要,增加複雜度
- ❌ 動態 HTML 重複分析: 與現有功能重複

---

## 📎 附錄: 程式碼片段對照

### A. Core Module - asyncio.sleep 所在位置

```python
# services/core/aiva_core/app.py:312
async def monitor_execution_status() -> None:
    while True:
        await asyncio.sleep(30)  # ⚠️ 硬編碼
        system_status = execution_monitor.get_system_health()
```

### B. Scan Module - shutdown 實作

```python
# services/scan/aiva_scan/dynamic_engine/headless_browser_pool.py:159-192
async def shutdown(self) -> None:
    # ✅ 完整的資源清理邏輯
    for page_instance in list(self._pages.values()):
        await self._close_page(page_instance.page_id)
    # ...
```

### C. Function Module - 配置類別

```python
# services/function/function_sqli/aiva_func_sqli/worker.py:48-58
@dataclass
class SqliEngineConfig:
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    enable_error_detection: bool = True
    # ⚠️ 啟動時固定,無法根據任務調整
```

### D. Integration Module - 資料庫層

```python
# services/integration/aiva_integration/reception/sql_result_database.py:148
async def get_finding(self, finding_id: str) -> FindingPayload | None:
    # ✅ 已統一回傳型別
    if record:
        return record.to_finding_payload()
    return None
```

---

**分析完成** | 作者: GitHub Copilot | 日期: 2025-10-15
