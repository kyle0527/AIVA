# Features 模組整合指南

## 📑 目錄

- [🎯 目標](#目標)
- [📦 核心概念](#核心概念)
  - [✅ **非侵入式整合**](#非侵入式整合)
  - [🔑 **關鍵原則**](#關鍵原則)
- [🚀 快速開始](#快速開始)
  - [步驟 1: 導入助手](#步驟-1-導入助手)
  - [步驟 2: 初始化（在 Worker 初始化時）](#步驟-2-初始化在-worker-初始化時)
  - [步驟 3: 在發布 Finding 時儲存](#步驟-3-在發布-finding-時儲存)
- [📝 實際範例](#實際範例)
  - [範例 1: SQLi Worker 整合](#範例-1-sqli-worker-整合)
  - [範例 2: XSS Worker 整合](#範例-2-xss-worker-整合)
  - [範例 3: 批量儲存（性能優化）](#範例-3-批量儲存性能優化)
- [🎛️ 配置選項](#配置選項)
  - [啟用/禁用整合](#啟用禁用整合)
  - [使用全局單例](#使用全局單例)
  - [使用便利函數](#使用便利函數)
- [🔍 進階用法](#進階用法)
  - [選擇性記錄經驗](#選擇性記錄經驗)
  - [條件性更新攻擊路徑](#條件性更新攻擊路徑)
  - [錯誤處理](#錯誤處理)
- [📊 對比：修改前後](#對比修改前後)
  - [❌ 修改前（無整合）](#修改前無整合)
  - [✅ 修改後（有整合）](#修改後有整合)
- [🎯 總結](#總結)
  - [✅ **優點**](#優點)
  - [📝 **實施檢查清單**](#實施檢查清單)
  - [🔧 **下一步**](#下一步)

---


## 🎯 目標

在**不修改現有檢測邏輯**的前提下，將 Features 模組的結果自動儲存到 Integration 模組。

---

## 📦 核心概念

### ✅ **非侵入式整合**

```
原始流程（保持不變）:
檢測 → 產生 Finding → 發布到 MQ → 完成

新增流程（可選啟用）:
檢測 → 產生 Finding → 發布到 MQ → 【同時儲存到資料庫】 → 完成
                              ↓
                        【記錄經驗】
                              ↓
                        【更新攻擊路徑】
```

### 🔑 **關鍵原則**

1. **原始功能不變**: 所有現有的檢測邏輯完全不改
2. **可選啟用**: 可以選擇是否使用整合功能
3. **失敗不影響**: 即使整合失敗，原始流程繼續
4. **最小修改**: 只需加 1-3 行代碼

---

## 🚀 快速開始

### 步驟 1: 導入助手

```python
# 在 worker.py 開頭添加
from services.features.base.integration_helper import IntegrationHelper
```

### 步驟 2: 初始化（在 Worker 初始化時）

```python
# 方案 A: 創建實例（推薦）
integration = IntegrationHelper()

# 方案 B: 使用全局單例
from services.features.base.integration_helper import get_integration_helper
integration = get_integration_helper()

# 方案 C: 禁用整合功能
integration = IntegrationHelper(enable_integration=False)
```

### 步驟 3: 在發布 Finding 時儲存

```python
# 原始代碼（保持不變）
for finding in context.findings:
    await publisher.publish_finding(finding, trace_id=trace_id)
    
    # ✨ 新增這一行（可選）
    await integration.save_finding(finding)
```

**就這麼簡單！** 不需要修改任何檢測邏輯。

---

## 📝 實際範例

### 範例 1: SQLi Worker 整合

```python
# services/features/function_sqli/worker.py

# ===== 原始代碼 =====
from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas.tasks import FunctionTaskPayload
from services.aiva_common.schemas.findings import FindingPayload
# ... 其他導入 ...

# ===== 新增：導入整合助手 =====
from services.features.base.integration_helper import IntegrationHelper

logger = get_logger(__name__)

# ===== 原始類別保持不變 =====
class SqliOrchestrator:
    def __init__(self, config: SqliEngineConfig | None = None):
        self.config = config or SqliEngineConfig()
        self._engines: dict[str, DetectionEngineProtocol] = {}
        self._setup_default_engines()


class SqliWorkerService:
    def __init__(self, config: SqliEngineConfig | None = None):
        self.config = config or SqliEngineConfig()
        self.orchestrator = SqliOrchestrator(self.config)
        
        # ✨ 新增：初始化整合助手
        self.integration = IntegrationHelper()
    
    async def process_task(
        self,
        task: FunctionTaskPayload,
        client: httpx.AsyncClient | None = None,
    ) -> SqliContext:
        # ... 原始檢測邏輯完全不變 ...
        context = await self.orchestrator.execute_detection(ctx, http_client)
        
        # ✨ 新增：記錄檢測經驗（可選）
        await self.integration.record_experience(
            task_id=task.task_id,
            action="sqli_detection",
            outcome={
                "findings_count": len(context.findings),
                "engines_used": list(self.orchestrator._engines.keys()),
            },
            metadata={
                "timeout": self.config.timeout_seconds,
                "target": task.target.url
            }
        )
        
        return context


# ===== 在消費者函數中（最小修改）=====
async def _consume_queue_with_service(
    queue: SqliTaskQueue, 
    publisher: SqliResultBinderPublisher
) -> None:
    service = SqliWorkerService()
    
    while True:
        queued: QueuedTask | None = await queue.get()
        if queued is None:
            return

        task = queued.task
        trace_id = queued.trace_id

        await publisher.publish_status(task, "IN_PROGRESS", trace_id=trace_id)

        try:
            context = await service.process_task(task)

            # ===== 原始代碼：發布結果 =====
            for finding in context.findings:
                await publisher.publish_finding(finding, trace_id=trace_id)
                
                # ✨ 新增：同時儲存到資料庫（1 行）
                await service.integration.save_finding(finding)
                
                # ✨ 新增：更新攻擊路徑（可選，1 行）
                if finding.severity in ["critical", "high"]:
                    await service.integration.update_attack_path(
                        finding, 
                        task.target.url
                    )

            # ===== 原始狀態發布保持不變 =====
            await publisher.publish_status(
                task,
                "COMPLETED",
                trace_id=trace_id,
                details=context.telemetry.to_details(len(context.findings)),
            )

        except Exception as exc:
            logger.exception(
                "Unhandled error while processing SQLi task",
                extra={"task_id": task.task_id},
            )
            await publisher.publish_error(task, exc, trace_id=trace_id)
```

**修改總結**:
- ✅ 導入 1 行
- ✅ 初始化 1 行
- ✅ 儲存 Finding 每個 1 行
- ✅ 記錄經驗 1 次
- ❌ **檢測邏輯 0 行修改**

---

### 範例 2: XSS Worker 整合

```python
# services/features/function_xss/worker.py

from services.features.base.integration_helper import IntegrationHelper

async def _execute_task(queued: QueuedTask, publisher: XssResultPublisher) -> None:
    task = queued.task
    trace_id = queued.trace_id
    
    # ✨ 創建整合助手
    integration = IntegrationHelper()

    await publisher.publish_status(task, "IN_PROGRESS", trace_id=trace_id)

    try:
        # ===== 原始檢測邏輯不變 =====
        result = await process_task(task)
    except Exception as exc:
        logger.exception(
            "Unhandled error while processing XSS task",
            extra={"task_id": task.task_id},
        )
        await publisher.publish_error(task, exc, trace_id=trace_id)
        return

    # ===== 原始發布邏輯 =====
    for finding in result.findings:
        await publisher.publish_finding(finding, trace_id=trace_id)
        
        # ✨ 新增：儲存到資料庫
        await integration.save_finding(finding)

    # ✨ 新增：記錄經驗
    await integration.record_experience(
        task_id=task.task_id,
        action="xss_detection",
        outcome={
            "findings_count": len(result.findings),
            "payloads_sent": result.telemetry.payloads_sent,
            "reflections": result.telemetry.reflections,
            "dom_escalations": result.telemetry.dom_escalations,
        }
    )

    await publisher.publish_status(
        task,
        "COMPLETED",
        trace_id=trace_id,
        details=result.telemetry.to_details(len(result.findings)),
    )
```

---

### 範例 3: 批量儲存（性能優化）

```python
# 如果有大量 Finding，可以批量儲存
integration = IntegrationHelper()

# 原始發布（保持不變）
for finding in findings:
    await publisher.publish_finding(finding, trace_id=trace_id)

# ✨ 批量儲存（更高效）
saved_count = await integration.save_findings_batch(findings)
logger.info(f"Saved {saved_count}/{len(findings)} findings to database")
```

---

## 🎛️ 配置選項

### 啟用/禁用整合

```python
# 完全啟用（預設）
integration = IntegrationHelper(enable_integration=True)

# 完全禁用
integration = IntegrationHelper(enable_integration=False)

# 通過環境變數控制
import os
enable = os.getenv("AIVA_ENABLE_INTEGRATION", "true").lower() == "true"
integration = IntegrationHelper(enable_integration=enable)
```

### 使用全局單例

```python
# 所有 Worker 共享同一個實例
from services.features.base.integration_helper import get_integration_helper

integration = get_integration_helper()
```

### 使用便利函數

```python
# 不需要創建實例
from services.features.base.integration_helper import save_finding, record_experience

# 直接調用
await save_finding(finding)
await record_experience(task_id, action, outcome)
```

---

## 🔍 進階用法

### 選擇性記錄經驗

```python
# 只記錄成功的檢測
if len(findings) > 0:
    await integration.record_experience(
        task_id=task.task_id,
        action="sqli_detection_success",
        outcome={"findings_count": len(findings)}
    )
else:
    await integration.record_experience(
        task_id=task.task_id,
        action="sqli_detection_no_findings",
        outcome={"payloads_tested": telemetry.payloads_sent}
    )
```

### 條件性更新攻擊路徑

```python
# 只為高危漏洞更新攻擊路徑
for finding in findings:
    await publisher.publish_finding(finding, trace_id=trace_id)
    await integration.save_finding(finding)
    
    # 只有嚴重和高危漏洞才加入攻擊路徑
    if finding.severity in ["critical", "high"]:
        await integration.update_attack_path(finding, target_url)
```

### 錯誤處理

```python
# 整合失敗不影響主流程
for finding in findings:
    # 主流程
    await publisher.publish_finding(finding, trace_id=trace_id)
    
    # 整合（失敗只記錄日誌）
    try:
        success = await integration.save_finding(finding)
        if not success:
            logger.warning(f"Failed to save finding {finding.finding_id}")
    except Exception as e:
        logger.error(f"Integration error: {e}")
        # 繼續執行，不中斷主流程
```

---

## 📊 對比：修改前後

### ❌ 修改前（無整合）

```python
async def process_task(task):
    # 檢測
    findings = await detect(task)
    
    # 發布
    for finding in findings:
        await publisher.publish_finding(finding)
    
    return findings
```

**問題**:
- ❌ Finding 只發布到 MQ，未儲存到資料庫
- ❌ 無法查詢歷史漏洞
- ❌ 沒有經驗學習
- ❌ 沒有攻擊路徑分析

### ✅ 修改後（有整合）

```python
async def process_task(task):
    integration = IntegrationHelper()  # +1 行
    
    # 檢測（不變）
    findings = await detect(task)
    
    # 發布（不變）+ 整合
    for finding in findings:
        await publisher.publish_finding(finding)
        await integration.save_finding(finding)  # +1 行
    
    # 記錄經驗
    await integration.record_experience(  # +4 行
        task_id=task.task_id,
        action="detection",
        outcome={"findings_count": len(findings)}
    )
    
    return findings
```

**改進**:
- ✅ Finding 自動儲存到資料庫（可查詢）
- ✅ 經驗自動記錄（AI 學習）
- ✅ 攻擊路徑自動更新（分析）
- ✅ **檢測邏輯 0 行修改**
- ✅ **總共只加 6 行代碼**

---

## 🎯 總結

### ✅ **優點**

| 項目 | 說明 |
|-----|------|
| **非侵入式** | 檢測邏輯完全不變 |
| **可選啟用** | 可以隨時開關 |
| **失敗安全** | 整合失敗不影響主流程 |
| **最小修改** | 每個 Worker 只需加 3-6 行 |
| **統一介面** | 所有功能模組使用相同方式 |

### 📝 **實施檢查清單**

為每個 Worker 模組實施整合：

- [ ] **SQLi Worker**: 3 行修改
  - [ ] 導入 `IntegrationHelper`
  - [ ] 初始化實例
  - [ ] 在發布 Finding 後調用 `save_finding()`

- [ ] **XSS Worker**: 3 行修改
  - [ ] 導入 `IntegrationHelper`
  - [ ] 初始化實例
  - [ ] 在發布 Finding 後調用 `save_finding()`

- [ ] **SSRF Worker**: 3 行修改
  - [ ] 導入 `IntegrationHelper`
  - [ ] 初始化實例
  - [ ] 在發布 Finding 後調用 `save_finding()`

- [ ] **IDOR Worker**: 3 行修改
  - [ ] 導入 `IntegrationHelper`
  - [ ] 初始化實例
  - [ ] 在發布 Finding 後調用 `save_finding()`

- [ ] **PostEx Worker**: 3 行修改
  - [ ] 導入 `IntegrationHelper`
  - [ ] 初始化實例
  - [ ] 在發布 Finding 後調用 `save_finding()`

### 🔧 **下一步**

1. **測試 IntegrationHelper**: 確保助手正常工作
2. **在一個 Worker 中試點**: 建議從 SQLi 開始
3. **驗證結果**: 確認 Finding 正確儲存到資料庫
4. **推廣到其他 Worker**: 複製相同模式
5. **監控和優化**: 觀察性能和錯誤率

---

**📅 文檔更新**: 2025年11月24日  
**📦 助手位置**: `services/features/base/integration_helper.py`  
**🎯 目標**: 在不影響檢測邏輯的前提下，統一整合到 Integration 模組
