# 整合實施範例 - SQLi Worker

這個文檔展示如何在 **SQLi Worker** 中實施整合功能，作為其他 Worker 的參考範本。

---

## 📋 修改清單

### 修改 1: 導入整合助手（1 行）

```python
# 在 services/features/function_sqli/worker.py 開頭
# 找到現有的導入區塊

from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas.tasks import FunctionTaskPayload
from services.aiva_common.schemas.findings import FindingPayload
from services.aiva_common.schemas.messaging import AivaMessage
from services.aiva_common.schemas.base import MessageHeader
from services.aiva_common.utils import get_logger, new_id
from services.features.common.worker_statistics import (
    StatisticsCollector,
    ErrorCategory,
)

# ✨ 新增這一行
from services.features.base.integration_helper import IntegrationHelper

# ... 其他導入保持不變 ...
```

---

### 修改 2: 在 SqliWorkerService 中初始化（1 行）

```python
# 在 SqliWorkerService 類別中
class SqliWorkerService:
    """SQLi Worker 服務 - 使用依賴注入的協調器"""

    def __init__(self, config: SqliEngineConfig | None = None):
        self.config = config or SqliEngineConfig()
        self.orchestrator = SqliOrchestrator(self.config)
        
        # ✨ 新增這一行：初始化整合助手
        self.integration = IntegrationHelper()

        logger.info(
            "SqliWorkerService initialized",
            extra={
                "config": {
                    "timeout": self.config.timeout_seconds,
                    "max_retries": self.config.max_retries,
                },
            },
        )
```

---

### 修改 3: 在發布 Finding 時儲存（每個 Finding 1 行）

找到這段代碼：

```python
# 原始代碼（約在第 450-460 行）
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

            # ===== 在這裡修改 =====
            # 發布結果
            for finding in context.findings:
                await publisher.publish_finding(finding, trace_id=trace_id)

            # ... 後續代碼 ...
```

修改為：

```python
            # 發布結果並儲存到資料庫
            for finding in context.findings:
                # 原始發布（不變）
                await publisher.publish_finding(finding, trace_id=trace_id)
                
                # ✨ 新增：儲存到資料庫
                await service.integration.save_finding(finding)
                
                # ✨ 新增：更新攻擊路徑（高危漏洞）
                if finding.severity in ["critical", "high"]:
                    await service.integration.update_attack_path(
                        finding, 
                        task.target.url
                    )
```

---

### 修改 4: 記錄檢測經驗（可選，建議加）

在 `SqliWorkerService.process_task()` 方法的最後：

```python
class SqliWorkerService:
    async def process_task(
        self,
        task: FunctionTaskPayload,
        client: httpx.AsyncClient | None = None,
    ) -> SqliContext:
        # ... 現有檢測邏輯完全不變 ...
        
        context = SqliContext(task=task, config=self.config)
        
        # 使用共享客戶端或創建新的
        if client is None:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=self.config.timeout_seconds
            ) as http_client:
                context = await self.orchestrator.execute_detection(context, http_client)
        else:
            context = await self.orchestrator.execute_detection(context, client)

        # ✨ 新增：記錄檢測經驗（在返回前）
        await self.integration.record_experience(
            task_id=task.task_id,
            action="sqli_detection",
            outcome={
                "findings_count": len(context.findings),
                "payloads_tested": context.telemetry.payloads_sent,
                "engines_used": [
                    name for name in self.orchestrator._engines.keys()
                ],
            },
            metadata={
                "target": task.target.url,
                "timeout": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
            }
        )

        return context
```

---

## 📊 完整修改對比

### 原始代碼（無整合）

```python
# services/features/function_sqli/worker.py

from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
# ... 其他導入 ...

class SqliWorkerService:
    def __init__(self, config: SqliEngineConfig | None = None):
        self.config = config or SqliEngineConfig()
        self.orchestrator = SqliOrchestrator(self.config)
    
    async def process_task(self, task, client=None):
        context = SqliContext(task=task, config=self.config)
        # ... 檢測邏輯 ...
        return context


async def _consume_queue_with_service(queue, publisher):
    service = SqliWorkerService()
    
    while True:
        queued = await queue.get()
        if queued is None:
            return

        task = queued.task
        trace_id = queued.trace_id

        try:
            context = await service.process_task(task)

            # 發布結果
            for finding in context.findings:
                await publisher.publish_finding(finding, trace_id=trace_id)

            await publisher.publish_status(
                task, "COMPLETED", trace_id=trace_id,
                details=context.telemetry.to_details(len(context.findings))
            )

        except Exception as exc:
            await publisher.publish_error(task, exc, trace_id=trace_id)
```

### 修改後代碼（有整合）

```python
# services/features/function_sqli/worker.py

from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
# ... 其他導入 ...

# ✨ 新增導入
from services.features.base.integration_helper import IntegrationHelper


class SqliWorkerService:
    def __init__(self, config: SqliEngineConfig | None = None):
        self.config = config or SqliEngineConfig()
        self.orchestrator = SqliOrchestrator(self.config)
        
        # ✨ 新增初始化
        self.integration = IntegrationHelper()
    
    async def process_task(self, task, client=None):
        context = SqliContext(task=task, config=self.config)
        # ... 檢測邏輯（不變）...
        
        # ✨ 新增：記錄經驗
        await self.integration.record_experience(
            task_id=task.task_id,
            action="sqli_detection",
            outcome={
                "findings_count": len(context.findings),
                "payloads_tested": context.telemetry.payloads_sent,
            }
        )
        
        return context


async def _consume_queue_with_service(queue, publisher):
    service = SqliWorkerService()
    
    while True:
        queued = await queue.get()
        if queued is None:
            return

        task = queued.task
        trace_id = queued.trace_id

        try:
            context = await service.process_task(task)

            # 發布結果並整合
            for finding in context.findings:
                # 原始發布（不變）
                await publisher.publish_finding(finding, trace_id=trace_id)
                
                # ✨ 新增：儲存到資料庫
                await service.integration.save_finding(finding)
                
                # ✨ 新增：更新攻擊路徑
                if finding.severity in ["critical", "high"]:
                    await service.integration.update_attack_path(
                        finding, task.target.url
                    )

            await publisher.publish_status(
                task, "COMPLETED", trace_id=trace_id,
                details=context.telemetry.to_details(len(context.findings))
            )

        except Exception as exc:
            await publisher.publish_error(task, exc, trace_id=trace_id)
```

---

## 📈 修改統計

| 項目 | 原始 | 修改後 | 變化 |
|-----|------|--------|------|
| **導入語句** | N 行 | N+1 行 | +1 行 |
| **初始化** | 2 行 | 3 行 | +1 行 |
| **發布 Finding** | 1 行/Finding | 3 行/Finding | +2 行/Finding |
| **記錄經驗** | 0 行 | 7 行 | +7 行 |
| **檢測邏輯** | X 行 | X 行 | **0 行** ✅ |

**總計**: 約 **10-15 行** 新增代碼，**0 行檢測邏輯修改**

---

## ✅ 驗證步驟

### 1. 確認助手正常工作

```python
# 在 Python REPL 或測試腳本中
from services.features.base.integration_helper import IntegrationHelper

integration = IntegrationHelper()
print(f"Integration enabled: {integration.enable_integration}")
print(f"Data manager: {integration._data_manager}")
```

### 2. 測試儲存 Finding

```python
# 創建測試 Finding
from services.aiva_common.schemas.findings import FindingPayload
from services.aiva_common.enums import Severity, Confidence

test_finding = FindingPayload(
    finding_id="test_finding_001",
    severity=Severity.HIGH,
    confidence=Confidence.HIGH,
    # ... 其他必要欄位 ...
)

# 測試儲存
success = await integration.save_finding(test_finding)
print(f"Save result: {success}")
```

### 3. 檢查資料庫

```python
# 查詢剛才儲存的 Finding
from services.integration.aiva_integration.unified_data_manager import UnifiedDataManager

data_manager = UnifiedDataManager()
finding = await data_manager.get_finding("test_finding_001")
print(f"Retrieved finding: {finding}")
```

### 4. 測試記錄經驗

```python
# 記錄測試經驗
success = await integration.record_experience(
    task_id="test_task_001",
    action="sqli_test",
    outcome={"findings_count": 1},
    metadata={"engine": "boolean"}
)
print(f"Experience recorded: {success}")

# 查詢經驗
experiences = await data_manager.query_experiences(task_id="test_task_001")
print(f"Retrieved experiences: {experiences}")
```

---

## 🚀 推廣到其他 Worker

完成 SQLi Worker 整合後，可以用相同方式推廣：

### XSS Worker
```python
# 完全相同的模式
from services.features.base.integration_helper import IntegrationHelper

class XssWorkerService:
    def __init__(self):
        self.integration = IntegrationHelper()
    
    # 在發布時儲存
    await self.integration.save_finding(finding)
```

### SSRF Worker
```python
# 完全相同的模式
from services.features.base.integration_helper import IntegrationHelper

integration = IntegrationHelper()
await integration.save_finding(finding)
```

### IDOR Worker
```python
# 完全相同的模式
from services.features.base.integration_helper import IntegrationHelper

integration = IntegrationHelper()
await integration.save_finding(finding)
```

---

## 📝 注意事項

### ✅ **DO（建議做）**

1. ✅ 在發布 Finding 後立即儲存
2. ✅ 記錄經驗到整合模組
3. ✅ 為高危漏洞更新攻擊路徑
4. ✅ 使用 try-except 包裹整合調用（避免影響主流程）

### ❌ **DON'T（避免做）**

1. ❌ 不要在檢測邏輯中調用整合功能
2. ❌ 不要讓整合失敗影響主流程
3. ❌ 不要為每個 Payload 都記錄經驗（太多）
4. ❌ 不要同步調用（使用 async/await）

---

## 🔧 故障排除

### 問題 1: 導入錯誤

```python
ImportError: cannot import name 'IntegrationHelper'
```

**解決方案**: 確認檔案存在
```bash
ls services/features/base/integration_helper.py
```

### 問題 2: UnifiedDataManager 未找到

```python
ImportError: cannot import name 'UnifiedDataManager'
```

**解決方案**: 檢查整合模組路徑
```bash
ls services/integration/aiva_integration/unified_data_manager.py
```

### 問題 3: 儲存失敗

```python
# 查看日誌
logger.error("Failed to save finding to database")
```

**解決方案**: 檢查資料庫連接和配置

---

## 📅 實施時間表

| 階段 | 任務 | 預計時間 |
|-----|------|---------|
| **階段 1** | 在 SQLi Worker 實施（試點） | 30 分鐘 |
| **階段 2** | 測試和驗證 | 1 小時 |
| **階段 3** | 推廣到 XSS Worker | 15 分鐘 |
| **階段 4** | 推廣到 SSRF Worker | 15 分鐘 |
| **階段 5** | 推廣到 IDOR Worker | 15 分鐘 |
| **階段 6** | 完整測試 | 1 小時 |

**總計**: 約 **3 小時**完成所有 Worker 整合

---

**📅 文檔創建**: 2025年11月24日  
**🎯 目標**: 在不影響檢測邏輯的前提下完成整合  
**✅ 狀態**: 助手已創建，等待實施
