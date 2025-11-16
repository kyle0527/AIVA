# AIVA Core 系統啟動指南

## 🎯 架構概覽

### 系統層次（明確的主從關係）

```
┌─────────────────────────────────────────────────────────┐
│  app.py (FastAPI Application)                          │  ← 系統唯一入口點
│  - HTTP 端點                                            │
│  - 啟動流程控制                                         │
│  - 持有所有後台任務                                     │
└──────────────────────┬──────────────────────────────────┘
                       │ 持有和調用
                       ↓
┌─────────────────────────────────────────────────────────┐
│  CoreServiceCoordinator                                 │  ← 狀態管理器（非主線程）
│  - 服務實例管理                                         │
│  - 狀態協調                                             │
│  - 配置管理                                             │
└──────────────────────┬──────────────────────────────────┘
                       │ 管理和提供
                       ↓
┌─────────────────────────────────────────────────────────┐
│  功能服務層                                              │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │ EnhancedDecisionAgent                           │  │  ← 決策代理
│  │   ↓ 調用                                        │  │
│  │ BioNeuronDecisionController                     │  │  ← AI 決策控制器
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  - StrategyGenerator (策略生成)                        │
│  - TaskExecutor (任務執行)                             │
│  - CapabilityRegistry (能力註冊表)                     │
│  - InternalLoopConnector (內部閉環)                    │
│  - ExternalLoopConnector (外部學習)                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 啟動流程

### 1. 啟動命令

```bash
# 方式 1: 直接啟動（開發模式）
cd /path/to/AIVA-git/services/core/aiva_core
uvicorn service_backbone.api.app:app --host 0.0.0.0 --port 8000 --reload

# 方式 2: 生產環境
uvicorn service_backbone.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info

# 方式 3: 使用 Gunicorn (推薦生產環境)
gunicorn service_backbone.api.app:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 120 \
  --log-level info
```

### 2. 啟動序列（詳細步驟）

當執行啟動命令後，系統按以下順序初始化：

#### Phase 1: FastAPI 應用初始化

```
[時間點 T0] FastAPI 應用創建
    ↓
[T0 + 0.1s] 載入路由和中間件
    ↓
[T0 + 0.2s] 進入 @app.on_event("startup")
```

#### Phase 2: CoreServiceCoordinator 初始化

```python
# 在 app.py 的 startup() 中
coordinator = AIVACoreServiceCoordinator()
await coordinator.start()
```

**CoreServiceCoordinator 啟動步驟：**
1. 初始化核心組件（CommandRouter, ContextManager, ExecutionPlanner）
2. 初始化共享服務（ConfigManager, CrossLangService, MonitoringService, SecurityManager）
3. 設置監控和配置
4. 啟動共享服務
5. 檢查核心組件就緒狀態

**時間消耗：** ~1-2 秒

#### Phase 3: 內部閉環啟動（P0 問題一）

```python
# 啟動內部探索更新循環
_background_tasks.append(asyncio.create_task(
    periodic_update(),
    name="internal_loop_update"
))
```

**功能：** 定期更新系統自我感知
- 掃描代碼庫能力
- 更新 CapabilityRegistry
- 維護能力元數據

**時間消耗：** 異步啟動，不阻塞主流程

#### Phase 4: 外部學習循環啟動（P0 問題二）

```python
# 啟動外部學習監聽器
external_connector = ExternalLoopConnector()
_background_tasks.append(asyncio.create_task(
    external_connector.start_listening(),
    name="external_learning_loop"
))
```

**功能：** 監聽外部學習信號
- 接收功能模組執行結果
- 觸發策略調整
- 更新決策知識庫

**時間消耗：** 異步啟動，不阻塞主流程

#### Phase 5: 核心處理循環啟動

```python
# 掃描結果處理
_background_tasks.append(asyncio.create_task(
    process_scan_results(),
    name="scan_results_processor"
))

# 功能結果處理
_background_tasks.append(asyncio.create_task(
    process_function_results(),
    name="function_results_processor"
))

# 執行狀態監控
_background_tasks.append(asyncio.create_task(
    monitor_execution_status(),
    name="execution_monitor"
))
```

**功能：**
- `process_scan_results`: 處理掃描模組結果
- `process_function_results`: 處理功能模組結果
- `monitor_execution_status`: 監控系統健康狀態

**時間消耗：** 異步啟動，不阻塞主流程

#### Phase 6: 系統就緒

```
[T0 + 2s] ✅ 所有組件初始化完成
    ↓
[T0 + 2.1s] 🎉 系統開始接受 HTTP 請求
```

### 3. 啟動日誌示例

```log
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     🚀 [啟動] AIVA Core Engine starting up...
INFO:     ✅ [啟動] CoreServiceCoordinator initialized (state manager mode)
INFO:     命令路由器就緒，支持 15 個命令
INFO:     ✅ [啟動] Internal exploration loop started
INFO:     ✅ [啟動] External learning listener started
INFO:     [統計] Initializing analysis components...
INFO:     [循環] Starting message processing loops...
INFO:     ✅ [啟動] All background tasks started
INFO:     🎉 [啟動] AIVA Core Engine ready to accept requests!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 🔌 健康檢查

### 檢查系統狀態

```bash
# 基本健康檢查
curl http://localhost:8000/health

# 響應示例
{
  "status": "healthy",
  "service": "aiva-core-engine",
  "components": {
    "scan_interface": "active",
    "analysis_engine": "active",
    "task_coordinator": "active",
    "state_manager": "active"
  }
}
```

### 檢查掃描狀態

```bash
# 獲取特定掃描的狀態
curl http://localhost:8000/status/{scan_id}

# 響應示例
{
  "scan_id": "abc123",
  "status": "processing",
  "progress": 0.65,
  "tasks_completed": 13,
  "tasks_total": 20
}
```

---

## 🛑 優雅關閉

### 關閉流程

當收到 SIGTERM 或 Ctrl+C 時：

#### Phase 1: 觸發 Shutdown Event

```python
@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("🛑 [關閉] AIVA Core Engine shutting down...")
```

#### Phase 2: 停止 CoreServiceCoordinator

```python
if coordinator:
    await coordinator.stop()
```

**停止步驟：**
1. 停止核心組件（清理上下文和會話）
2. 停止共享服務（SecurityManager → CrossLangService → MonitoringService）
3. 記錄關閉統計

#### Phase 3: 取消後台任務

```python
# 所有後台任務會自動被取消（asyncio 機制）
for task in _background_tasks:
    task.cancel()
```

#### Phase 4: 關閉完成

```log
INFO:     🛑 [關閉] AIVA Core Engine shutting down...
INFO:     ✅ [關閉] CoreServiceCoordinator stopped
INFO:     👋 [關閉] AIVA Core Engine shutdown complete
INFO:     Shutting down
INFO:     Finished server process [12345]
```

---

## 📦 組件職責明細

### app.py（系統唯一入口）

**職責：**
- ✅ 作為 FastAPI 應用的主入口
- ✅ 控制啟動和關閉流程
- ✅ 持有 CoreServiceCoordinator 實例
- ✅ 管理所有後台任務
- ✅ 提供 HTTP API 端點

**不負責：**
- ❌ 業務邏輯處理
- ❌ 狀態管理（由 Coordinator 負責）
- ❌ 服務實例化（由 Coordinator 負責）

### CoreServiceCoordinator（狀態管理器）

**職責：**
- ✅ 管理服務實例和狀態
- ✅ 協調命令處理流程
- ✅ 管理執行上下文和會話
- ✅ 配置管理和監控

**不負責：**
- ❌ 系統主線程（不運行 `run()` 循環）
- ❌ HTTP 端點提供
- ❌ 主動任務調度

### BioNeuronDecisionController（AI 決策控制器）

**職責：**
- ✅ 管理 BioNeuronRAGAgent（5M 參數神經網路）
- ✅ 提供 AI 決策服務
- ✅ 支持三種操作模式（UI/AI/Chat）
- ✅ RAG 知識檢索

**不負責：**
- ❌ 系統協調
- ❌ 服務啟動
- ❌ 任務執行
- ❌ 資源管理

---

## 🔧 故障排除

### 問題 1: 啟動超時

**現象：** 系統啟動超過 10 秒仍未就緒

**可能原因：**
1. CoreServiceCoordinator 初始化失敗
2. 共享服務（MonitoringService, SecurityManager）啟動失敗
3. 網路連接問題（MQ, 外部服務）

**解決方法：**
```bash
# 檢查日誌
tail -f logs/aiva_core.log

# 檢查依賴服務
docker ps  # 檢查 RabbitMQ, Redis 等
```

### 問題 2: 後台任務異常退出

**現象：** 內部閉環或外部學習循環停止

**可能原因：**
1. 未處理的異常導致任務崩潰
2. 資源不足（記憶體、CPU）

**解決方法：**
```python
# 檢查任務狀態
async def check_background_tasks():
    for task in _background_tasks:
        if task.done():
            try:
                task.result()
            except Exception as e:
                logger.error(f"Background task failed: {e}")
```

### 問題 3: 重複啟動

**現象：** CoreServiceCoordinator 已在運行警告

**原因：** 多次調用 `startup()`

**解決方法：** 確保只有一個 FastAPI 應用實例

---

## 📚 相關文檔

- **架構分析**: `ARCHITECTURE_GAPS_ANALYSIS.md`
- **P0 完成報告**: `ARCHITECTURE_FIXES_P0_COMPLETION_REPORT.md`
- **P1 完成報告**: `ARCHITECTURE_FIXES_P1_COMPLETION_REPORT.md`
- **API 文檔**: `http://localhost:8000/docs` (Swagger UI)
- **決策合約**: `services/aiva_common/schemas/decision.py`
- **能力註冊表**: `core_capabilities/capability_registry.py`

---

## 🎯 關鍵要點

### ✅ 明確的主從關係

1. **app.py** 是系統唯一入口點
2. **CoreServiceCoordinator** 是狀態管理器（非主線程）
3. **BioNeuronDecisionController** 是 AI 決策控制器（非系統 Master）

### ✅ 啟動流程清晰

1. FastAPI 應用初始化
2. CoreServiceCoordinator 初始化
3. 內部閉環啟動
4. 外部學習啟動
5. 核心處理循環啟動
6. 系統就緒

### ✅ 職責明確

- **app.py**: 入口點、流程控制
- **CoreServiceCoordinator**: 狀態管理、服務工廠
- **BioNeuronDecisionController**: AI 決策、RAG

### ✅ 優雅關閉

1. 觸發 shutdown event
2. 停止 CoreServiceCoordinator
3. 取消後台任務
4. 記錄統計並退出

---

**版本**: 3.0.0  
**最後更新**: 2025-01-XX  
**維護**: AIVA Core Team
