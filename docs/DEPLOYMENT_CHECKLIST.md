---
Created: 2025-11-25
Document Type: Deployment Checklist
Status: Pre-Release Planning
---

# AIVA 部署前檢查清單

## 📖 Integration 模組運作說明

### 架構定位
**Integration 模組** 是 AIVA 系統的**內部整合中樞**，負責：
- ✅ 接收來自 Scan/Features 模組的掃描結果（透過 RabbitMQ）
- ✅ 資料庫比對與歷史分析
- ✅ 風險評估與優先級排序
- ✅ 產出最終報告

### 部署架構
```
外部世界
    ↓
🌐 API Gateway (對外接口)
    ↓ (RabbitMQ)
📦 Core 模組 (AI 決策引擎)
    ↓ (RabbitMQ)
🔍 Scan 模組 (資產發現)
🎯 Features 模組 (漏洞檢測)
    ↓ (RabbitMQ)
🔧 Integration 模組 (內部整合) ← 本模組
    ↓
💾 PostgreSQL (漏洞發現資料庫)
```

### 關鍵設計決策
1. **不對外暴露**: Integration 模組只接收內部訊息，不直接對外提供 API
2. **訊息驅動**: 透過 RabbitMQ 訂閱 `Topic.LOG_RESULTS_ALL` 接收漏洞發現
3. **資料庫獨立**: 使用獨立的 PostgreSQL 連線，不依賴容器網路
4. **背景處理**: `_consume_logs()` 協程持續運行，處理訊息佇列

### 為什麼 localhost 連線不是問題？
在**開發階段**，Integration 模組：
- ✅ 運行在本機 (localhost)
- ✅ 連線本機 PostgreSQL (localhost:5432)
- ✅ 連線本機 RabbitMQ (localhost:5672)

在**生產階段**，部署方會提供：
- 📌 完整的 PostgreSQL 連線字串（包含外部主機地址）
- 📌 完整的 RabbitMQ 連線字串
- 📌 不需要 Docker 容器化（直接運行在生產伺服器）

### 原本建議的問題
第三方建議提到的「容器化網路問題」是基於以下**錯誤假設**：
- ❌ 假設 Integration 會在 Docker Compose 中運行
- ❌ 假設需要容器間網路通訊 (localhost → postgres 容器)
- ❌ 假設需要環境變數配置

**實際情況**：
- ✅ Integration 是獨立服務，不在 Docker Compose 中
- ✅ 生產環境直接連線外部 PostgreSQL（由部署方提供）
- ✅ 不需要複雜的環境變數，直接修改配置檔即可

---

## 🔴 P0 - 發布前必須修復（嚴重）

### 1. 資料庫連線配置問題
**問題描述**:
- Integration 模組 PostgreSQL 連線硬編碼為 `postgres:postgres@localhost:5432`
- 與 Docker Compose 的實際配置 `aiva:aiva_secure_password@postgres:5432` 不一致
- 開發階段使用本機 PostgreSQL，生產環境需使用外部提供的連線字串

**影響範圍**:
- 🟡 **開發階段**: 無影響（本機 PostgreSQL 運行正常）
- 🔴 **生產部署**: 連線失敗（憑證不一致）

**受影響檔案**:
```
services/integration/aiva_integration/config.py:86
services/integration/aiva_integration/app.py:37
services/integration/alembic/env.py:32
```

**修復方案**（發布前執行）:
```python
# services/integration/aiva_integration/config.py
# 使用外部提供的連線字串（由部署方提供）

# 開發階段（本機）:
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/aiva_db"

# 生產階段（由部署方提供完整連線字串）:
# DATABASE_URL = os.getenv("DATABASE_URL")  # 從環境變數讀取
# 或
# DATABASE_URL = "postgresql://user:pass@external-db-host:5432/aiva"
```

**重要說明**:
- ✅ **開發階段**: 保持 localhost 連線（本機測試正常）
- ⚠️ **生產階段**: 由部署方提供完整連線字串（包含主機、憑證）
- ❌ **不使用環境變數**: 避免配置複雜化，直接修改 config.py 即可

---

### 2. 背景任務錯誤處理缺失
**問題描述**:
- `_consume_logs()` 協程無任何錯誤處理
- `AivaMessage.model_validate_json()` 可能拋出 `ValidationError`
- `FindingPayload(**msg.payload)` 可能拋出 `ValidationError`
- `recv.store_finding()` 可能拋出資料庫異常

**影響範圍**:
- 🔴 **單點失敗**: 一條格式錯誤訊息會永久終止日誌處理
- 🟡 **隱蔽故障**: FastAPI 主服務正常，但監控數據無聲停止
- 🟡 **難排查**: 需手動重啟容器，無錯誤日誌

**受影響檔案**:
```
services/integration/aiva_integration/app.py:93-99
```

**修復方案**（發布前執行）:
```python
# services/integration/aiva_integration/app.py
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

async def _consume_logs() -> None:
    """
    背景任務：持續消費日誌訊息並存儲漏洞發現
    增加錯誤處理確保單一錯誤不會終止整個協程
    """
    broker = await get_broker()
    subscriber = await broker.subscribe(Topic.LOG_RESULTS_ALL)
    
    async for mqmsg in subscriber:
        try:
            # 解析訊息
            msg = AivaMessage.model_validate_json(mqmsg.body)
            finding = FindingPayload(**msg.payload)
            
            # 存儲漏洞發現
            await recv.store_finding(finding)
            logger.debug(f"Successfully stored finding: {finding.finding_id}")
            
        except ValidationError as e:
            logger.error(f"Invalid message format, skipping: {e}")
            logger.debug(f"Raw message body: {mqmsg.body[:200]}")  # 記錄前 200 字元
            continue  # 跳過此訊息，繼續處理下一條
            
        except Exception as e:
            logger.error(f"Failed to process log message: {e}", exc_info=True)
            continue  # 跳過此訊息，繼續處理下一條
    
    logger.warning("Log consumer exited (subscriber closed)")
```

**驗證步驟**:
1. 發送格式錯誤的測試訊息
2. 檢查日誌顯示錯誤但服務繼續運行
3. 發送正常訊息驗證處理恢復
4. 模擬資料庫斷線測試錯誤處理

---

## 🟡 P2 - 建議修復（中等優先級）

### 3. SQLite 併發鎖死風險
**問題描述**:
- 已使用 `asyncio.to_thread` 避免主執行緒阻塞（✅ 正確）
- 但缺少 `sqlite3.OperationalError` 錯誤處理
- 缺少 Retry 機制
- 缺少 Timeout 設定

**影響範圍**:
- 🟢 **已部分緩解**: `asyncio.to_thread` 避免主執行緒阻塞
- 🟡 **高併發風險**: 10+ 微服務同時註冊仍可能鎖死，但無 retry
- 🟢 **開發階段影響小**: 單機開發很少遇到

**受影響檔案**:
```
services/integration/capability/registry.py:282, 286, 336, 638
```

**修復方案**（發布前執行）:
```python
# services/integration/capability/registry.py
import sqlite3
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class CapabilityRegistry:
    # ... existing code ...
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
        retry=lambda e: isinstance(e, sqlite3.OperationalError) and "locked" in str(e)
    )
    async def _execute_with_retry(self, operation):
        """
        執行 SQLite 操作並在鎖定時重試
        
        Args:
            operation: 要執行的同步操作（函數）
            
        Returns:
            操作結果
            
        Raises:
            sqlite3.OperationalError: 重試 3 次後仍失敗
        """
        try:
            return await asyncio.to_thread(operation)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.warning(f"Database locked, will retry: {e}")
                raise  # 觸發 tenacity 重試
            else:
                logger.error(f"SQLite error (non-lockable): {e}")
                raise
    
    async def store(self, capability: CapabilityRecord) -> None:
        """存儲能力記錄（帶重試）"""
        def _store():
            self._capabilities[capability.id] = capability
            # 持久化到 SQLite
            conn = sqlite3.connect(self.db_path, timeout=5.0)  # 設定 timeout
            try:
                # ... existing store logic ...
                pass
            finally:
                conn.close()
        
        await self._execute_with_retry(_store)
```

**驗證步驟**:
1. 啟動 10 個微服務同時註冊能力
2. 檢查日誌是否有 "Database locked, will retry" 訊息
3. 確認所有註冊最終成功（有重試）
4. 壓力測試: 20 個併發註冊

**依賴新增**:
```toml
# pyproject.toml
[project]
dependencies = [
    "tenacity>=8.0.0",  # Retry 機制
    # ... existing dependencies ...
]
```

---

## 📋 發布前檢查清單

### 資料庫連線配置
- [ ] 由部署方提供完整 PostgreSQL 連線字串（包含主機、端口、用戶名、密碼、資料庫名）
- [ ] 更新 `services/integration/aiva_integration/config.py:86`
- [ ] 更新 `services/integration/aiva_integration/app.py:37`
- [ ] 更新 `services/integration/alembic/env.py:32`
- [ ] 測試連線正常

### 錯誤處理
- [ ] 所有背景任務已加入 try-except
- [ ] 所有資料庫操作已加入錯誤處理
- [ ] 所有網路請求已加入 timeout 和 retry
- [ ] 日誌記錄完整（包含 exc_info=True）

### 測試驗證
- [ ] Docker Compose 完整啟動測試
- [ ] 服務間通訊測試
- [ ] 資料庫連線測試
- [ ] 錯誤場景測試（格式錯誤訊息、資料庫斷線等）
- [ ] 併發壓力測試

### 文檔更新
- [ ] 部署文檔更新環境變數說明
- [ ] README 更新 Docker 啟動指令
- [ ] 故障排查文檔更新常見錯誤
- [ ] 配置檔案範例更新

---

## 🔍 驗證命令

### 測試資料庫連線
```powershell
# 方法 1: 使用 Python 測試腳本
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://user:pass@host:5432/aiva'); conn = engine.connect(); print('✅ 連線成功'); conn.close()"

# 方法 2: 使用 psql 命令列工具
psql -h <主機> -U <用戶名> -d aiva -c "SELECT version();"
```

### 測試背景任務錯誤處理
```python
# 發送測試訊息到 RabbitMQ
import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 發送格式錯誤的訊息
channel.basic_publish(
    exchange='',
    routing_key='log_results_all',
    body='{"invalid": "json"}'  # 格式錯誤
)
print("✅ 已發送測試訊息，檢查 Integration 日誌是否有錯誤處理")
```

---

## 📝 修復進度追蹤

| 問題編號 | 問題描述 | 優先級 | 狀態 | 預計完成 | 實際完成 |
|---------|---------|-------|------|---------|---------||
| P0-1 | 資料庫連線配置 | 🔴 P0 | ⏳ 待修復 | 發布前 | - |
| P0-2 | 背景任務錯誤處理 | 🔴 P0 | ⏳ 待修復 | 發布前 | - |
| P2-3 | SQLite 併發處理 | 🟡 P2 | ⏳ 待修復 | 下個版本 | - |

---

## 🎯 快速修復腳本（發布時執行）

```powershell
# 1. 備份原始檔案
$files = @(
    "services/integration/aiva_integration/config.py",
    "services/integration/aiva_integration/app.py",
    "services/integration/capability/registry.py"
)
foreach ($file in $files) {
    Copy-Item $file "$file.backup"
}

# 2. 手動編輯檔案（套用上述修復方案）
# - 更新資料庫連線字串
# - 增加錯誤處理
# - 增加 SQLite retry 機制

# 3. 驗證修復
python -m pytest tests/integration/  # 執行整合測試
python services/integration/aiva_integration/app.py  # 啟動服務測試

# 4. 如果失敗，回滾
# foreach ($file in $files) {
#     Move-Item "$file.backup" $file -Force
# }
```

---

## 📚 相關文檔

### 安裝與部署指南
- [系統安裝指南](../guides/deployment/SYSTEM_INSTALLATION_GUIDE.md) - 完整生產環境安裝 (Python/Node/Go/Rust/PostgreSQL/Redis)
- [生產環境故障排除指南](../guides/troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md) - 運行時問題解決
- [BUILD_GUIDE.md](../guides/deployment/BUILD_GUIDE.md) - 構建流程
- [DOCKER_KUBERNETES_GUIDE.md](../guides/deployment/DOCKER_KUBERNETES_GUIDE.md) - 容器化部署

### 開發環境指南
- [INSTALLATION_GUIDE.md](../guides/deployment/INSTALLATION_GUIDE.md) - Python 開發環境安裝 (全域環境、可編輯安裝)

### 架構與進度
- [DEPLOYMENT_PROGRESS_RECORD.md](../reports/project_status/DEPLOYMENT_PROGRESS_RECORD.md) - 部署進度記錄
- [ARCHITECTURE_COMPLETE_DESIGN.md](./ARCHITECTURE_COMPLETE_DESIGN.md) - 完整架構設計
- [Docker Compose 配置](../docker/compose/docker-compose.yml) - 容器編排配置

---

**最後更新**: 2025-11-25  
**更新者**: GitHub Copilot  
**版本**: 1.0.0
