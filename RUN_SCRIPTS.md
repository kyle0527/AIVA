# AIVA 各模組執行腳本 (MVP 版本)

**文件目的**: 提供每個模組按語言分類的快速啟動腳本  
**建立日期**: 2025-10-13  
**適用環境**: Windows PowerShell

---

## 📁 目錄結構總覽

```plaintext
services/
├── aiva_common/          [Python] 共享 Schema/MQ/Config
├── core/aiva_core/       [Python] 智慧分析與協調中心
├── scan/aiva_scan/       [Python] 爬蟲引擎 (未來 → Node.js Playwright)
├── function/
│   ├── function_xss/     [Python] XSS 探測器 (未來 → Node.js Playwright)
│   ├── function_sqli/    [Python] SQLi 探測器 (未來 → Go)
│   ├── function_ssrf/    [Python] SSRF 探測器 (未來 → Go)
│   └── function_idor/    [Python] IDOR 探測器 (未來 → Go)
└── integration/          [Python] 報告整合與分析
    └── aiva_integration/
```

---

## 🐍 Python 模組 (現有實作)

### 前置要求

```powershell
# 1. 確認 Python 版本
python --version  # 需要 3.11+

# 2. 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 3. 安裝依賴 (如果還沒安裝)
pip install -e .
```

### 啟動基礎設施

```powershell
# 啟動 RabbitMQ + PostgreSQL (Docker)
docker-compose -f docker\docker-compose.yml up -d

# 等待服務就緒
Start-Sleep -Seconds 10

# 確認服務狀態
docker ps
# 應該看到: rabbitmq, postgres
```

### 模組 1: Core (智慧分析引擎)

**路徑**: `services/core/aiva_core/`  
**入口**: `app.py`  
**功能**: 攻擊面分析、策略生成、任務協調

```powershell
# 方式 1: 直接運行
cd services\core\aiva_core
python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload

# 方式 2: 使用 Windows 背景啟動
Start-Process pwsh -ArgumentList "-Command", "cd services\core\aiva_core; python -m uvicorn app:app --host 0.0.0.0 --port 8001"

# 檢查 API 文檔
# 瀏覽器打開: http://localhost:8001/docs
```

**環境變數** (可選):

```powershell
$env:RABBITMQ_URL = "amqp://aiva:dev_password@localhost:5672/"
$env:LOG_LEVEL = "DEBUG"
```

---

### 模組 2: Scan (爬蟲引擎)

**路徑**: `services/scan/aiva_scan/`  
**入口**: `worker.py`  
**功能**: URL 發現、靜態內容解析、指紋識別

```powershell
# 運行 Worker
cd services\scan\aiva_scan
python worker.py

# 或使用模組運行
python -m services.scan.aiva_scan.worker
```

**測試掃描任務**:

```powershell
# 發送測試訊息到 RabbitMQ
python -c "
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, ScanStartPayload, MessageHeader
from services.aiva_common.enums import Topic, ModuleName
from services.aiva_common.utils import new_id
import asyncio
import json

async def send_test():
    broker = await get_broker()
    payload = ScanStartPayload(
        scan_id=new_id('scan'),
        target_url='https://example.com',
        max_depth=2,
        max_pages=10
    )
    msg = AivaMessage(
        header=MessageHeader(
            message_id=new_id('msg'),
            trace_id=new_id('trace'),
            correlation_id=payload.scan_id,
            source_module=ModuleName.CORE
        ),
        topic=Topic.TASK_SCAN_START,
        payload=payload.model_dump()
    )
    await broker.publish(Topic.TASK_SCAN_START, json.dumps(msg.model_dump()).encode())
    print(f'✅ 已發送測試掃描任務: {payload.scan_id}')

asyncio.run(send_test())
"
```

---

### 模組 3: Function - XSS 探測器

**路徑**: `services/function/function_xss/aiva_func_xss/`  
**入口**: `worker.py`  
**功能**: 反射型/儲存型/DOM 型 XSS 檢測

```powershell
# 運行 XSS Worker
cd services\function\function_xss\aiva_func_xss
python worker.py

# 背景運行
Start-Process pwsh -ArgumentList "-Command", "cd services\function\function_xss\aiva_func_xss; python worker.py"
```

**依賴**:

- Playwright (瀏覽器自動化)
- 需先安裝瀏覽器: `playwright install chromium`

---

### 模組 4: Function - SQLi 探測器

**路徑**: `services/function/function_sqli/aiva_func_sqli/`  
**入口**: `worker.py`  
**功能**: SQL 注入檢測 (時間盲注、錯誤注入、布林注入)

```powershell
# 運行 SQLi Worker
cd services\function\function_sqli\aiva_func_sqli
python worker.py
```

---

### 模組 5: Function - SSRF 探測器

**路徑**: `services/function/function_ssrf/aiva_func_ssrf/`  
**入口**: `worker.py`  
**功能**: 服務端請求偽造檢測

```powershell
# 運行 SSRF Worker
cd services\function\function_ssrf\aiva_func_ssrf
python worker.py
```

---

### 模組 6: Function - IDOR 探測器

**路徑**: `services/function/function_idor/aiva_func_idor/`  
**入口**: `worker.py`  
**功能**: 不安全直接物件參照檢測

```powershell
# 運行 IDOR Worker
cd services\function\function_idor\aiva_func_idor
python worker.py
```

---

### 模組 7: Integration (報告整合)

**路徑**: `services/integration/aiva_integration/`  
**入口**: `app.py`  
**功能**: 漏洞關聯分析、風險評估、報告生成

```powershell
# 運行 Integration API
cd services\integration\aiva_integration
python -m uvicorn app:app --host 0.0.0.0 --port 8003 --reload

# 檢查 API
# http://localhost:8003/docs
```

---

## 🚀 一鍵啟動所有模組 (MVP)

建立 `start_all.ps1`:

```powershell
# start_all.ps1
Write-Host "🚀 啟動 AIVA 完整系統..." -ForegroundColor Green

# 1. 啟動基礎設施
Write-Host "`n📦 啟動 Docker 服務..." -ForegroundColor Cyan
docker-compose -f docker\docker-compose.yml up -d
Start-Sleep -Seconds 15

# 2. 啟動 Core
Write-Host "`n🧠 啟動 Core 模組..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd services\core\aiva_core; python -m uvicorn app:app --host 0.0.0.0 --port 8001"

# 3. 啟動 Scan
Write-Host "`n🕷️  啟動 Scan 模組..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd services\scan\aiva_scan; python worker.py"

# 4. 啟動 Function Workers
Write-Host "`n🔍 啟動 Function 模組..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd services\function\function_xss\aiva_func_xss; python worker.py"
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd services\function\function_sqli\aiva_func_sqli; python worker.py"
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd services\function\function_ssrf\aiva_func_ssrf; python worker.py"
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd services\function\function_idor\aiva_func_idor; python worker.py"

# 5. 啟動 Integration
Write-Host "`n📊 啟動 Integration 模組..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd services\integration\aiva_integration; python -m uvicorn app:app --host 0.0.0.0 --port 8003"

Write-Host "`n✅ 所有模組已啟動!" -ForegroundColor Green
Write-Host "📍 Core API: http://localhost:8001/docs" -ForegroundColor Yellow
Write-Host "📍 Integration API: http://localhost:8003/docs" -ForegroundColor Yellow
Write-Host "📍 RabbitMQ 管理介面: http://localhost:15672 (帳號: aiva / dev_password)" -ForegroundColor Yellow
```

**使用方式**:

```powershell
.\start_all.ps1
```

---

## 🛑 停止所有服務

建立 `stop_all.ps1`:

```powershell
# stop_all.ps1
Write-Host "🛑 停止 AIVA 系統..." -ForegroundColor Red

# 停止所有 Python 進程
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# 停止 Docker 服務
docker-compose -f docker\docker-compose.yml down

Write-Host "✅ 所有服務已停止" -ForegroundColor Green
```

---

## 🟢 Node.js 模組 (規劃中 - MVP 範例)

### 未來模組: Scan (Playwright 版本)

**路徑**: `services/scan/aiva_scan_node/`  
**入口**: `src/index.ts`

```powershell
# 安裝依賴
cd services\scan\aiva_scan_node
npm install

# 安裝 Playwright 瀏覽器
npx playwright install --with-deps chromium

# 開發模式
npm run dev

# 生產模式
npm run build
npm start
```

**package.json** (MVP):

```json
{
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js"
  }
}
```

---

## 🦀 Go 模組 (規劃中 - MVP 範例)

### 未來模組: Function - SSRF (Go 版本)

**路徑**: `services/function/function_ssrf_go/`  
**入口**: `cmd/worker/main.go`

```powershell
# 初始化 Go 模組
cd services\function\function_ssrf_go
go mod init github.com/aiva/function-ssrf-go
go mod tidy

# 運行
go run cmd/worker/main.go

# 編譯
go build -o ssrf_worker.exe cmd/worker/main.go

# 執行編譯後的執行檔
.\ssrf_worker.exe
```

**目錄結構** (MVP):

```plaintext
function_ssrf_go/
├── cmd/
│   └── worker/
│       └── main.go          # 主入口
├── internal/
│   ├── detector/
│   │   └── ssrf.go          # SSRF 檢測邏輯
│   └── grpc/
│       └── client.go        # gRPC 客戶端
├── proto/                   # Proto 定義
│   └── aiva/
│       └── v1/
│           └── scan.proto
└── go.mod
```

---

## 🦀 Rust 模組 (規劃中 - MVP 範例)

### 未來模組: Info Gatherer (敏感資訊掃描)

**路徑**: `services/scan/info_gatherer_rust/`  
**入口**: `src/main.rs`

```powershell
# 初始化 Rust 專案
cd services\scan\info_gatherer_rust
cargo init --name aiva-info-gatherer

# 運行
cargo run

# 釋出編譯 (優化)
cargo build --release

# 執行編譯後的執行檔
.\target\release\aiva-info-gatherer.exe
```

**Cargo.toml** (MVP):

```toml
[package]
name = "aiva-info-gatherer"
version = "1.0.0"
edition = "2021"

[dependencies]
regex = "1.10"
aho-corasick = "1.1"
tokio = { version = "1.35", features = ["full"] }
```

---

## 🧪 測試各模組

### 端到端測試流程

```powershell
# 1. 啟動所有服務
.\start_all.ps1

# 2. 等待服務就緒
Start-Sleep -Seconds 20

# 3. 發送測試掃描請求
Invoke-RestMethod -Method POST -Uri "http://localhost:8001/scan" -Body (@{
    target_url = "https://testphp.vulnweb.com"
    max_depth = 2
    max_pages = 10
} | ConvertTo-Json) -ContentType "application/json"

# 4. 查看 RabbitMQ 訊息流
# 瀏覽器開啟: http://localhost:15672
# 帳號: aiva / 密碼: dev_password
# 檢查 Queues 頁面的訊息流動

# 5. 查看結果
Invoke-RestMethod -Uri "http://localhost:8003/findings"
```

---

## 📊 監控與除錯

### 查看 RabbitMQ 訊息

```powershell
# 安裝 RabbitMQ 管理工具 (可選)
# 使用 Web UI: http://localhost:15672

# 或使用 Python 查看訊息
python -c "
from services.aiva_common.mq import get_broker
from services.aiva_common.enums import Topic
import asyncio

async def monitor():
    broker = await get_broker()
    print('📡 監聽訊息...')
    async for msg in broker.subscribe(Topic.RESULTS_SCAN_COMPLETED):
        print(f'收到訊息: {msg.body.decode()[:200]}...')

asyncio.run(monitor())
"
```

### 查看 PostgreSQL 資料

```powershell
# 連線到資料庫
docker exec -it aiva-postgres psql -U aiva -d aiva_dev

# SQL 查詢
# SELECT * FROM findings LIMIT 10;
# \q (離開)
```

### 查看日誌

```powershell
# 各模組的日誌輸出在啟動的 PowerShell 視窗中

# 或查看 Docker 日誌
docker logs aiva-rabbitmq
docker logs aiva-postgres
```

---

## 🔧 常見問題排查

### 問題 1: RabbitMQ 連線失敗

```powershell
# 確認 RabbitMQ 運行中
docker ps | Select-String rabbitmq

# 檢查連線
Test-NetConnection localhost -Port 5672
Test-NetConnection localhost -Port 15672

# 重啟 RabbitMQ
docker restart aiva-rabbitmq
```

### 問題 2: Python 模組無法啟動

```powershell
# 確認虛擬環境已啟動
Get-Command python | Select-Object -ExpandProperty Source
# 應該顯示 .venv 路徑

# 重新安裝依賴
pip install --force-reinstall -e .
```

### 問題 3: 埠號衝突

```powershell
# 檢查埠號佔用
netstat -ano | Select-String "8001"
netstat -ano | Select-String "5672"

# 殺掉佔用進程
Stop-Process -Id <PID> -Force
```

---

## 📝 開發工作流程

### 1. 修改代碼後重新載入

```powershell
# FastAPI 應用 (Core/Integration) - 自動重載 (--reload)
# 無需手動重啟

# Worker 進程 - 需要手動重啟
# 在對應的 PowerShell 視窗按 Ctrl+C 停止
# 然後重新運行 python worker.py
```

### 2. 新增依賴

```powershell
# 安裝新套件
pip install <package-name>

# 更新 pyproject.toml
# 在 dependencies = [...] 中新增

# 重新安裝
pip install -e .
```

### 3. 資料庫遷移 (Alembic)

```powershell
# 建立遷移檔
cd services\integration
alembic revision --autogenerate -m "描述變更"

# 執行遷移
alembic upgrade head

# 回滾
alembic downgrade -1
```

---

## 🎯 快速參考

| 模組 | 埠號 | 語言 | 啟動命令 |
|-----|------|------|---------|
| **Core** | 8001 | Python | `uvicorn app:app --port 8001` |
| **Scan** | - | Python | `python worker.py` |
| **XSS** | - | Python | `python worker.py` |
| **SQLi** | - | Python | `python worker.py` |
| **SSRF** | - | Python | `python worker.py` |
| **IDOR** | - | Python | `python worker.py` |
| **Integration** | 8003 | Python | `uvicorn app:app --port 8003` |
| **RabbitMQ** | 5672, 15672 | - | `docker-compose up` |
| **PostgreSQL** | 5432 | - | `docker-compose up` |

---

**文件結束**  
**維護者**: AIVA 開發團隊  
**最後更新**: 2025-10-13
