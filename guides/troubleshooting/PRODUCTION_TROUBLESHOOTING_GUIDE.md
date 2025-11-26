---
Created: 2025-11-25
Document Type: Troubleshooting Guide
Status: Production Ready
Category: Troubleshooting
---

# AIVA 生產環境故障排除指南

## 📑 目錄

- [📋 文檔說明](#文檔說明)
- [🚨 緊急故障快速診斷](#緊急故障快速診斷)
  - [30 秒健康檢查](#30-秒健康檢查)
- [🔧 常見問題與解決方案](#常見問題與解決方案)
  - [問題 1: 資料庫連線失敗 ⭐⭐⭐ 高頻](#問題-1-資料庫連線失敗-高頻)
    - [症狀](#症狀)
    - [原因分析](#原因分析)
    - [解決步驟](#解決步驟)
  - [問題 2: Playwright 瀏覽器啟動失敗 ⭐⭐ 常見](#問題-2-playwright-瀏覽器啟動失敗-常見)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
  - [問題 3: 記憶體不足 (OOM) ⭐⭐ 常見](#問題-3-記憶體不足-oom-常見)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
  - [問題 4: TypeScript Engine 無法啟動 ⭐⭐ 常見](#問題-4-typescript-engine-無法啟動-常見)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
  - [問題 5: Go 模組執行失敗 ⭐ 偶爾](#問題-5-go-模組執行失敗-偶爾)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
  - [問題 6: 背景任務無聲停止 ⭐⭐ 隱蔽](#問題-6-背景任務無聲停止-隱蔽)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
  - [問題 7: SQLite 資料庫鎖定 ⭐ 偶爾](#問題-7-sqlite-資料庫鎖定-偶爾)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
  - [問題 8: Python 套件衝突 ⭐ 偶爾](#問題-8-python-套件衝突-偶爾)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
  - [問題 9: Redis 連線逾時 ⭐ 偶爾](#問題-9-redis-連線逾時-偶爾)
    - [症狀](#症狀-1)
    - [原因分析](#原因分析-1)
    - [解決步驟](#解決步驟-1)
- [🔍 日誌分析](#日誌分析)
  - [關鍵日誌位置](#關鍵日誌位置)
  - [常見錯誤模式](#常見錯誤模式)
    - [資料庫錯誤](#資料庫錯誤)
    - [Playwright 錯誤](#playwright-錯誤)
    - [記憶體錯誤](#記憶體錯誤)
    - [網路錯誤](#網路錯誤)
- [📊 效能監控](#效能監控)
  - [資源使用監控](#資源使用監控)
  - [資料庫效能](#資料庫效能)
- [🚀 效能優化建議](#效能優化建議)
  - [1. 資料庫優化](#1-資料庫優化)
  - [2. 記憶體優化](#2-記憶體優化)
  - [3. 網路優化](#3-網路優化)
- [📋 健康檢查清單](#健康檢查清單)
  - [每日檢查](#每日檢查)
  - [每週檢查](#每週檢查)
  - [每月檢查](#每月檢查)
- [🔗 相關文檔](#相關文檔)
- [📞 支援聯絡](#支援聯絡)

---
---
---
---

## 📋 文檔說明

**前提條件**: 所有安裝需求已完成 (參考 [系統安裝指南](./SYSTEM_INSTALLATION_GUIDE.md))  
**適用場景**: 系統已部署，準備實際運行時遇到的問題  
**不包含**: 開發環境問題、IDE 配置問題

---

## 🚨 緊急故障快速診斷

### 30 秒健康檢查

```powershell
# 1. 檢查 Python 服務
python -c "import fastapi; print('✅ FastAPI OK')"

# 2. 檢查資料庫連線
psql -h <HOST> -U <USER> -d aiva -c "SELECT 1;"

# 3. 檢查 Redis 連線
redis-cli -h <HOST> ping

# 4. 檢查 TypeScript Engine
cd services/scan/engines/typescript_engine
node -e "console.log('✅ Node.js OK')"

# 5. 檢查編譯產物
Test-Path services/features/function_authn_go/worker.exe
Test-Path services/scan/engines/typescript_engine/dist/index.js
```

---

## 🔧 常見問題與解決方案

### 問題 1: 資料庫連線失敗 ⭐⭐⭐ 高頻

#### 症狀
```
sqlalchemy.exc.OperationalError: could not connect to server
psycopg2.OperationalError: connection refused
```

#### 原因分析
1. **錯誤的連線字串**: 配置檔中硬編碼 localhost
2. **資料庫未啟動**: PostgreSQL 服務未運行
3. **憑證錯誤**: 用戶名密碼不正確
4. **網路問題**: 防火牆阻擋、網路不通

#### 解決步驟

**步驟 1: 驗證資料庫服務**
```powershell
# Windows (Docker)
docker ps | Select-String "postgres"

# Windows (本機服務)
Get-Service postgresql*

# Linux (Docker)
docker ps | grep postgres

# Linux (系統服務)
sudo systemctl status postgresql
```

**步驟 2: 測試連線**
```powershell
# 使用 psql 測試
psql -h <HOST> -p 5432 -U <USER> -d aiva

# 如果連線失敗，檢查:
# 1. HOST 是否正確 (生產環境不是 localhost)
# 2. 端口是否正確 (預設 5432)
# 3. 用戶名密碼是否正確
```

**步驟 3: 修正配置**
```python
# 修改 services/integration/aiva_integration/config.py:86
# 修改 services/integration/aiva_integration/app.py:37
# 修改 services/integration/alembic/env.py:32

# 替換為部署方提供的實際連線字串
DATABASE_URL = "postgresql://prod_user:prod_pass@prod-db-host:5432/aiva_prod"
```

**步驟 4: 驗證修正**
```python
# 測試連線
python -c "
from sqlalchemy import create_engine
url = 'postgresql://prod_user:prod_pass@prod-db-host:5432/aiva_prod'
engine = create_engine(url)
with engine.connect() as conn:
    result = conn.execute('SELECT 1')
    print('✅ 資料庫連線成功')
"
```

---

### 問題 2: Playwright 瀏覽器啟動失敗 ⭐⭐ 常見

#### 症狀
```
playwright._impl._errors.Error: Executable doesn't exist
Browser was not installed
```

#### 原因分析
1. **未安裝瀏覽器**: 忘記執行 `playwright install`
2. **缺少系統依賴**: Linux 缺少必要的系統庫
3. **權限問題**: 無法執行瀏覽器二進位檔

#### 解決步驟

**步驟 1: 安裝瀏覽器**
```powershell
# Python Playwright
playwright install chromium

# TypeScript Engine
cd services/scan/engines/typescript_engine
npx playwright install chromium
```

**步驟 2: Linux 額外依賴**
```bash
# Ubuntu/Debian
sudo apt install -y \
  libnss3 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libcups2 \
  libdrm2 \
  libxkbcommon0 \
  libxcomposite1 \
  libxdamage1 \
  libxfixes3 \
  libxrandr2 \
  libgbm1 \
  libasound2

# 或使用完整安裝
playwright install --with-deps chromium
```

**步驟 3: 驗證安裝**
```python
# Python 測試
python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    print('✅ Chromium 啟動成功')
    browser.close()
"
```

```javascript
// TypeScript 測試
node -e "
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  console.log('✅ Chromium 啟動成功');
  await browser.close();
})();
"
```

---

### 問題 3: 記憶體不足 (OOM) ⭐⭐ 常見

#### 症狀
```
MemoryError: Unable to allocate array
Killed (OOM Killer)
```

#### 原因分析
1. **Playwright 佔用過高**: 併發多個瀏覽器實例
2. **Python 記憶體洩漏**: 大量資料未釋放
3. **系統記憶體不足**: 總記憶體 < 8GB

#### 解決步驟

**步驟 1: 檢查記憶體使用**
```powershell
# Windows
Get-Process python, node, chromium | Select-Object Name, @{N='MemoryMB';E={$_.WorkingSet64/1MB}}

# Linux
ps aux | grep -E 'python|node|chromium' | awk '{print $11, $6/1024 " MB"}'
```

**步驟 2: 限制併發數**
```python
# services/scan/engines/python_engine/worker.py
# 修改併發設定
MAX_CONCURRENT_BROWSERS = 2  # 從 5 降至 2
```

```javascript
// services/scan/engines/typescript_engine/src/index.ts
// 修改併發設定
const MAX_CONCURRENT_PAGES = 2;  // 從 5 降至 2
```

**步驟 3: 增加系統記憶體**
- 建議最低 16GB 記憶體
- 或使用更強大的伺服器

**步驟 4: 啟用記憶體監控**
```python
# 新增記憶體監控
import psutil
import logging

def check_memory():
    mem = psutil.virtual_memory()
    if mem.percent > 90:
        logging.warning(f"⚠️ 記憶體使用過高: {mem.percent}%")
```

---

### 問題 4: TypeScript Engine 無法啟動 ⭐⭐ 常見

#### 症狀
```
Error: Cannot find module 'playwright'
SyntaxError: Unexpected token
```

#### 原因分析
1. **未安裝依賴**: 忘記執行 `npm install`
2. **未編譯 TypeScript**: 缺少 dist/ 目錄
3. **Node.js 版本過低**: < 20.0

#### 解決步驟

**步驟 1: 檢查 Node.js 版本**
```powershell
node --version  # 應 ≥ v20.0.0
npm --version   # 應 ≥ 10.0.0
```

**步驟 2: 重新安裝依賴**
```powershell
cd services/scan/engines/typescript_engine

# 清理舊依賴
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json

# 重新安裝
npm install
```

**步驟 3: 編譯 TypeScript**
```powershell
# 編譯
npm run build

# 驗證產物
Test-Path dist/index.js  # 應回傳 True
```

**步驟 4: 測試啟動**
```powershell
# 直接執行測試
node dist/index.js --help
```

---

### 問題 5: Go 模組執行失敗 ⭐ 偶爾

#### 症狀
```
exec: "worker.exe": executable file not found
panic: runtime error
```

#### 原因分析
1. **未編譯**: 缺少 worker.exe
2. **編譯錯誤**: Go 編譯失敗但未注意
3. **缺少依賴**: go.mod 依賴未下載

#### 解決步驟

**步驟 1: 檢查編譯產物**
```powershell
# Windows
Test-Path services/features/function_authn_go/worker.exe
Test-Path services/features/function_sca_go/worker.exe
Test-Path services/features/function_cspm_go/worker.exe
Test-Path services/features/function_ssrf_go/worker.exe

# Linux
test -f services/features/function_authn_go/worker
test -f services/features/function_sca_go/worker
```

**步驟 2: 重新編譯**
```powershell
cd services/features/function_authn_go

# 下載依賴
go mod download

# 編譯 (Windows)
go build -o worker.exe

# 編譯 (Linux)
go build -o worker

# 驗證
./worker.exe --version  # Windows
./worker --version      # Linux
```

**步驟 3: 檢查 Go 環境**
```powershell
go version         # ≥ 1.21
go env GOPATH      # 確認 GOPATH 設置正確
go env GOROOT      # 確認 GOROOT 設置正確
```

---

### 問題 6: 背景任務無聲停止 ⭐⭐ 隱蔽

#### 症狀
- Integration 服務運行正常
- 但漏洞發現無法儲存
- 無明顯錯誤訊息

#### 原因分析
`_consume_logs()` 協程遇到錯誤後終止 (P0-2 問題)

#### 解決步驟

**步驟 1: 檢查 app.py 是否已修復**
```python
# services/integration/aiva_integration/app.py:93-99
# 應包含 try-except 錯誤處理

async def _consume_logs() -> None:
    broker = await get_broker()
    subscriber = await broker.subscribe(Topic.LOG_RESULTS_ALL)
    async for mqmsg in subscriber:
        try:  # ← 必須有這行
            msg = AivaMessage.model_validate_json(mqmsg.body)
            finding = FindingPayload(**msg.payload)
            await recv.store_finding(finding)
        except ValidationError as e:  # ← 必須有錯誤處理
            logger.error(f"Invalid message: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to process: {e}")
            continue
```

**步驟 2: 檢查日誌**
```powershell
# 查找協程停止的證據
Select-String -Path logs/integration.log -Pattern "consumer.*exit|subscriber.*close"
```

**步驟 3: 臨時修復**
```powershell
# 重啟 Integration 服務
# Docker
docker restart aiva-integration

# 本機服務
# 停止並重新啟動 Python 進程
```

**步驟 4: 永久修復**
參考 [部署檢查清單](../DEPLOYMENT_CHECKLIST.md) P0-2 修復方案

---

### 問題 7: SQLite 資料庫鎖定 ⭐ 偶爾

#### 症狀
```
sqlite3.OperationalError: database is locked
```

#### 原因分析
多個服務同時註冊能力，SQLite 併發寫入衝突 (P2-3 問題)

#### 解決步驟

**步驟 1: 臨時解決 - 減少併發**
```powershell
# 依序啟動服務，避免同時註冊
Start-Sleep -Seconds 5  # 每個服務間隔 5 秒
```

**步驟 2: 檢查是否已使用 asyncio.to_thread**
```python
# services/integration/capability/registry.py
# 應使用 asyncio.to_thread 避免阻塞
await asyncio.to_thread(store)
```

**步驟 3: 永久修復 (可選)**
參考 [部署檢查清單](../DEPLOYMENT_CHECKLIST.md) P2-3 修復方案
- 增加 retry 機制
- 增加 timeout 設定
- 使用 tenacity 庫

---

### 問題 8: Python 套件衝突 ⭐ 偶爾

#### 症狀
```
ImportError: cannot import name 'xxx'
AttributeError: module 'xxx' has no attribute 'yyy'
```

#### 原因分析
1. **版本衝突**: 套件版本不相容
2. **快取問題**: __pycache__ 快取過時
3. **安裝不完整**: pip install 中斷

#### 解決步驟

**步驟 1: 清理快取**
```powershell
# 刪除所有 __pycache__
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# 刪除 .pyc 文件
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

**步驟 2: 重新安裝套件到全域環境**
```powershell
# 升級 pip
pip install --upgrade pip

# 重新安裝所有依賴
pip install -r requirements.txt --force-reinstall

# 重新安裝專案本身
pip install -e . --force-reinstall
```

**步驟 3: 驗證安裝**
```powershell
pip list | Select-String "fastapi|playwright|beautifulsoup4|sqlalchemy"
```

---

### 問題 9: Redis 連線逾時 ⭐ 偶爾

#### 症狀
```
redis.exceptions.TimeoutError: Timeout reading from socket
redis.exceptions.ConnectionError: Error connecting to Redis
```

#### 原因分析
1. **Redis 未啟動**: 服務未運行
2. **網路問題**: 防火牆阻擋
3. **記憶體不足**: Redis 記憶體耗盡

#### 解決步驟

**步驟 1: 檢查 Redis 服務**
```powershell
# Docker
docker ps | Select-String "redis"

# 本機服務
redis-cli ping  # 應回應 PONG
```

**步驟 2: 檢查 Redis 記憶體**
```powershell
redis-cli INFO memory | Select-String "used_memory_human"
```

**步驟 3: 重啟 Redis**
```powershell
# Docker
docker restart redis

# 本機服務 (Linux)
sudo systemctl restart redis
```

---

## 🔍 日誌分析

### 關鍵日誌位置

```
logs/
├── integration.log           # Integration 模組日誌
├── scan_python.log          # Python Scan Engine
├── scan_typescript.log      # TypeScript Scan Engine
├── feature_authn_go.log     # Go Feature 模組
└── error.log                # 全域錯誤日誌
```

### 常見錯誤模式

#### 資料庫錯誤
```
grep -i "sqlalchemy.exc\|psycopg2.OperationalError" logs/integration.log
```

#### Playwright 錯誤
```
grep -i "playwright.*error\|browser.*crash" logs/scan_*.log
```

#### 記憶體錯誤
```
grep -i "memoryerror\|oom\|killed" logs/*.log
```

#### 網路錯誤
```
grep -i "timeout\|connection refused\|network" logs/*.log
```

---

## 📊 效能監控

### 資源使用監控

```powershell
# CPU 使用率
Get-Process python, node | Select-Object Name, CPU

# 記憶體使用
Get-Process python, node | Select-Object Name, @{N='MemoryGB';E={$_.WorkingSet64/1GB}}

# 磁碟使用
Get-PSDrive C | Select-Object Used, Free

# 網路連線
Get-NetTCPConnection | Where-Object {$_.State -eq "Established"} | Measure-Object
```

### 資料庫效能

```sql
-- PostgreSQL 活動連線
SELECT count(*) FROM pg_stat_activity;

-- 長時間執行的查詢
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC;

-- 資料庫大小
SELECT pg_size_pretty(pg_database_size('aiva'));
```

---

## 🚀 效能優化建議

### 1. 資料庫優化

```sql
-- 建立索引
CREATE INDEX idx_findings_scan_id ON findings(scan_id);
CREATE INDEX idx_findings_severity ON findings(severity);
CREATE INDEX idx_findings_created_at ON findings(created_at);

-- 定期清理
VACUUM ANALYZE findings;
```

### 2. 記憶體優化

```python
# 限制 Playwright 併發數
MAX_CONCURRENT_BROWSERS = 3  # 根據記憶體調整

# 啟用垃圾回收
import gc
gc.collect()
```

### 3. 網路優化

```python
# 增加連線池大小
from sqlalchemy.pool import QueuePool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)

# 增加 HTTP 超時
httpx.Client(timeout=30.0)
```

---

## 📋 健康檢查清單

### 每日檢查
- [ ] 資料庫連線正常
- [ ] Redis 連線正常
- [ ] 磁碟空間充足 (>20%)
- [ ] 記憶體使用合理 (<80%)
- [ ] 無錯誤日誌堆積

### 每週檢查
- [ ] 資料庫大小合理
- [ ] 日誌輪轉正常
- [ ] 備份完整
- [ ] 更新套件版本

### 每月檢查
- [ ] 效能指標正常
- [ ] 資源使用趨勢
- [ ] 安全更新
- [ ] 資料清理

---

## 🔗 相關文檔

- [系統安裝指南](./SYSTEM_INSTALLATION_GUIDE.md) - 完整安裝清單
- [部署檢查清單](../DEPLOYMENT_CHECKLIST.md) - 發布前修復項目
- [架構完整設計](../../docs/ARCHITECTURE_COMPLETE_DESIGN.md) - 系統架構說明

---

## 📞 支援聯絡

如果上述方案無法解決問題:
1. 收集完整日誌 (`logs/*.log`)
2. 記錄錯誤訊息和重現步驟
3. 檢查系統資源使用狀況
4. 聯絡技術支援團隊

---

**最後更新**: 2025-11-25  
**維護者**: AIVA 開發團隊  
**版本**: 1.0.0
