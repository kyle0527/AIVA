# AIVA TypeScript 掃描引擎 - 完整使用手冊

**更新日期**: 2025-11-18  
**版本**: 1.0.0  
**技術棧**: Node.js 20+ | TypeScript 5.3+ | Playwright 1.41+

使用 Node.js + Playwright 實現的高性能動態網頁掃描引擎，專為 SPA 應用、AJAX 請求、WebSocket 檢測設計。

---

## 📏 目錄

- [功能特性](#功能特性)
- [環境要求](#環境要求)
- [快速開始](#快速開始)
- [配置說明](#配置說明)
- [使用方式](#使用方式)
- [測試驗證](#測試驗證)
- [故障排除](#故障排除)
- [架構說明](#架構說明)

---

## 🎯 功能特性

### Phase1 深度掃描能力
- ✅ **真實瀏覽器渲染**: 使用 Playwright Chromium 引擎
- ✅ **SPA 框架檢測**: React、Vue、Angular、Svelte
- ✅ **動態路由發現**: History API 監聽、Hash 路由提取
- ✅ **AJAX 攔截**: XHR、Fetch API 完整捕獲
- ✅ **WebSocket 檢測**: 實時連接監控
- ✅ **表單與輸入框**: 自動提取所有互動元素
- ✅ **網路請求分析**: API 端點識別、請求模式分析
- ✅ **深度爬取**: 可配置最大深度和頁面數

---

## 📋 環境要求

### 必需依賴
- **Node.js**: >= 20.0.0
- **npm**: >= 10.0.0
- **RabbitMQ**: 3.12+ (運行中)
- **Python**: 3.11+ (用於 worker.py)

### 系統要求
- **記憶體**: >= 2GB (Chromium 需要)
- **磁碟**: >= 500MB (Playwright 瀏覽器)
- **作業系統**: Windows 10+, Linux, macOS

---

## 🚀 快速開始

### 步驟 1: 安裝依賴

```powershell
# 進入 TypeScript 引擎目錄
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 安裝 Node.js 依賴
npm install

# 安裝 Playwright Chromium 瀏覽器
npm run install:browsers
```

### 步驟 2: 編譯 TypeScript

```powershell
# 編譯為 JavaScript (輸出到 dist/)
npm run build

# 驗證編譯產物
ls dist\index.js
```

**預期輸出**:
```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---         2025/11/18   下午 2:30      10240 index.js
```

### 步驟 3: 配置說明

**研發階段無需配置**：所有連接使用預設值，開箱即用。

預設配置：
```javascript
// 自動使用以下預設值
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
// 無需設置任何環境變數
```

**生產環境部署時**（未來）才需要覆蓋預設值。

### 步驟 4: 啟動引擎

**選項 A: 直接啟動 Node.js** (獨立模式)
```powershell
# 必須在 typescript_engine 目錄下執行
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 直接啟動（無需設置環境變數）
node dist/index.js
```
```

**選項 B: 通過 Python Worker** (推薦，整合模式)
```powershell
# 從專案根目錄執行
cd C:\D\fold7\AIVA-git
$env:PYTHONPATH="C:\D\fold7\AIVA-git"

# 啟動 Python Worker (會自動調用 Node.js)
python -m services.scan.engines.typescript_engine.worker
```

---

## ⚙️ 配置說明

**研發階段**：無需任何配置，直接使用預設值。

**預設配置**：
```javascript
const RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
```

**生產環境**（未來部署時才需要）：
```bash
export RABBITMQ_URL="amqp://prod_user:password@prod-host:5672/"
```
| `TASK_QUEUE` | 任務佇列名稱 | `task.scan.dynamic` | `task.scan.phase1` |
| `RESULT_QUEUE` | 結果佇列名稱 | `findings.new` | `results.scan.completed` |
| `LOG_LEVEL` | 日誌級別 | `info` | `debug` |

### 完整 URL 方式 (替代方案)

```powershell
# 使用完整 URL (會覆蓋其他配置)
$env:RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
```

---

## 📖 使用方式

### 1. 開發模式

```powershell
# 熱重載開發 (修改代碼自動重啟)
npm run dev
```

### 2. 生產模式

```powershell
# 編譯
npm run build

# 啟動 (需要環境變數)
npm start
```

### 3. 程式碼品質檢查

```powershell
# ESLint 檢查
npm run lint

# Prettier 格式化
npm run format
```

```powershell
# Prettier 格式化
npm run format
```

---

## 🧪 測試驗證

### 測試 1: 驗證環境配置

```powershell
# 1. 檢查 Node.js 版本
node --version
# 預期: v20.x.x 或更高

# 2. 檢查 npm 版本
npm --version
# 預期: 10.x.x 或更高

# 3. 檢查 RabbitMQ 狀態
docker ps --filter "name=rabbitmq"
# 預期: aiva-rabbitmq 容器運行中

# 4. 驗證編譯產物
Test-Path "dist\index.js"
# 預期: True
```

### 測試 2: 獨立啟動測試

```powershell
# 切換到正確目錄 (❗ 重要)
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 直接啟動（無需設置環境變數）
node dist/index.js
```

**預期輸出**:
```
{"level":30,"time":"2025-11-18T...","msg":"🚀 初始化 AIVA Scan Node..."}
{"level":30,"time":"2025-11-18T...","msg":"🌐 啟動 Chromium 瀏覽器..."}
{"level":30,"time":"2025-11-18T...","msg":"✅ 瀏覽器已啟動"}
{"level":30,"time":"2025-11-18T...","msg":"📡 連接 RabbitMQ..."}
{"level":30,"time":"2025-11-18T...","msg":"✅ RabbitMQ 已連接"}
{"level":30,"time":"2025-11-18T...","msg":"✅ 初始化完成,開始監聽任務..."}
```

### 測試 3: 靶場掃描測試

**前置條件**: Juice Shop 運行在 http://localhost:3000

```powershell
# 使用 Python 測試腳本
cd C:\D\fold7\AIVA-git
$env:PYTHONPATH="C:\D\fold7\AIVA-git"

# 直接執行測試（無需設置連接環境變數）
python services/scan/engines/typescript_engine/test_typescript_engine.py
```

**預期結果**:
- ✅ Node.js 可用性檢查通過
- ✅ 編譯產物存在
- ✅ 靶場連接成功
- ✅ 掃描任務完成
- ✅ 發現資產 (forms, inputs, links, apis)

---

## 🔧 故障排除

### 問題 1: `Error: Cannot find module 'C:\D\fold7\AIVA-git\dist\index.js'`

**原因**: 當前工作目錄不正確

**解決**:
```powershell
# 必須在 typescript_engine 目錄下執行
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine
node dist/index.js
```

### 問題 2: RabbitMQ 連接錯誤

**原因**: RabbitMQ 服務未啟動

**解決**:
```powershell
# 確認 RabbitMQ 狀態
docker ps --filter "name=rabbitmq"

# 如果未運行，啟動 RabbitMQ
docker start aiva-rabbitmq

# 檢查埠號
netstat -an | Select-String "5672"
```

**解決**:
```powershell
# 啟動 RabbitMQ
docker start aiva-rabbitmq

# 或從頭啟動
cd C:\D\fold7\AIVA-git
docker-compose up -d rabbitmq
```

### 問題 4: Playwright 瀏覽器未安裝

**錯誤**: `browserType.launch: Executable doesn't exist`

**解決**:
```powershell
npm run install:browsers
```

### 問題 5: Python Worker 找不到模組

**錯誤**: `ModuleNotFoundError: No module named 'services'`

**解決**:
```powershell
# 設置 PYTHONPATH
$env:PYTHONPATH="C:\D\fold7\AIVA-git"

# 確認虛擬環境已啟動
& C:/D/fold7/AIVA-git/.venv/Scripts/Activate.ps1
```

---

## 🏗️ 架構說明

### 目錄結構

```
typescript_engine/
├── src/                        # TypeScript 源代碼
│   ├── index.ts               # 主入口 (RabbitMQ 監聽)
│   ├── services/              # 掃描服務
│   │   ├── scan-service.ts   # 核心掃描邏輯
│   │   └── network-interceptor.service.ts  # 網路攔截
│   ├── interfaces/            # 型別定義
│   └── utils/                 # 工具函數
├── dist/                       # 編譯產物 (JavaScript)
├── worker.py                   # Python Worker (整合層)
├── package.json               # Node.js 配置
├── tsconfig.json              # TypeScript 配置
└── README.md                  # 本文件
```

### 資料流程

```
Phase1 請求 (RabbitMQ)
    ↓
Python Worker (worker.py)
    ↓
啟動 Node.js 子進程 (dist/index.js)
    ↓
Playwright 瀏覽器自動化
    ↓
ScanService 掃描邏輯
    ├─ 頁面訪問
    ├─ SPA 檢測
    ├─ 網路攔截 (NetworkInterceptor)
    ├─ WebSocket 監聽
    └─ 資產提取
    ↓
返回掃描結果
    ↓
Python Worker 處理
    ↓
發送結果 (RabbitMQ)
```

### 核心組件

| 組件 | 檔案 | 說明 |
|------|------|------|
| **入口** | `src/index.ts` | RabbitMQ 連接、任務監聽 |
| **掃描服務** | `src/services/scan-service.ts` | Playwright 掃描邏輯、SPA 檢測 |
| **網路攔截** | `src/services/network-interceptor.service.ts` | AJAX、API 請求攔截 |
| **Python 橋接** | `worker.py` | Python ↔ Node.js 橋接層 |

---

## 📊 效能指標

| 指標 | 數值 | 說明 |
|------|------|------|
| **頁面載入** | ~2s/頁 | 含 JavaScript 渲染 |
| **深度 3 掃描** | ~15-30s | 取決於目標網站 |
| **記憶體使用** | ~300-500MB | 含 Chromium |
| **CPU 使用** | ~10-30% | 單核心 |

---

## 🔗 相關文件

- [AIVA Common 規範](../../../aiva_common/README.md)
- [掃描流程圖](../SCAN_FLOW_DIAGRAMS.md)
- [引擎完成度分析](../ENGINE_COMPLETION_ANALYSIS.md)
- [Playwright 官方文檔](https://playwright.dev/)

---

## ✅ 檢查清單

使用前請確認:

- [ ] Node.js >= 20.0.0
- [ ] npm install 完成
- [ ] Playwright 瀏覽器已安裝
- [ ] RabbitMQ 容器運行中
- [ ] TypeScript 編譯完成 (dist/ 存在)
- [ ] 環境變數已設置 (USER, PASSWORD)
- [ ] 當前目錄正確 (typescript_engine/)

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2025-11-18  
**問題回報**: GitHub Issues

