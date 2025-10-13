# 🔍 AIVA 多語言系統依賴分析報告

> **產生時間**: 2025-01-21  
> **分析目的**: 確認所有語言環境的依賴安裝狀態,識別缺失的套件

---

\n## 📊 執行環境檢查結果\n
- ❌ `github.com/rabbitmq/amqp091-go v1.9.0` - RabbitMQ 官方客戶端

\n### ✅ 已安裝的工具鏈\n

| 語言 | 需求版本 | 實際版本 | 狀態 |
|------|---------|---------|------|
| **Python** | 3.13+ | **3.12.10** | ⚠️ 版本略低但可用 |
| **Node.js** | 20+ | **22.19.0** | ✅ 符合需求 |
| **Go** | 1.21+ | **1.25.0** | ✅ 符合需求 |
| **Rust** | 1.70+ | **1.90.0** | ✅ 符合需求 |

\n### ❌ 缺失的環境與套件\n

| 項目 | 狀態 | 缺失內容 |
|------|------|---------|
| **Python 虛擬環境** | ❌ 不存在 | `.venv` 目錄未建立 |
| **Python 套件** | ❌ 未安裝 | 所有 `pyproject.toml` 中的套件 |
| **Node.js 模組** | ❌ 不存在 | `node_modules` 目錄未建立 |
| **Go 模組快取** | ❓ 未確認 | 需執行 `go mod download` |
| **Rust 依賴快取** | ❓ 未確認 | 需執行 `cargo build` |

---

\n## 🐍 Python 依賴狀態\n
- ❌ `go.uber.org/multierr v1.11.0` - Zap 的依賴項

\n### 核心運行時依賴 (18 個)\n

**Web 框架與 HTTP:**

- ❌ `fastapi >= 0.115.0` - 主要 Web 框架
- ❌ `uvicorn >= 0.30.0` - ASGI 服務器 (含 standard extra)
- ❌ `httpx >= 0.27.0` - 異步 HTTP 客戶端


**數據驗證與處理:**

- ❌ `pydantic >= 2.7.0` - 數據模型驗證
- ❌ `pydantic-settings >= 2.2.0` - 配置管理


**消息隊列:**

- ❌ `aio-pika >= 9.4.0` - RabbitMQ 異步客戶端


**Web 抓取與解析:**

- ❌ `beautifulsoup4 >= 4.12.2` - HTML 解析
- ❌ `lxml >= 5.0.0` - XML/HTML 處理器
- ❌ `selenium >= 4.18.0` - 瀏覽器自動化


**日誌與配置:**

- ❌ `structlog >= 24.1.0` - 結構化日誌
- ❌ `python-dotenv >= 1.0.0` - 環境變量管理


**數據庫:**

- ❌ `redis >= 5.0.0` - Redis 客戶端
- ❌ `sqlalchemy >= 2.0.31` - ORM 框架
- ❌ `asyncpg >= 0.29.0` - PostgreSQL 異步驅動
- ❌ `alembic >= 1.13.2` - 數據庫遷移工具
- ❌ `neo4j >= 5.23.0` - Neo4j 圖數據庫客戶端


**其他工具:**

- ❌ `charset-normalizer >= 3.3.0` - 字符編碼檢測
- ❌ `jinja2 >= 3.1.0` - 模板引擎


\n### 開發工具依賴 (11 個)\n


**測試框架:**

- ❌ `pytest >= 8.0.0` - 單元測試框架
- ❌ `pytest-asyncio >= 0.23.0` - 異步測試支援
- ❌ `pytest-cov >= 4.0.0` - 測試覆蓋率
- ❌ `pytest-mock >= 3.12.0` - Mock 支援
- ❌ `httpx >= 0.27.0` - HTTP 測試客戶端


**代碼質量工具:**

- ❌ `black >= 24.0.0` - 代碼格式化
- ❌ `isort >= 5.13.0` - Import 排序
- ❌ `ruff >= 0.3.0` - 快速 Linter
- ❌ `mypy >= 1.8.0` - 靜態類型檢查


**文檔工具:**

- ❌ `sphinx >= 7.2.0` - 文檔生成
- ❌ `sphinx-rtd-theme >= 2.0.0` - ReadTheDocs 主題


\n### 📌 安裝命令\n

```powershell
# 1. 創建虛擬環境
python -m venv .venv

# 2. 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 3. 升級 pip
python -m pip install --upgrade pip

# 4. 安裝所有依賴 (包含開發工具)
pip install -e ".[dev]"

# 或僅安裝運行時依賴
pip install -e .
```

\n### ✔️ 驗證安裝\n

```powershell
# 檢查已安裝套件
pip list

# 檢查關鍵套件版本
pip show fastapi uvicorn pydantic aio-pika sqlalchemy
```

---

\n## 📦 Node.js 依賴狀態\n


\n### 專案位置\n
`services/scan/aiva_scan_node/`

\n### 運行時依賴 (4 個)\n

- ❌ `amqplib ^0.10.3` - RabbitMQ 客戶端
- ❌ `playwright ^1.41.0` - 瀏覽器自動化 (支援 Chromium/Firefox/WebKit)
- ❌ `pino ^8.17.0` - 高性能日誌框架
- ❌ `pino-pretty ^10.3.0` - 日誌美化輸出

\n### 開發工具依賴 (9 個)\n

**TypeScript 支援:**

- ❌ `typescript ^5.3.3` - TypeScript 編譯器
- ❌ `@types/node ^20.11.0` - Node.js 類型定義
- ❌ `@types/amqplib ^0.10.4` - amqplib 類型定義
- ❌ `tsx ^4.7.0` - TypeScript 直接執行器

**測試工具:**

- ❌ `vitest ^1.2.0` - 單元測試框架
- ❌ `@vitest/ui ^1.2.0` - 測試 UI 界面

**代碼質量:**

- ❌ `eslint ^8.56.0` - JavaScript/TypeScript Linter
- ❌ `@typescript-eslint/parser ^6.19.0` - ESLint TypeScript 解析器
- ❌ `@typescript-eslint/eslint-plugin ^6.19.0` - ESLint TypeScript 規則


\n### 📌 安裝命令\n

```powershell
# 進入專案目錄
cd services\scan\aiva_scan_node

# 安裝所有依賴
npm install

# 安裝 Playwright 瀏覽器
npx playwright install chromium
```

\n### ✔️ 驗證安裝\n

```powershell
# 檢查已安裝套件
npm list --depth=0

# 編譯 TypeScript 檢查
npm run build

# 執行測試
npm test
```


---

\n## 🐹 Go 依賴狀態\n


\n### 專案位置\n
`services/function/function_ssrf_go/`


\n### 直接依賴 (2 個)\n


- ❌ `github.com/rabbitmq/amqp091-go v1.9.0` - RabbitMQ 官方客戶端
- ❌ `go.uber.org/zap v1.26.0` - 高性能結構化日誌


\n### 間接依賴 (1 個)\n


- ❌ `go.uber.org/multierr v1.11.0` - Zap 的依賴項

\n### 📌 安裝命令\n

```powershell
# 進入專案目錄
cd services\function\function_ssrf_go

# 下載所有依賴
go mod download

# 整理模組文件
go mod tidy

# 驗證依賴完整性
go mod verify
```

\n### ✔️ 驗證安裝\n

```powershell
# 檢查模組依賴樹
go mod graph

# 編譯檢查 (不產生二進制文件)
go build -o NUL .\cmd\worker

# 執行測試
go test ./...
```

---

\n## 🦀 Rust 依賴狀態\n

\n### 專案位置\n
`services/scan/info_gatherer_rust/`

\n### 運行時依賴 (10 個)\n

**正則表達式引擎:**
- ❌ `regex = "1.10"` - 標準正則表達式
- ❌ `aho-corasick = "1.1"` - 多模式字符串匹配 (高效能)

**並發處理:**
- ❌ `rayon = "1.8"` - 數據並行處理框架
- ❌ `tokio = { version = "1.35", features = ["full"] }` - 異步運行時

**序列化:**
- ❌ `serde = { version = "1.0", features = ["derive"] }` - 序列化框架
- ❌ `serde_json = "1.0"` - JSON 支援

**消息隊列:**
- ❌ `lapin = "2.3"` - RabbitMQ 異步客戶端
- ❌ `futures = "0.3"` - Future trait 和工具

**日誌:**
- ❌ `tracing = "0.1"` - 應用級追蹤框架
- ❌ `tracing-subscriber = { version = "0.3", features = ["env-filter"] }` - 日誌訂閱器

\n### 開發依賴 (1 個)\n

- ❌ `criterion = "0.5"` - 性能基準測試框架

\n### 📌 安裝命令\n

```powershell
# 進入專案目錄
cd services\scan\info_gatherer_rust

# 檢查依賴 (不編譯)
cargo check

# 編譯 (Debug 模式)
cargo build

# 編譯 (Release 模式,啟用優化)
cargo build --release
```

\n### ✔️ 驗證安裝\n

```powershell
# 檢查依賴樹
cargo tree

# 執行測試
cargo test

# 執行基準測試
cargo bench
```

---

## 🐳 基礎設施依賴

### Docker 服務 (2 個)

**RabbitMQ:**
- 映像: `rabbitmq:3.13-management-alpine`
- 端口: 5672 (AMQP), 15672 (Management UI)
- 狀態: ❓ 需檢查

**PostgreSQL:**
- 映像: `postgres:16-alpine`
- 端口: 5432
- 狀態: ❓ 需檢查

### Docker 啟動命令

```powershell
# 啟動所有服務
docker-compose -f docker\docker-compose.yml up -d

# 檢查服務狀態
docker-compose -f docker\docker-compose.yml ps

# 查看日誌
docker-compose -f docker\docker-compose.yml logs -f
```

### Docker 驗證服務

```powershell
# 測試 RabbitMQ 連接
curl http://localhost:15672

# 測試 PostgreSQL 連接 (需安裝 psql)
psql -h localhost -U aiva_user -d aiva_db
```

---

## 🚀 一鍵安裝腳本

我們已經提供 `setup_multilang.ps1`,它會自動執行以下所有步驟:

```powershell
# 執行自動化安裝腳本
.\setup_multilang.ps1
```

### 安裝腳本執行內容

1. ✅ **檢查工具鏈**: 驗證 Python/Node.js/Go/Rust 是否已安裝
2. 🐍 **Python 環境**: 創建 `.venv` + 安裝所有 pip 套件
3. 📦 **Node.js 模組**: 安裝 npm 套件 + Playwright 瀏覽器
4. 🐹 **Go 模組**: 下載並驗證 Go 依賴
5. 🦀 **Rust 依賴**: 編譯 Release 版本
6. 🐳 **Docker 服務**: 啟動 RabbitMQ + PostgreSQL
7. ✔️ **健康檢查**: 驗證所有服務可用性

---

## 📋 依賴清單總結

### 依賴套件數量總結

| 語言 | 運行時依賴 | 開發依賴 | 總計 |
|------|-----------|---------|------|
| **Python** | 18 | 11 | **29** |
| **Node.js** | 4 | 9 | **13** |
| **Go** | 2 | 0 | **2** |
| **Rust** | 10 | 1 | **11** |
| **總計** | **34** | **21** | **55** |

### 依賴安裝預估時間

| 步驟 | 預估時間 | 網絡需求 |
|------|---------|---------|
| Python 套件安裝 | 5-10 分鐘 | 200-500 MB |
| Node.js 模組安裝 | 3-5 分鐘 | 150-300 MB |
| Playwright 瀏覽器 | 5-10 分鐘 | 400-600 MB |
| Go 模組下載 | 1-2 分鐘 | 10-20 MB |
| Rust 編譯 | 5-15 分鐘 | 50-100 MB |
| Docker 映像拉取 | 3-5 分鐘 | 200-300 MB |
| **總計** | **22-47 分鐘** | **1-2 GB** |

---

## ⚠️ 常見問題與解決方案

### Python 版本問題

**問題**: 當前 Python 3.12.10,專案需求 3.13+
```powershell
# 解決方案 1: 放寬版本限制 (修改 pyproject.toml)
# requires-python = ">=3.12"  # 原為 >=3.13

# 解決方案 2: 安裝 Python 3.13 (從 python.org 下載)
# 然後使用 py -3.13 指定版本
py -3.13 -m venv .venv
```

### Node.js Playwright 安裝失敗

```powershell
# 如果 npx playwright install 失敗,手動下載瀏覽器
$env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
npm install
npx playwright install --with-deps chromium
```

### Go 模組下載緩慢

```powershell
# 配置 Go 代理 (使用阿里雲鏡像)
go env -w GOPROXY=https://mirrors.aliyun.com/goproxy/,direct
go env -w GOSUMDB=sum.golang.google.cn
go mod download
```

### Rust 編譯時間過長

```powershell
# 使用 sccache 加速編譯 (Windows)
cargo install sccache
$env:RUSTC_WRAPPER="sccache"
cargo build --release
```

### Docker 服務無法啟動

```powershell
# 檢查端口衝突
netstat -ano | findstr "5672 15672 5432"

# 如果端口被佔用,修改 docker-compose.yml 中的端口映射
# 例如: "5673:5672" 代替 "5672:5672"
```

---

## ✅ 下一步行動

### 推薦執行順序

1. **立即執行**: `.\setup_multilang.ps1` (一鍵安裝所有依賴)
2. **驗證安裝**: `.\check_status.ps1` (檢查所有服務狀態)
3. **啟動系統**: `.\start_all_multilang.ps1` (啟動多語言系統)
4. **測試功能**: `.\test_scan.ps1` (發送測試任務)
5. **查看日誌**: 檢查各服務日誌確認運行正常

### 手動安裝（自動化腳本失敗時）

```powershell
# 1. Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Node.js
cd services\scan\aiva_scan_node
npm install
npx playwright install chromium
cd ..\..\..

# 3. Go
cd services\function\function_ssrf_go
go mod download
go mod tidy
cd ..\..\..

# 4. Rust
cd services\scan\info_gatherer_rust
cargo build --release
cd ..\..\..

# 5. Docker
docker-compose -f docker\docker-compose.yml up -d
```

---

## 📊 總結

- **工具鏈狀態**: ✅ 4/4 已安裝 (Python/Node/Go/Rust)
- **Python 套件**: ❌ 0/29 已安裝 (虛擬環境未建立)
- **Node.js 模組**: ❌ 0/13 已安裝 (node_modules 不存在)
- **Go 模組**: ❌ 未確認 (需執行 go mod download)
- **Rust 依賴**: ❌ 未確認 (需執行 cargo build)
- **Docker 服務**: ❌ 未確認 (需啟動容器)

**結論**: 所有開發工具鏈已就緒,但所有語言的依賴套件都需要安裝。**建議直接執行 `.\setup_multilang.ps1` 完成全部設置。**
