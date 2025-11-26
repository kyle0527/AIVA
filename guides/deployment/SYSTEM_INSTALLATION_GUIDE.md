---
Created: 2025-11-25
Document Type: Installation Guide
Status: Production Ready
Category: Deployment
---

# AIVA 系統安裝指南

## 📑 目錄

- [📋 文檔說明](#-文檔說明)
- [🖥️ 系統需求](#️-系統需求)
  - [硬體需求](#硬體需求)
  - [作業系統](#作業系統)
- [🔧 核心運行時環境](#-核心運行時環境)
  - [1. Python 環境](#1-python-環境--必需)
  - [2. Node.js 環境](#2-nodejs-環境--必需-typescript-engine)
  - [3. Go 環境](#3-go-環境--重要-部分-feature-模組)
  - [4. Rust 環境](#4-rust-環境--可選-sastscan-engine)
- [🗄️ 資料庫與中介軟體](#️-資料庫與中介軟體)
  - [1. PostgreSQL](#1-postgresql--必需)
  - [2. RabbitMQ](#2-rabbitmq--可選-已部分棄用)
  - [3. Redis](#3-redis--重要-快取與任務佇列)
- [🔨 編譯工具](#-編譯工具)
  - [Windows](#windows)
  - [Linux](#linux)
- [📦 系統特定依賴](#-系統特定依賴)
  - [Playwright 瀏覽器依賴](#playwright-瀏覽器依賴)
- [🚀 完整安裝流程](#-完整安裝流程)
  - [Windows 環境](#windows-環境)
  - [Linux 環境](#linux-環境)
- [✅ 驗證清單](#-驗證清單)
  - [基礎環境驗證](#基礎環境驗證)
  - [Python 套件驗證](#python-套件驗證)
  - [資料庫連線驗證](#資料庫連線驗證)
  - [Playwright 瀏覽器驗證](#playwright-瀏覽器驗證)
- [📋 依賴摘要](#-依賴摘要)
- [📝 注意事項](#-注意事項)
- [🔗 相關文檔](#-相關文檔)

---

## 📋 文檔說明

**用途**: 列出 AIVA 系統實際運行所需的所有軟體、依賴、工具  
**適用場景**: 生產環境部署、新環境設置、系統遷移  
**不包含**: 開發工具（IDE、Linter）、測試工具、文檔生成工具

---

## 🖥️ 系統需求

### 硬體需求
- **CPU**: 4 核心以上（建議 8 核心）
- **記憶體**: 8GB 以上（建議 16GB）
  - Python 引擎: ~1GB
  - TypeScript/Playwright: ~2GB
  - Go 服務: ~500MB
  - Rust 引擎: ~500MB
  - PostgreSQL: ~1GB
  - RabbitMQ: ~500MB
- **磁碟空間**: 10GB 以上（建議 20GB）
  - 系統程式碼: ~2GB
  - Playwright 瀏覽器: ~500MB
  - PostgreSQL 資料: ~2GB
  - 日誌與報告: ~2GB

### 作業系統
- **Windows**: Windows 10/11 或 Windows Server 2019+
- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+
- **macOS**: macOS 11+ (Big Sur 或更新)

---

## 🔧 核心運行時環境

### 1. Python 環境 ⭐⭐⭐ 必需

#### 版本要求
- **Python 3.13+** (已測試: 3.13.9)
- 不支援 Python 3.12 以下版本

#### 安裝方式
**Windows**:
```powershell
# 下載並安裝 Python 3.13
# https://www.python.org/downloads/

# 驗證安裝
python --version  # 應顯示 Python 3.13.x
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.13 python3.13-venv python3-pip

# 驗證安裝
python3.13 --version
```

#### Python 套件依賴

##### 核心依賴 (生產必需)
```bash
# Web 框架
fastapi>=0.104.0              # API 服務框架
uvicorn[standard]>=0.24.0     # ASGI 伺服器
pydantic>=2.0.0               # 資料驗證

# HTTP 客戶端
httpx>=0.25.0                 # 異步 HTTP 客戶端
aiohttp>=3.9.0                # 異步 HTTP 框架
requests>=2.31.0              # 同步 HTTP 客戶端

# HTML 解析 (Python Engine 必需)
beautifulsoup4>=4.12.0        # HTML 解析器
lxml>=4.9.3                   # XML/HTML 處理引擎

# 瀏覽器自動化 (Python Engine 必需)
playwright>=1.40.0            # 瀏覽器自動化

# 資料庫
sqlalchemy>=2.0.0             # ORM 框架
psycopg2-binary>=2.9.9        # PostgreSQL 驅動
alembic>=1.12.0               # 資料庫遷移工具

# 訊息佇列
pika>=1.3.0                   # RabbitMQ 客戶端 (已棄用但保留相容性)

# 快取
redis>=5.0.0                  # Redis 客戶端

# 資料處理
pandas>=2.1.0                 # 資料分析 (報告生成)
networkx>=3.2.0               # 圖計算 (攻擊路徑分析)

# 安全相關
cryptography>=41.0.0          # 加密工具
pyjwt>=2.8.0                  # JWT 處理

# 日誌與監控
structlog>=23.2.0             # 結構化日誌
```

##### 安裝命令
```bash
# 安裝核心依賴到全域環境
pip install -r requirements.txt

# 將專案本身以可編輯模式安裝到全域
pip install -e .

# 安裝 Playwright 瀏覽器
playwright install chromium
```

---

### 2. Node.js 環境 ⭐⭐⭐ 必需 (TypeScript Engine)

#### 版本要求
- **Node.js 20.0.0+**
- **npm 10.0.0+**

#### 安裝方式
**Windows**:
```powershell
# 下載並安裝 Node.js LTS
# https://nodejs.org/

# 驗證安裝
node --version  # 應顯示 v20.x.x
npm --version   # 應顯示 10.x.x
```

**Linux**:
```bash
# 使用 NodeSource 倉庫
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 驗證安裝
node --version
npm --version
```

#### Node.js 套件依賴

##### TypeScript Engine 依賴
```bash
cd services/scan/engines/typescript_engine

# 核心生產依賴
npm install playwright@1.56.1      # 瀏覽器自動化 (關鍵)
npm install amqplib@0.10.9         # RabbitMQ 客戶端 (已棄用)
npm install pino@8.21.0            # 日誌框架
npm install pino-pretty@11.2.2     # 日誌格式化

# 開發依賴 (生產環境可選)
npm install --save-dev typescript@5.9.3
npm install --save-dev @types/node@22.10.1

# 安裝 Playwright 瀏覽器
npx playwright install chromium
```

---

### 3. Go 環境 ⭐⭐ 重要 (部分 Feature 模組)

#### 版本要求
- **Go 1.21+** (建議 Go 1.23+)

#### 安裝方式
**Windows**:
```powershell
# 下載並安裝 Go
# https://go.dev/dl/

# 驗證安裝
go version  # 應顯示 go1.21 或更高
```

**Linux**:
```bash
# 下載並安裝 Go
wget https://go.dev/dl/go1.23.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.23.linux-amd64.tar.gz

# 設置環境變數
export PATH=$PATH:/usr/local/go/bin

# 驗證安裝
go version
```

#### Go 模組依賴
```bash
# 以下模組使用 Go 實現
cd services/features/function_authn_go
go mod download

cd services/features/function_sca_go
go mod download

cd services/features/function_cspm_go
go mod download

cd services/features/function_ssrf_go
go mod download
```

---

### 4. Rust 環境 ⭐ 可選 (SAST、Scan Engine)

#### 版本要求
- **Rust 1.70+** (建議最新穩定版)

#### 安裝方式
**Windows**:
```powershell
# 下載並安裝 rustup
# https://rustup.rs/

# 驗證安裝
rustc --version
cargo --version
```

**Linux**:
```bash
# 安裝 rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 驗證安裝
rustc --version
cargo --version
```

#### Rust 模組依賴
```bash
# SAST 模組
cd services/features/function_sast_rust
cargo build --release

# Rust Scan Engine (如果使用)
cd services/scan/engines/rust_engine
cargo build --release
```

---

## 🗄️ 資料庫與中介軟體

### 1. PostgreSQL ⭐⭐⭐ 必需

#### 版本要求
- **PostgreSQL 15+** (建議 PostgreSQL 16)

#### 用途
- Integration 模組漏洞發現資料庫
- 歷史掃描結果儲存
- 風險評估資料

#### 安裝方式
**Windows**:
```powershell
# 下載並安裝 PostgreSQL
# https://www.postgresql.org/download/windows/

# 或使用 Docker
docker run -d --name postgres `
  -e POSTGRES_USER=aiva `
  -e POSTGRES_PASSWORD=aiva_secure_password `
  -e POSTGRES_DB=aiva `
  -p 5432:5432 `
  postgres:16
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-16

# 或使用 Docker
docker run -d --name postgres \
  -e POSTGRES_USER=aiva \
  -e POSTGRES_PASSWORD=aiva_secure_password \
  -e POSTGRES_DB=aiva \
  -p 5432:5432 \
  postgres:16
```

#### 配置需求
```sql
-- 建立資料庫
CREATE DATABASE aiva;

-- 建立用戶 (由部署方提供憑證)
CREATE USER aiva_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE aiva TO aiva_user;
```

---

### 2. RabbitMQ ⭐ 可選 (已部分棄用)

#### 版本要求
- **RabbitMQ 3.12+** (如果使用舊版訊息佇列)

#### 狀態
- ⚠️ **正在遷移**: Core 模組已移除 MQ 依賴
- ✅ **仍支援**: TypeScript Engine 仍使用 RabbitMQ
- 📅 **未來**: 將完全移除

#### 安裝方式 (可選)
**Docker**:
```bash
docker run -d --name rabbitmq \
  -e RABBITMQ_DEFAULT_USER=aiva \
  -e RABBITMQ_DEFAULT_PASS=aiva_mq_password \
  -e RABBITMQ_DEFAULT_VHOST=aiva \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3.13-management
```

---

### 3. Redis ⭐⭐ 重要 (快取與任務佇列)

#### 版本要求
- **Redis 7+**

#### 用途
- 掃描任務快取
- 臨時資料儲存
- 分散式鎖

#### 安裝方式
**Windows**:
```powershell
# 使用 Docker
docker run -d --name redis `
  -p 6379:6379 `
  redis:7
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server

# 或使用 Docker
docker run -d --name redis \
  -p 6379:6379 \
  redis:7
```

---

## 🔨 編譯工具

### Windows

#### Visual Studio Build Tools ⭐⭐ 重要
- **用途**: 編譯 Python C 擴展 (lxml, cryptography)
- **下載**: https://visualstudio.microsoft.com/downloads/
- **組件**: "Desktop development with C++"

### Linux

#### 編譯工具鏈 ⭐⭐ 重要
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential gcc g++ make

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
```

#### 系統依賴庫
```bash
# Ubuntu/Debian
sudo apt install -y \
  libssl-dev \
  libffi-dev \
  libpq-dev \
  libxml2-dev \
  libxslt1-dev \
  zlib1g-dev

# CentOS/RHEL
sudo yum install -y \
  openssl-devel \
  libffi-devel \
  postgresql-devel \
  libxml2-devel \
  libxslt-devel \
  zlib-devel
```

---

## 📦 系統特定依賴

### Playwright 瀏覽器依賴

#### Linux 額外依賴
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
```

---

## 🚀 完整安裝流程

### Windows 環境

```powershell
# 1. 安裝運行時
# - Python 3.13+
# - Node.js 20+
# - Go 1.21+
# - Rust (可選)

# 2. 安裝資料庫
docker run -d --name postgres `
  -e POSTGRES_USER=aiva `
  -e POSTGRES_PASSWORD=aiva_secure_password `
  -e POSTGRES_DB=aiva `
  -p 5432:5432 `
  postgres:16

docker run -d --name redis `
  -p 6379:6379 `
  redis:7

# 3. 安裝 Python 依賴
cd C:\D\fold7\AIVA-git
pip install -r requirements.txt
pip install -e .
playwright install chromium

# 4. 安裝 TypeScript Engine
cd services\scan\engines\typescript_engine
npm install
npx playwright install chromium

# 5. 編譯 Go 模組
cd services\features\function_authn_go
go build -o worker.exe

cd ..\function_sca_go
go build -o worker.exe

cd ..\function_cspm_go
go build -o worker.exe

cd ..\function_ssrf_go
go build -o worker.exe

# 6. 編譯 Rust 模組 (可選)
cd ..\..\..\..\scan\engines\rust_engine
cargo build --release
```

### Linux 環境

```bash
# 1. 安裝系統依賴
sudo apt update
sudo apt install -y \
  build-essential \
  libssl-dev \
  libffi-dev \
  libpq-dev \
  python3.13 \
  python3-pip \
  nodejs \
  npm \
  golang-go \
  cargo

# 2. 安裝資料庫 (Docker)
docker run -d --name postgres \
  -e POSTGRES_USER=aiva \
  -e POSTGRES_PASSWORD=aiva_secure_password \
  -e POSTGRES_DB=aiva \
  -p 5432:5432 \
  postgres:16

docker run -d --name redis \
  -p 6379:6379 \
  redis:7

# 3. 安裝 Python 依賴
cd /path/to/AIVA-git
pip install -r requirements.txt
pip install -e .
playwright install --with-deps chromium

# 4. 安裝 TypeScript Engine
cd services/scan/engines/typescript_engine
npm install
npx playwright install chromium

# 5. 編譯 Go 模組
cd services/features/function_authn_go
go build -o worker

cd ../function_sca_go
go build -o worker

cd ../function_cspm_go
go build -o worker

cd ../function_ssrf_go
go build -o worker

# 6. 編譯 Rust 模組 (可選)
cd ../../../../scan/engines/rust_engine
cargo build --release
```

---

## ✅ 驗證清單

### 基礎環境驗證
```powershell
# 檢查 Python
python --version                    # ≥ 3.13

# 檢查 Node.js
node --version                      # ≥ 20.0
npm --version                       # ≥ 10.0

# 檢查 Go
go version                          # ≥ 1.21

# 檢查 Rust (可選)
rustc --version                     # ≥ 1.70
cargo --version

# 檢查 PostgreSQL
psql --version                      # ≥ 15

# 檢查 Redis
redis-cli --version                 # ≥ 7
```

### Python 套件驗證
```python
# 驗證關鍵套件
python -c "import fastapi; print('✅ FastAPI:', fastapi.__version__)"
python -c "import playwright; print('✅ Playwright:', playwright.__version__)"
python -c "import bs4; print('✅ BeautifulSoup4:', bs4.__version__)"
python -c "import sqlalchemy; print('✅ SQLAlchemy:', sqlalchemy.__version__)"
python -c "import psycopg2; print('✅ psycopg2:', psycopg2.__version__)"
```

### 資料庫連線驗證
```powershell
# PostgreSQL
psql -h localhost -U aiva -d aiva -c "SELECT version();"

# Redis
redis-cli ping  # 應回應 PONG
```

### Playwright 瀏覽器驗證
```powershell
# 檢查 Chromium 是否安裝
playwright install chromium --dry-run
```

---

## 📋 依賴摘要

### 運行時環境
| 組件 | 版本 | 必需性 | 用途 |
|------|------|--------|------|
| Python | 3.13+ | ⭐⭐⭐ 必需 | 核心引擎、API、Integration |
| Node.js | 20.0+ | ⭐⭐⭐ 必需 | TypeScript Scan Engine |
| Go | 1.21+ | ⭐⭐ 重要 | 部分 Feature 模組 |
| Rust | 1.70+ | ⭐ 可選 | SAST、Scan Engine |

### 資料庫與中介軟體
| 組件 | 版本 | 必需性 | 用途 |
|------|------|--------|------|
| PostgreSQL | 15+ | ⭐⭐⭐ 必需 | 漏洞發現資料庫 |
| Redis | 7+ | ⭐⭐ 重要 | 快取與任務佇列 |
| RabbitMQ | 3.12+ | ⭐ 可選 | 訊息佇列 (部分棄用) |

### 關鍵 Python 套件
| 套件 | 版本 | 必需性 | 用途 |
|------|------|--------|------|
| fastapi | 0.104+ | ⭐⭐⭐ | API 框架 |
| playwright | 1.40+ | ⭐⭐⭐ | 瀏覽器自動化 |
| beautifulsoup4 | 4.12+ | ⭐⭐⭐ | HTML 解析 |
| sqlalchemy | 2.0+ | ⭐⭐⭐ | ORM 框架 |
| httpx | 0.25+ | ⭐⭐ | HTTP 客戶端 |
| pydantic | 2.0+ | ⭐⭐ | 資料驗證 |

### 關鍵 Node.js 套件
| 套件 | 版本 | 必需性 | 用途 |
|------|------|--------|------|
| playwright | 1.56+ | ⭐⭐⭐ | 瀏覽器自動化 |
| typescript | 5.9+ | ⭐⭐ | TypeScript 編譯 |
| @types/node | 22.10+ | ⭐⭐ | Node.js 類型定義 |

---

## 📝 注意事項

### 生產環境特別注意

1. **資料庫憑證**: 
   - 不使用硬編碼密碼
   - 由部署方提供完整連線字串
   - 定期輪換密碼

2. **Playwright 瀏覽器**:
   - Linux 需安裝額外系統依賴
   - Docker 環境使用 `--with-deps` 安裝

3. **記憶體管理**:
   - Playwright 需要 2GB+ 記憶體
   - 併發掃描建議 16GB+ 記憶體

4. **網路需求**:
   - 需要存取外部網路 (Python/npm 套件下載)
   - 需要存取資料庫 (PostgreSQL)
   - 需要存取 Redis

5. **權限需求**:
   - 讀寫系統日誌目錄
   - 讀寫資料庫
   - 執行編譯過的二進位檔

---

## 🔗 相關文檔

- [生產環境故障排除指南](./PRODUCTION_TROUBLESHOOTING_GUIDE.md) - 運行時問題解決
- [部署檢查清單](../DEPLOYMENT_CHECKLIST.md) - 發布前修復項目
- [架構完整設計](../../docs/ARCHITECTURE_COMPLETE_DESIGN.md) - 系統架構說明
- [README.md](../../README.md) - 專案概覽

---

**最後更新**: 2025-11-25  
**維護者**: AIVA 開發團隊  
**版本**: 1.0.0
