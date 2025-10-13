# AIVA 專案使用指南

# =================

## 📋 目錄

1. [環境設置](#環境設置)
2. [依賴安裝](#依賴安裝)
3. [專案分析工具](#專案分析工具)
4. [多語言開發](#多語言開發)
5. [測試與品質檢查](#測試與品質檢查)
6. [部署與運行](#部署與運行)

## 🚀 環境設置

### 1. 系統需求

- **Python**: 3.11+
- **Node.js**: 18+ (TypeScript 模組)
- **Go**: 1.21+ (Go 模組)
- **Rust**: 1.70+ (Rust 模組)
- **Docker**: 24+ (容器化部署)

### 2. 開發環境設置

#### 自動設置 (推薦)

```bash
# 使用 PowerShell 腳本 (Windows)
.\setup_multilang.ps1

# 或使用 Bash 腳本 (Linux/Mac)
./generate_project_report.sh
```

#### 手動設置

```bash
# 1. 安裝 Python 依賴
pip install -e .[dev]

# 2. 安裝 Node.js 依賴
cd services/scan/aiva_scan_node && npm install

# 3. 安裝 Go 依賴
cd services/function/function_ssrf_go && go mod download

# 4. 安裝 Rust 依賴
cd services/function/function_sast_rust && cargo build --release

# 5. 啟動 Docker 服務
docker-compose -f docker/docker-compose.yml up -d
```

## 📦 依賴安裝

### Python 依賴

#### 核心依賴 (已包含)

```bash
pip install -e .
```

#### 開發依賴 (包含測試、格式化等)

```bash
pip install -e .[dev]
```

#### AI/ML 依賴 (需要額外安裝)

```bash
# 安裝機器學習依賴
pip install scikit-learn joblib

# 或安裝完整 AI 套件
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
```

### 多語言依賴

#### Go 模組

```bash
# 統一安裝所有 Go 模組依賴
./init_go_deps.ps1

# 或手動安裝
cd services/function/function_authn_go && go mod tidy
cd services/function/function_cspm_go && go mod tidy
# ... 其他 Go 模組
```

#### Rust 模組

```bash
# 統一編譯所有 Rust 模組
./build_rust_modules.ps1

# 或手動編譯
cd services/function/function_sast_rust && cargo build --release
cd services/scan/info_gatherer_rust && cargo build --release
```

#### Node.js/TypeScript 模組

```bash
# 安裝 Node.js 依賴
cd services/scan/aiva_scan_node && npm install

# 安裝 TypeScript 編譯器 (全域)
npm install -g typescript
```

## 🔍 專案分析工具

### 1. 程式碼分析工具

#### 自動分析 (推薦)

```bash
# 使用 Python 分析工具
python tools/analyze_codebase.py

# 或使用 PowerShell 腳本
.\generate_project_report.ps1
```

#### 手動分析選項

```bash
# Python 程式碼品質檢查
mypy services/ tools/                    # 類型檢查
ruff check services/ tools/             # 程式碼檢查
black services/ tools/                  # 格式化

# 多語言程式碼檢查
golangci-lint run                       # Go 程式碼檢查
cargo clippy                            # Rust 程式碼檢查
npx eslint .                            # TypeScript 檢查
```

### 2. 樹狀架構圖生成

#### 自動生成

```bash
# 使用 Bash 腳本 (Linux/Mac)
./generate_project_report.sh

# 或使用 PowerShell 腳本 (Windows)
.\generate_project_report.ps1
```

#### 手動生成選項

```bash
# 生成不同格式的樹狀圖
tree -I '__pycache__|node_modules|.git' > _out/tree_ascii.txt
tree -I '__pycache__|node_modules|.git' -H . > _out/tree.html
```

### 3. Mermaid 圖表生成

#### 自動生成

```bash
# 使用專用工具
python tools/generate_mermaid_diagrams.py
```

#### 手動編輯 Mermaid 圖表

```bash
# 在 VS Code 中開啟並編輯 .mmd 檔案
code docs/diagrams/Module.mmd
```

## 🌐 多語言開發

### 語言版本要求

| 語言 | 版本 | 安裝檢查 |
|------|------|----------|
| Python | 3.11+ | `python --version` |
| Go | 1.21+ | `go version` |
| Rust | 1.70+ | `rustc --version` |
| Node.js | 18+ | `node --version` |
| TypeScript | 5.0+ | `tsc --version` |

### 跨語言通訊

AIVA 使用 **RabbitMQ** 作為跨語言通訊中樞：

```python
# Python 發送訊息
import pika
# ... 連接並發送

# Go 接收訊息
import "github.com/streadway/amqp"
// ... 連接並接收
```

### 資料庫整合

所有語言共用 **PostgreSQL** 資料庫：

```sql
-- 統一資料結構
CREATE TABLE scan_results (
    id SERIAL PRIMARY KEY,
    language VARCHAR(50),  -- 'python', 'go', 'rust', 'typescript'
    module_name VARCHAR(100),
    result JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🧪 測試與品質檢查

### 單元測試

```bash
# Python 測試
pytest -v

# Go 測試
go test ./...

# Rust 測試
cargo test

# TypeScript 測試
npm test
```

### 整合測試

```bash
# 啟動所有服務
./start_all_multilang.ps1

# 運行整合測試
pytest tests/integration/
```

### 程式碼品質檢查

```bash
# Python
mypy services/ tools/
ruff check services/ tools/
black --check services/ tools/

# Go
golangci-lint run

# Rust
cargo clippy

# TypeScript
npx eslint services/scan/aiva_scan_node/src/
```

## 🚀 部署與運行

### 開發模式

```bash
# 啟動所有服務
./start_all_multilang.ps1

# 或個別啟動
./start_dev.bat          # Python 核心服務
go run main.go          # Go 服務
cargo run               # Rust 服務
npm start               # Node.js 服務
```

### 生產部署

```bash
# 使用 Docker Compose
docker-compose -f docker/docker-compose.production.yml up -d

# 或使用 Kubernetes
kubectl apply -f k8s/
```

### 監控與日誌

```bash
# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f

# 健康檢查
curl http://localhost:8000/health
```

## 📊 專案統計

### 程式碼規模

- **總檔案數**: 235 個
- **程式碼檔案**: 221 個
- **總程式碼行數**: 33,318 行

### 語言分佈

- **Python**: 27,015 行 (81.1%)
- **Go**: 2,972 行 (8.9%)
- **Rust**: 1,552 行 (4.7%)
- **TypeScript**: 352 行 (1.1%)

### 品質指標

- **類型提示覆蓋率**: 74.8%
- **文檔字串覆蓋率**: 81.9%
- **平均複雜度**: 11.94
- **編碼相容性**: 100%

## 🐛 常見問題

### 依賴問題

```bash
# sklearn 缺失
pip install scikit-learn joblib

# Go 模組依賴
go mod download && go mod tidy

# Node.js 依賴
npm install
```

### 編譯錯誤

```bash
# Rust 編譯問題
rustup update
cargo clean && cargo build

# Go 編譯問題
go mod tidy
go build
```

### 網路問題

```bash
# RabbitMQ 連接問題
docker-compose restart rabbitmq

# 資料庫連接問題
docker-compose restart postgres
```

## 📚 相關文檔

- [專案架構分析](COMPREHENSIVE_PROJECT_ANALYSIS.md)
- [快速開始指南](QUICK_START.md)
- [API 文檔](docs/api/)
- [部署指南](docs/deployment/)

---

*最後更新: 2025-10-13*
