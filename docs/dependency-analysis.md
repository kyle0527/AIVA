# AIVA 依賴分析報告

**分析日期**: 2026-03-16
**清理日期**: 2026-03-16
**狀態**: ✅ 已完成清理

---

## 變更摘要

| 項目 | 變更前 | 變更後 |
|------|--------|--------|
| 依賴總數 (requirements.txt) | 57 個 | 46 個 |
| 未使用依賴 | 8 個 | 0 個（已移除） |
| 重疊依賴 | 2 組 | 1 組（`python-jose` 已移除，HTTP client 待中期整合）|
| 缺少宣告 | 2 個 | 0 個（已補上 `streamlit`、`PyYAML`）|

---

## 已執行的變更

### ✅ 移除 9 個未使用/重疊依賴

| # | 套件 | 移除原因 |
|---|------|---------|
| 1 | `openai>=1.0.0` | 零 import |
| 2 | `nltk>=3.8.0` | 零 import |
| 3 | `spacy>=3.6.0` | 零 import，最大的幽靈依賴 (~500MB+) |
| 4 | `sentence-transformers>=2.2.0` | 零 import，embedding 已用 transformers 自行實作 |
| 5 | `gymnasium>=0.29.0` | 零 import，RL 已用 PyTorch 原生實作 |
| 6 | `passlib[bcrypt]>=1.7.4` | 零 import |
| 7 | `python-dotenv>=1.0.1` | 零 import，設定由 pydantic-settings 處理 |
| 8 | `orjson>=3.10.0` | 零 import |
| 9 | `python-jose[cryptography]>=3.3.0` | 零 import，與 PyJWT 功能重疊 |

### ✅ 新增 2 個缺少宣告的依賴

| # | 套件 | 引用數 | 說明 |
|---|------|--------|------|
| 1 | `streamlit>=1.28.0` | 5 檔 | dashboard UI 框架 |
| 2 | `PyYAML>=6.0` | 5 檔 | YAML 設定檔解析 |

### ✅ 移至 optional/dev 的依賴

| # | 套件 | 移動到 | 原因 |
|---|------|--------|------|
| 1 | `grpcio-tools>=1.60.0` | dev | 建置工具，非 runtime |
| 2 | `alembic>=1.13.2` | optional (註解) | 僅封存碼使用 |
| 3 | `psycopg2-binary>=2.9.0` | optional (註解) | 已被 asyncpg 取代 |

---

## 保留但待觀察的依賴

| 套件 | 引用數 | 狀態 | 備註 |
|------|--------|------|------|
| `redis` | 0 檔 | ⚠️ 保留 | config 有設定，可能透過其他方式使用（如 docker-compose） |
| `requests` | 14 檔 | ⚠️ 保留 | 可被 httpx 取代，但需逐步遷移 14 個檔案 |

---

## 中期規劃（未執行）

1. **統一 HTTP client**：將 `requests` 的 14 處遷移至 `httpx`，減少一個依賴
2. **確認 Redis 狀態**：有設定但零 import，確認是否透過其他方式使用或可移除

---

## 現有依賴使用狀態（清理後）

### Core Framework（4 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `fastapi` | 3 檔 | API 框架核心 |
| `uvicorn` | 2 檔 | ASGI 伺服器 |
| `pydantic` | **71 檔** | 全專案最多引用的依賴 |
| `pydantic-settings` | 1 檔 | 設定管理 |

### Async & Message Queue（4 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `aio-pika` | 2 檔 | RabbitMQ 非同步客戶端 |
| `aiohttp` | 13 檔 | async HTTP + WebSocket |
| `aiofiles` | 4 檔 | 非同步檔案 I/O |
| `httpx` | 22 檔 | 主力 HTTP client |

### Database & Storage（3 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `sqlalchemy` | 3 檔 | ORM 核心 |
| `asyncpg` | 2 檔 | PostgreSQL async driver |
| `redis` | 0 檔 | 待確認（保留） |

### gRPC（2 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `grpcio` | 4 檔 | gRPC runtime |
| `protobuf` | 4 檔 | Protocol Buffers |

### AI / Neural Network（3 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `torch` | 8 檔 | 5M 神經網路 + RL |
| `numpy` | 14 檔 | 數值計算 |
| `scikit-learn` | 1 檔 | model_trainer.py |

### NLP & Embeddings（1 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `transformers` | 2 檔 | embedding 模型載入 |

### Web Scraping & Scanning（4 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `requests` | 14 檔 | sync HTTP（待遷移至 httpx） |
| `beautifulsoup4` | 2 檔 | HTML 解析 |
| `lxml` | 1 檔 | XXE 偵測 |
| `dnspython` | 3 檔 | DNS 解析/子網域掃描 |

### Security（2 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `PyJWT` | 1 檔 | JWT 處理 |
| `cryptography` | 2 檔 | 加密操作 |

### Graph & Visualization（3 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `networkx` | 1 檔 | skill graph 路徑分析 |
| `plotly` | 1 檔 | 權限矩陣視覺化 |
| `pandas` | 4 檔 | dashboard + 資料處理 |

### CLI & UX（2 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `click` | 3 檔 | CLI 框架 |
| `rich` | **21 檔** | terminal UI |

### Configuration & Serialization（2 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `PyYAML` | 5 檔 | YAML 設定解析 (新增) |
| `jinja2` | 1 檔 | 報告模板 |

### Monitoring & Utilities（3 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `structlog` | 6 檔 | 結構化日誌 |
| `psutil` | 5 檔 | 系統監控 |
| `tenacity` | 1 檔 | 重試機制 |

### Dashboard（1 個）

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `streamlit` | 5 檔 | Dashboard UI (新增) |
