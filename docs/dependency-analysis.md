# AIVA 依賴分析報告

**分析日期**: 2026-03-16
**依賴總數**: 57 個 (requirements.txt)
**實際有使用**: 39 個
**完全未使用**: 8 個
**可合併/替代**: 4 組
**缺少宣告**: 1 個

---

## 一、完全未使用的依賴（建議移除或確認未來計畫）

| # | 套件 | 大小影響 | 說明 |
|---|------|---------|------|
| 1 | `openai>=1.0.0` | 中 | 零 import。若未來不接 OpenAI API 可移除 |
| 2 | `nltk>=3.8.0` | 大（含語料庫） | 零 import。NLP 功能未使用此庫 |
| 3 | `spacy>=3.6.0` | 很大（~500MB+模型） | 零 import。最肥的未使用依賴 |
| 4 | `sentence-transformers>=2.2.0` | 大 | 零 import。embedding 已用 transformers 自行實作 |
| 5 | `gymnasium>=0.29.0` | 中 | 零 import。RL 已用 PyTorch 原生實作 |
| 6 | `passlib[bcrypt]>=1.7.4` | 小 | 零 import。密碼雜湊未使用此庫 |
| 7 | `python-dotenv>=1.0.1` | 小 | 零 import。設定由 pydantic-settings 處理 |
| 8 | `orjson>=3.10.0` | 小 | 零 import。JSON 處理用標準庫/pydantic |

> **移除這 8 個可減少約 1-2GB 安裝空間**（主要來自 spacy + nltk + sentence-transformers）

---

## 二、僅在封存程式碼中使用（可考慮移至 optional）

| # | 套件 | 使用位置 | 說明 |
|---|------|---------|------|
| 1 | `alembic>=1.13.2` | `_archive/` 目錄 | 僅在封存的 migration 中使用，services/ 無引用 |
| 2 | `redis>=5.0.0` | 零 import | config 中有設定但程式碼未直接 import |
| 3 | `psycopg2-binary>=2.9.0` | 零 import | 已被 asyncpg 取代，services/ 無引用 |

---

## 三、功能重疊 — 可合併的依賴組

### 3.1 HTTP Client 重疊：`httpx` + `aiohttp` + `requests`（三擇二）

| 套件 | 引用檔案數 | 用途 |
|------|-----------|------|
| `httpx` | **22 檔** | 主要 async HTTP client，安全測試引擎 |
| `aiohttp` | **13 檔** | async HTTP，XSS/SQLi/Web 掃描 |
| `requests` | **14 檔** | sync HTTP，dashboard/偵察/爬蟲 |

**分析**：
- `httpx` 同時支援 sync 和 async，理論上可以取代 `requests`
- `aiohttp` 用於 WebSocket 和某些特定場景，與 `httpx` 有部分重疊
- **建議**：保留 `httpx`（主力）+ `aiohttp`（WebSocket 場景），將 `requests` 的 14 處逐步遷移到 `httpx`

### 3.2 JWT 重疊：`PyJWT` vs `python-jose`

| 套件 | 引用檔案數 | 用途 |
|------|-----------|------|
| `PyJWT` | **1 檔** (`security.py` 用 `import jwt`) | JWT 編解碼 |
| `python-jose[cryptography]` | **0 檔** | 零使用 |

**建議**：移除 `python-jose`，僅保留 `PyJWT`

---

## 四、每個依賴的詳細使用狀態

### Core Framework（6 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `fastapi` | 3 檔 | ✅ 必要 | API 框架核心 |
| `uvicorn` | 2 檔 | ✅ 必要 | ASGI 伺服器 |
| `pydantic` | **71 檔** | ✅ 核心 | 全專案最多引用的依賴 |
| `pydantic-settings` | 1 檔 | ✅ 必要 | 設定管理 |
| `python-dotenv` | 0 檔 | ❌ 未使用 | pydantic-settings 已內建處理 |
| `orjson` | 0 檔 | ❌ 未使用 | 未引用 |

### Async & Message Queue（4 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `aio-pika` | 2 檔 | ✅ 必要 | RabbitMQ 非同步客戶端 |
| `aiohttp` | 13 檔 | ✅ 使用中 | async HTTP + WebSocket |
| `aiofiles` | 4 檔 | ✅ 使用中 | 非同步檔案 I/O |
| `httpx` | 22 檔 | ✅ 核心 | 主力 HTTP client |

### Database & Storage（5 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `sqlalchemy` | 3 檔 | ✅ 必要 | ORM 核心 |
| `asyncpg` | 2 檔 | ✅ 必要 | PostgreSQL async driver |
| `psycopg2-binary` | 0 檔 | ⚠️ 未使用 | 被 asyncpg 取代 |
| `alembic` | 0 檔 (僅 _archive/) | ⚠️ 封存 | 遷移工具，僅在封存碼中 |
| `redis` | 0 檔 | ⚠️ 未使用 | 有設定但無 import |

### gRPC（3 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `grpcio` | 4 檔 | ✅ 必要 | gRPC runtime |
| `grpcio-tools` | 0 檔 | ⚠️ 建置工具 | 用於編譯 .proto，非 runtime |
| `protobuf` | 4 檔 | ✅ 必要 | Protocol Buffers |

### AI / Neural Network（3 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `torch` | 8 檔 | ✅ 核心 | 5M 神經網路 + RL |
| `numpy` | 14 檔 | ✅ 核心 | 數值計算 |
| `scikit-learn` | 1 檔 | ✅ 使用中 | model_trainer.py 中的 ML 模型 |

### NLP & Embeddings（5 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `transformers` | 2 檔 | ✅ 必要 | embedding 模型載入 |
| `sentence-transformers` | 0 檔 | ❌ 未使用 | 已自行實作 |
| `openai` | 0 檔 | ❌ 未使用 | |
| `nltk` | 0 檔 | ❌ 未使用 | |
| `spacy` | 0 檔 | ❌ 未使用 | |

### Reinforcement Learning（1 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `gymnasium` | 0 檔 | ❌ 未使用 | RL 用 PyTorch 原生實作 |

### Web Scraping（3 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `beautifulsoup4` | 2 檔 | ✅ 使用中 | HTML 解析 |
| `lxml` | 1 檔 | ✅ 使用中 | XXE 偵測 |
| `dnspython` | 3 檔 | ✅ 使用中 | DNS 解析/子網域掃描 |

### Security（4 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `PyJWT` | 1 檔 | ✅ 使用中 | JWT 處理 |
| `python-jose` | 0 檔 | ❌ 與 PyJWT 重疊 | 可移除 |
| `passlib` | 0 檔 | ❌ 未使用 | |
| `cryptography` | 2 檔 | ✅ 使用中 | 加密操作 |

### Graph & Visualization（3 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `networkx` | 1 檔 | ✅ 使用中 | skill graph 路徑分析 |
| `plotly` | 1 檔 | ✅ 使用中 | 權限矩陣視覺化 |
| `pandas` | 4 檔 | ✅ 使用中 | dashboard + 資料處理 |

### CLI & UX（2 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `click` | 3 檔 | ✅ 使用中 | CLI 框架 |
| `rich` | **21 檔** | ✅ 核心 | terminal UI |

### Monitoring & Utilities（4 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `structlog` | 6 檔 | ✅ 使用中 | 結構化日誌 |
| `psutil` | 5 檔 | ✅ 使用中 | 系統監控 |
| `tenacity` | 1 檔 | ✅ 使用中 | 重試機制 |
| `jinja2` | 1 檔 | ✅ 使用中 | 報告模板 |

### Sync HTTP（1 個）

| 套件 | 引用數 | 狀態 | 說明 |
|------|--------|------|------|
| `requests` | 14 檔 | ⚠️ 可被 httpx 取代 | sync HTTP |

---

## 五、缺少宣告的依賴

| 套件 | 引用數 | 說明 |
|------|--------|------|
| `streamlit` | **5 檔** | dashboard 使用但未列在 requirements.txt |

---

## 六、總結建議

### 可立即行動
1. **移除 8 個未使用依賴**：`openai`、`nltk`、`spacy`、`sentence-transformers`、`gymnasium`、`passlib`、`python-dotenv`、`orjson`
2. **移除 1 個重疊依賴**：`python-jose`（PyJWT 已覆蓋）
3. **補上缺少的依賴**：`streamlit`
4. **移至 optional**：`grpcio-tools`（建置工具）、`alembic`（封存）

### 中期規劃
5. **統一 HTTP client**：將 `requests` 的 14 處遷移至 `httpx`，減少一個依賴
6. **確認 Redis 狀態**：有設定但零 import，確認是否透過其他方式使用或可移除
7. **確認 psycopg2 狀態**：已被 asyncpg 取代，確認無其他需求後移除

### 依賴精簡後
- **移除前**: 57 個依賴
- **移除後**: ~45 個依賴（含新增 streamlit）
- **節省空間**: ~1-2GB（主要來自 spacy/nltk/sentence-transformers）
