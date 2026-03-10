# Phase 2: 部署基礎建設修復 — Docker + 資料庫

> 優先級: P1  
> 目標: `docker build` 成功 + 資料庫能自動初始化  
> 前置條件: Phase 1 完成  
> 驗證方式: `docker build` → `docker-compose up` → `/health` 返回 200

---

## 修復 2.1: Dockerfile 修正

### 問題

`docker/Dockerfile.complete` 是主要的生產環境 Dockerfile，但引用了 **6 個不存在的目錄**：

| COPY 來源 | 狀態 | 說明 |
|-----------|------|------|
| `api/` | ❌ 不存在 | 可能曾是獨立的 API 層，目前已整合到 services/ |
| `examples/` | ❌ 不存在 | 範例程式碼，曾存在但已歸檔 |
| `models/` | ❌ 不存在 | 模型檔案存在於 `ai_models/` |
| `utilities/` | ❌ 不存在 | 工具程式，目前在 `_dev_tools/` |
| `testing/` | ❌ 不存在 | 測試程式在 `tests/` |
| `web/` | ❌ 不存在 | 前端/Web 介面，目前不存在 |

另外還有：
- `aiva_complete_launcher.py` — ENTRYPOINT CMD 引用但不存在
- `docker/core/entrypoint.sh` — ENTRYPOINT 腳本不存在
- 使用 `python:3.11-slim` 但專案要求 `python>=3.13`

### 修復動作

#### A. 清理不存在的 COPY 指令

```dockerfile
# 改前 (Dockerfile.complete):
COPY api/ api/
COPY examples/ examples/
COPY utilities/ utilities/
COPY testing/ testing/
COPY web/ web/
COPY models/ models/

# 改後:
# (刪除以上 6 行，替換為實際存在的目錄)
COPY ai_models/ ai_models/
COPY tests/ tests/
COPY _dev_tools/ _dev_tools/
```

#### B. 修正 Python 版本

```dockerfile
# 改前:
FROM python:3.11-slim as builder
FROM python:3.11-slim

# 改後:
FROM python:3.13-slim as builder
FROM python:3.13-slim
```

#### C. 建立入口點腳本

建立 `docker/core/entrypoint.sh`：

```bash
#!/bin/bash
set -e

echo "🚀 AIVA Core Engine starting..."

# 等待 PostgreSQL（如果配置了的話）
if [ -n "$DATABASE_URL" ]; then
    echo "⏳ Waiting for database..."
    python -c "
import time, socket
host = '${DB_HOST:-localhost}'
port = int('${DB_PORT:-5432}')
for i in range(30):
    try:
        s = socket.socket()
        s.connect((host, port))
        s.close()
        print('✅ Database is ready')
        break
    except:
        time.sleep(1)
else:
    print('⚠️ Database not available, continuing anyway')
"
fi

# 啟動 AIVA
exec python -m uvicorn services.core.aiva_core.service_backbone.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers ${WORKERS:-1} \
    --log-level ${LOG_LEVEL:-info}
```

#### D. 替代方案 — 使用 Dockerfile.core.minimal

`docker/core/Dockerfile.core.minimal` 是目前**唯一可用**的 Dockerfile。
它只做最基本的事情，可作為快速啟動方案：

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/ services/
COPY config/ config/
CMD ["python", "-m", "uvicorn", "services.core.aiva_core.service_backbone.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 建議策略

1. **短期**: 修復 `Dockerfile.core.minimal`，保證能 build + run
2. **中期**: 大幅簡化 `Dockerfile.complete`，移除不存在的目錄
3. **長期**: 當 Go/Rust 工具鏈實際可用時，再恢復多階段建置

### 驗證指令

```powershell
# 建置最小映像
docker build -f docker/core/Dockerfile.core.minimal -t aiva-core:test .

# 執行
docker run -p 8000:8000 aiva-core:test

# 測試
curl http://localhost:8000/health
```

---

## 修復 2.2: Docker Compose 修正

### 問題

- `docker/compose/docker-compose.yml` — 基礎版本（839 bytes），結構簡單
- `docker/compose/docker-compose.production.yml` — 生產版本（5,463 bytes），服務較多
- `docker/docker-compose.complete.yml` — 完整版本（5,878 bytes），包含所有服務

### 修復動作

確認 compose 檔案中引用的映像和服務都是可建置的。
至少需要下列服務能啟動：

```yaml
# docker-compose.minimal.yml (需新建)
version: '3.8'

services:
  aiva-core:
    build:
      context: ../..
      dockerfile: docker/core/Dockerfile.core.minimal
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///data/aiva.db
      - LOG_LEVEL=info
    volumes:
      - aiva-data:/app/data

  # PostgreSQL（可選）
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: aiva
      POSTGRES_USER: aiva
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-aiva_dev}
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ../infrastructure/initdb:/docker-entrypoint-initdb.d

volumes:
  aiva-data:
  postgres-data:
```

---

## 修復 2.3: 資料庫初始化修復

### 問題

資料庫初始化有三條路徑，目前都不完整：

| 路徑 | 狀態 | 問題 |
|------|------|------|
| Docker initdb/ SQL | ✅ 存在 | 只有 Docker 環境能用（PostgreSQL 容器啟動時自動執行） |
| Alembic 遷移 | ❌ 已刪除 | `alembic.ini` 存在但 `alembic/` 目錄已刪 |
| App 自動建表 | ❌ 不存在 | app.py 中沒有任何 DB init 邏輯 |

### Schema 檔案

```
docker/infrastructure/initdb/
├── 001_schema.sql          ← 基礎 4 表 (scans, reports, task_executions, findings)
└── 002_enhanced_schema.sql ← 增強 2 表 (assets, vulnerabilities) + triggers
```

### 修復動作

#### 選項 A（推薦）: 在 app.py startup 中加入 SQLite 自動建表

對於本地開發和快速啟動，支援 SQLite + 自動建表：

```python
# 在 app.py startup() 中添加
async def _init_database():
    """初始化資料庫 — 支援 SQLite 和 PostgreSQL"""
    import os
    db_url = os.getenv("DATABASE_URL", "sqlite:///data/database/aiva.db")
    
    if db_url.startswith("sqlite"):
        import sqlite3
        db_path = db_url.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        # 從 initdb SQL 建表（轉換 PostgreSQL 語法為 SQLite）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                target TEXT NOT NULL,
                scan_type TEXT DEFAULT 'quick',
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                scan_id TEXT REFERENCES scans(id),
                vuln_type TEXT,
                severity TEXT,
                url TEXT,
                evidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"✅ SQLite database initialized: {db_path}")
    else:
        # PostgreSQL — 假設 initdb/ SQL 已執行（Docker 環境）
        logger.info(f"📦 Using PostgreSQL: {db_url[:30]}...")
```

#### 選項 B: 恢復 Alembic

重建已刪除的檔案：
- `services/integration/alembic/env.py`
- `services/integration/alembic/versions/001_initial_schema.py`

在 startup 中呼叫 `alembic upgrade head`。

### 建議

短期用**選項 A**（SQLite 自動建表），中期補回 Alembic。

---

## 修復 2.4: .env 和環境變數配置

### 現狀

- `.env` 檔案存在 ✅
- `.env.docker` 檔案存在 ✅
- `.env.example` 檔案存在 ✅

### 確認項目

確認以下環境變數在 `.env` 中都有設定：

```bash
# 核心必要
DATABASE_URL=sqlite:///data/database/aiva.db
LOG_LEVEL=info
PORT=8000

# 可選（RabbitMQ）
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# 可選（外部 API）
OPENAI_API_KEY=
SHODAN_API_KEY=
```

### 驗證指令

```powershell
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
required = ['DATABASE_URL', 'LOG_LEVEL']
optional = ['RABBITMQ_URL', 'OPENAI_API_KEY', 'SHODAN_API_KEY']
for k in required:
    v = os.getenv(k, '(未設定)')
    print(f'  [必要] {k} = {v[:30]}...' if len(v) > 30 else f'  [必要] {k} = {v}')
for k in optional:
    v = os.getenv(k, '(未設定)')
    print(f'  [可選] {k} = {v[:10]}...' if len(v) > 10 else f'  [可選] {k} = {v}')
"
```

---

## 修復 2.5: main.py 安全閘道確認

### 現狀

`services/core/main.py` 是 port 9000 的安全閘道，負責：
1. 接收外部請求
2. 檢查惡意 payload
3. 轉發到 app.py (port 8000)

### 確認項目

1. main.py 能否獨立啟動？
2. 是否正確轉發到 port 8000？
3. 安全檢查邏輯是否完整？

### 驗證指令

```powershell
# 測試 main.py 匯入
python -c "import sys; sys.path.insert(0,'.'); import services.core.main; print('OK')"

# 完整啟動測試（需要先啟動 app.py 在 port 8000）
# 終端 1:
python -m uvicorn services.core.aiva_core.service_backbone.api.app:app --port 8000
# 終端 2:
python services/core/main.py  # port 9000
# 終端 3:
curl http://localhost:9000/health
```

---

## 完成清單

```
2.1 [ ] 清理 Dockerfile.complete 中不存在的 COPY
2.1 [ ] 修正 Python 版本為 3.13
2.1 [ ] 建立 entrypoint.sh
2.1 [ ] 建置 Dockerfile.core.minimal 測試
2.2 [ ] 建立最小 docker-compose.minimal.yml
2.3 [ ] 在 app.py 加入 SQLite 自動建表
2.4 [ ] 確認 .env 配置完整
2.5 [ ] 驗證 main.py → app.py 轉發鏈
2.X [ ] docker build 成功
2.X [ ] docker-compose up 成功
2.X [ ] /health 返回 200
```
