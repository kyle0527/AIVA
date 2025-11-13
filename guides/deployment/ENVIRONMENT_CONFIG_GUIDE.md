---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 環境變數配置指南 ✅ 11/10驗證 (10/31實測驗證)

## 📑 目錄

- [📁 配置文件說明](#-配置文件說明)
- [⚙️ 配置項目統一說明](#-配置項目統一說明)
- [🚀 快速設定](#-快速設定)
- [🔒 安全配置](#-安全配置)
- [🐛 故障排除](#-故障排除)
- [🔗 相關資源](#-相關資源)

## 配置文件說明

AIVA 專案有三個標準化的環境配置文件：

### 1. `.env` - 本地開發配置 (當前使用)
- **用途**: 在本地主機運行 AIVA 服務，連接到 Docker 容器
- **場景**: 開發調試時使用
- **特點**: 所有服務地址都是 `localhost`

### 2. `.env.docker` - Docker 容器配置
- **用途**: 在 Docker Compose 網絡內運行所有服務
- **場景**: 完整容器化部署
- **特點**: 使用容器服務名稱 (postgres, rabbitmq, redis, neo4j)

### 3. `.env.example` - 生產環境範本
- **用途**: 生產環境配置參考
- **場景**: 正式部署時使用
- **特點**: 包含安全配置和性能優化參數

## 配置項目統一說明

### 資料庫配置 (PostgreSQL + pgvector)
```bash
# 主要配置
AIVA_DATABASE_URL=postgresql://postgres:aiva123@localhost:5432/aiva_db
AIVA_DB_TYPE=postgres

# 詳細配置
AIVA_POSTGRES_HOST=localhost
AIVA_POSTGRES_PORT=5432
AIVA_POSTGRES_DB=aiva_db
AIVA_POSTGRES_USER=postgres
AIVA_POSTGRES_PASSWORD=aiva123

# 傳統配置支援（向後兼容）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=aiva_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=aiva123
```

### 消息隊列配置 (RabbitMQ)
```bash
# 主要配置
AIVA_RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# 詳細配置
AIVA_RABBITMQ_HOST=localhost
AIVA_RABBITMQ_PORT=5672
AIVA_RABBITMQ_USER=guest
AIVA_RABBITMQ_PASSWORD=guest
AIVA_RABBITMQ_VHOST=/

# 隊列配置
AIVA_MQ_EXCHANGE=aiva.topic
AIVA_MQ_DLX=aiva.dlx
```

### Redis 配置
```bash
AIVA_REDIS_URL=redis://localhost:6379/0
AIVA_REDIS_HOST=localhost
AIVA_REDIS_PORT=6379
```

### Neo4j 配置
```bash
AIVA_NEO4J_URL=bolt://neo4j:aiva1234@localhost:7687
AIVA_NEO4J_HOST=localhost
AIVA_NEO4J_PORT=7687
AIVA_NEO4J_USER=neo4j
AIVA_NEO4J_PASSWORD=aiva1234
```

## 使用方式

### 本地開發 (推薦)
```bash
# 使用當前的 .env 配置
cp .env .env.backup  # 備份
# .env 已經是本地配置，直接使用
```

### Docker 部署
```bash
# 切換到 Docker 配置
cp .env.docker .env
docker-compose up -d
```

### 生產環境
```bash
# 基於範本創建生產配置
cp .env.example .env.production
# 修改 .env.production 中的密碼和地址
```

## 測試配置
確保 Docker 服務正在運行：
```bash
docker-compose ps
```

測試連接：
```bash
python -c "
import os
from services.integration.aiva_integration.reception.unified_storage_adapter import UnifiedStorageAdapter
adapter = UnifiedStorageAdapter()
print('配置成功！')
"
```

## 配置優先級
1. 直接傳入的參數 (最高優先級)
2. AIVA_* 環境變數
3. 傳統環境變數 (向後兼容)
4. 預設值 (最低優先級)

## 常見問題

### Q: 為什麼有多個環境變數名稱？
A: 為了支援不同的部署場景和向後兼容性。

### Q: Docker 容器無法連接？
A: 檢查 Docker Compose 服務是否健康運行：
```bash
docker-compose ps
docker-compose logs postgres
```

### Q: 本地開發連接失敗？
A: 確保使用 localhost 地址且 Docker 服務端口已映射。