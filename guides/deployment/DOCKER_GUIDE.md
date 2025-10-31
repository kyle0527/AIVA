---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Docker 基礎設施指南
---

# AIVA Docker 基礎設施使用指南

## 📑 目錄

- [📊 當前狀態](#-當前狀態)
- [🏗️ 架構概覽](#-架構概覽)
- [📁 目錄結構說明](#-目錄結構說明)
- [🚀 快速開始](#-快速開始)
- [🔧 服務管理](#-服務管理)
- [📊 監控與日誌](#-監控與日誌)
- [🐛 故障排除](#-故障排除)
- [⚡ 性能優化](#-性能優化)
- [🔒 安全設定](#-安全設定)
- [🔗 相關資源](#-相關資源)

本指南基於 Docker 基礎設施分析報告和 aiva_common 標準編寫。

## 📊 當前狀態

- **Docker 文件總數**: 18 → 重組後分類管理
- **複雜度評分**: 35/100 → 預期降低至 25/100
- **增長預測**: 高 → 結構化管理後可控
- **重組狀態**: ✅ 已完成

## 🏗️ 架構概覽

AIVA 採用微服務架構，基於以下容器化策略：

### 核心服務 (永遠運行)
- **aiva-core**: 核心 AI 服務，包含對話助理、經驗管理器
- **基礎設施**: PostgreSQL, Redis, RabbitMQ, Neo4j

### 功能組件 (按需啟動)
- **22個功能組件**: 各種掃描器、測試工具、分析器
- **動態調度**: 根據任務需求啟動相應組件

## 📁 目錄結構說明

```
docker/
├── core/                    # 核心服務容器配置
│   ├── Dockerfile.core      # 主要核心服務
│   ├── Dockerfile.core.minimal  # 最小化版本
│   └── Dockerfile.patch     # 增量更新版本
│
├── components/              # 功能組件容器配置
│   └── Dockerfile.component # 通用組件容器
│
├── infrastructure/          # 基礎設施服務配置
│   ├── Dockerfile.integration  # 整合服務
│   ├── entrypoint.integration.sh  # 啟動腳本
│   └── initdb/             # 數據庫初始化
│       ├── 001_schema.sql
│       └── 002_enhanced_schema.sql
│
├── compose/                 # Docker Compose 配置
│   ├── docker-compose.yml  # 主要配置
│   └── docker-compose.production.yml  # 生產環境配置
│
├── k8s/                     # Kubernetes 部署配置
│   ├── 00-namespace.yaml   # 命名空間
│   ├── 01-configmap.yaml   # 配置管理
│   ├── 02-storage.yaml     # 存儲配置
│   ├── 10-core-deployment.yaml     # 核心服務部署
│   └── 20-components-jobs.yaml     # 組件任務配置
│
└── helm/                    # Helm Charts
    └── aiva/
        ├── Chart.yaml
        └── values.yaml
```

## 🚀 使用方式

### 開發環境

```bash
# 啟動完整開發環境
docker compose -f docker/compose/docker-compose.yml up -d

# 只啟動基礎設施
docker compose -f docker/compose/docker-compose.yml up -d postgres redis rabbitmq neo4j

# 啟動核心服務
docker compose -f docker/compose/docker-compose.yml up -d aiva-core

# 查看服務狀態
docker compose -f docker/compose/docker-compose.yml ps
```

### 生產環境

```bash
# 使用生產配置
docker compose -f docker/compose/docker-compose.production.yml up -d

# Kubernetes 部署
kubectl apply -f docker/k8s/

# Helm 部署
helm install aiva docker/helm/aiva/
```

### 單獨構建映像

```bash
# 構建核心服務
docker build -f docker/core/Dockerfile.core -t aiva-core:latest .

# 構建功能組件
docker build -f docker/components/Dockerfile.component -t aiva-component:latest .

# 構建最小化版本
docker build -f docker/core/Dockerfile.core.minimal -t aiva-core:minimal .

# 構建整合服務
docker build -f docker/infrastructure/Dockerfile.integration -t aiva-integration:latest .
```

## 🔧 配置說明

### 環境變量

核心服務支援以下環境變量配置：

```bash
# 模式配置
AIVA_MODE=production
AIVA_ENVIRONMENT=docker

# 數據庫配置
AIVA_POSTGRES_HOST=postgres
AIVA_POSTGRES_PORT=5432
AIVA_POSTGRES_USER=postgres
AIVA_POSTGRES_PASSWORD=aiva123
AIVA_POSTGRES_DB=aiva_db

# Redis 配置
AIVA_REDIS_HOST=redis
AIVA_REDIS_PORT=6379

# RabbitMQ 配置
AIVA_RABBITMQ_HOST=rabbitmq
AIVA_RABBITMQ_PORT=5672
AIVA_RABBITMQ_USER=guest
AIVA_RABBITMQ_PASSWORD=guest

# Neo4j 配置
AIVA_NEO4J_HOST=neo4j
AIVA_NEO4J_PORT=7687
AIVA_NEO4J_USER=neo4j
AIVA_NEO4J_PASSWORD=aiva123
```

### 端口映射

| 服務 | 內部端口 | 外部端口 | 說明 |
|------|---------|---------|------|
| AIVA Core | 8000 | 8000 | 主 API |
| AIVA Core | 8001 | 8001 | 管理 API |
| AIVA Core | 8002 | 8002 | WebSocket |
| PostgreSQL | 5432 | 5432 | 數據庫 |
| Redis | 6379 | 6379 | 緩存 |
| RabbitMQ | 5672 | 5672 | 消息隊列 |
| RabbitMQ UI | 15672 | 15672 | 管理界面 |
| Neo4j | 7687 | 7687 | 圖數據庫 |
| Neo4j UI | 7474 | 7474 | 管理界面 |

## 🔍 故障排除

### 常見問題

1. **容器啟動失敗**
   ```bash
   # 檢查日誌
   docker compose -f docker/compose/docker-compose.yml logs aiva-core
   
   # 檢查資源使用
   docker stats
   ```

2. **服務連接問題**
   ```bash
   # 檢查網絡連通性
   docker compose -f docker/compose/docker-compose.yml exec aiva-core ping postgres
   
   # 檢查端口
   docker compose -f docker/compose/docker-compose.yml port aiva-core 8000
   ```

3. **數據持久化問題**
   ```bash
   # 檢查數據卷
   docker volume ls
   docker volume inspect aiva-git_postgres-data
   ```

### 性能優化

1. **資源限制**
   - 核心服務: 2GB RAM, 1 CPU
   - 功能組件: 1GB RAM, 0.5 CPU
   - 基礎設施: 根據負載調整

2. **網絡優化**
   - 使用內部網絡通信
   - 啟用連接池
   - 配置健康檢查

## 📊 基於 aiva_common 的整合

本 Docker 基礎設施與 aiva_common 深度整合：

### 服務發現

aiva_common 提供統一的服務發現機制：

```python
# services/aiva_common/continuous_components_sot.json
{
  "integration_points": {
    "docker_integration": {
      "enabled": true,
      "docker_socket": "/var/run/docker.sock",
      "container_health_check": true,
      "auto_container_restart": true,
      "config_directory": "docker/",
      "compose_files": {
        "development": "docker/compose/docker-compose.yml",
        "production": "docker/compose/docker-compose.production.yml"
      },
      "k8s_directory": "docker/k8s/",
      "helm_chart": "docker/helm/aiva/"
    }
  }
}
```

### 枚舉支援

容器相關的標準枚舉：

```python
from aiva_common.enums.assets import AssetType
from aiva_common.enums.security import VulnerabilityType

# 容器資產類型
AssetType.CONTAINER  # "container"

# 容器相關漏洞
VulnerabilityType.CONTAINER_ESCAPE  # "container_escape"
```

### 消息隊列整合

統一的消息隊列配置支援容器化部署：

```python
from aiva_common.mq import MQClient
from aiva_common import Topic, ModuleName

# 容器環境中的 MQ 連接
mq = MQClient(
    host=os.getenv('AIVA_RABBITMQ_HOST', 'rabbitmq'),
    port=int(os.getenv('AIVA_RABBITMQ_PORT', '5672'))
)
```

## 📚 相關文檔

- [AIVA 架構文檔](../reports/architecture/ARCHITECTURE_SUMMARY.md)
- [Docker 基礎設施分析報告](../reports/architecture/docker_infrastructure_analysis_20251030_113318.md)
- [Docker 基礎設施更新報告](../reports/architecture/docker_infrastructure_update_report_20251030_114200.md)
- [aiva_common 開發指南](../services/aiva_common/README.md)
- [aiva_common Docker 整合](../services/aiva_common/README.md#docker-整合)

## 🔄 版本記錄

- **2025-10-30**: 
  - 初始版本，基於基礎設施分析報告創建
  - 完成文件重組，從 18 個散布文件整理為結構化目錄
  - 整合 aiva_common 標準和配置
  - 添加基礎設施服務支援 (initdb, integration)

## 🚀 未來計劃

1. **容器優化**
   - 多階段構建減少映像大小
   - 安全掃描整合
   - 資源使用監控

2. **部署模式擴展**
   - 支援 Docker Swarm
   - 混合雲部署
   - 邊緣計算支援

3. **自動化改進**
   - CI/CD 管道整合
   - 自動擴展配置
   - 災難恢復機制

---

## 📞 支援

如有問題或建議，請參考：
- [問題追蹤](https://github.com/kyle0527/aiva/issues)
- [討論區](https://github.com/kyle0527/aiva/discussions)
- 內部開發團隊: AIVA DevOps

*最後更新: 2025-10-30T11:41:00+08:00*