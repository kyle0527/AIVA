---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 微服務部署指南

## 📑 目錄

- [🏗️ 架構概述](#架構概述)
  - [Layer 0: 基礎設施層（永遠運行）](#layer-0-基礎設施層永遠運行)
  - [Layer 1: 核心 AI 服務（永遠運行）](#layer-1-核心-ai-服務永遠運行)
  - [Layer 2: 功能組件（按需啟動，最多 22 個）](#layer-2-功能組件按需啟動最多-22-個)
- [🐳 Docker Compose 部署（本地開發）](#docker-compose-部署本地開發)
  - [1. 前置要求](#1-前置要求)
  - [2. 快速啟動](#2-快速啟動)
    - [只啟動核心服務和基礎設施](#只啟動核心服務和基礎設施)
    - [啟動特定組件](#啟動特定組件)
  - [3. 查看服務狀態](#3-查看服務狀態)
  - [4. 訪問服務](#4-訪問服務)
  - [5. 停止服務](#5-停止服務)
- [☸️ Kubernetes 部署（生產環境）](#kubernetes-部署生產環境)
  - [1. 前置要求](#1-前置要求)
  - [2. 使用原生 Kubernetes Manifests](#2-使用原生-kubernetes-manifests)
    - [部署核心服務](#部署核心服務)
    - [啟動功能組件（按需）](#啟動功能組件按需)
    - [查看服務狀態](#查看服務狀態)
    - [訪問服務](#訪問服務)
  - [3. 使用 Helm Chart（推薦）](#3-使用-helm-chart推薦)
    - [安裝 AIVA](#安裝-aiva)
    - [升級 AIVA](#升級-aiva)
    - [查看狀態](#查看狀態)
    - [卸載 AIVA](#卸載-aiva)
- [🔧 配置說明](#配置說明)
  - [環境變數配置](#環境變數配置)
    - [核心服務環境變數](#核心服務環境變數)
    - [組件連接配置](#組件連接配置)
  - [資源配置建議](#資源配置建議)
- [📊 監控和日誌](#監控和日誌)
  - [Docker Compose](#docker-compose)
  - [Kubernetes](#kubernetes)
- [🚀 動態組件管理](#動態組件管理)
  - [使用 Docker Compose](#使用-docker-compose)
  - [使用 Kubernetes](#使用-kubernetes)
- [🛠️ 故障排查](#故障排查)
  - [核心服務無法啟動](#核心服務無法啟動)
  - [組件無法連接核心服務](#組件無法連接核心服務)
  - [數據庫連接問題](#數據庫連接問題)
- [📚 更多資源](#更多資源)
- [💡 最佳實踐](#最佳實踐)

---

## 🏗️ 架構概述

AIVA 採用分層微服務架構：

### Layer 0: 基礎設施層（永遠運行）
- **PostgreSQL**: 主數據庫
- **Redis**: 緩存和會話存儲
- **RabbitMQ**: 消息隊列
- **Neo4j**: 圖數據庫

### Layer 1: 核心 AI 服務（永遠運行）
- **AIVA Core**: AI 對話助理、經驗管理器、AI 引擎
- 提供 REST API、管理接口、WebSocket 連接
- 端口: 8000 (API), 8001 (Admin), 8002 (WebSocket)

### Layer 2: 功能組件（按需啟動，最多 22 個）
- SQL 注入掃描器
- XSS 掃描器
- 自主測試循環
- 系統探索器
- 功能驗證器
- 綜合滲透測試
- ...等其他組件

---

## 🐳 Docker Compose 部署（本地開發）

### 1. 前置要求
```bash
# 安裝 Docker Desktop
# Windows: https://www.docker.com/products/docker-desktop/
# 確保 Docker Compose 已安裝
docker-compose --version
```

### 2. 快速啟動

#### 只啟動核心服務和基礎設施
```bash
cd C:\D\fold7\AIVA-git
docker-compose up -d
```

這將啟動：
- ✅ PostgreSQL (5432)
- ✅ Redis (6379)
- ✅ RabbitMQ (5672, 15672)
- ✅ Neo4j (7474, 7687)
- ✅ AIVA Core (8000, 8001, 8002)

#### 啟動特定組件
```bash
# 啟動掃描器組件
docker-compose --profile scanners up -d

# 啟動測試組件
docker-compose --profile testing up -d

# 啟動所有組件
docker-compose --profile all up -d
```

### 3. 查看服務狀態
```bash
# 查看運行中的容器
docker-compose ps

# 查看核心服務日誌
docker-compose logs -f aiva-core

# 查看特定組件日誌
docker-compose logs -f scanner-sqli
```

### 4. 訪問服務

| 服務 | URL | 說明 |
|------|-----|------|
| AIVA Core API | http://localhost:8000 | 核心 API 端點 |
| AIVA Admin | http://localhost:8001 | 管理界面 |
| RabbitMQ Management | http://localhost:15672 | 消息隊列管理 (guest/guest) |
| Neo4j Browser | http://localhost:7474 | 圖數據庫瀏覽器 (neo4j/aiva123) |

### 5. 停止服務
```bash
# 停止所有服務
docker-compose down

# 停止並刪除數據卷
docker-compose down -v
```

---

## ☸️ Kubernetes 部署（生產環境）

### 1. 前置要求
```bash
# 安裝 kubectl
kubectl version --client

# 確保已連接到 K8s 集群
kubectl cluster-info

# （可選）安裝 Helm
helm version
```

### 2. 使用原生 Kubernetes Manifests

#### 部署核心服務
```bash
cd C:\D\fold7\AIVA-git\k8s

# 創建命名空間
kubectl apply -f 00-namespace.yaml

# 創建配置和密鑰
kubectl apply -f 01-configmap.yaml

# 創建存儲
kubectl apply -f 02-storage.yaml

# 部署核心 AI 服務（永遠運行）
kubectl apply -f 10-core-deployment.yaml

# 檢查部署狀態
kubectl get pods -n aiva-system
kubectl get svc -n aiva-system
```

#### 啟動功能組件（按需）
```bash
# 啟動掃描器 Job
kubectl create job --from=cronjob/aiva-scanner-sqli manual-scan-1 -n aiva-system

# 啟動測試組件（CronJob 會自動執行）
kubectl apply -f 20-components-jobs.yaml

# 手動觸發測試
kubectl create job --from=cronjob/aiva-testing-autonomous manual-test-1 -n aiva-system
```

#### 查看服務狀態
```bash
# 查看所有 Pods
kubectl get pods -n aiva-system

# 查看核心服務日誌
kubectl logs -f deployment/aiva-core -n aiva-system

# 查看 Job 日誌
kubectl logs job/aiva-scanner-sqli -n aiva-system

# 查看服務
kubectl get svc -n aiva-system
```

#### 訪問服務
```bash
# 端口轉發到本地
kubectl port-forward -n aiva-system svc/aiva-core-service 8000:8000

# 或使用 LoadBalancer 的外部 IP
kubectl get svc aiva-core-external -n aiva-system
```

### 3. 使用 Helm Chart（推薦）

#### 安裝 AIVA
```bash
cd C:\D\fold7\AIVA-git

# 安裝完整的 AIVA 系統
helm install aiva ./helm/aiva \
  --namespace aiva-system \
  --create-namespace

# 使用自定義配置
helm install aiva ./helm/aiva \
  --namespace aiva-system \
  --create-namespace \
  --values custom-values.yaml
```

#### 升級 AIVA
```bash
helm upgrade aiva ./helm/aiva \
  --namespace aiva-system
```

#### 查看狀態
```bash
helm status aiva -n aiva-system
helm list -n aiva-system
```

#### 卸載 AIVA
```bash
helm uninstall aiva -n aiva-system
```

---

## 🔧 配置說明

### 環境變數配置

#### 核心服務環境變數
```yaml
AIVA_MODE: production
AIVA_ENVIRONMENT: docker|kubernetes
AIVA_POSTGRES_HOST: postgres
AIVA_POSTGRES_PORT: 5432
AIVA_REDIS_HOST: redis
AIVA_REDIS_PORT: 6379
AIVA_RABBITMQ_URL: amqp://guest:guest@rabbitmq:5672/
AIVA_NEO4J_HOST: neo4j
AIVA_NEO4J_PORT: 7687
```

#### 組件連接配置
```yaml
AIVA_CORE_URL: http://aiva-core:8000  # Docker Compose
AIVA_CORE_URL: http://aiva-core-service:8000  # Kubernetes
```

### 資源配置建議

| 組件類型 | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|-------------|-----------|----------------|--------------|
| 核心服務 | 500m | 2000m | 512Mi | 2Gi |
| 掃描器 | 200m | 500m | 256Mi | 512Mi |
| 測試組件 | 500m | 1000m | 512Mi | 1Gi |
| 滲透測試 | 500m | 1000m | 512Mi | 1Gi |

---

## 📊 監控和日誌

### Docker Compose
```bash
# 查看所有容器狀態
docker-compose ps

# 查看資源使用
docker stats

# 實時日誌
docker-compose logs -f --tail=100
```

### Kubernetes
```bash
# 查看 Pod 狀態
kubectl get pods -n aiva-system -w

# 查看資源使用
kubectl top pods -n aiva-system

# 查看事件
kubectl get events -n aiva-system

# 描述 Pod（查看詳細信息）
kubectl describe pod <pod-name> -n aiva-system
```

---

## 🚀 動態組件管理

### 使用 Docker Compose
```bash
# 啟動特定組件
docker-compose up -d scanner-sqli

# 停止特定組件
docker-compose stop scanner-sqli

# 重啟組件
docker-compose restart scanner-sqli

# 縮放組件（如果支持）
docker-compose up -d --scale scanner-sqli=3
```

### 使用 Kubernetes
```bash
# 創建 Job
kubectl create job my-scan --from=cronjob/aiva-scanner-sqli -n aiva-system

# 刪除 Job
kubectl delete job my-scan -n aiva-system

# 暫停 CronJob
kubectl patch cronjob aiva-testing-autonomous -p '{"spec":{"suspend":true}}' -n aiva-system

# 恢復 CronJob
kubectl patch cronjob aiva-testing-autonomous -p '{"spec":{"suspend":false}}' -n aiva-system
```

---

## 🛠️ 故障排查

### 核心服務無法啟動
```bash
# 檢查依賴服務
docker-compose ps  # Docker
kubectl get pods -n aiva-system  # K8s

# 查看詳細日誌
docker-compose logs aiva-core  # Docker
kubectl logs deployment/aiva-core -n aiva-system  # K8s

# 檢查健康檢查
curl http://localhost:8000/health
```

### 組件無法連接核心服務
```bash
# 檢查網絡連接
docker-compose exec scanner-sqli ping aiva-core  # Docker
kubectl exec -it <pod-name> -n aiva-system -- ping aiva-core-service  # K8s

# 檢查環境變數
docker-compose exec scanner-sqli env | grep AIVA  # Docker
kubectl exec -it <pod-name> -n aiva-system -- env | grep AIVA  # K8s
```

### 數據庫連接問題
```bash
# 檢查 PostgreSQL
docker-compose exec postgres psql -U postgres -d aiva_db  # Docker
kubectl exec -it <postgres-pod> -n aiva-system -- psql -U postgres -d aiva_db  # K8s

# 檢查 Redis
docker-compose exec redis redis-cli ping  # Docker
kubectl exec -it <redis-pod> -n aiva-system -- redis-cli ping  # K8s
```

---

## 📚 更多資源

- [Docker Compose 文檔](https://docs.docker.com/compose/)
- [Kubernetes 文檔](https://kubernetes.io/docs/)
- [Helm 文檔](https://helm.sh/docs/)
- AIVA 項目文檔: `README.md`

---

## 💡 最佳實踐

1. **本地開發**: 使用 Docker Compose，快速啟動和測試
2. **生產部署**: 使用 Kubernetes + Helm，實現高可用和自動擴展
3. **核心服務**: 永遠保持運行，確保系統穩定性
4. **功能組件**: 按需啟動，節省資源
5. **監控**: 配置 Prometheus + Grafana 進行監控
6. **日誌**: 使用 ELK Stack 或雲端日誌服務
7. **備份**: 定期備份 PostgreSQL 和 Neo4j 數據

---

**部署完成後，AIVA 核心服務將持續運行，其他 22 個組件可以隨時動態啟動和停止！** 🎯
