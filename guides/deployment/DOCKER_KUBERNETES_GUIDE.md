---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# 🎯 AIVA 微服務部署 - 完整方案

## 📑 目錄

- [� 目錄](#目錄)
- [�📦 你獲得了什麼](#你獲得了什麼)
  - [✅ **雙模式部署架構**](#雙模式部署架構)
- [📂 文件結構](#文件結構)
- [🚀 快速開始（3 步驟）](#快速開始3-步驟)
  - [步驟 1: 啟動核心服務（永遠運行）](#步驟-1-啟動核心服務永遠運行)
  - [步驟 2: 訪問服務](#步驟-2-訪問服務)
  - [步驟 3: 啟動功能組件（按需）](#步驟-3-啟動功能組件按需)
- [🎯 核心理念](#核心理念)
  - [**分層架構**](#分層架構)
  - [**關鍵特性**](#關鍵特性)
- [📋 可用的組件 Profiles](#可用的組件-profiles)
- [🔧 常用命令](#常用命令)
  - [Docker Compose 模式](#docker-compose-模式)
  - [Kubernetes 模式](#kubernetes-模式)
- [🏗️ 生產部署（Kubernetes）](#生產部署kubernetes)
  - [準備工作](#準備工作)
  - [部署步驟](#部署步驟)
  - [使用 Helm（推薦）](#使用-helm推薦)
- [📊 資源需求](#資源需求)
  - [最小配置（核心服務）](#最小配置核心服務)
  - [推薦配置（核心 + 全部組件）](#推薦配置核心-全部組件)
- [🛠️ 故障排查](#故障排查)
  - [核心服務無法啟動](#核心服務無法啟動)
  - [組件無法連接核心](#組件無法連接核心)
  - [健康檢查失敗](#健康檢查失敗)
- [📚 文檔索引](#文檔索引)
- [💡 最佳實踐](#最佳實踐)
  - [本地開發](#本地開發)
  - [生產部署](#生產部署)
- [🎉 總結](#總結)
- [🤝 支持](#支持)
- [🔗 相關資源](#相關資源)
  - [部署指南](#部署指南)
  - [開發指南](#開發指南)
  - [故障排除](#故障排除)
  - [使用者手冊](#使用者手冊)

---
## � 目錄

- [📦 你獲得了什麼](#-你獲得了什麼)
- [📂 文件結構](#-文件結構)
- [🚀 快速開始](#-快速開始)
- [🐳 Docker Compose 部署](#-docker-compose-部署)
- [☸️ Kubernetes 部署](#-kubernetes-部署)
- [📊 監控與維護](#-監控與維護)
- [🐛 故障排除](#-故障排除)
- [🔗 相關資源](#-相關資源)

## �📦 你獲得了什麼

### ✅ **雙模式部署架構**
完美結合 Docker Compose 本地開發 + Kubernetes 生產部署！

```
本地開發                        生產部署
    ↓                             ↓
Docker Compose    ──────→    Kubernetes + Helm
(快速迭代)                    (企業級高可用)
```

---

## 📂 文件結構

```
AIVA-git/
├── Dockerfile.core              # 核心 AI 服務容器
├── Dockerfile.component         # 功能組件容器
├── docker-compose.yml          # Docker Compose 配置
├── start-aiva.ps1              # Windows 快速啟動
├── start-aiva.sh               # Linux/macOS 快速啟動
├── DEPLOYMENT.md               # 詳細部署指南
├── ARCHITECTURE_SUMMARY.md     # 架構總結
│
├── k8s/                        # Kubernetes 清單
│   ├── 00-namespace.yaml       # 命名空間
│   ├── 01-configmap.yaml       # 配置和密鑰
│   ├── 02-storage.yaml         # 持久化存儲
│   ├── 10-core-deployment.yaml # 核心服務部署
│   └── 20-components-jobs.yaml # 功能組件 Jobs
│
└── helm/aiva/                  # Helm Chart
    ├── Chart.yaml              # Chart 元數據
    └── values.yaml             # 配置參數
```

---

## 🚀 快速開始（3 步驟）

### 步驟 1: 啟動核心服務（永遠運行）
```powershell
# Windows
.\start-aiva.ps1 -Action core

# Linux/macOS
./start-aiva.sh core
```

等待 60 秒，核心服務將包括：
- ✅ PostgreSQL (數據庫)
- ✅ Redis (緩存)
- ⚠️ RabbitMQ (消息佇列 - v2.0已改用命令系統，可選)
- ✅ Neo4j (圖數據庫)
- ✅ **AIVA Core AI Service** (核心大腦)

### 步驟 2: 訪問服務
打開瀏覽器：
- 🌐 **AIVA API**: http://localhost:8000/health
- 🌐 **RabbitMQ 管理** (可選): http://localhost:15672 (guest/guest)
- 🌐 **Neo4j 瀏覽器**: http://localhost:7474 (neo4j/aiva123)

### 步驟 3: 啟動功能組件（按需）
```powershell
# 啟動掃描器組件
.\start-aiva.ps1 -Action scanners

# 啟動測試組件
.\start-aiva.ps1 -Action testing

# 啟動所有 22 個組件
.\start-aiva.ps1 -Action all
```

---

## 🎯 核心理念

### **分層架構**
```
Layer 2: 功能組件（按需啟動，最多 22 個）
         ↓ 連接
Layer 1: 核心 AI 服務（永遠運行，系統大腦）
         ↓ 依賴
Layer 0: 基礎設施（永遠運行）
```

### **關鍵特性**
1. **核心服務永不停止** - 確保系統穩定性
2. **組件動態管理** - 按需啟動/停止 22 個功能組件
3. **環境零配置** - 自動設置所有環境變數
4. **雙模式支持** - 本地開發 + 生產部署
5. **健康監控** - 自動檢測和重啟

---

## 📋 可用的組件 Profiles

| Profile | 組件 | 用途 |
|---------|------|------|
| `scanners` | SQLi, XSS 掃描器 | 漏洞掃描 |
| `testing` | 自主測試循環 | AI 自動化測試 |
| `explorers` | 系統探索器 | 代碼分析 |
| `validators` | 功能驗證器 | 功能測試 |
| `pentest` | 綜合滲透測試 | 完整滲透測試 |
| `all` | 所有組件 | 全功能模式 |

---

## 🔧 常用命令

### Docker Compose 模式

```powershell
# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f aiva-core

# 重啟核心服務
docker-compose restart aiva-core

# 停止所有服務
docker-compose down

# 完全清理（包括數據卷）
docker-compose down -v
```

### Kubernetes 模式

```bash
# 部署核心服務
kubectl apply -f k8s/

# 查看 Pod 狀態
kubectl get pods -n aiva-system

# 查看日誌
kubectl logs -f deployment/aiva-core -n aiva-system

# 啟動掃描 Job
kubectl create job my-scan --from=cronjob/aiva-scanner-sqli -n aiva-system

# 使用 Helm 安裝
helm install aiva ./helm/aiva --namespace aiva-system --create-namespace
```

---

## 🏗️ 生產部署（Kubernetes）

### 準備工作
1. 確保有可用的 Kubernetes 集群
2. 配置 kubectl 連接
3. （可選）安裝 Helm

### 部署步驟
```bash
# 1. 創建命名空間和配置
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/02-storage.yaml

# 2. 部署核心服務（永遠運行）
kubectl apply -f k8s/10-core-deployment.yaml

# 3. 部署功能組件（按需）
kubectl apply -f k8s/20-components-jobs.yaml

# 4. 檢查狀態
kubectl get all -n aiva-system
```

### 使用 Helm（推薦）
```bash
# 一鍵部署
helm install aiva ./helm/aiva \
  --namespace aiva-system \
  --create-namespace

# 升級
helm upgrade aiva ./helm/aiva -n aiva-system

# 卸載
helm uninstall aiva -n aiva-system
```

---

## 📊 資源需求

### 最小配置（核心服務）
- CPU: 4 核心
- 內存: 8 GB
- 磁盤: 50 GB

### 推薦配置（核心 + 全部組件）
- CPU: 8 核心
- 內存: 16 GB
- 磁盤: 100 GB

---

## 🛠️ 故障排查

### 核心服務無法啟動
```bash
# 檢查依賴服務
docker-compose ps

# 查看詳細日誌
docker-compose logs aiva-core

# 重啟服務
docker-compose restart aiva-core
```

### 組件無法連接核心
```bash
# 檢查網絡
docker-compose exec scanner-sqli ping aiva-core

# 檢查環境變數
docker-compose exec scanner-sqli env | grep AIVA
```

### 健康檢查失敗
```bash
# 手動測試健康端點
curl http://localhost:8000/health

# 查看容器日誌
docker-compose logs --tail=100 aiva-core
```

---

## 📚 文檔索引

1. **[DEPLOYMENT.md](./DEPLOYMENT.md)** - 詳細部署指南
2. **[ARCHITECTURE_SUMMARY.md](./ARCHITECTURE_SUMMARY.md)** - 架構設計總結
3. **[docker-compose.yml](./docker-compose.yml)** - 完整配置文件
4. **[k8s/](./k8s/)** - Kubernetes 清單目錄
5. **[helm/aiva/](./helm/aiva/)** - Helm Chart 目錄

---

## 💡 最佳實踐

### 本地開發
```powershell
# 1. 只啟動核心服務進行開發
.\start-aiva.ps1 -Action core

# 2. 根據需要啟動特定組件
.\start-aiva.ps1 -Action scanners

# 3. 開發完成後停止
.\start-aiva.ps1 -Action stop
```

### 生產部署
```bash
# 1. 使用 Helm 部署
helm install aiva ./helm/aiva -n aiva-system --create-namespace

# 2. 配置監控
# 啟用 Prometheus + Grafana

# 3. 設置自動擴展
kubectl autoscale deployment aiva-core --cpu-percent=70 --min=1 --max=3 -n aiva-system
```

---

## 🎉 總結

你現在擁有：

✅ **完整的容器化方案** - Docker + Docker Compose  
✅ **生產級 K8s 部署** - Kubernetes + Helm  
✅ **核心服務永遠運行** - 系統穩定性保證  
✅ **22 個動態組件** - 按需啟動/停止  
✅ **零配置啟動** - 自動環境設置  
✅ **詳細文檔** - 完整的部署指南  
✅ **快速啟動腳本** - 一鍵操作  

**AIVA 已經準備好了！開始你的 AI 安全測試之旅吧！** 🚀

---

## 🤝 支持

有問題？查看文檔：
- 📖 [DEPLOYMENT.md](./DEPLOYMENT.md)
- 📖 [ARCHITECTURE_SUMMARY.md](./ARCHITECTURE_SUMMARY.md)

或聯繫團隊！

---

**現在就開始：`.\start-aiva.ps1 -Action core`** 🎯


---

## 🔗 相關資源

### 部署指南
- 📖 [安裝指南](./INSTALLATION_GUIDE.md) - 系統安裝
- 📖 [環境配置指南](./ENVIRONMENT_CONFIG_GUIDE.md) - 環境設置
- 📖 [構建指南](./BUILD_GUIDE.md) - 項目構建
- 📖 [Docker/K8s 指南](./DOCKER_KUBERNETES_GUIDE.md) - 容器化部署

### 開發指南
- 📖 [開發快速指南](../development/DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)

### 故障排除
- 📖 [性能優化指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)
- 📖 [測試重現指南](../troubleshooting/TESTING_REPRODUCTION_GUIDE.md)

### 使用者手冊
- 📚 [AIVA 使用者手冊](../../docs/user_guides/00_general/AIVA_USER_MANUAL.md)

