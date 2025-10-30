# AIVA Docker 映像檔建立指南

> 📅 建立日期: 2025-10-30  
> 🏷️ 版本: v1.0  
> 📋 狀態: 完整可用

## 🎯 概覽

AIVA 採用微服務架構，提供多種 Docker 映像檔建立方式：

### 📦 映像檔類型

| 映像檔類型 | 用途 | 大小 | 建議場景 |
|-----------|------|------|---------|
| `aiva-core` | 核心 AI 服務 | ~800MB | 生產環境必需 |
| `aiva-component` | 功能組件 | ~600MB | 按需啟動 |
| `aiva-core-minimal` | 最小化版本 | ~400MB | 資源受限環境 |
| `aiva-integration` | 整合服務 | ~500MB | 企業級整合 |

## 🚀 快速建立

### 1. 建立所有映像檔（一鍵建立）

```bash
# 切換到項目根目錄
cd "c:\D\fold7\AIVA-git"

# 建立所有映像檔
docker compose -f docker/compose/docker-compose.yml build

# 或者使用 PowerShell 腳本
.\docker\build-all-images.ps1
```

### 2. 個別建立映像檔

```bash
# 建立核心服務映像檔
docker build -f docker/core/Dockerfile.core -t aiva-core:latest .

# 建立功能組件映像檔
docker build -f docker/components/Dockerfile.component -t aiva-component:latest .

# 建立最小化版本
docker build -f docker/core/Dockerfile.core.minimal -t aiva-core:minimal .

# 建立整合服務映像檔
docker build -f docker/infrastructure/Dockerfile.integration -t aiva-integration:latest .
```

## 🔧 建立選項與優化

### 高級建立參數

```bash
# 使用多階段建立減少映像檔大小
docker build \
  --target production \
  -f docker/core/Dockerfile.core \
  -t aiva-core:optimized \
  --build-arg BUILD_ENV=production \
  .

# 指定平台建立（支援多架構）
docker build \
  --platform linux/amd64,linux/arm64 \
  -f docker/core/Dockerfile.core \
  -t aiva-core:multi-arch \
  .

# 使用建立緩存加速
docker build \
  --cache-from aiva-core:latest \
  -f docker/core/Dockerfile.core \
  -t aiva-core:cache-optimized \
  .
```

### 環境變數自訂

```bash
# 為特定環境建立
docker build \
  --build-arg ENVIRONMENT=production \
  --build-arg DEBUG=false \
  --build-arg LOG_LEVEL=INFO \
  -f docker/core/Dockerfile.core \
  -t aiva-core:production \
  .
```

## 📁 映像檔內容說明

### 核心服務映像檔 (`aiva-core`)

```
/app/
├── services/
│   ├── aiva_common/         # 共用模組
│   ├── core/               # 核心 AI 服務
│   └── features/           # 功能組件
├── aiva_launcher.py        # 啟動器
├── requirements.txt        # Python 依賴
└── .env                   # 環境配置
```

**功能特色**：
- ✅ AI 對話助理
- ✅ 經驗管理器
- ✅ 服務健康監控
- ✅ API 服務 (端口 8000, 8001, 8002)

### 功能組件映像檔 (`aiva-component`)

```
/app/
├── services/              # 所有服務模組
├── config/               # 配置文件
├── api/                  # API 定義
└── *.py                  # 50+ 運行時腳本
```

**功能特色**：
- ✅ 22 個掃描器
- ✅ 安全測試工具
- ✅ 報告生成器
- ✅ 按需啟動

## 🔍 映像檔驗證

### 建立完成後驗證

```bash
# 查看已建立的映像檔
docker images | grep aiva

# 檢查映像檔詳細資訊
docker inspect aiva-core:latest

# 測試映像檔運行
docker run --rm aiva-core:latest python --version

# 健康檢查
docker run -d --name aiva-test aiva-core:latest
sleep 30
docker inspect --format='{{.State.Health.Status}}' aiva-test
docker rm -f aiva-test
```

### 映像檔大小優化驗證

```bash
# 比較映像檔大小
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep aiva

# 檢查映像檔層次
docker history aiva-core:latest --no-trunc
```

## 🚢 部署與運行

### 使用 Docker Compose 部署

```bash
# 啟動完整服務（包含基礎設施）
docker compose -f docker/compose/docker-compose.yml up -d

# 僅啟動核心服務
docker compose -f docker/compose/docker-compose.yml up -d aiva-core

# 生產環境部署
docker compose -f docker/compose/docker-compose.production.yml up -d
```

### 使用 Kubernetes 部署

```bash
# 創建命名空間
kubectl apply -f docker/k8s/00-namespace.yaml

# 部署配置
kubectl apply -f docker/k8s/01-configmap.yaml

# 部署存儲
kubectl apply -f docker/k8s/02-storage.yaml

# 部署核心服務
kubectl apply -f docker/k8s/10-core-deployment.yaml

# 部署功能組件
kubectl apply -f docker/k8s/20-components-jobs.yaml
```

### 使用 Helm 部署

```bash
# 安裝 AIVA Helm Chart
helm install aiva docker/helm/aiva/ \
  --set image.tag=latest \
  --set environment=production

# 升級部署
helm upgrade aiva docker/helm/aiva/ \
  --set image.tag=v1.1.0
```

## 🔧 故障排除

### 常見建立問題

1. **依賴安裝失敗**
   ```bash
   # 清理建立緩存
   docker builder prune -a
   
   # 重新建立不使用緩存
   docker build --no-cache -f docker/core/Dockerfile.core -t aiva-core:latest .
   ```

2. **映像檔太大**
   ```bash
   # 使用最小化版本
   docker build -f docker/core/Dockerfile.core.minimal -t aiva-core:minimal .
   
   # 檢查映像檔內容
   docker run -it --rm aiva-core:latest sh
   ```

3. **權限問題**
   ```bash
   # Windows 下確保 Docker Desktop 權限正確
   # 檢查文件共享設置
   ```

### 性能優化建議

1. **使用 .dockerignore**
   ```
   # .dockerignore 文件內容
   **/__pycache__
   **/*.pyc
   **/node_modules
   .git
   .pytest_cache
   docs/
   reports/
   ```

2. **多階段建立**
   ```dockerfile
   FROM python:3.11-slim as builder
   # 建立階段...
   
   FROM python:3.11-slim as production
   # 最終階段...
   ```

3. **層次優化**
   - 將較少變動的指令放在前面
   - 合併 RUN 指令減少層數
   - 使用特定版本標籤避免意外更新

## 📊 統一環境變數

根據前面完成的環境變數統一，建立時使用以下標準配置：

```dockerfile
# 統一環境變數（無 AIVA_ 前綴）
ENV DATABASE_URL=postgresql://aiva:aiva_secure_password@postgres:5432/aiva \
    RABBITMQ_URL=amqp://aiva:aiva_mq_password@rabbitmq:5672/aiva \
    REDIS_URL=redis://:aiva_redis_password@redis:6379/0 \
    NEO4J_URL=bolt://neo4j:password@neo4j:7687 \
    LOG_LEVEL=INFO \
    AUTO_MIGRATE=1
```

## 🔄 CI/CD 整合

### GitHub Actions 建立範例

```yaml
name: Build AIVA Docker Images

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build Core Image
      run: |
        docker build -f docker/core/Dockerfile.core -t aiva-core:${{ github.sha }} .
    
    - name: Build Component Image
      run: |
        docker build -f docker/components/Dockerfile.component -t aiva-component:${{ github.sha }} .
    
    - name: Test Images
      run: |
        docker run --rm aiva-core:${{ github.sha }} python --version
        docker run --rm aiva-component:${{ github.sha }} python --version
```

## 📚 相關資源

- [Docker 指南](docker/DOCKER_GUIDE.md)
- [環境變數配置](../.env.docker)
- [Docker Compose 配置](docker/compose/docker-compose.yml)
- [Kubernetes 配置](docker/k8s/)
- [Helm Chart](docker/helm/aiva/)

## 🎯 最佳實踐總結

1. **映像檔標籤管理**
   - 使用語義化版本 (v1.0.0)
   - 包含建立時間戳
   - 標記環境類型 (dev, staging, prod)

2. **安全考量**
   - 定期更新基礎映像檔
   - 掃描漏洞
   - 使用非 root 使用者運行

3. **效能優化**
   - 合理使用建立緩存
   - 最小化映像檔大小
   - 優化層次結構

4. **維護性**
   - 統一建立流程
   - 自動化測試
   - 完整的文檔記錄

---

## 📞 支援

如需協助，請參考：
- [問題追蹤](https://github.com/kyle0527/aiva/issues)
- [討論區](https://github.com/kyle0527/aiva/discussions)
- 內部文檔: [Docker 指南](docker/DOCKER_GUIDE.md)

*最後更新: 2025-10-30*