---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 配置指南 - 生產環境部署專用

> ⚠️ **重要說明**: 本指南僅適用於**生產環境部署**  
> 📘 **研發階段**: 無需任何環境變數配置，直接使用預設值開發

## 📑 目錄

- [📁 研發 vs 生產配置說明](#-研發-vs-生產配置說明)
- [⚙️ 生產環境配置項目](#-生產環境配置項目)
- [🚀 生產環境部署流程](#-生產環境部署流程)
- [🔒 安全配置指南](#-安全配置指南)
- [🐛 故障排除](#-故障排除)

## 研發 vs 生產配置說明

### 🛠️ 研發階段（當前使用）
**完全無需配置**，AIVA 自動使用安全的預設值：

```bash
# 自動使用的預設連接（無需手動設置）
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/aiva_db"
RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
LOG_LEVEL="INFO"
ENVIRONMENT="development"
```

### 🏭 生產環境（未來部署時）
以下配置**僅在生產部署時**才需要設置：

```bash
# 生產資料庫配置
DATABASE_URL="postgresql://prod_user:secure_password@prod-db:5432/aiva_prod"
RABBITMQ_URL="amqp://prod_user:secure_password@prod-mq:5672/"
LOG_LEVEL="WARN"
ENVIRONMENT="production"
```

```

## 生產環境部署流程

### 步驟 1: 準備生產配置
```bash
# 基於範本創建生產配置
cp .env.example .env.production
# 修改 .env.production 中的連接參數
```

### 步驟 2: 設置安全連接
```bash
# 設置生產環境變數
export DATABASE_URL="postgresql://prod_user:secure_password@prod-db:5432/aiva_prod"
export RABBITMQ_URL="amqp://prod_user:secure_password@prod-mq:5672/"
export LOG_LEVEL="WARN"
export ENVIRONMENT="production"
```

### 步驟 3: 部署驗證
```bash
# 測試連接
docker-compose -f docker-compose.prod.yml up -d
```

## 安全配置指南

### 密碼安全
- 使用強密碼（至少 16 字符）
- 定期輪換密碼
- 避免在程式碼中寫死密碼

### 網絡安全
- 使用 TLS 加密連接
- 限制資料庫訪問 IP
- 設定防火牆規則

## 故障排除

### 連接問題
- 檢查服務是否運行: `docker ps`
- 驗證網絡連通性: `ping <host>`
- 查看服務日誌: `docker logs <container>`
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