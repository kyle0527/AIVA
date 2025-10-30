# AIVA 安全配置指南

## 概述

此文檔說明 AIVA 平台如何遵循 12-factor app 原則，移除硬編碼憑證，並實施安全的環境變數配置。

## 修復的安全問題

### 1. 硬編碼 RabbitMQ 憑證
**問題**: 多個 workers 中硬編碼了 RabbitMQ 連接字串，包含明文用戶名和密碼。

**修復**:
- 移除所有硬編碼的認證資訊
- 實施環境變數配置，支援完整 URL 或組合式配置
- 統一所有語言的配置邏輯 (Go, Rust, Node.js, Python)

### 2. 配置標準化
**改進**:
- 統一使用 `AIVA_` 前綴的環境變數
- 實施回退配置邏輯
- 提供清晰的錯誤訊息

## 環境變數配置

### RabbitMQ 配置

#### 選項 1: 完整 URL (推薦)
```bash
AIVA_RABBITMQ_URL=amqp://username:password@host:port/vhost
```

#### 選項 2: 組合式配置
```bash
AIVA_RABBITMQ_HOST=localhost
AIVA_RABBITMQ_PORT=5672
AIVA_RABBITMQ_USER=aiva_user
AIVA_RABBITMQ_PASSWORD=secure_password
AIVA_RABBITMQ_VHOST=/
```

### 其他配置
```bash
AIVA_MQ_EXCHANGE=aiva.topic
AIVA_MQ_DLX=aiva.dlx
AIVA_LOG_LEVEL=info
AIVA_ENVIRONMENT=production
```

## 部署指南

### 1. 本地開發
```bash
# 複製環境變數模板
cp .env.example .env

# 編輯 .env 檔案，填入實際配置值
nano .env

# 載入環境變數
source .env
```

### 2. 容器化部署
```dockerfile
# 使用環境變數而非硬編碼值
ENV AIVA_RABBITMQ_USER=aiva_user
ENV AIVA_RABBITMQ_PASSWORD=secure_password
```

### 3. Kubernetes 部署
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aiva-config
type: Opaque
data:
  rabbitmq-user: <base64-encoded-username>
  rabbitmq-password: <base64-encoded-password>
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: aiva-worker
        env:
        - name: AIVA_RABBITMQ_USER
          valueFrom:
            secretKeyRef:
              name: aiva-config
              key: rabbitmq-user
        - name: AIVA_RABBITMQ_PASSWORD
          valueFrom:
            secretKeyRef:
              name: aiva-config
              key: rabbitmq-password
```

## 安全最佳實踐

1. **永不硬編碼憑證**: 所有敏感資訊都應通過環境變數配置
2. **使用強密碼**: RabbitMQ 和資料庫密碼應該足夠複雜
3. **定期輪換憑證**: 建議定期更換密碼和 API 金鑰
4. **最小權限原則**: 為每個服務分配最小必要權限
5. **監控和審計**: 記錄所有配置變更和存取

## 配置驗證

系統啟動時會驗證所有必需的環境變數是否已設置：

```bash
# 測試配置
docker-compose config

# 驗證 worker 啟動
docker-compose up -d
docker-compose logs
```

## 故障排除

### 常見錯誤

1. **錯誤**: `AIVA_RABBITMQ_URL or AIVA_RABBITMQ_USER/AIVA_RABBITMQ_PASSWORD must be set`
   **解決**: 設置正確的環境變數

2. **錯誤**: `Failed to connect to RabbitMQ`
   **解決**: 檢查 RabbitMQ 服務是否正在運行，網絡是否可達

3. **錯誤**: `Authentication failed`
   **解決**: 驗證用戶名和密碼是否正確

### 配置檢查清單

- [ ] 所有必需的環境變數已設置
- [ ] RabbitMQ 服務正在運行
- [ ] 網絡連接正常
- [ ] 用戶權限配置正確
- [ ] 隊列名稱標準化完成

## 相關文檔

- [12-Factor App 配置原則](https://12factor.net/config)
- [AIVA 架構文檔](./docs/architecture.md)
- [部署指南](./docs/deployment.md)