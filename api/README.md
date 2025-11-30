# AIVA Security Platform API

## 📑 目錄

- [� 修復原則](#-修復原則)
- [�🚀 快速開始](#-快速開始)
  - [1. 安裝依賴](#1-安裝依賴)
  - [2. 啟動 API 服務](#2-啟動-api-服務)
  - [3. 訪問 API 文檔](#3-訪問-api-文檔)
- [🔐 認證系統](#-認證系統)
  - [登入範例](#登入範例)
- [💎 高價值功能模組 API](#-高價值功能模組-api)
  - [Mass Assignment 檢測](#mass-assignment-檢測)
  - [JWT 混淆攻擊檢測](#jwt-混淆攻擊檢測)
  - [OAuth 配置錯誤檢測](#oauth-配置錯誤檢測)
  - [GraphQL 權限檢測](#graphql-權限檢測)
  - [SSRF OOB 檢測](#ssrf-oob-檢測)
- [📊 掃描管理](#-掃描管理)
  - [查看掃描狀態](#查看掃描狀態)
  - [列出所有掃描](#列出所有掃描)
- [🛠 系統管理](#-系統管理)
  - [健康檢查](#健康檢查)
  - [系統統計（管理員）](#系統統計管理員)
- [🧪 測試 API](#-測試-api)
- [📁 API 結構](#-api-結構)
- [🌟 商業價值](#-商業價值)
- [📈 部署建議](#-部署建議)
  - [開發環境](#開發環境)
  - [生產環境](#生產環境)
  - [Docker 部署](#docker-部署)
- [🔒 安全考量](#-安全考量)
- [📞 支援](#-支援)

---

完整的 REST API 系統，為 AIVA 高價值安全功能模組提供商業化接口。

## � 修復原則

**保留未使用函數原則**: 在程式碼修復過程中，若發現有定義但尚未使用的函數或方法，只要不影響程式正常運作，建議予以保留。這些函數可能是：
- 預留的 API 端點或介面
- 未來功能的基礎架構
- 測試或除錯用途的輔助函數
- 向下相容性考量的舊版介面

說不定未來會用到，保持程式碼的擴展性和靈活性。

## 📍 定位說明

> **API 層的角色**  
> 本 API 系統是 `services/features/` 安全測試模組的 REST 包裝層。  
> 核心業務邏輯 (93.8%) 位於 `services/` 目錄。

---

## 🚀 快速開始

### 1. 安裝依賴
```bash
cd api
pip install -r requirements.txt
```

### 2. 啟動 API 服務
```bash
# 使用啟動腳本（推薦）
python start_api.py

# 或直接使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 訪問 API 文檔
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔐 認證系統

API 使用 JWT 令牌認證。默認用戶帳戶：

| 用戶名 | 密碼 | 角色 | 權限 |
|--------|------|------|------|
| admin | aiva-admin-2025 | 管理員 | 完全權限 |
| user | aiva-user-2025 | 一般用戶 | 讀寫權限 |
| viewer | aiva-viewer-2025 | 檢視者 | 唯讀權限 |

### 登入範例
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "aiva-admin-2025"}'
```

## 💎 高價值功能模組 API

### Mass Assignment 檢測
```bash
curl -X POST "http://localhost:8000/api/v1/security/mass-assignment" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://example.com",
    "update_endpoint": "/api/users/update",
    "auth_headers": {"Authorization": "Bearer user-token"},
    "test_fields": ["admin", "role", "is_admin"]
  }'
```

### JWT 混淆攻擊檢測
```bash
curl -X POST "http://localhost:8000/api/v1/security/jwt-confusion" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://example.com",
    "victim_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  }'
```

### OAuth 配置錯誤檢測
```bash
curl -X POST "http://localhost:8000/api/v1/security/oauth-confusion" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://example.com",
    "client_id": "your-client-id",
    "legitimate_redirect": "https://legitimate.com/callback",
    "attacker_redirect": "https://attacker.com/callback"
  }'
```

### GraphQL 權限檢測
```bash
curl -X POST "http://localhost:8000/api/v1/security/graphql-authz" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://example.com/graphql",
    "user_headers": {"Authorization": "Bearer user-token"},
    "test_queries": [
      "query { users { id email } }",
      "query { adminUsers { id email role } }"
    ]
  }'
```

### SSRF OOB 檢測
```bash
curl -X POST "http://localhost:8000/api/v1/security/ssrf-oob" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://example.com",
    "oob_callback": "https://your-oob-server.com/callback"
  }'
```

## 📊 掃描管理

### 查看掃描狀態
```bash
curl -X GET "http://localhost:8000/api/v1/scans/SCAN_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 列出所有掃描
```bash
curl -X GET "http://localhost:8000/api/v1/scans" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🛠 系統管理

### 健康檢查
```bash
curl -X GET "http://localhost:8000/health"
```

### 系統統計（管理員）
```bash
curl -X GET "http://localhost:8000/api/v1/admin/stats" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

## 🧪 測試 API

使用內建的測試腳本：

```bash
# 執行完整測試
python test_api.py

# 執行特定測試
python test_api.py --test health
python test_api.py --test auth --username admin --password aiva-admin-2025
python test_api.py --test scan

# 測試不同環境
python test_api.py --host production-server.com --port 443
```

## 📁 API 結構

```
api/
├── main.py              # 主應用入口
├── requirements.txt     # Python 依賴
├── start_api.py        # 啟動腳本
├── test_api.py         # API 測試工具
├── routers/            # 路由模組
│   ├── auth.py         # 認證端點
│   ├── security.py     # 高價值功能模組端點
│   └── admin.py        # 系統管理端點
└── README.md           # 本文檔
```

## 🌟 商業價值

每個高價值模組的預期市場價值：

- **Mass Assignment**: $2.1K-$8.2K
- **JWT Confusion**: $1.8K-$7.5K  
- **OAuth Confusion**: $2.5K-$10.2K
- **GraphQL AuthZ**: $1.9K-$7.8K
- **SSRF OOB**: $2.2K-$8.7K

**總潛在價值**: $10.5K-$41K+ 每次成功的漏洞發現

## 📈 部署建議

### 開發環境
```bash
python start_api.py --reload --log-level debug
```

### 生產環境
```bash
python start_api.py --workers 4 --log-level warning
```

### Docker 部署
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "start_api.py", "--workers", "4"]
```

## 🔒 安全考量

1. **生產環境必須更改默認密碼**
2. **使用 HTTPS 進行 API 通信**
3. **定期輪換 JWT 密鑰**
4. **實施 API 速率限制**
5. **監控和記錄 API 使用情況**

## 📞 支援

- 如需技術支援，請查看 `/docs` 端點的 API 文檔
- 使用 `test_api.py` 診斷連接問題
- 檢查 `health` 端點確認系統狀態