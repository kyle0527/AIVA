# AIVA 搜索功能 API Keys 配置指南

## 概述

本指南說明如何獲取各種搜索 API 的密鑰並配置到 AIVA 系統中。

## 必需的 API Keys

### 1. Google Custom Search API

**免費額度**: 每天 100 次查詢

**獲取步驟**:
1. 訪問 [Google Cloud Console](https://console.cloud.google.com/)
2. 創建新項目或選擇現有項目
3. 啟用 "Custom Search API"
4. 創建憑據 → API Key
5. 訪問 [Programmable Search Engine](https://programmablesearchengine.google.com/)
6. 創建搜索引擎，獲取 Search Engine ID (cx)

**配置**:
```bash
export GOOGLE_API_KEY="AIzaSy..."
export GOOGLE_SEARCH_ENGINE_ID="017576662512468239146:..."
```

---

### 2. GitHub API

**免費額度**: 
- 未認證: 60 次/小時
- 認證: 5,000 次/小時

**獲取步驟**:
1. 登錄 GitHub
2. 訪問 [Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
3. 點擊 "Generate new token (classic)"
4. 選擇權限: `public_repo`, `read:user`
5. 生成並複製 token

**配置**:
```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
```

---

### 3. NVD (National Vulnerability Database) API

**免費額度**: 
- 未認證: 5 次/30秒
- 認證: 50 次/30秒

**獲取步驟**:
1. 訪問 [NVD API Key Request](https://nvd.nist.gov/developers/request-an-api-key)
2. 填寫申請表單
3. 通過郵箱驗證
4. 獲取 API Key

**配置**:
```bash
export NVD_API_KEY="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

---

## 可選的 API Keys

### 4. Shodan API

**費用**: 需要付費會員（$59/月）

**獲取步驟**:
1. 註冊 [Shodan 賬號](https://account.shodan.io/)
2. 購買會員
3. 訪問 [Account](https://account.shodan.io/) 頁面查看 API Key

**配置**:
```bash
export SHODAN_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

### 5. VirusTotal API

**免費額度**: 每天 500 次查詢

**獲取步驟**:
1. 註冊 [VirusTotal](https://www.virustotal.com/gui/join-us)
2. 登錄後訪問 [API Key](https://www.virustotal.com/gui/user/your-username/apikey)
3. 複製 API Key

**配置**:
```bash
export VIRUSTOTAL_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

### 6. AbuseIPDB API

**免費額度**: 每天 1,000 次查詢

**獲取步驟**:
1. 註冊 [AbuseIPDB](https://www.abuseipdb.com/register)
2. 驗證郵箱
3. 訪問 [API](https://www.abuseipdb.com/account/api) 頁面
4. 創建 API Key

**配置**:
```bash
export ABUSEIPDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## DuckDuckGo (無需 API Key)

DuckDuckGo Instant Answer API 是完全免費的，無需註冊或 API Key。

---

## 配置方法

### 方法 1: 環境變量 (.env 文件)

創建 `.env` 文件：

```bash
# 複製示例文件
cp .env.example .env

# 編輯並填入你的 API Keys
nano .env
```

### 方法 2: 系統環境變量

**Windows (PowerShell)**:
```powershell
$env:GOOGLE_API_KEY="your_key_here"
$env:GITHUB_TOKEN="your_token_here"
```

**Linux/macOS**:
```bash
export GOOGLE_API_KEY="your_key_here"
export GITHUB_TOKEN="your_token_here"
```

### 方法 3: 配置文件

編輯 `config/search_config.yaml`:

```yaml
search:
  google:
    api_key: "your_google_api_key"
    search_engine_id: "your_search_engine_id"
  
  github:
    token: "your_github_token"
  
  nvd:
    api_key: "your_nvd_api_key"
```

---

## 測試配置

運行測試腳本驗證配置:

```bash
python test_command_optimization.py
```

成功輸出示例:
```
✅ Google API Key: 已配置
✅ GitHub Token: 已配置
✅ NVD API Key: 已配置
⚠️  Shodan API Key: 未配置（可選）
⚠️  VirusTotal API Key: 未配置（可選）
```

---

## 安全建議

1. **永遠不要**將 API Keys 提交到 Git 倉庫
2. 將 `.env` 添加到 `.gitignore`
3. 定期輪換 API Keys
4. 為不同環境使用不同的 Keys（開發/測試/生產）
5. 限制 API Key 的權限範圍
6. 監控 API 使用量，防止濫用

---

## API 使用限制對照表

| API 服務 | 免費額度 | 付費計劃 | 是否必需 |
|---------|---------|---------|---------|
| Google Custom Search | 100次/天 | $5/1000次 | ⭐ 推薦 |
| GitHub | 5000次/小時 | - | ⭐ 推薦 |
| NVD | 50次/30秒 | - | ⭐ 推薦 |
| DuckDuckGo | 無限制 | - | ✅ 免費 |
| Shodan | - | $59/月 | ⚪ 可選 |
| VirusTotal | 500次/天 | $100+/月 | ⚪ 可選 |
| AbuseIPDB | 1000次/天 | $20+/月 | ⚪ 可選 |

---

## 常見問題

### Q: 不配置 API Key 會怎樣？
A: 系統會自動跳過需要 API Key 的搜索，使用 DuckDuckGo 等免費替代方案。

### Q: API Key 洩露了怎麼辦？
A: 立即在對應平台撤銷舊 Key，生成新 Key。

### Q: 如何查看 API 使用量？
A: 訪問各個平台的控制台查看配額使用情況。

### Q: 可以使用代理嗎？
A: 可以，在配置中添加 `HTTP_PROXY` 和 `HTTPS_PROXY` 環境變量。

---

## 相關資源

- [Google Custom Search 文檔](https://developers.google.com/custom-search/v1/overview)
- [GitHub API 文檔](https://docs.github.com/en/rest)
- [NVD API 文檔](https://nvd.nist.gov/developers)
- [DuckDuckGo API 文檔](https://duckduckgo.com/api)
- [Shodan API 文檔](https://developer.shodan.io/)
- [VirusTotal API 文檔](https://developers.virustotal.com/reference/overview)
- [AbuseIPDB API 文檔](https://docs.abuseipdb.com/)
