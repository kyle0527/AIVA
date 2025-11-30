# 🚀 AIVA 快速使用指南

> **v2.1.1 架構：AI 驅動的動態調用系統**

## 🎯 核心概念

AIVA v2.1.1 採用**分層架構**，不需要預先啟動所有模組：

```
Layer 2: 功能模組 (AI 按需動態調用)
         ↓
Layer 1: AI 核心 (持續運作)
         ↓  
Layer 0: 基礎設施 (PostgreSQL、Redis)
```

## 📦 正式啟動方式

### 🐳 方式一：Docker 部署 (推薦)

#### 1. 啟動基礎設施
```bash
# 啟動 PostgreSQL 和 Redis
docker-compose -f docker/compose/docker-compose.yml up -d postgres redis

# 檢查服務狀態
docker-compose ps
```

#### 2. 啟動 AI 核心
```bash
# 方式 A: Docker 啟動 (生產環境)
docker-compose -f docker/compose/docker-compose.yml up -d aiva-core

# 方式 B: 本地啟動 (開發環境)
python scripts/startup/start_ai_service.py --mode api --port 8000
```

### 🖥️ 方式二：Windows 一鍵啟動
```batch
# 雙擊啟動 (自動啟動 API 服務)
.\啟動AI服務.bat

# 這會啟動持續運作的 API 服務
# 訪問: http://localhost:8000
```

### 🔧 方式三：Linux/macOS 腳本
```bash
# 啟動核心服務
./scripts/startup/start-aiva.sh core

# 查看服務狀態
./scripts/startup/start-aiva.sh status
```

## 🔄 AI 核心的四種運作模式

### 1. 📡 API 服務模式 (預設)
```bash
python scripts/startup/start_ai_service.py --mode api
```
- **用途**: 提供 REST API 服務
- **持續運作**: 是
- **訪問**: http://localhost:8000
- **文檔**: http://localhost:8000/docs

### 2. 🔍 監控模式 (自動掃描)
```bash
python scripts/startup/start_ai_service.py --mode monitor
```
- **用途**: 後台自動掃描指定目標
- **持續運作**: 是
- **掃描間隔**: 1小時 (可配置)
- **適合**: 持續安全監控

### 3. 💬 互動模式 (命令列)
```bash
python scripts/startup/start_ai_service.py --mode interactive
```
- **用途**: 命令列互動式控制台
- **可用命令**:
  - `scan <url>` - 掃描網站
  - `status` - 查看系統狀態
  - `help` - 顯示幫助
  - `quit` - 退出

### 4. 🤖 守護進程模式 (背景執行)
```bash
python scripts/startup/start_ai_service.py --mode daemon
```
- **用途**: 後台守護進程
- **持續運作**: 是
- **無終端輸出**: 適合系統服務

## 💡 核心優勢：AI 動態調用

### ❌ 舊架構 (v1.x)
```
需要預先啟動所有模組:
├── Scan Engine    (佔用 200MB)
├── Attack Module  (佔用 150MB)
├── Feature Tests  (佔用 180MB)
├── Integration    (佔用 120MB)
└── ... 更多模組   (佔用 300MB+)

總計: 2-4 GB 記憶體
啟動時間: 30-60秒
```

### ✅ 新架構 (v2.1.1)
```
只啟動 AI 核心:
├── PostgreSQL     (基礎設施)
├── Redis          (快取)
└── AI Core        (核心大腦)

AI 根據需求動態調用功能模組
├─→ 需要掃描時 → 調用 Scan Module
├─→ 需要攻擊時 → 調用 Attack Module
└─→ 需要報告時 → 調用 Report Module

總計: 500-800 MB 記憶體
啟動時間: 5-10秒
```

### 📊 優勢對比

| 特性 | 預先全部啟動 | AI 動態調用 (v2.1.1) |
|------|------------|---------------------|
| 記憶體消耗 | 2-4 GB | 500-800 MB |
| 啟動時間 | 30-60秒 | 5-10秒 |
| 資源利用 | 低 (閒置浪費) | 高 (按需使用) |
| 維護複雜度 | 高 (多進程) | 低 (單一核心) |
| 擴展性 | 困難 | 簡單 (新增即可用) |

## 🎯 實際使用範例

### 範例 1: 基本 API 模式
```bash
# 1. 啟動基礎設施
docker-compose -f docker/compose/docker-compose.yml up -d postgres redis

# 2. 啟動 AI 核心
python scripts/startup/start_ai_service.py --mode api

# 3. 等待啟動 (約 10 秒)
# 看到 "Uvicorn running on http://localhost:8000"

# 4. 測試 API
curl http://localhost:8000/health

# 5. 查看 API 文檔
# 訪問: http://localhost:8000/docs
```

### 範例 2: 自動監控模式
```bash
# 啟動自動監控 (每小時掃描一次)
python scripts/startup/start_ai_service.py --mode monitor

# AI 會自動:
# - 每小時掃描指定目標
# - 發現漏洞自動記錄
# - 持續運作無需人工干預
```

### 範例 3: 互動式控制台
```bash
# 啟動互動模式
python scripts/startup/start_ai_service.py --mode interactive

# 在控制台輸入命令:
AIVA> scan https://testphp.vulnweb.com
AIVA> status
AIVA> engines
AIVA> quit
```

### 範例 4: CLI 查詢
```bash
# 查詢可用能力
python aiva_cli.py --query "SQL 注入"

# 查看系統統計
python aiva_cli.py --stats

# 同步能力到 RAG 知識庫
python aiva_cli.py --sync
```

## 🔧 服務管理

### 🟢 啟動服務
```bash
# Docker 方式 (推薦)
docker-compose -f docker/compose/docker-compose.yml up -d postgres redis

# 本地方式
python scripts/startup/start_ai_service.py --mode api
```

### 🔄 健康檢查
```bash
# 檢查 Docker 服務
docker-compose -f docker/compose/docker-compose.yml ps

# 檢查 AI 核心健康
curl http://localhost:8000/health

# 檢視即時日誌
docker-compose -f docker/compose/docker-compose.yml logs -f
```

### 🔴 停止服務
```bash
# 停止 Docker 服務
docker-compose -f docker/compose/docker-compose.yml down

# 停止本地服務 (按 Ctrl+C)
```

## ⚠️ 常見問題

### Q: 端口已被占用？
```powershell
# Windows 檢查端口
netstat -ano | findstr :8000

# 更換端口啟動
python scripts/startup/start_ai_service.py --mode api --port 8001
```

### Q: PostgreSQL 連接失敗？
```bash
# 檢查 PostgreSQL 是否運行
docker ps | grep postgres

# 重啟 PostgreSQL
docker-compose -f docker/compose/docker-compose.yml restart postgres
```

### Q: Redis 連接失敗？
```bash
# 檢查 Redis 狀態
docker exec -it redis redis-cli ping

# 應該返回: PONG
```

### Q: 需要完整部署 (包含所有組件)？
```bash
# 使用完整 Docker Compose (包含 22 個功能組件)
docker-compose -f docker/docker-compose.complete.yml up -d

# 或使用 Kubernetes 部署
# 參考: reports/architecture/DOCKER_KUBERNETES_GUIDE.md
```

## 📚 相關文檔

- **完整部署指南**: `reports/architecture/DOCKER_KUBERNETES_GUIDE.md`
- **AI 啟動策略**: `reports/architecture/AI_STARTUP_AND_DIAGNOSTIC_CLARIFICATION.md`
- **構建指南**: `reports/architecture/BUILD_GUIDE.md`
- **依賴管理**: `reports/architecture/DEPENDENCY_MANAGEMENT_GUIDE.md`

---

## 🎯 重點總結

1. **最小啟動**: 只需 PostgreSQL + Redis + AI Core
2. **持續運作**: AI 核心提供 REST API 服務
3. **動態調用**: 功能模組由 AI 按需調用，無需預先啟動
4. **四種模式**: API、監控、互動、守護進程
5. **資源效率**: 記憶體消耗僅 500-800 MB

**版本**: AIVA v2.1.1  
**更新日期**: 2025-11-29  
**架構**: AI 驅動的動態調用系統