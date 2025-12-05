# 🚀 AIVA 快速入門指南

> **適用對象**: 所有 AIVA 使用者  
> **預計時間**: 10 分鐘  
> **最後驗證**: 2025-11-29

---

## 📑 目錄

1. [什麼是 AIVA](#什麼是-aiva)
2. [快速啟動](#快速啟動)
3. [使用方式](#使用方式)
4. [運作模式比較](#運作模式比較)
5. [常見問題](#常見問題)
6. [下一步](#下一步)

---

## 什麼是 AIVA

AIVA (AI-driven Vulnerability Assessment) 是一個 AI 驅動的智能漏洞評估平台，整合了：

- 🤖 **782 個能力模組** (已驗證)
- 🔍 **智能掃描** - 自動識別漏洞
- 🎯 **攻擊路徑分析** - AI 推薦測試流程
- 📊 **持續監控** - 自動化安全評估

**系統架構** (v2.1.1):
```
Layer 0: 基礎設施
  ├── PostgreSQL (資料庫)
  └── Redis (快取)

Layer 1: AI Core
  └── 動態能力調用中心

Layer 2: 功能模組
  ├── Scan (掃描: 286個能力, 36.6%)
  ├── Core (核心: 207個能力, 26.5%)
  ├── Integration (整合: 111個能力, 14.2%)
  └── Features (功能: 98個能力, 12.5%)
```

---

## 快速啟動

### Windows 用戶 (最簡單)

**方式 1: 一鍵啟動** ⭐ 推薦
```batch
# 雙擊此檔案即可
啟動AI服務.bat
```

啟動後會看到:
```
========================================
  AIVA AI Service - API Mode
========================================

Starting AIVA AI Service...
API will be available at: http://localhost:8000
API Documentation: http://localhost:8000/docs

Press Ctrl+C to stop the service
```

**方式 2: 命令行啟動**
```powershell
# 進入專案目錄
cd C:\D\fold7\AIVA-git

# 啟動 API 服務（推薦）
python scripts/startup/start_ai_service.py --mode api

# 或啟動交互式模式
python scripts/startup/start_ai_service.py --mode interactive
```

### Linux / macOS 用戶

**方式 1: Docker Compose** ⭐ 推薦
```bash
# 啟動所有服務
./scripts/startup/start-aiva.sh core

# 重新構建並啟動
./scripts/startup/start-aiva.sh --build

# 查看服務狀態
docker-compose ps
```

**方式 2: 直接運行**
```bash
# 啟動 API 服務
python scripts/startup/start_ai_service.py --mode api
```

---

## 使用方式

### 1. CLI 命令行介面 (推薦新手)

#### 啟動交互式選單
```bash
python aiva_cli.py
```

會顯示:
```
╔══════════════════════════════════════════════════════════════╗
║                     AIVA CLI 主選單                          ║
╚══════════════════════════════════════════════════════════════╝

[1] 快速查詢能力      - 輸入問題查找相關功能
[2] 查看系統統計      - 顯示 782 個能力的分布
[3] 獲取工作流推薦    - AI 推薦任務執行步驟
[4] 同步能力資料      - 更新 RAG 知識庫
[5] 運行測試驗證      - 驗證 AI 分析能力
[0] 退出

選擇功能 >
```

#### 直接執行命令

**查詢能力** (已驗證)
```bash
# 查詢攻擊相關功能
python aiva_cli.py --query "攻擊工具"

# 查詢掃描功能
python aiva_cli.py --query "掃描"
```

**AI 執行攻擊** (主要功能)
```bash
# 讓 AI 自動執行掃描
python aiva_cli.py --attack "幫我掃描 http://localhost:3000"

# SQL 注入測試
python aiva_cli.py --attack "對 http://example.com 執行 SQL 注入測試"
```

**查看系統統計** (已驗證)
```bash
python aiva_cli.py --stats
```

實際輸出:
```
        模組分布 (Top 10)        

  模組             數量    佔比 
 ───────────────────────────────
  scan              286   36.6%
  core/aiva_core    207   26.5%
  integration       111   14.2%
  features           98   12.5%

╭─────────────── 系統摘要 ───────────────╮
│ 總計: 782 個能力                       │
│ 模組數: 16                             │
│ 語言數: 4 (Python, Rust, TS, Go)      │
╰────────────────────────────────────────╯
```

**獲取工作流推薦**
```bash
python aiva_cli.py --workflow "web 應用滲透測試"
```

### 2. API 服務模式

#### 啟動 API 服務
```bash
# 方式 1: 使用 BAT 檔案 (Windows)
啟動AI服務.bat

# 方式 2: 命令行
python scripts/startup/start_ai_service.py --mode api --port 8000
```

#### 訪問 API
- **API 基礎地址**: http://localhost:8000
- **API 文檔**: http://localhost:8000/docs
- **健康檢查**: http://localhost:8000/health

#### 預設帳號
```
Admin 帳號:
  用戶名: admin
  密碼: aiva-admin-2025

User 帳號:
  用戶名: user
  密碼: aiva-user-2025
```

#### API 使用範例
```bash
# 健康檢查
curl http://localhost:8000/health

# 執行掃描 (需要認證)
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"target": "http://localhost:3000"}'
```

### 3. 監控模式 (自動化)

```bash
# 啟動後台監控，每小時自動掃描
python scripts/startup/start_ai_service.py \
  --mode monitor \
  --targets http://localhost:3000 http://example.com \
  --interval 3600
```

**功能**:
- ✅ 自動定期掃描
- ✅ 異常自動告警
- ✅ 結果自動儲存

### 4. 守護進程模式 (推薦生產環境)

```bash
# 同時提供 API 和監控功能
python scripts/startup/start_ai_service.py \
  --mode daemon \
  --port 8000 \
  --targets http://localhost:3000 \
  --interval 1800
```

**適用場景**:
- 🏢 生產環境部署
- 🔄 需要持續監控
- 🌐 對外提供 API 服務

---

## 運作模式比較

| 模式 | 適用場景 | 資源占用 | 難度 |
|------|---------|---------|-----|
| **CLI 交互式** | 學習、測試 | 低 | ⭐ 簡單 |
| **API 服務** | 團隊協作、整合 | 中 | ⭐⭐ 中等 |
| **監控模式** | 自動化、定期掃描 | 中 | ⭐⭐ 中等 |
| **守護進程** | 生產環境 | 高 | ⭐⭐⭐ 進階 |

---

## 常見問題

### Q1: 如何確認 AIVA 已啟動？

**A**: 檢查以下幾點:

```bash
# 1. 查看進程
# Windows
tasklist | findstr python

# Linux/macOS
ps aux | grep python

# 2. 測試 API (如果啟動了 API 模式)
curl http://localhost:8000/health

# 3. 查看日誌
# 日誌位置: logs/aiva_*.log
```

### Q2: 啟動失敗怎麼辦？

**A**: 按以下順序排查:

```bash
# 1. 檢查 Python 版本
python --version  # 需要 3.11+

# 2. 檢查依賴
pip list | grep fastapi
pip list | grep uvicorn

# 3. 檢查端口占用
# Windows
netstat -ano | findstr :8000

# Linux/macOS
lsof -i :8000

# 4. 查看錯誤日誌
cat logs/aiva_error.log
```

### Q3: 如何停止 AIVA？

**A**: 

```bash
# 方式 1: Ctrl+C (如果在前台運行)
按 Ctrl+C

# 方式 2: 殺掉進程
# Windows
taskkill /F /IM python.exe

# Linux/macOS
pkill -f start_ai_service.py

# 方式 3: Docker 環境
docker-compose down
```

### Q4: 782 個能力都是什麼？

**A**: 使用統計命令查看:

```bash
python aiva_cli.py --stats
```

能力分類:
- **Scan (286個, 36.6%)**: 掃描、探測、指紋識別
- **Core (207個, 26.5%)**: AI 分析、決策、學習
- **Integration (111個, 14.2%)**: 工具整合、資料處理
- **Features (98個, 12.5%)**: 攻擊模組、漏洞利用

### Q5: 需要什麼環境？

**A**: 基本需求:

```yaml
必需:
  - Python: 3.11+
  - 記憶體: 4GB+ (推薦 8GB)
  - 硬碟: 5GB 可用空間

可選 (Docker 部署):
  - Docker: 20.10+
  - Docker Compose: 2.0+

資料庫 (API/Daemon 模式):
  - PostgreSQL: 14+
  - Redis: 6+
```

### Q6: 如何更新能力資料？

**A**:

```bash
# 同步能力到 RAG 知識庫
python aiva_cli.py --sync

# 或在交互式選單選擇 [4]
python aiva_cli.py
> 選擇 4
```

### Q7: 日誌檔案在哪裡？

**A**:

```
logs/
├── aiva_api.log        # API 服務日誌
├── aiva_scan.log       # 掃描日誌
├── aiva_error.log      # 錯誤日誌
└── aiva_debug.log      # 調試日誌 (需開啟 DEBUG 模式)
```

查看即時日誌:
```bash
# Windows
Get-Content logs/aiva_api.log -Wait -Tail 50

# Linux/macOS
tail -f logs/aiva_api.log
```

---

## 下一步

✅ 已完成快速入門？繼續學習:

1. **深入 CLI 使用** → 查看 [CLI_COMPLETE_GUIDE.md](CLI_COMPLETE_GUIDE.md)
2. **API 整合** → 查看 [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)
3. **掃描功能** → 查看 [SCAN_USAGE_GUIDE.md](SCAN_USAGE_GUIDE.md)
4. **常見問題** → 查看 [FAQ.md](FAQ.md)

---

## 需要幫助？

- 📖 **文檔中心**: [docs/user-guides/README.md](README.md)
- 🐛 **問題回報**: GitHub Issues
- 💬 **社群討論**: GitHub Discussions

---

**最後更新**: 2025-11-29  
**驗證狀態**: ✅ 所有命令已實際測試
