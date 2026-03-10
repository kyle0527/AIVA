# AIVA Dashboard 安裝與使用指南

> **版本**: 1.0  
> **日期**: 2026-02-13  
> **狀態**: Phase 1 完成 - 基礎功能可用

---

## 📋 目錄

1. [系統需求](#系統需求)
2. [安裝步驟](#安裝步驟)
3. [啟動方式](#啟動方式)
4. [功能概覽](#功能概覽)
5. [常見問題](#常見問題)
6. [開發狀態](#開發狀態)

---

## 系統需求

### 必要組件
- **Python**: 3.11+
- **AIVA Core API**: 必須運行
- **套件依賴**: 見 `requirements-dashboard.txt`

### 可選組件（Phase 2/3 功能）
- **Plotly**: 圖表視覺化
- **Graphviz**: 攻擊路徑圖

---

## 安裝步驟

### 1. 安裝依賴套件

```bash
# 從專案根目錄執行
cd C:\D\fold7\AIVA-git

# 安裝 Dashboard 依賴
pip install -r requirements-dashboard.txt
```

**核心依賴清單**:
```
streamlit>=1.31.0
requests>=2.31.0
pandas>=2.1.0
pyyaml>=6.0.1
sse-starlette>=1.8.2
```

### 2. 檢查配置檔

配置檔位置：`config/dashboard_config.yaml`

預設配置（無需修改）:
```yaml
api:
  base_url: http://localhost:8000  # AIVA Core API 地址
  timeout: 30
  retry_attempts: 3

dashboard:
  title: AIVA Security Dashboard
  port: 8501
  theme: dark
```

---

## 啟動方式

### 方式 1：使用啟動腳本（推薦）

```bash
# 一鍵啟動 API + Dashboard
python scripts/start_dashboard.py
```

此腳本會：
1. 檢查依賴套件
2. 啟動 AIVA Core API (Port 8000)
3. 啟動 Streamlit Dashboard (Port 8501)
4. 自動進行健康檢查

### 方式 2：分別啟動

```bash
# Terminal 1: 啟動 API
cd services/core/aiva_core/service_backbone/api
python app.py

# Terminal 2: 啟動 Dashboard
streamlit run services/dashboard/streamlit_app.py
```

### 方式 3：手動指定參數

```bash
streamlit run services/dashboard/streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0
```

### 訪問 Dashboard

啟動後訪問：**http://localhost:8501**

---

## 功能概覽

### 🎯 掃描控制台
**路徑**: `pages/1_🎯_掃描控制台.py`

**功能**:
- ✅ 啟動新掃描（輸入目標 URL/IP）
- ✅ 配置掃描參數（類型、深度、超時）
- ✅ 查看當前執行中的掃描
- ✅ 停止掃描任務
- ✅ 快速統計（總數、執行中、已完成）

**使用步驟**:
1. 輸入目標 URL: `https://example.com`
2. 選擇掃描類型: 快速/標準/深度
3. 點擊「🚀 啟動掃描」
4. 記下返回的 `scan_id`
5. 前往「即時監控」查看進度

---

### 📊 即時監控
**路徑**: `pages/2_📊_即時監控.py`

**功能**:
- ✅ 選擇掃描任務
- ✅ 顯示掃描進度（百分比、階段）
- ✅ 顯示發現漏洞數
- ⚠️  日誌串流（SSE）- 需要 API 支援
- ⚠️  任務佇列狀態 - 待實作
- ⚠️  系統資源監控 - 待實作

**使用步驟**:
1. 從下拉選單選擇掃描任務
2. 查看即時進度和狀態
3. （可選）啟用「日誌串流」查看詳細日誌
4. 勾選「自動刷新」持續更新狀態

---

### 🔍 結果分析
**路徑**: `pages/3_🔍_結果分析.py`

**功能**:
- ✅ 選擇已完成的掃描
- ✅ 查看掃描概覽（漏洞數、資產數、風險評級）
- ✅ 漏洞清單（表格顯示）
- ✅ 漏洞詳情（JSON 格式）
- ⚠️  嚴重度圖表 - Phase 2 實作
- ⚠️  CVSS 分布圖 - Phase 2 實作
- ⚠️  攻擊路徑圖 - Phase 2 實作
- ⚠️  報告匯出 - Phase 3 實作

**使用步驟**:
1. 選擇已完成的掃描結果
2. 查看概覽統計
3. 瀏覽漏洞清單
4. 展開詳情查看具體資訊

---

### 📜 歷史記錄
**路徑**: `pages/4_📜_歷史記錄.py`

**功能**:
- ✅ 查詢歷史掃描列表
- ✅ 篩選（狀態、時間範圍、搜尋）
- ✅ 選擇掃描查看詳情
- ✅ 刪除掃描記錄
- ⚠️  結果對比 - Phase 3 實作
- ⚠️  批次清理 - Phase 3 實作

**使用步驟**:
1. 使用篩選器縮小範圍
2. 在表格中選擇掃描
3. 點擊「查看詳情」或「刪除記錄」

---

## 常見問題

### Q1: Dashboard 顯示「API 無法連線」

**原因**: AIVA Core API 未啟動或配置錯誤

**解決方案**:
```bash
# 1. 確認 API 是否運行
curl http://localhost:8000/health

# 2. 如果沒有運行，啟動 API
cd services/core/aiva_core/service_backbone/api
python app.py

# 3. 檢查配置檔
cat config/dashboard_config.yaml
# 確認 api.base_url 正確
```

---

### Q2: 日誌串流無法使用

**原因**: SSE 功能需要 `sse-starlette` 套件

**解決方案**:
```bash
# 檢查是否已安裝
pip show sse-starlette

# 如果未安裝
pip install sse-starlette>=1.8.2

# 重啟 API
python app.py
```

---

### Q3: 圖表功能顯示「尚未實作」

**原因**: Phase 2 功能，需要額外套件

**解決方案**:
```bash
# 安裝視覺化套件（Phase 2）
pip install plotly>=5.18.0
pip install graphviz>=0.20.1

# 等待 Phase 2 實作完成
```

---

### Q4: 掃描列表為空

**可能原因**:
1. SessionStateManager 未實作 `list_all_sessions()` 方法
2. 從未執行過掃描

**解決方案**:
```bash
# 1. 先啟動一次掃描
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"target": "https://example.com", "scan_type": "quick"}'

# 2. 檢查 SessionStateManager 實作
# 查看 services/core/aiva_core/service_backbone/state/session_state_manager.py
```

---

### Q5: 導入錯誤 `module 'services.dashboard' not found`

**原因**: Python 路徑問題

**解決方案**:
```bash
# 從專案根目錄啟動
cd C:\D\fold7\AIVA-git
streamlit run services/dashboard/streamlit_app.py

# 或設定 PYTHONPATH
set PYTHONPATH=C:\D\fold7\AIVA-git\services
streamlit run services/dashboard/streamlit_app.py
```

---

## 開發狀態

### ✅ Phase 1: 基礎設施（已完成）

**完成內容**:
- [x] SSE 端點處理器 (`sse.py`)
- [x] app.py API 端點擴展（6 個新端點）
- [x] Dashboard 目錄結構
- [x] API 客戶端封裝 (`api_client.py`)
- [x] Streamlit 主程式 (`streamlit_app.py`)
- [x] 配置管理 (`config.py`, `dashboard_config.yaml`)
- [x] 4 個主要頁面（掃描/監控/結果/歷史）

**檔案清單**:
```
services/dashboard/
├── __init__.py
├── streamlit_app.py          # 主程式
├── api_client.py              # API 客戶端
├── config.py                  # 配置管理
└── pages/
    ├── 1_🎯_掃描控制台.py
    ├── 2_📊_即時監控.py
    ├── 3_🔍_結果分析.py
    └── 4_📜_歷史記錄.py

services/core/aiva_core/service_backbone/api/
├── app.py                     # 擴展 API 端點
└── sse.py                     # SSE 串流處理

config/
└── dashboard_config.yaml      # Dashboard 配置

scripts/
└── start_dashboard.py         # 一鍵啟動腳本

requirements-dashboard.txt     # 依賴清單
```

---

### ⚠️ Phase 2: 視覺化功能（計劃中）

**待實作**:
- [ ] Plotly 圖表整合
  - 嚴重度圓餅圖
  - CVSS 分數分布直方圖
  - 時間序列圖表
- [ ] Graphviz 攻擊路徑圖
- [ ] 資料視覺化組件庫

**預估工期**: 1-2 天

---

### ⚠️ Phase 3: 進階功能（計劃中）

**待實作**:
- [ ] 報告匯出（JSON/HTML/PDF）
- [ ] 掃描結果對比
- [ ] 批次操作
- [ ] WebSocket 即時通訊
- [ ] 系統資源監控

**預估工期**: 2-3 天

---

## 已知限制

### 1. API 方法未實作

**影響功能**:
- 掃描列表為空（`SessionStateManager.list_all_sessions()` 未實作）
- 完整結果無法顯示（`SimpleDataManager.get_task_result()` 未實作）
- 刪除功能無效（刪除方法未實作）

**解決方案**: 需要在 `SessionStateManager` 和 `SimpleDataManager` 中實作對應方法

---

### 2. SSE 串流可能不穩定

**原因**: 日誌檔案讀取方式為同步 I/O

**建議**: Phase 2 改用異步檔案 API (`aiofiles`)

---

### 3. 缺少身份驗證

**風險**: Dashboard 無認證機制

**建議**: Phase 3 實作 API Token 認證

---

## 技術架構

### 整體架構圖

```
┌─────────────────────────────────────────────────┐
│              使用者界面層                         │
│                                                 │
│   Streamlit Dashboard (Port 8501)              │
│   ├─ 掃描控制台 (啟動/管理掃描)                   │
│   ├─ 即時監控 (進度/日誌)                        │
│   ├─ 結果分析 (漏洞/圖表)                        │
│   └─ 歷史記錄 (查詢/對比)                        │
└─────────────────────────────────────────────────┘
                      ↓ HTTP/SSE
┌─────────────────────────────────────────────────┐
│            FastAPI Core API (Port 8000)         │
│                                                 │
│   REST API:                                     │
│   ├─ POST /scan              - 啟動掃描          │
│   ├─ GET  /status/{id}       - 查詢狀態          │
│   ├─ GET  /api/scans         - 掃描列表          │
│   ├─ GET  /api/scans/{id}/results - 完整結果    │
│   ├─ POST /api/scans/{id}/stop - 停止掃描       │
│   └─ DELETE /api/scans/{id}  - 刪除記錄          │
│                                                 │
│   SSE Streaming:                                │
│   ├─ GET /api/logs/stream    - 日誌串流          │
│   └─ GET /api/status/stream  - 狀態串流          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│               AIVA Core Engine                  │
│                                                 │
│   ├─ EnhancedDecisionAgent (AI 大腦)            │
│   ├─ CommanderCoordinator (指揮官)              │
│   ├─ UnifiedExecutor (執行器)                   │
│   ├─ SessionStateManager (狀態管理)             │
│   └─ SimpleDataManager (資料管理)               │
└─────────────────────────────────────────────────┘
```

---

## 下一步計劃

### 短期（1 週內）
1. ✅ Phase 1 基礎功能（已完成）
2. ⏳ 修復 SessionStateManager 方法缺失
3. ⏳ 實作完整的日誌讀取
4. ⏳ 測試 SSE 串流穩定性

### 中期（2-3 週）
1. Phase 2 視覺化功能
2. Phase 3 進階功能
3. 效能優化與測試
4. 文檔完善

### 長期（1 個月+）
1. 使用者認證與授權
2. 多租戶支援
3. 國際化（i18n）
4. Docker 容器化部署

---

## 相關文件

- [實施計劃](docs/UI_DASHBOARD_IMPLEMENTATION_PLAN.md) - 完整技術規劃
- [AIVA Common README](services/aiva_common/README.md) - API 規範
- [使用手冊](guides/user_manuals/) - 系統操作指南

---

## 貢獻指南

如需貢獻代碼或回報問題：

1. Fork 本專案
2. 創建特性分支 (`git checkout -b feature/dashboard-enhancement`)
3. 提交變更 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/dashboard-enhancement`)
5. 創建 Pull Request

---

**文件版本**: 1.0  
**最後更新**: 2026-02-13  
**維護者**: AIVA Team
