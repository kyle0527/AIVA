# AIVA BioNeuronRAGAgent API 服務

## 📑 目錄

- [簡介](#簡介)
- [功能特色](#功能特色)
- [快速開始](#快速開始)
  - [1. 安裝依賴](#1-安裝依賴)
  - [2. 啟動服務](#2-啟動服務)
  - [3. 查看 API 文件](#3-查看-api-文件)
- [API 端點](#api-端點)
  - [基本端點](#基本端點)
  - [核心端點](#核心端點)
- [使用範例](#使用範例)
  - [1. 程式碼分析](#1-程式碼分析)
  - [2. 漏洞掃描](#2-漏洞掃描)
  - [3. 系統命令執行](#3-系統命令執行)
  - [4. 檔案讀取](#4-檔案讀取)
  - [5. 程式碼寫入](#5-程式碼寫入)
- [回應格式](#回應格式)
  - [成功回應範例](#成功回應範例)
  - [錯誤回應範例](#錯誤回應範例)
- [參數說明](#參數說明)
  - [InvokeRequest 參數](#invokerequest-參數)
- [注意事項](#注意事項)
- [進階功能](#進階功能)
  - [知識庫統計](#知識庫統計)
  - [執行歷史](#執行歷史)
  - [健康檢查](#健康檢查)
- [疑難排解](#疑難排解)
  - [常見問題](#常見問題)
  - [日誌查看](#日誌查看)
- [開發建議](#開發建議)

---

## 簡介

將原本的 `demo_bio_neuron_agent.py` 示範程式改寫為實際的 RESTful API 服務，提供 AIVA 核心 AI 代理功能。

## 功能特色

- **程式碼分析**: 分析程式碼結構、函數、類別等
- **檔案操作**: 讀取、寫入程式碼檔案
- **漏洞掃描**: 觸發安全掃描作業
- **系統命令**: 執行系統命令並取得結果
- **知識庫查詢**: 檢索程式碼知識庫
- **執行歷史**: 追蹤 AI 代理的操作記錄

## 快速開始

### 1. 安裝依賴
```bash
# 確保所有依賴已安裝 (FastAPI, uvicorn 等)
pip install -r requirements.txt
```

### 2. 啟動服務
```bash
# 方法 1: 直接執行
python examples/demo_bio_neuron_agent.py

# 方法 2: 使用 uvicorn
uvicorn examples.demo_bio_neuron_agent:app --reload --host 127.0.0.1 --port 8000
```

### 3. 查看 API 文件
啟動後，可在以下位置查看 API 文件：
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## API 端點

### 基本端點

- `GET /` - 服務狀態檢查
- `GET /health` - 健康檢查
- `GET /stats` - 知識庫統計資訊
- `GET /history` - 執行歷史

### 核心端點

- `POST /invoke` - 呼叫 AI 代理執行任務

## 使用範例

### 1. 程式碼分析
```bash
curl -X POST "http://127.0.0.1:8000/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "分析核心模組的應用程式結構",
    "path": "services/core/aiva_core/app.py"
  }'
```

### 2. 漏洞掃描
```bash
curl -X POST "http://127.0.0.1:8000/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "對目標網站執行完整的安全掃描",
    "target_url": "https://example.com",
    "scan_type": "full"
  }'
```

### 3. 系統命令執行
```bash
curl -X POST "http://127.0.0.1:8000/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "檢查 Python 版本",
    "command": "python --version"
  }'
```

### 4. 檔案讀取
```bash
curl -X POST "http://127.0.0.1:8000/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "讀取掃描器的主要入口檔案",
    "path": "services/scan/aiva_scan/worker.py"
  }'
```

### 5. 程式碼寫入
```bash
curl -X POST "http://127.0.0.1:8000/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "建立一個簡單的測試 Python 檔案",
    "path": "test_api_generated.py",
    "content": "def hello():\n    print(\"Hello from API!\")\n\nif __name__ == \"__main__\":\n    hello()"
  }'
```

## 回應格式

### 成功回應範例
```json
{
  "status": "success",
  "tool_used": "code_analyzer",
  "confidence": 0.95,
  "tool_result": {
    "status": "success",
    "total_lines": 150,
    "functions": 8,
    "classes": 3
  }
}
```

### 錯誤回應範例
```json
{
  "status": "error",
  "message": "檔案不存在或無法讀取",
  "confidence": 0.0
}
```

## 參數說明

### InvokeRequest 參數

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `query` | string | ✅ | 要執行的查詢或指令 |
| `path` | string | ❌ | 檔案路徑 (用於程式碼讀取/寫入/分析) |
| `target_url` | string | ❌ | 目標 URL (用於掃描) |
| `scan_type` | string | ❌ | 掃描類型 (如: full, quick) |
| `command` | string | ❌ | 系統命令 (用於命令執行) |
| `content` | string | ❌ | 檔案內容 (用於檔案寫入) |

## 注意事項

1. **初始化時間**: 首次啟動時 BioNeuronRAGAgent 需要索引整個程式碼庫，可能需要幾分鐘
2. **路徑設定**: 確保程式碼庫路徑 (`codebase_path`) 設定正確
3. **權限管理**: 檔案寫入和命令執行功能需要適當的系統權限
4. **安全考量**: 在生產環境中應該加入適當的驗證和授權機制

## 進階功能

### 知識庫統計
```bash
curl "http://127.0.0.1:8000/stats"
```

### 執行歷史
```bash
curl "http://127.0.0.1:8000/history"
```

### 健康檢查
```bash
curl "http://127.0.0.1:8000/health"
```

## 疑難排解

### 常見問題

1. **服務啟動失敗**: 檢查依賴是否已安裝 (`pip install -r requirements.txt`)
2. **代理初始化失敗**: 檢查程式碼庫路徑是否正確
3. **工具執行失敗**: 檢查系統權限和環境設定

### 日誌查看
服務會輸出詳細的日誌資訊，包括：
- 🚀 服務啟動狀態
- 🔍 請求處理過程
- ✅ 執行成功結果
- ❌ 錯誤訊息和堆疊追蹤

## 開發建議

1. 使用 `--reload` 參數啟動服務以支援熱重載
2. 檢查 Swagger UI 文件了解完整的 API 規格
3. 使用健康檢查端點監控服務狀態
4. 定期查看執行歷史了解 AI 代理的行為模式