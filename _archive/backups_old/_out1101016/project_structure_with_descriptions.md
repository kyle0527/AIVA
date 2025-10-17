# AIVA 專案結構說明

生成時間: 2025-10-16 22:54:00

## 📁 目錄結構

### 🏗️ 核心服務模組 (services/)

#### aiva_common/ - 共用元件庫
- `enums/` - 列舉定義（模組、安全、資產等）
- `schemas/` - 資料結構定義（訊息、任務、遙測等）
- `utils/` - 工具函數（去重、網路、日誌等）

#### core/aiva_core/ - 核心引擎
- `ai_engine/` - AI 引擎（生物神經元、知識庫）
- `messaging/` - 訊息處理（任務分發、結果收集）
- `planner/` - 任務規劃器
- `execution/` - 任務執行引擎
- `storage/` - 儲存管理
- `rag/` - RAG 檢索增強

#### scan/aiva_scan/ - 掃描引擎
- `core_crawling_engine/` - 核心爬蟲
- `dynamic_engine/` - 動態內容掃描
- `info_gatherer/` - 資訊收集

#### function/ - 檢測功能模組
- `function_sqli/` - SQL 注入檢測（Python）
- `function_xss/` - XSS 檢測（Python）
- `function_ssrf/` - SSRF 檢測（Python + Go）
- `function_idor/` - IDOR 檢測（Python）
- `function_sca_go/` - 軟體組成分析（Go）
- `function_sast_rust/` - 靜態分析（Rust）
- `function_cspm_go/` - 雲端安全（Go）

#### integration/aiva_integration/ - 整合服務
- `reporting/` - 報告生成
- `analysis/` - 風險分析
- `remediation/` - 修復建議
- `attack_path_analyzer/` - 攻擊路徑分析

### 🛠️ 工具與腳本

#### tools/ - 分析工具
- `analyze_codebase.py` - 程式碼分析工具
- `generate_complete_architecture.py` - 架構圖生成

#### scripts/ - 維護腳本
- `maintenance/` - 維護腳本（樹狀圖生成等）
- `setup/` - 安裝設定腳本
- `deployment/` - 部署腳本

### 📊 文件與範例

#### docs/ - 文件
- `ARCHITECTURE/` - 架構文件
- `DEPLOYMENT/` - 部署文件
- `DEVELOPMENT/` - 開發文件

#### examples/ - 範例程式
- 各種功能示範程式

### 🧪 測試

#### tests/ - 測試程式
- 整合測試、單元測試

## 🌍 多語言架構

- **Python**: 核心邏輯、AI 引擎（273 檔案，63,981 行）
- **Go**: 高效能功能模組（18 檔案，3,065 行）
- **Rust**: 安全關鍵模組（10 檔案，1,552 行）
- **TypeScript**: 動態掃描引擎（8 檔案，1,872 行）

## 🔗 通訊架構

- **訊息佇列**: RabbitMQ
- **交換器類型**: TOPIC
- **訊息格式**: AivaMessage + FunctionTaskPayload

## 📈 專案統計

- **總檔案數**: 309
- **總程式碼行數**: 70,470
- **平均複雜度**: 12.73
- **類型覆蓋率**: 72.9%
- **文檔覆蓋率**: 90.8%

## 🎯 關鍵特性

1. **多語言協同**: Python/Go/Rust/TypeScript 混合架構
2. **分散式架構**: RabbitMQ 訊息驅動
3. **AI 驅動**: 生物神經元核心 + RAG 增強
4. **模組化設計**: Core/Scan/Function/Integration 四大模組
5. **智慧檢測**: 每個檢測引擎都具備智慧決策能力
