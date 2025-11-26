---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 專案快速參考 (2025-10-28 更新)

---

## 📑 目錄

- [🚀 快速啟動](#快速啟動)
  - [方式一: 離線模式 (推薦)](#方式一-離線模式-推薦)
  - [方式二: 完整環境](#方式二-完整環境)
- [📂 核心目錄](#核心目錄)
- [🔧 關鍵檔案](#關鍵檔案)
  - [啟動器與環境](#啟動器與環境)
  - [AI 實戰工具 (新增)](#ai-實戰工具-新增)
  - [環境配置](#環境配置)
  - [AI 核心](#ai-核心)
  - [學習數據](#學習數據)
- [🎯 模組功能](#模組功能)
- [📋 常用命令](#常用命令)

---
---
---
---

## 🚀 快速啟動

### 方式一: 離線模式 (推薦)
```bash
# 一鍵啟動離線環境
python launch_offline_mode.py

# 系統健康檢查
python health_check.py

# AI 實戰安全測試
python ai_security_test.py --target http://localhost:3000

# AI 自主學習測試
python ai_autonomous_testing_loop.py --target http://localhost:3000
```

### 方式二: 完整環境
```bash
# Docker 環境啟動
cd docker && docker compose up -d

# 環境自動修復
python fix_environment_dependencies.py

# 統一啟動介面
python scripts/launcher/aiva_launcher.py

# API 服務
python api/start_api.py
```

## 📂 核心目錄

| 目錄 | 用途 | 主要檔案 |
|------|------|----------|
| `services/aiva_common/` | 通用基礎模組 | 共享結構、工具函數 |
| `services/core/aiva_core/` | 核心業務邏輯 | AI 引擎、決策系統 |
| `services/scan/aiva_scan/` | 掃描檢測 | 漏洞掃描、環境檢測 |
| `services/integration/` | 整合服務 | API 閘道、監控系統 |
| `services/features/` | 功能檢測 | XSS、SQLi、IDOR 等 |
| `api/` | API 服務 | FastAPI 後端 |
| `scripts/launcher/` | 啟動腳本 | 統一啟動介面 |

## 🔧 關鍵檔案

### 啟動器與環境
- `launch_offline_mode.py` - 離線模式啟動器 (推薦)
- `fix_offline_dependencies.py` - 離線環境修復
- `fix_environment_dependencies.py` - 完整環境修復
- `health_check.py` - 系統健康檢查

### AI 實戰工具 (新增)
- `ai_security_test.py` - AI 實戰安全測試
- `ai_autonomous_testing_loop.py` - AI 自主學習循環
- `ai_component_explorer.py` - AI 組件探索
- `ai_system_explorer_v3.py` - 系統自我分析

### 環境配置
- `.env` - 環境變數配置 (自動生成)
- `services/aiva_common/config/unified_config.py` - 統一配置 (已修補)

### AI 核心
- `services/core/aiva_core/bio_neuron_master.py` - BioNeuron 主控
- `services/core/aiva_core/ai_engine/anti_hallucination_module.py` - 抗幻覺

### 學習數據
- `reports/ai_diagnostics/exploration.db` - 學習數據庫 (58.9MB)
- `reports/ai_diagnostics/` - AI 診斷報告目錄

## 🎯 模組功能

| 模組 | 主要功能 |
|------|----------|
| **aiva_common** | 基礎工具、共享結構 |
| **core** | AI 引擎、決策代理、BioNeuron |
| **scan** | 漏洞掃描、環境檢測、指紋識別 |
| **integration** | API 閘道、監控、報告生成 |
| **features** | 漏洞檢測功能實現 |

## 📋 常用命令

```bash
# 檢查環境
python aiva_package_validator.py

# 啟動完整系統
python scripts/launcher/aiva_launcher.py

# 單獨測試 API
python api/test_api.py
```