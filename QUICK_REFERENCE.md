# AIVA 專案快速參考

## 🚀 快速啟動
```bash
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

### 啟動器
- `scripts/launcher/aiva_launcher.py` - 主啟動器
- `api/start_api.py` - API 服務啟動

### AI 核心
- `services/core/aiva_core/bio_neuron_master.py` - BioNeuron 主控
- `services/core/aiva_core/ai_engine/anti_hallucination_module.py` - 抗幻覺

### 整合服務
- `services/integration/aiva_integration/trigger_ai_continuous_learning.py` - AI 學習
- `services/integration/aiva_integration/integrated_ai_trainer.py` - AI 訓練

### 檢測功能
- `services/features/smart_detection_manager.py` - 檢測管理
- `services/features/high_value_manager.py` - 高價值管理

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