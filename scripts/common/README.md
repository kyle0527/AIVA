# 🔗 Common - 通用服務腳本

> **版本**: v2.0  
> **更新日期**: 2026年1月12日  
> **檔案數量**: 10 個腳本  
> **架構更新**: 移除多餘 FastAPI 服務，簡化為單一入口點

---

## 📋 目錄概述

Common 目錄包含 AIVA 系統的通用服務工具，包括系統啟動、CLI 介面、驗證工具等基礎設施腳本。

---

## 📂 腳本說明

### 🚀 啟動腳本

| 腳本 | 功能說明 |
|------|----------|
| `start_ai_service.py` | ✅ 啟動 Core API 服務（系統唯一入口點） |
| `start_ai_simple.py` | 簡化版 AI 服務啟動器 |
| `run_aiva_cli.bat` | Windows 批次檔啟動 AIVA CLI |
| `run_aiva_cli.sh` | Linux/Mac shell 啟動 AIVA CLI |
| `run_capability_cli.bat` | Windows 批次檔啟動能力 CLI |
| `run_capability_cli.sh` | Linux/Mac shell 啟動能力 CLI |

### 🖥️ CLI 介面工具

| 腳本 | 功能說明 |
|------|----------|
| `aiva_cli.py` | AIVA 主要命令列介面 |
| `aiva_ai_menu.py` | AI 功能選單介面 |

### ✅ 驗證工具

| 腳本 | 功能說明 |
|------|----------|
| `validate_coordinator_drives_engines.py` | 驗證協調器驅動引擎的正確性 |
| `validate_scan_system.py` | 驗證掃描系統的完整性 |

---

## 🚀 使用方式

### 啟動 AI 服務

```bash
# Python 啟動
python start_ai_service.py

# Windows 批次檔
.\run_aiva_cli.bat

# Linux/Mac
./run_aiva_cli.sh
```

### 使用 CLI

```bash
# 啟動 AIVA CLI
python aiva_cli.py

# 啟動 AI 選單
python aiva_ai_menu.py
```

### 執行驗證

```bash
# 驗證協調器
python validate_coordinator_drives_engines.py

# 驗證掃描系統
python validate_scan_system.py
```

---

## 📝 注意事項

- 啟動腳本需確保相關服務依賴已安裝
- CLI 工具支援互動式操作
- 驗證工具建議在系統更新後執行
