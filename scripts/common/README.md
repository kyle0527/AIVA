# 🔗 Common Service Scripts

## 📑 目錄

- [📋 目錄概述](#-目錄概述)
- [🗂️ 目錄結構](#-目錄結構)
- [🚀 啟動器工具](#-啟動器工具)
  - [🎯 AIVA 統一啟動入口](#-aiva-統一啟動入口)
  - [⚡ AI 持續訓練啟動器](#-ai-持續訓練啟動器)
  - [💬 智能通信選擇器](#-智能通信選擇器)
- [🔧 維護工具](#-維護工具)
  - [🛠️ 系統修復工具](#-系統修復工具)
- [⚙️ 環境設置](#-環境設置)
  - [🐍 Python 路徑設置](#-python-路徑設置)
- [✅ 驗證工具](#-驗證工具)
  - [📦 套件驗證器](#-套件驗證器)
- [🎯 使用流程](#-使用流程)
  - [🚀 系統首次啟動](#-系統首次啟動)
  - [🔧 系統維護流程](#-系統維護流程)

---

> **通用服務腳本目錄** - AIVA 通用基礎設施工具集  
> **服務對應**: AIVA Common Services  
> **腳本數量**: 6個通用工具

---

## 📋 目錄概述

Common 服務腳本提供 AIVA 系統的基礎設施功能，包括系統啟動、環境配置、套件驗證、系統維護等通用功能。這些工具為其他所有服務提供基礎支援。

---

## 🗂️ 目錄結構

```
common/
├── 📋 README.md                     # 本文檔
│
├── 🚀 launcher/                     # 系統啟動器 (3個)
│   ├── 🎯 aiva_launcher.py          # AIVA 統一啟動入口
│   ├── ⚡ start_ai_continuous_training.py # AI 持續訓練啟動
│   └── 💬 smart_communication_selector.py # 智能通信選擇器
│
├── 🔧 maintenance/                  # 系統維護 (1個)
│   └── 🛠️ system_repair_tool.py    # 系統修復工具
│
├── ⚙️ setup/                        # 環境設置 (1個)
│   └── 🐍 setup_python_path.py     # Python 路徑設置
│
└── ✅ validation/                   # 系統驗證 (1個)
    └── 📦 aiva_package_validator.py # 套件驗證器
```

---

## 🚀 啟動器工具

### 🎯 AIVA 統一啟動入口
**檔案**: `launcher/aiva_launcher.py`
```bash
cd launcher
python aiva_launcher.py [options]
```

**功能**:
- 🎯 統一 AIVA 系統啟動入口
- 🔧 自動檢測系統環境
- ⚙️ 配置服務啟動順序
- 📊 啟動過程監控與日誌

**參數選項**:
```bash
python aiva_launcher.py --mode development    # 開發模式
python aiva_launcher.py --mode production     # 生產模式
python aiva_launcher.py --services core,scan  # 指定啟動服務
python aiva_launcher.py --config custom.yaml  # 自訂配置檔
```

### ⚡ AI 持續訓練啟動器
**檔案**: `launcher/start_ai_continuous_training.py`
```bash
cd launcher
python start_ai_continuous_training.py
```

**功能**:
- ⚡ 啟動 AI 持續學習系統
- 🧠 配置訓練參數
- 📈 監控訓練進度
- 💾 自動保存訓練模型

### 💬 智能通信選擇器
**檔案**: `launcher/smart_communication_selector.py`
```bash
cd launcher
python smart_communication_selector.py
```

**功能**:
- 💬 智能選擇最佳通信協議
- 🌐 動態負載平衡
- 🔒 通信安全驗證
- 📡 連線品質監控

---

## 🔧 維護工具

### 🛠️ 系統修復工具
**檔案**: `maintenance/system_repair_tool.py`
```bash
cd maintenance
python system_repair_tool.py [repair_type]
```

**功能**:
- 🛠️ 自動檢測並修復系統問題
- 🔧 修復損壞的配置檔案
- 📂 清理暫存檔案和日誌
- 🔄 重新建立索引和快取

**修復類型**:
```bash
python system_repair_tool.py --type config     # 修復配置檔案
python system_repair_tool.py --type database   # 修復資料庫問題
python system_repair_tool.py --type cache      # 清理快取
python system_repair_tool.py --type all        # 全面系統修復
```

---

## ⚙️ 環境設置

### 🐍 Python 路徑設置
**檔案**: `setup/setup_python_path.py`
```bash
cd setup
python setup_python_path.py
```

**功能**:
- 🐍 自動配置 Python 環境路徑
- 📦 檢查必要套件依賴
- ⚙️ 設置環境變數
- 🔧 修復路徑衝突問題

---

## ✅ 驗證工具

### 📦 套件驗證器
**檔案**: `validation/aiva_package_validator.py`
```bash
cd validation
python aiva_package_validator.py
```

**功能**:
- 📦 驗證 AIVA 所有套件完整性
- 🔍 檢查版本相容性
- ⚠️ 識別缺失或損壞的套件
- 📋 生成驗證報告

---

## 🎯 使用流程

### 🚀 系統首次啟動
```bash
# 1. 設置 Python 環境
cd setup
python setup_python_path.py

# 2. 驗證套件完整性
cd ../validation
python aiva_package_validator.py

# 3. 啟動 AIVA 系統
cd ../launcher
python aiva_launcher.py --mode development
```

### 🔧 系統維護流程
```bash
# 1. 檢查系統健康狀態
cd maintenance
python system_repair_tool.py --type all

# 2. 重新驗證套件
cd ../validation
python aiva_package_validator.py

# 3. 重啟系統服務
cd ../launcher
python aiva_launcher.py --mode production
```

---

**維護者**: AIVA Common Services Team  
**最後更新**: 2025-11-17  
**服務狀態**: ✅ 所有工具已重組並驗證

---

[← 返回 Scripts 主目錄](../README.md)
