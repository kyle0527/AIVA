# 🤖 Core Service Scripts

## 📑 目錄

- [📋 目錄概述](#-目錄概述)
- [🗂️ 目錄結構](#-目錄結構)
- [🚀 工具說明](#-工具說明)
  - [🔄 自我感知更新工具](#-自我感知更新工具)
  - [🧠 AI 分析工具集](#-ai-分析工具集)
- [📊 報告系統](#-報告系統)
  - [📈 AI 分析報告](#-ai-分析報告)
  - [🏢 企業級 AI 洞察報告](#-企業級-ai-洞察報告)
- [🎯 使用情境](#-使用情境)
  - [🚀 系統啟動時](#-系統啟動時)
  - [🔧 維護作業](#-維護作業)
  - [📊 分析需求](#-分析需求)
- [⚡ 效能考量](#-效能考量)
- [🔗 服務整合](#-服務整合)

---

> **核心服務腳本目錄** - AIVA 核心 AI 系統相關工具集  
> **服務對應**: AIVA Core Services  
> **腳本數量**: 10個核心工具

---

## 📋 目錄概述

Core 服務腳本專門處理 AIVA 的核心 AI 功能，包括 AI 分析、系統自我感知、智能元件探索等核心能力。這些腳本是 AIVA 智能化運作的基礎工具。

---

## 🗂️ 目錄結構

```
core/
├── 📋 README.md                     # 本文檔
├── 🔄 update_self_awareness.py      # 自我感知更新工具
│
└── 🧠 ai_analysis/                  # AI 分析工具集 (9個)
    ├── 🔍 ai_component_explorer.py  # AI 元件探索器
    ├── 🚀 ai_system_explorer_v2.py  # AI 系統探索器 v2
    ├── ⚡ aiva_continuous_ai_manager.py # 持續 AI 管理器
    ├── 🏢 enterprise_ai_manager.py   # 企業級 AI 管理器
    ├── 🎯 production_ai_manager_v2.py # 生產環境 AI 管理器 v2
    ├── 📈 real_time_ai_analysis.py  # 即時 AI 分析
    ├── 💡 smart_ai_integrator.py    # 智能 AI 整合器
    ├── 🔗 ai_system_bridge.py       # AI 系統橋接器
    ├── 🎨 ai_enhanced_generator.py  # AI 增強生成器
    │
    └── 📊 reporting/                # AI 分析報告
        ├── ai_analysis_report.json  # AI 分析結果報告
        └── enterprise_ai_insights.md # 企業級 AI 洞察報告
```

---

## 🚀 工具說明

### 🔄 自我感知更新工具

**檔案**: `update_self_awareness.py`
```bash
python update_self_awareness.py
```

**功能**:
- ✅ 更新 AIVA 系統的自我認知資訊
- ✅ 同步系統元件狀態
- ✅ 調整 AI 行為參數
- ✅ 優化系統回應機制

---

### 🧠 AI 分析工具集

#### 🔍 AI 元件探索器
**檔案**: `ai_analysis/ai_component_explorer.py`
```bash
cd ai_analysis
python ai_component_explorer.py
```

**功能**:
- 🔍 深度分析 AIVA 各 AI 元件
- 📊 生成元件性能報告
- 🔗 檢測元件間依賴關係
- 💡 提供優化建議

#### 🚀 AI 系統探索器 v2
**檔案**: `ai_analysis/ai_system_explorer_v2.py`
```bash
cd ai_analysis
python ai_system_explorer_v2.py
```

**功能**:
- 🎯 全面分析 AIVA AI 系統架構
- 📈 評估系統整體性能
- 🔄 識別瓶頸與優化機會
- 📋 生成系統健康報告

#### ⚡ 持續 AI 管理器
**檔案**: `ai_analysis/aiva_continuous_ai_manager.py`
```bash
cd ai_analysis
python aiva_continuous_ai_manager.py
```

**功能**:
- ⚡ 實時監控 AI 系統運行狀態
- 🎯 動態調整 AI 參數設定
- 📊 收集 AI 性能數據
- 🔧 自動執行系統優化

#### 🏢 企業級 AI 管理器
**檔案**: `ai_analysis/enterprise_ai_manager.py`
```bash
cd ai_analysis
python enterprise_ai_manager.py
```

**功能**:
- 🏢 企業級 AI 部署管理
- 🔒 AI 系統安全性檢查
- 📈 企業 AI 資源配置
- 📋 合規性驗證報告

#### 🎯 生產環境 AI 管理器 v2
**檔案**: `ai_analysis/production_ai_manager_v2.py`
```bash
cd ai_analysis
python production_ai_manager_v2.py
```

**功能**:
- 🎯 生產環境 AI 系統監控
- ⚡ 高可用性部署管理
- 📊 生產數據分析處理
- 🔧 故障自動恢復機制

#### 📈 即時 AI 分析
**檔案**: `ai_analysis/real_time_ai_analysis.py`
```bash
cd ai_analysis
python real_time_ai_analysis.py
```

**功能**:
- 📈 即時 AI 數據流分析
- ⚡ 快速異常檢測
- 🎯 實時決策支持
- 📊 動態報告生成

#### 💡 智能 AI 整合器
**檔案**: `ai_analysis/smart_ai_integrator.py`
```bash
cd ai_analysis
python smart_ai_integrator.py
```

**功能**:
- 💡 智能整合多個 AI 服務
- 🔗 自動化 API 對接
- ⚙️ 動態負載平衡
- 📋 整合效果評估

#### 🔗 AI 系統橋接器
**檔案**: `ai_analysis/ai_system_bridge.py`
```bash
cd ai_analysis
python ai_system_bridge.py
```

**功能**:
- 🔗 連接不同 AI 系統模組
- 🌐 跨平台 AI 通信
- 📡 數據格式轉換
- ⚙️ 協議適配處理

#### 🎨 AI 增強生成器
**檔案**: `ai_analysis/ai_enhanced_generator.py`
```bash
cd ai_analysis
python ai_enhanced_generator.py
```

**功能**:
- 🎨 AI 驅動內容生成
- 💡 智能模板創建
- 📝 自動化文檔生成
- 🔧 個性化輸出調整

---

## 📊 報告系統

### 📈 AI 分析報告
**檔案**: `ai_analysis/reporting/ai_analysis_report.json`

包含 AI 系統的詳細分析結果，供其他服務使用。

### 🏢 企業級 AI 洞察報告
**檔案**: `ai_analysis/reporting/enterprise_ai_insights.md`

提供企業級的 AI 性能分析和戰略建議。

---

## 🎯 使用情境

### 🚀 系統啟動時
1. 執行 `update_self_awareness.py` 更新系統狀態
2. 運行 `ai_system_explorer_v2.py` 進行系統檢查
3. 啟動 `aiva_continuous_ai_manager.py` 開始監控

### 🔧 維護作業
1. 使用 `ai_component_explorer.py` 分析個別元件
2. 運行 `production_ai_manager_v2.py` 檢查生產環境
3. 執行 `enterprise_ai_manager.py` 進行合規檢查

### 📊 分析需求
1. 執行 `real_time_ai_analysis.py` 獲取即時數據
2. 使用 `smart_ai_integrator.py` 整合分析結果
3. 查看 `reporting/` 目錄中的詳細報告

---

## ⚡ 效能考量

- **並行執行**: 大部分工具支援並行運行
- **資源消耗**: AI 分析工具需要充足的 CPU 和記憶體
- **數據存儲**: 分析結果會存儲在 `reporting/` 目錄
- **執行時間**: 完整分析通常需要 5-15 分鐘

---

## 🔗 服務整合

- **與 Common 服務**: 使用共用的啟動器和驗證工具
- **與 Scan 服務**: 提供 AI 分析結果給掃描報告
- **與 Integration 服務**: 支援跨語言 AI 功能整合
- **與 Testing 服務**: 提供 AI 系統測試數據

---

**維護者**: AIVA Core AI Team  
**最後更新**: 2025-11-17  
**服務狀態**: ✅ 所有工具已重組並驗證

---

[← 返回 Scripts 主目錄](../README.md)