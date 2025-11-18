# 🛠️ Utilities - 精簡工具腳本

> **實用工具目錄** - AIVA 精選實用工具集  
> **清理成果**: 移除重複工具 80%+，保留最佳版本  
> **腳本數量**: 12個高品質實用工具

---

## 📋 目錄概述

Utilities 目錄經過重組後，移除了大量重複和過時的工具，保留最實用和高品質的工具腳本。這些工具為 AIVA 系統提供日常維護、調試、生成和清理等基礎功能支援。

---

## 🗂️ 目錄結構

```
utilities/
├── 📋 README.md                     # 本文檔
│
├── 🏥 health_check.py               # 系統健康檢查 (整合最佳版本)
├── 🔧 debug_fixer.py                # 調試修復器 (統一所有修復功能)
├── 🧹 cleanup_diagram_output.py     # 圖表輸出清理工具
├── 🛡️ safe_batch_repair.py          # 安全批次修復工具
├── 🎨 diagram_auto_composer.py      # 自動圖表組成器
│
├── 📝 generate_ai_analysis.py       # AI 分析報告生成器
├── 📊 generate_enterprise_report.py # 企業級報告生成器
├── 🔍 generate_intelligent_analysis.py # 智能分析生成器
├── 📈 generate_system_report.py     # 系統報告生成器
├── 🎯 generate_ultimate_analysis.py # 終極分析生成器
└── 📋 generate_ultimate_report.py   # 終極報告生成器
```

---

## 🏥 系統維護工具

### 🏥 系統健康檢查 (整合版)
**檔案**: `health_check.py`
```bash
python health_check.py [check_type] [options]
```

**功能** (整合所有重複版本的最佳功能):
- 🏥 系統整體健康狀況檢查
- 💾 記憶體與 CPU 使用監控
- 🌐 網路連接與服務狀態檢查
- 📊 系統效能基準測試
- ⚠️ 問題自動檢測與警報

**檢查類型**:
```bash
# 快速健康檢查
python health_check.py --mode quick

# 完整系統檢查
python health_check.py --mode comprehensive --detailed

# 特定服務檢查
python health_check.py --service core,scan,integration

# 持續監控模式
python health_check.py --monitor --interval 5m
```

### 🔧 調試修復器 (統一版)
**檔案**: `debug_fixer.py`  
```bash
python debug_fixer.py [fix_type] [options]
```

**功能** (統一所有 debug_fixer 版本):
- 🔧 自動檢測並修復常見問題
- 🐛 Python 導入錯誤修復
- 📂 檔案路徑問題自動修正
- 🔄 設定檔損壞修復
- 💾 快取清理與重建

**修復類型**:
```bash
# 自動診斷並修復
python debug_fixer.py --auto-fix

# Python 導入問題修復
python debug_fixer.py --fix imports --recursive

# 設定檔修復
python debug_fixer.py --fix config --backup

# 快取問題修復
python debug_fixer.py --fix cache --rebuild
```

### 🛡️ 安全批次修復工具
**檔案**: `safe_batch_repair.py`
```bash
python safe_batch_repair.py [target] [options]
```

**功能**:
- 🛡️ 安全模式的批次檔案修復
- 📁 大量檔案的批次處理
- 🔄 自動備份與還原機制
- ✅ 修復前後的完整性驗證

---

## 🧹 清理與整理工具

### 🧹 圖表輸出清理工具
**檔案**: `cleanup_diagram_output.py`
```bash
python cleanup_diagram_output.py [cleanup_type] [path]
```

**功能**:
- 🧹 清理過期的圖表輸出檔案
- 📊 整理 Mermaid 圖表檔案
- 🗂️ 依日期歸檔圖表檔案
- 💾 壓縮舊的圖表檔案

---

## 🎨 圖表與可視化工具

### 🎨 自動圖表組成器
**檔案**: `diagram_auto_composer.py`
```bash
python diagram_auto_composer.py [diagram_type] [options]
```

**功能**:
- 🎨 自動組成複雜的系統架構圖
- 📊 根據程式碼自動生成流程圖
- 🔗 服務關係圖自動繪製
- 📈 資料流向圖自動生成

**圖表類型**:
```bash
# 系統架構圖
python diagram_auto_composer.py --type architecture --auto-layout

# 服務關係圖  
python diagram_auto_composer.py --type services --relationships

# 資料流程圖
python diagram_auto_composer.py --type dataflow --trace-paths
```

---

## 📝 報告生成工具集

### 📝 AI 分析報告生成器
**檔案**: `generate_ai_analysis.py`
```bash
python generate_ai_analysis.py [analysis_type] [options]
```

**功能**:
- 🤖 生成 AI 系統分析報告
- 📊 AI 效能評估報告
- 🧠 AI 學習進度分析
- 💡 AI 優化建議報告

### 📊 企業級報告生成器
**檔案**: `generate_enterprise_report.py`
```bash
python generate_enterprise_report.py [report_scope] [options]
```

**功能**:
- 🏢 企業級綜合系統報告
- 📈 業務指標與 KPI 報告
- 🔒 安全性與合規性報告
- 💼 管理層決策支援報告

### 🔍 智能分析生成器
**檔案**: `generate_intelligent_analysis.py`
```bash
python generate_intelligent_analysis.py [analysis_depth] [options]
```

**功能**:
- 🔍 深度智能系統分析
- 🧠 行為模式識別分析
- 📈 趨勢預測與建議
- 💡 自動化洞察發現

### 📈 系統報告生成器
**檔案**: `generate_system_report.py`
```bash
python generate_system_report.py [system_scope] [options]
```

**功能**:
- 📈 系統狀態綜合報告
- ⚡ 效能指標詳細報告
- 🔧 系統組態與設定報告
- 📊 使用統計分析報告

### 🎯 終極分析生成器
**檔案**: `generate_ultimate_analysis.py`
```bash
python generate_ultimate_analysis.py [target] [options]
```

**功能**:
- 🎯 最深度的系統分析
- 🔬 多維度數據交叉分析
- 🧠 AI 輔助的洞察生成
- 📋 綜合優化建議報告

### 📋 終極報告生成器
**檔案**: `generate_ultimate_report.py`
```bash
python generate_ultimate_report.py [report_level] [options]
```

**功能**:
- 📋 最全面的系統報告
- 🌐 跨服務綜合分析
- 📊 多格式輸出支援
- 🎨 專業級報告排版

---

## 🎯 使用情境

### 🚀 日常維護作業
```bash
# 1. 系統健康檢查
python health_check.py --mode comprehensive

# 2. 清理暫存檔案
python cleanup_diagram_output.py --cleanup temp --auto

# 3. 生成日常系統報告
python generate_system_report.py --scope daily --auto-send
```

### 🔧 問題診斷與修復
```bash
# 1. 自動診斷問題
python debug_fixer.py --auto-diagnose --verbose

# 2. 安全修復問題
python safe_batch_repair.py --target all --safe-mode

# 3. 生成修復報告
python generate_intelligent_analysis.py --focus repair_results
```

### 📊 報告與分析
```bash
# 1. 生成 AI 分析報告
python generate_ai_analysis.py --comprehensive --charts

# 2. 企業級報告
python generate_enterprise_report.py --scope quarterly --executive

# 3. 終極分析報告
python generate_ultimate_analysis.py --all-services --deep
```

### 🎨 圖表生成
```bash
# 1. 自動組成架構圖
python diagram_auto_composer.py --type architecture --complete

# 2. 清理舊圖表檔案
python cleanup_diagram_output.py --archive --compress
```

---

## ⚡ 工具最佳化特色

### 🔄 重複工具整合成果
- **health_check.py**: 整合 5 個重複版本的最佳功能
- **debug_fixer.py**: 統一所有調試修復器的功能
- **移除冗餘**: 清除 80%+ 的重複工具
- **保留精華**: 只保留最實用和高品質的工具

### 🛠️ 效能最佳化
- **快取機制**: 常用檢查結果快取
- **並行處理**: 多項檢查並行執行
- **增量更新**: 只處理變更的部分
- **記憶體優化**: 大檔案串流處理

### 📊 輸出格式標準化
- **多格式支援**: HTML、PDF、JSON、Excel
- **統一樣式**: 所有工具使用一致的輸出格式
- **圖表整合**: 自動嵌入相關圖表
- **互動功能**: 支援互動式報告

---

## 🔧 配置與自訂

### ⚙️ 工具配置
```yaml
# utilities_config.yaml
utilities:
  health_check:
    check_interval: 5m
    alert_threshold: 80%
    
  debug_fixer:
    auto_backup: true
    fix_confidence: 0.8
    
  report_generators:
    default_format: html
    include_charts: true
```

### 📊 報告樣板
- **企業樣板**: 正式企業報告格式
- **技術樣板**: 詳細技術分析格式
- **摘要樣板**: 簡潔執行摘要格式
- **自訂樣板**: 可自訂的報告樣板

---

## 🔗 工具整合使用

### 🔄 工具鏈整合
```bash
# 完整維護工具鏈
python health_check.py --comprehensive > health_report.json
python debug_fixer.py --auto-fix --report health_report.json
python generate_system_report.py --include-fixes --format pdf
```

### 📊 批次報告生成
```bash
# 生成所有類型報告
python generate_ai_analysis.py --batch &
python generate_enterprise_report.py --batch &
python generate_system_report.py --batch &
python generate_ultimate_report.py --consolidate all
```

---

## 📋 故障排除

### 常見問題與解決方案

#### 🏥 健康檢查失敗
```bash
# 重置健康檢查環境
python health_check.py --reset --reconfigure
```

#### 🔧 修復工具無效
```bash  
# 清除修復工具快取
python debug_fixer.py --clear-cache --reinitialize
```

#### 📊 報告生成錯誤
```bash
# 重建報告樣板
python generate_system_report.py --rebuild-templates
```

---

## 📅 工具維護排程

### 🔄 自動化維護
- **每日**: 健康檢查與基礎清理
- **每週**: 深度分析與報告生成
- **每月**: 工具更新與效能優化
- **季度**: 工具整合度評估與改進

---

**維護者**: AIVA Utilities Team  
**最後更新**: 2025-11-17  
**工具狀態**: ✅ 精簡完成，80%+ 重複工具已移除

---

[← 返回 Scripts 主目錄](../README.md)