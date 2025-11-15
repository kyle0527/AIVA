# AIVA 圖表目錄索引

**更新日期**: 2025年11月14日  
**組織結構**: 組合圖 vs 單獨圖表分類管理

---

## 📁 目錄結構

### 📊 **composite/** - 組合圖表
**用途**: 整合多個分析面向的綜合性圖表  
**特點**: 高層次概覽、多維度整合、決策支援導向

- `AIVA_CORE_AI_ARCHITECTURE_COMPOSITE.md` - AI 架構組合圖
  - AI 引擎架構概覽
  - 攻擊規劃系統圖
  - 分析能力組合圖
  - 系統協調架構圖
  - 知識管理架構圖
  - 執行引擎組合圖

- `AST_MERMAID_GENERATION_WORKFLOW.md` - AST 解析與 Mermaid 生成工作流程
  - Python AST 解析流程
  - Mermaid 語法生成邏輯
  - 複雜度分析工作流
  - 程式碼結構視覺化流程

### 📋 **individual/** - 單獨圖表
**用途**: 詳細的個別功能分析圖表  
**特點**: 細節深入、函數級分析、開發調試導向

- `aiva_core_analysis/` - AIVA Core 詳細分析 (538個圖表)
  - 每個模組的詳細流程圖
  - 每個函數的執行邏輯圖
  - 類別結構和方法關係圖
  - 模組間依賴關係圖

### 📄 **根目錄報告**
- `AIVA_CORE_ANALYSIS_REPORT.md` - 分析總報告
- `AIVA_CORE_COMPOSITE_ANALYSIS_COMPLETION_REPORT.md` - 組合分析完成報告

---

## 🎯 使用指南

### **查看組合圖** (推薦用於決策和概覽)
```bash
# 瀏覽高層次架構
cd docs/diagrams/composite/
```

**適用場景**:
- 系統架構理解
- 技術決策支援
- 專案規劃和設計
- 對外技術展示

### **查看單獨圖表** (推薦用於開發和調試)
```bash
# 瀏覽詳細實現
cd docs/diagrams/individual/aiva_core_analysis/
```

**適用場景**:
- 程式碼開發和維護
- Bug 調試和追蹤
- 功能實現細節研究
- 程式碼審查和優化

---

## 🔄 圖表更新流程

### **組合圖更新**
1. 使用 `scripts/generate_organization_diagrams.py` 生成組織層級圖表
2. 使用 `tools/common/development/generate_complete_architecture.py` 生成架構圖
3. 手動整合和優化組合圖表

### **單獨圖表更新**
1. 使用 `tools/py2mermaid.py` 對目標程式碼進行 AST 分析
2. 自動生成詳細的函數級流程圖
3. 批量處理和分類存放

---

## 📈 圖表統計

| 類型 | 數量 | 用途 | 更新頻率 |
|------|------|------|----------|
| 組合圖 | 5 | 架構概覽、決策支援 | 手動/週期性 |
| 單獨圖表 | 538+ | 開發調試、詳細分析 | 自動/程式碼變更時 |
| 報告檔案 | 2 | 分析總結、完成狀態 | 專案里程碑 |

---

## 🔍 圖表查找

### **按功能查找組合圖**
- **AI 架構**: `composite/AIVA_CORE_AI_ARCHITECTURE_COMPOSITE.md`
- **程式碼分析流程**: `composite/AST_MERMAID_GENERATION_WORKFLOW.md`
- **權限控制分析**: `composite/core_authorization_analysis_composite.mmd` ⭐ **新增**
- **AI智能引擎**: `composite/ai_engine_intelligence_composite.mmd` ⭐ **新增**  
- **攻擊系統編排**: `composite/attack_system_orchestration_composite.mmd` ⭐ **新增**

### **按模組查找單獨圖表**
```bash
# 查找特定模組的圖表
find individual/aiva_core_analysis/ -name "*模組名稱*" -type f
```

### **按功能查找單獨圖表**
```bash
# 查找特定功能的圖表  
grep -r "功能關鍵字" individual/aiva_core_analysis/
```

---

**維護負責**: AIVA 開發團隊  
**文件版本**: v1.0  
**下次更新**: 依據程式碼變更和架構演進