# AIVA 程式碼分析與 Mermaid 流程圖生成腳本完整清單報告

**報告日期**: 2025年11月14日  
**專案**: AIVA AI-Driven Security Testing Platform  
**目的**: 整合階段充分運用現有腳本資源  

## 📑 目錄

- [📋 核心分析工具](#核心分析工具)
- [📊 專案架構分析工具](#專案架構分析工具)
- [🛠️ 自動化工具](#自動化工具)
- [🔍 進階分析工具](#進階分析工具)
- [📊 許可、文件生成工具](#許可文件生成工具)
- [🐛 測試與修復工具](#測試與修復工具)
- [📈 分析結果統計](#分析結果統計)
- [🚀 快速使用指南](#快速使用指南)

---

## 📋 核心分析工具 (Core Analysis Tools)

### 1. **py2mermaid.py**
- **位置**: `tools/common/development/py2mermaid.py`
- **主要功能**: Python AST 解析與 Mermaid 流程圖產生
- **核心能力**:
  - 🧠 完整 Python AST 解析 (Abstract Syntax Tree)
  - 📊 自動將函數和模組轉換為 Mermaid 流程圖
  - 🎯 支援複雜控制流 (if/else、while、for、try/except)
  - 🔧 命令列介面，支援批次處理和自訂配置
  - 📁 目錄遞迴掃描，可處理整個專案
- **輸出格式**: `.mmd` 檔案 (Mermaid 語法)
- **使用場景**: 程式碼架構分析、函數流程可視化、程式碼文檔生成

### 2. **generate_mermaid_diagrams.py**
- **位置**: `tools/common/development/generate_mermaid_diagrams.py`
- **主要功能**: AIVA 專案架構圖生成器
- **核心能力**:
  - 🏗️ 多語言架構概覽圖 (Python, Go, Rust, TypeScript)
  - 📈 程式碼分布統計圓餅圖
  - 🔄 模組關係圖和依賴關係
  - 🚀 技術棧選擇決策流程圖
  - 🔄 掃描工作流程圖 (Sequence Diagram)
  - 📊 資料流程圖和部署架構圖
- **輸出格式**: Markdown 檔案包含 Mermaid 圖表
- **使用場景**: 專案架構文檔、技術選型說明、系統設計文檔

### 3. **mermaid_optimizer.py**
- **位置**: `tools/features/mermaid_optimizer.py`
- **主要功能**: Mermaid 圖表優化器 v2.0
- **核心能力**:
  - ✅ 符合 Mermaid.js v11.12.0+ 官方語法規範
  - 🧠 智能錯誤檢測和自動修復
  - 📊 支援所有官方圖表類型
  - 🎨 現代主題配置和自訂樣式系統
  - ♿ 無障礙功能和語意化標籤
  - 📱 響應式佈局和高 DPI 支援
- **輸出格式**: 最佳化的 Mermaid 程式碼
- **使用場景**: 圖表品質保證、語法驗證、樣式標準化

### 4. **generate_complete_architecture.py**
- **位置**: `tools/common/development/generate_complete_architecture.py`
- **主要功能**: 完整架構圖生成器
- **核心能力**:
  - 🔍 從模組程式碼自動解析架構關係
  - 🌐 多格式匯出 (PNG、SVG、PDF)
  - 🌏 中英文雙語標籤支援
  - 🔒 安全檢測流程圖 (XSS、SSRF、SQL注入等)
  - 📊 資料流程圖和系統整合圖
- **輸出格式**: 多種圖像格式 + Mermaid 原始碼
- **使用場景**: 正式文檔製作、架構設計評審、安全流程規劃

---

## 🛠️ 輔助分析工具 (Auxiliary Analysis Tools)

### 5. **refactor_imports_and_cleanup.py**
- **位置**: `tools/integration/aiva-schemas-plugin/scripts/refactor_imports_and_cleanup.py`
- **主要功能**: AST 重構與程式碼清理
- **核心能力**:
  - 🔄 Import 語句重寫和標準化
  - 🧹 程式碼清理和優化
  - 📦 模組依賴關係重構
- **使用場景**: 程式碼重構、依賴關係梳理

### 6. **generate_organization_diagrams.py**
- **位置**: `scripts/generate_organization_diagrams.py`
- **主要功能**: 組織結構圖表生成
- **核心能力**:
  - 📊 144 種組織方式分析
  - 🏗️ 專案結構可視化
  - 📈 複雜度和依賴關係分析
- **使用場景**: 專案結構優化、組織方式評估

### 7. **generate_advanced_diagrams.py**
- **位置**: `scripts/generate_advanced_diagrams.py`
- **主要功能**: 進階圖表生成器
- **核心能力**:
  - 🔧 技術棧分析和可視化
  - 📊 複雜度矩陣圖
  - 🎯 多維度分析圖表
- **使用場景**: 技術決策支援、複雜度分析

### 8. **diagram_auto_composer.py**
- **位置**: `scripts/diagram_auto_composer.py`
- **主要功能**: 圖表自動組合引擎
- **核心能力**:
  - 🤖 模組整合架構自動推導
  - 🔗 組件關係自動發現
  - 📊 多層級圖表自動生成
- **使用場景**: 自動化架構文檔生成

---

## 🔧 Mermaid 專用工具 (Mermaid-Specific Tools)

### 9. **smart_repair_engine.py**
- **位置**: `tools/mermaid/smart_repair_engine.py`
- **主要功能**: Mermaid 智能驗證與修復
- **核心能力**:
  - 🔍 圖表類型自動檢測
  - 🛠️ 語法錯誤自動修復
  - 📋 診斷報告生成
- **使用場景**: 圖表品質保證、錯誤修復

### 10. **mermaid_diagnostic_system.py**
- **位置**: `tools/mermaid/mermaid_diagnostic_system.py`
- **主要功能**: Mermaid 診斷系統
- **核心能力**:
  - 🔍 深度錯誤檢測
  - 💡 自動修復建議
  - 📊 診斷報告和統計
- **使用場景**: 圖表維護、問題診斷

---

## 🎯 核心解析器 (Core Parsers)

### 11. **ast_parser.py**
- **位置**: `services/core/aiva_core/planner/ast_parser.py`
- **主要功能**: 攻擊流程 AST 解析器
- **核心能力**:
  - 🔒 攻擊流程圖生成
  - 📋 安全測試規劃
  - 🎯 攻擊路徑可視化
- **使用場景**: 安全測試規劃、攻擊流程設計

---

## 🔗 AIVA Core 整合應用分析

### 目前在 `C:\D\fold7\AIVA-git\services\core\aiva_core` 整合中使用的腳本:

#### **直接使用 (Direct Usage)**

1. **ast_parser.py** 
   - 位置: `services/core/aiva_core/planner/ast_parser.py`
   - 整合狀態: ✅ **核心組件** - 直接整合在 aiva_core 中
   - 使用方式: 作為 planner 模組的核心解析器
   - 功能: 攻擊流程的 AST 解析和圖表生成

#### **間接支援 (Indirect Support)**

2. **py2mermaid.py**
   - 整合狀態: 🔄 **開發支援** - 用於分析 aiva_core 模組
   - 使用方式: 分析 aiva_core 內部模組結構和函數流程
   - 應用場景: 生成 aiva_core 各模組的流程圖文檔

3. **mermaid_optimizer.py**
   - 整合狀態: 🎨 **品質保證** - 優化 aiva_core 生成的圖表
   - 使用方式: 確保 aiva_core 輸出的 Mermaid 圖表符合標準
   - 應用場景: 圖表驗證、樣式標準化

#### **架構文檔支援 (Architecture Documentation)**

4. **generate_mermaid_diagrams.py**
   - 整合狀態: 📊 **文檔生成** - 為 aiva_core 生成整體架構圖
   - 使用方式: 生成 aiva_core 的多語言架構概覽
   - 應用場景: aiva_core 架構文檔、技術選型說明

5. **generate_complete_architecture.py**
   - 整合狀態: 🏗️ **完整文檔** - 生成 aiva_core 完整架構設計
   - 使用方式: 自動解析 aiva_core 模組關係並生成架構圖
   - 應用場景: 正式架構文檔、設計評審

---

## 📈 建議整合策略

### **短期整合 (立即可用)**
1. 使用 `py2mermaid.py` 分析所有 aiva_core 模組
2. 使用 `mermaid_optimizer.py` 優化現有圖表
3. 使用 `generate_mermaid_diagrams.py` 生成架構概覽

### **中期整合 (開發階段)**
1. 整合 `smart_repair_engine.py` 到 aiva_core 的圖表生成流程
2. 使用 `generate_complete_architecture.py` 自動生成設計文檔
3. 建立自動化圖表生成 pipeline

### **長期整合 (系統級整合)**
1. 將圖表生成能力整合到 aiva_core 的核心功能
2. 建立統一的圖表標準和樣式系統
3. 實現即時圖表更新和維護機制

---

**報告總結**: AIVA 專案擁有完整的程式碼分析和圖表生成工具鏈，其中 `ast_parser.py` 已直接整合在 aiva_core 中，其他工具可在整合階段充分運用以提升開發效率和文檔品質。

**建議**: 優先運用 `py2mermaid.py` 和 `mermaid_optimizer.py` 進行 aiva_core 的分析和優化工作。