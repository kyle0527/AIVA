# AIVA Token最佳化開發指南 (10/31實測驗證)

## 📋 目錄

- [📑 目錄](#目錄)
- [🎯 Token最佳化策略](#token最佳化策略)
  - [1. 📋 使用完整補包文件](#1-使用完整補包文件)
  - [2. 🔧 善用自動化工具](#2-善用自動化工具)
  - [3. 📊 標準化開發流程](#3-標準化開發流程)
- [🚀 快速參考命令](#快速參考命令)
  - [開發前檢查](#開發前檢查)
  - [開發中工具](#開發中工具)
  - [問題診斷](#問題診斷)
- [📚 智慧文件導覽](#智慧文件導覽)
  - [🔍 按問題類型查找](#按問題類型查找)
  - [🎯 按開發階段查找](#按開發階段查找)
- [⚡ 效率開發技巧](#效率開發技巧)
  - [1. 避免重複性問題](#1-避免重複性問題)
  - [2. 善用既有架構](#2-善用既有架構)
  - [3. 自動化優先](#3-自動化優先)
  - [4. 測試驅動開發](#4-測試驅動開發)
- [📊 Phase I開發檢查清單](#phase-i開發檢查清單)
  - [Week 1: AI攻擊計畫對映器](#week-1-ai攻擊計畫對映器)
  - [Week 2: 進階SSRF偵測](#week-2-進階ssrf偵測)
  - [Weeks 3-4: 客戶端授權繞過](#weeks-3-4-客戶端授權繞過)
- [🔧 故障排除快速指南](#故障排除快速指南)
  - [Schema問題](#schema問題)
  - [模組通連問題](#模組通連問題)
  - [環境問題](#環境問題)
- [💡 Token節約最佳實踐](#token節約最佳實踐)
- [🔗 相關資源](#相關資源)
  - [開發指南](#開發指南)
  - [架構指南](#架構指南)
  - [故障排除](#故障排除)
  - [使用者手冊](#使用者手冊)

## 📑 目錄

- [🎯 Token最佳化策略](#-token最佳化策略)
- [📋 開發工具鏈](#-開發工具鏈)
- [🔧 自動化工具應用](#-自動化工具應用)
- [📊 標準化開發流程](#-標準化開發流程)
- [⚡ 效能最佳化技巧](#-效能最佳化技巧)
- [🐛 常見問題解決](#-常見問題解決)
- [🔗 相關資源](#-相關資源)

本指南專為最小化開發過程中的Token使用而設計，提供高效的開發工作流程和最佳實踐。

## 🎯 Token最佳化策略

### 1. 📋 使用完整補包文件
替代頻繁詢問，直接參考：
- `README_PACKAGE.md` - 快速導覽和問題解決
- `AIVA_IMPLEMENTATION_PACKAGE.md` - 完整技術規格
- `AIVA_PHASE_0_COMPLETE_PHASE_I_ROADMAP.md` - 開發路線圖

### 2. 🔧 善用自動化工具
```bash
# 系統狀態檢查 (取代多次詢問系統狀態)
python aiva_package_validator.py

# Schema自動生成 (避免手動同步)
python services/aiva_common/tools/schema_codegen_tool.py

# 通連性驗證 (快速問題診斷)
python services/aiva_common/tools/module_connectivity_tester.py
```

### 3. 📊 標準化開發流程

#### Phase I開發標準流程:
1. **Week 1**: AI攻擊計畫對映器
   - 參考: `services/core/aiva_core/ai_engine/`
   - Schema: 使用自動生成的messaging.py
   - 測試: 執行connectivity_tester驗證

2. **Week 2**: 進階SSRF偵測
   - 參考: `services/scan/aiva_scan/plugins/`
   - 整合: 使用統一的findings.py schema
   - 效能: Rust橋接器已準備就緒

3. **Weeks 3-4**: 客戶端授權繞過
   - 基礎: `services/features/detection/`
   - 跨語言: Go微服務架構已建立
   - ROI目標: $5,000-$25,000 Bug Bounty

## 🚀 快速參考命令

### 開發前檢查
```bash
# 一鍵系統健康檢查
python aiva_package_validator.py

# 預期結果: 🟢 優秀 (4/4)
```

### 開發中工具
```bash
# Schema修改後重新生成
python services/aiva_common/tools/schema_codegen_tool.py --output-dir services/aiva_common/schemas/generated

# 跨語言一致性檢查
python services/aiva_common/tools/module_connectivity_tester.py

# 即時語法驗證
python services/aiva_common/tools/schema_validator.py services/aiva_common/core_schema_sot.yaml
```

### 問題診斷
```bash
# 詳細診斷報告
python aiva_package_validator.py --detailed

# 匯出完整狀態
python aiva_package_validator.py --export-report
```

## 📚 智慧文件導覽

### 🔍 按問題類型查找

#### 架構相關問題
→ `ARCHITECTURE_CONTRACT_COMPLIANCE_REPORT.md`

#### Schema/通信問題  
→ `services/aiva_common/core_schema_sot.yaml` + 自動化工具

#### 模組整合問題
→ `services/aiva_common/tools/module_connectivity_tester.py` 結果

#### 商業/ROI問題
→ `COMMERCIAL_READINESS_ASSESSMENT.md`

### 🎯 按開發階段查找

#### 計畫階段
1. `AIVA_PHASE_0_COMPLETE_PHASE_I_ROADMAP.md` - 整體規劃
2. `AIVA_IMPLEMENTATION_PACKAGE.md` - 技術細節
3. `aiva_package_validator.py` - 現況評估

#### 開發階段
1. Schema修改 → 自動化工具重新生成
2. 模組開發 → 參考現有結構
3. 整合測試 → connectivity_tester驗證

#### 測試階段
1. 自動驗證 → `aiva_package_validator.py`
2. 手動測試 → 參考現有測試模式
3. 部署準備 → 商業化評估文件

## ⚡ 效率開發技巧

### 1. 避免重複性問題
✅ **正確做法**: 先查文件，再使用工具驗證
❌ **避免**: 反覆詢問相同的系統狀態

### 2. 善用既有架構
✅ **正確做法**: 基於五大模組架構擴展
❌ **避免**: 重新設計基礎架構

### 3. 自動化優先
✅ **正確做法**: 使用Schema自動生成
❌ **避免**: 手動維護跨語言一致性

### 4. 測試驅動開發
✅ **正確做法**: 先跑connectivity_tester
❌ **避免**: 開發完才發現整合問題

## 📊 Phase I開發檢查清單

### Week 1: AI攻擊計畫對映器
- [ ] 系統健康檢查 (`aiva_package_validator.py`)
- [ ] AI引擎模組檢視 (`services/core/aiva_core/ai_engine/`)
- [ ] Schema使用確認 (messaging.py, tasks.py)
- [ ] 基礎功能實作
- [ ] 通連性測試驗證

### Week 2: 進階SSRF偵測  
- [ ] 掃描模組架構檢視 (`services/scan/`)
- [ ] Plugin架構整合
- [ ] Rust橋接器測試
- [ ] 效能基準測試
- [ ] 安全測試驗證

### Weeks 3-4: 客戶端授權繞過
- [ ] 功能檢測模組檢視 (`services/features/`)
- [ ] Go微服務整合
- [ ] 跨語言通信測試
- [ ] Bug Bounty準備
- [ ] 最終整合測試

## 🔧 故障排除快速指南

### Schema問題
```bash
# 重新生成所有Schema
python services/aiva_common/tools/schema_codegen_tool.py

# 驗證Schema一致性
python services/aiva_common/tools/schema_validator.py services/aiva_common/core_schema_sot.yaml
```

### 模組通連問題
```bash
# 完整通連性測試
python services/aiva_common/tools/module_connectivity_tester.py

# 檢查個別模組狀態
python aiva_package_validator.py --detailed
```

### 環境問題
```bash
# 系統整體健康檢查
python aiva_package_validator.py

# 如果狀態不是"優秀"，檢查具體問題並修復
```

## 💡 Token節約最佳實踐

1. **批次查詢**: 一次性獲取完整資訊而非多次小查詢
2. **工具優先**: 使用自動化工具替代重複性問題
3. **文件導向**: 先查閱補包文件再進行具體詢問  
4. **標準流程**: 遵循既定開發流程避免探索性開發
5. **預防性檢查**: 開發前先驗證環境避免後期問題

---

🎯 **目標**: 以最少的Token投入實現$5,000-$25,000的Bug Bounty收益  
⚡ **策略**: 充分利用補包資源，避免重複性工作，專注高價值功能開發

---

## 🔗 相關資源

### 開發指南
- 📖 [開發快速指南](./DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [開發任務指南](./DEVELOPMENT_TASKS_GUIDE.md)
- 📖 [開發者指南](./DEVELOPER_GUIDE.md)
- 📖 [依賴管理指南](./DEPENDENCY_MANAGEMENT_GUIDE.md)
- 📖 [API 驗證指南](./API_VERIFICATION_GUIDE.md)
- 📖 [Schema 導入指南](./SCHEMA_IMPORT_GUIDE.md)

### 架構指南
- 📖 [Schema 統一指南](../architecture/SCHEMA_GUIDE.md)
- 📖 [架構兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 故障排除
- 📖 [導入問題解決](../troubleshooting/IMPORT_ISSUES_RESOLUTION_GUIDE.md)
- 📖 [性能優化指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)

### 使用者手冊
- 📚 [Core 模組手冊](../../docs/user_guides/01_core/AIVA_CORE_使用者手冊.md)

