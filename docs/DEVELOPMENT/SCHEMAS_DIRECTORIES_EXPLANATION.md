# AIVA 目錄結構說明：schemas 和 enums 處理方案

## 📋 現況分析

您詢問的兩個目錄有不同的用途和狀態：

### 1. `services/aiva_common/schemas/` - ✅ **保留**
**用途**: Python 原始 Schema 定義（單一真實來源）
**狀態**: 正常運作，是整個系統的核心
**內容**:
```
├── __init__.py          # 統一導出介面
├── ai.py               # AI 相關模型
├── api_testing.py      # API 測試模型  
├── assets.py           # 資產管理
├── base.py             # 基礎模型
├── enhanced.py         # 增強型模型
├── findings.py         # 漏洞發現
├── languages.py        # 程式語言支援
├── messaging.py        # 訊息系統
├── references.py       # 外部引用（CVE/CWE等）
├── risk.py             # 風險評估
├── system.py           # 系統狀態
├── tasks.py            # 任務定義
└── telemetry.py        # 遙測數據
```

### 2. `services/aiva_common/enums/` - ✅ **保留**
**用途**: Python 枚舉類型定義（單一真實來源）
**狀態**: 正常運作，支援所有模組
**內容**:
```
├── __init__.py         # 統一導出介面
├── assets.py           # 資產相關枚舉
├── common.py           # 通用枚舉（Severity、Confidence等）
├── modules.py          # 模組枚舉（ModuleName、Topic等）
└── security.py         # 安全測試枚舉
```

### 3. `schemas/` (根目錄) - ✅ **保留**
**用途**: 多語言生成檔案（從 aiva_common/schemas 生成）
**狀態**: 官方工具生成，剛完成
**內容**:
```
├── aiva_schemas.json   # JSON Schema (270KB)
├── aiva_schemas.d.ts   # TypeScript 定義 (51KB)
├── aiva_schemas.go     # Go 結構 (155KB)
├── aiva_schemas.rs     # Rust 結構 (107KB)
└── enums.ts            # TypeScript 枚舉 (14KB)
```

## 🎯 建議處理方式

### ✅ 全部保留 - 這是正確的架構

這兩個目錄都是 AIVA 系統的重要組成部分：

1. **`aiva_common/schemas/` 和 `aiva_common/enums/`** 
   - 是 Python 生態系統的單一真實來源（Single Source of Truth）
   - 被整個 AIVA 專案引用
   - 通過官方 Pydantic API 生成多語言合約

2. **根目錄 `schemas/`**
   - 是從 Python 定義生成的多語言檔案
   - 供 Go、Rust、TypeScript 專案使用
   - 自動化生成，不應手動修改

## 📊 資料流向

```
services/aiva_common/schemas/ (Python原始定義)
         ↓ (官方工具生成)
schemas/ (多語言生成檔案)
         ↓ (各語言專案使用)
Go/Rust/TypeScript 專案
```

## 🔧 維護建議

### 日常維護
1. **僅修改** `services/aiva_common/schemas/` 和 `services/aiva_common/enums/`
2. **重新生成** 根目錄 `schemas/` 檔案：
   ```powershell
   .\tools\generate-official-contracts.ps1 -GenerateAll
   ```
3. **不要手動修改** 根目錄 `schemas/` 中的生成檔案

### 版本控制
- ✅ 提交 `services/aiva_common/schemas/` 和 `services/aiva_common/enums/` 的變更
- ✅ 提交根目錄 `schemas/` 的生成檔案（作為正式發佈）
- ❌ 不要忽略生成檔案，它們是其他語言專案的依賴

## ✨ 總結

**結論**: 兩個目錄都應該保留，它們是 AIVA 多語言架構的重要組成部分。

- `services/aiva_common/schemas/` 和 `enums/` = 權威定義
- 根目錄 `schemas/` = 多語言分發版本

這個架構設計是正確且現代化的，符合企業級微服務最佳實踐。