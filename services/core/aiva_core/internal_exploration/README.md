# 🧭 Internal Exploration - 內部探索

> **路徑**: `internal_exploration/`  
> **狀態**: ✅ Production Ready | **最後更新**: 2026-01-21  
> **子模組**: 2 個 | **總文件數**: 16 | **架構版本**: v3.1（多語言工具已優化）  
> **父模組**: [AIVA Core](../README.md)

## 概述

**Internal Exploration** 是 AIVA 五大核心模組之一，作為自我認知系統。提供**4種語言**的 AST 分析、數據流追蹤、自動化分類和自我執行能力。

**核心特色**：
- 🔍 **多語言 AST 解析** - Python, Go, Rust, TypeScript 統一 JSON 輸出
- 📊 **數據流視覺化** - 自動生成 Mermaid 流程圖
- 🏷️ **智能分類系統** - 內部模組/外部模組自動分類
- 🔧 **自我修復診斷** - 自動檢測數據流斷點和架構問題
- ⚡ **動態執行系統** - 支援多語言 subprocess 執行

---

## 語言工具狀態總覽

| 語言 | 分析能力 | struct/CLI 支援 | 狀態 |
|------|----------|----------------|------|
| **Python** | ✅ 完整 AST | ✅ 函數參數 | 207 flows |
| **Go** | ✅ 完整 AST + 語義 | ✅ struct tags | 5 flows |
| **Rust** | ⚠️ 語法解析 | ❌ Clap macros | 1 flow |
| **TypeScript** | ✅ 完整 AST | ✅ interface/type | 待測試 |

---

## 🎯 架構設計 (v3.0)

### 雙層架構

```
語言層 (Language Layer) - 只做 AST 解析
  ├── python_tools/aiva_flow_analyzer.py
  ├── go_tools/go2mermaid.go
  ├── rust_tools/src/main.rs
  └── typescript_tools/ts2mermaid.ts
        ↓ 輸出統一 JSON
        
業務邏輯層 (Business Logic Layer) - 分類與執行
  ├── aiva_internal_classifier.py    (AI Core 分類)
  ├── aiva_internal_executor.py      (AI Core 執行)
  ├── aiva_external_classifier.py    (Features/Scan 分類)
  └── aiva_external_executor.py      (Features/Scan 執行)
```

### Internal vs External CLI

| 類型 | 目標模組 | 通信方式 | 分類器 | 執行器 |
|------|---------|---------|--------|--------|
| **Internal CLI** | AI Core 模組 | 直接導入 | `aiva_internal_classifier.py` | `aiva_internal_executor.py` |
| **External CLI** | Features/Scan | subprocess + JSON | `aiva_external_classifier.py` | `aiva_external_executor.py` |

---

## 📂 子模組 (Submodules)

- [go_tools](./go_tools/README.md)
- [python_tools](./python_tools/README.md)
- [rust_tools](./rust_tools/README.md)
- [self_healing](./self_healing/README.md)
- [typescript_tools](./typescript_tools/README.md)

## 📄 檔案概覽 (Files Overview)

- `aiva_external_classifier.py` - AIVA External Module Multi-Language Classifier (v3.3)
- `aiva_external_executor.py`
- `aiva_internal_classifier.py` - AIVA Core 數據流分類分析器 (完整版 v3.3)
- `aiva_internal_executor.py`
- `system_self_explorer.py` - SystemSelfExplorer - 系統自我探索器

