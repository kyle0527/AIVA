# Dialog 對話模組

> **路徑**: `core_capabilities/dialog/`  
> **狀態**: ✅ 正常 | **Python 文件數**: 3 | **最後更新**: 2026-04-05  
> **父模組**: [Core Capabilities](../README.md)

## 概述

AIVA 對話助理模組，實現 AI 對話層，支援自然語言問答、智能選單和一鍵執行功能。

## 核心組件

### ai_menu.py ⭐ 新增
- `AIVAIntelligentMenu` - AI 智能選單系統
  - 動態選單生成
  - 意圖識別
  - 能力查詢整合
  - 696 行代碼

### assistant.py
- `DialogIntent` - 對話意圖識別器
  - 意圖模式匹配：list_capabilities、explain_capability、run_scan、compare_capabilities、generate_cli、system_status
  - `classify_command()` - 識別用戶指令類型
  
- `AIVACommandProcessor` - AIVA 命令處理器
  - 處理對話請求
  - 調用 CapabilityRegistry 執行功能
  - 格式化輸出結果

- `_LazyDialogAssistant` - 延遲初始化的對話助理（單例模式）

## 支援的意圖類型

| 意圖 | 觸發詞彙 |
|------|----------|
| 列出能力 | 現在系統會什麼、你會什麼、有什麼功能 |
| 解釋能力 | 解釋、說明、介紹 |
| 執行掃描 | 掃描、scan、測試、attack |
| 比較能力 | 比較、差異、對比 |
| 生成 CLI | 產生 CLI、輸出指令 |
| 系統狀態 | 系統狀況、健康檢查 |

## 依賴關係

- `aiva_common.utils.logging` - 統一日誌
- `integration.capability` - CapabilityRegistry 能力註冊表
- `re` - 正則表達式意圖匹配
