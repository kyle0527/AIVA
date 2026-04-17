# Dialog 對話模組

> **路徑**: `core_capabilities/dialog/`  
> **狀態**: ✅ 正常 | **Python 文件數**: 3 | **最後更新**: 2026-04-05  
> **父模組**: [Core Capabilities](../README.md)

## 概述

AIVA 對話助理模組，實現 AI 對話層，支援自然語言問答、智能選單和一鍵執行功能。

## 📄 檔案詳細資訊 (Files Details)

### `ai_menu.py`
**說明**: AIVA AI 智能選單系統

**類別 (Classes)**:
- `AIVAIntelligentMenu` - AIVA 智能選單系統

### `assistant.py`
**說明**: AIVA 對話助理模組

**類別 (Classes)**:
- `DialogIntent` - 對話意圖識別
- `AIVACommandProcessor` - AIVA 指令處理器
- `_LazyDialogAssistant` - 延遲載入的對話助理包裝器
**函式 (Functions)**:
- `get_dialog_assistant()` - 獲取對話助理實例（延遲初始化）

