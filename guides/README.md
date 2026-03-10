# AIVA 指南中心

**最後更新**: 2026-03-10

---

## 文檔結構

```
docs/technical_manuals/   ← 各模組技術手冊（架構、資料流、介面）
guides/user_manuals/      ← 操作手冊（操作步驟、使用範例）
```

---

## 操作手冊（user_manuals/）

| 冊數 | 主題 | 閱讀對象 |
|---|---|---|
| [閱讀指南](./user_manuals/使用者手冊_閱讀指南.md) | 建議閱讀順序 | 所有人 |
| [第 1 冊](./user_manuals/使用者手冊_第1冊_系統入門與架構.md) | 系統入門與架構 | 新使用者 |
| [第 2 冊](./user_manuals/使用者手冊_第2冊_AI決策流程.md) | AI 決策流程 | 操作人員 |
| [第 2-1 冊](./user_manuals/使用者手冊_第2-1冊_策略系統更新指南.md) | 策略系統更新指南 | 操作人員 |
| [第 2-2 冊](./user_manuals/使用者手冊_第2-2冊_13步驟黑盒測試架構詳解.md) | 13 步驟黑盒測試架構 | 操作人員 |
| [第 3 冊](./user_manuals/使用者手冊_第3冊_執行與適應.md) | 執行與適應 | 操作人員 |
| [第 4 冊](./user_manuals/使用者手冊_第4冊_功能模組操作.md) | 功能模組操作 | 操作人員 |
| [第 4-1 冊](./user_manuals/使用者手冊_第4-1冊_RAG_P1驗證指南.md) | RAG P1 驗證指南 | 操作人員 |
| [第 5 冊](./user_manuals/使用者手冊_第5冊_數據流分析與執行器.md) | 數據流分析與執行器 | 進階使用者 |
| [第 6 冊](./user_manuals/使用者手冊_第6冊_進階開發.md) | 進階開發 | 開發者 |

---

## 技術手冊（docs/technical_manuals/）

| 手冊 | 對應模組 |
|---|---|
| [01 Core 模組](../docs/technical_manuals/01_CORE_MODULE_TECHNICAL_MANUAL.md) | `services/core/` |
| [02 Features 模組](../docs/technical_manuals/02_FEATURES_MODULE_TECHNICAL_MANUAL.md) | `services/features/` |
| [03 Scan 模組](../docs/technical_manuals/03_SCAN_MODULE_TECHNICAL_MANUAL.md) | `services/scan/` |
| [04 Integration 模組](../docs/technical_manuals/04_INTEGRATION_MODULE_TECHNICAL_MANUAL.md) | `services/integration/` |
| [05 AIVA Common](../docs/technical_manuals/05_AIVA_COMMON_TECHNICAL_MANUAL.md) | `services/aiva_common/` |
| [06 Cognitive Core](../docs/technical_manuals/06_COGNITIVE_CORE_TECHNICAL_MANUAL.md) | `cognitive_core/` |
| [07 RAG 系統](../docs/technical_manuals/07_RAG_SYSTEM_TECHNICAL_MANUAL.md) | `cognitive_core/rag/` |
| [08 雙閉環系統](../docs/technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md) | 跨模組 |

---

## 快速導航

| 需求 | 看哪裡 |
|---|---|
| 我是新手，從哪開始？ | [第 1 冊](./user_manuals/使用者手冊_第1冊_系統入門與架構.md) |
| 我要操作系統 | 操作手冊 第 2-4 冊 |
| 我要了解某模組的架構 | 對應的技術手冊 |
| 我要開發新功能 | [05 AIVA Common](../docs/technical_manuals/05_AIVA_COMMON_TECHNICAL_MANUAL.md) + [第 6 冊](./user_manuals/使用者手冊_第6冊_進階開發.md) |
| 雙閉環如何運作？ | [08 雙閉環技術手冊](../docs/technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md) |

---

## 歸檔文件

舊版指南、分析報告、驗證報告已移至 `_archive/guides_20260310/`。
