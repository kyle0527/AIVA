# AIVA 文檔索引

**最後更新**: 2026-03-10

---

## 技術手冊（docs/technical_manuals/）

各模組的架構原理、資料流、元件介面，供開發者參考。

| 手冊 | 說明 |
|---|---|
| [01 Core 模組](./technical_manuals/01_CORE_MODULE_TECHNICAL_MANUAL.md) | 主 AI 決策中樞、13 步驟工作流、Bug Bounty 決策引擎 |
| [02 Features 模組](./technical_manuals/02_FEATURES_MODULE_TECHNICAL_MANUAL.md) | 17 個安全測試功能模組（SQLi, XSS, SSRF 等） |
| [03 Scan 模組](./technical_manuals/03_SCAN_MODULE_TECHNICAL_MANUAL.md) | Go/Rust/TypeScript/Python 四大掃描引擎 |
| [04 Integration 模組](./technical_manuals/04_INTEGRATION_MODULE_TECHNICAL_MANUAL.md) | 7 層協調中樞、攻擊路徑圖、報告生成 |
| [05 AIVA Common](./technical_manuals/05_AIVA_COMMON_TECHNICAL_MANUAL.md) | 共用基礎設施、統一 enums/schemas/config |
| [06 Cognitive Core](./technical_manuals/06_COGNITIVE_CORE_TECHNICAL_MANUAL.md) | AI 大腦、嵌入知識庫、5M 決策引擎、反幻覺 |
| [07 RAG 系統](./technical_manuals/07_RAG_SYSTEM_TECHNICAL_MANUAL.md) | 向量知識庫、去語意化協定、經驗同步 |
| [08 雙閉環系統](./technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md) | 內/外閉環自我優化架構 |

---

## 操作手冊（guides/user_manuals/）

操作步驟、工作流、使用範例，供操作者閱讀。

| 冊數 | 主題 |
|---|---|
| [第 1 冊](../guides/user_manuals/使用者手冊_第1冊_系統入門與架構.md) | 系統入門與架構 |
| [第 2 冊](../guides/user_manuals/使用者手冊_第2冊_AI決策流程.md) | AI 決策流程 |
| [第 2-1 冊](../guides/user_manuals/使用者手冊_第2-1冊_策略系統更新指南.md) | 策略系統更新指南 |
| [第 2-2 冊](../guides/user_manuals/使用者手冊_第2-2冊_13步驟黑盒測試架構詳解.md) | 13 步驟黑盒測試架構 |
| [第 3 冊](../guides/user_manuals/使用者手冊_第3冊_執行與適應.md) | 執行與適應 |
| [第 4 冊](../guides/user_manuals/使用者手冊_第4冊_功能模組操作.md) | 功能模組操作 |
| [第 4-1 冊](../guides/user_manuals/使用者手冊_第4-1冊_RAG_P1驗證指南.md) | RAG P1 驗證指南 |
| [第 5 冊](../guides/user_manuals/使用者手冊_第5冊_數據流分析與執行器.md) | 數據流分析與執行器 |
| [第 6 冊](../guides/user_manuals/使用者手冊_第6冊_進階開發.md) | 進階開發 |
| [閱讀指南](../guides/user_manuals/使用者手冊_閱讀指南.md) | 建議閱讀順序 |

---

## 快速導航

**「這個模組是做什麼的？」** → 看對應的技術手冊

**「我要怎麼執行這個功能？」** → 看操作手冊

**「整個系統怎麼運作？」** → [第 1 冊](../guides/user_manuals/使用者手冊_第1冊_系統入門與架構.md) + [06 Cognitive Core](./technical_manuals/06_COGNITIVE_CORE_TECHNICAL_MANUAL.md)

**「雙閉環是什麼？」** → [08 雙閉環技術手冊](./technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md) + [第 3 冊](../guides/user_manuals/使用者手冊_第3冊_執行與適應.md)

**「如何開發新模組？」** → [05 AIVA Common](./technical_manuals/05_AIVA_COMMON_TECHNICAL_MANUAL.md) + [第 6 冊](../guides/user_manuals/使用者手冊_第6冊_進階開發.md)

---

## 歸檔文件

舊版分析報告、驗證報告、實作計畫等文件已移至：
- `_archive/docs_20260310/`
- `_archive/guides_20260310/`
