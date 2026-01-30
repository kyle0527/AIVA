# _dev_tools

AIVA 開發工具集中區 - 整合自 `tools/`、`plugins/`、`src/`。

## 目錄結構

```
_dev_tools/
├── common/                 # 通用工具
│   ├── automation/         # PowerShell 自動化腳本 (5)
│   └── development/        # AST/Mermaid 開發工具 (16)
├── converters/             # 格式轉換器
│   ├── converters/         # SARIF, Task, DOCX 轉換 (4)
│   ├── core/               # Schema 代碼生成核心 (5)
│   └── templates/          # Jinja2 多語言模板 (4)
├── integration/            # 跨語言整合插件 (4)
│   ├── aiva-contracts-tooling/
│   ├── aiva-enums-plugin/
│   ├── aiva-go-plugin/
│   └── aiva-schemas-plugin/
└── [根目錄 Python]         # AI 核心模組 (6)
```

## 子目錄

| 目錄 | 說明 | 檔案數 |
|------|------|--------|
| [common/](common/README.md) | 通用開發工具與自動化 | 21 |
| [converters/](converters/README.md) | 格式轉換與代碼生成 | 17 |
| [integration/](integration/README.md) | 跨語言整合插件 | 14 |

## 根目錄 Python 檔案

| 檔案 | 說明 | 行數 |
|------|------|------|
| `aiva_capability_orchestrator.py` | 能力與 5M 神經網路串接器 | 799 |
| `real_ai_core.py` | 真實 AI 核心 (500萬參數神經網路) | 577 |
| `aiva_model_manager.py` | AI 模型管理器 | 410 |
| `aiva_5M_replacement_evaluation.py` | 5M 替換評估工具 | 375 |
| `scan_real_targets.py` | 真實目標掃描 | 255 |
| `debug_rust_call.py` | Rust 呼叫除錯 | 55 |

## 功能總覽

### 🧠 AI 核心 (根目錄)
- 能力編排器：整合靜態分析、動態掃描、風險評估、攻擊編排
- 5M 神經網路：500萬參數的真實 AI 實作

### 🔄 格式轉換 (converters/)
- SARIF 安全報告
- Word → Markdown
- 多語言 Schema 代碼生成

### 🔧 開發工具 (common/)
- Flow 分析器
- Mermaid 圖表生成 (Python/TS/Go/Rust)
- 自動化腳本

### 🔗 跨語言整合 (integration/)
- JSON Schema 匯出
- TypeScript 型別生成
- Enum 集中管理

---

## 來源對應

| 原目錄 | 整合到 |
|--------|--------|
| `tools/` | `common/`, 根目錄 |
| `plugins/aiva_converters/` | `converters/` |
| `tools/integration/` | `integration/` |
| `src/core/` | 根目錄 Python |

---
*整理日期: 2026-01-03*
