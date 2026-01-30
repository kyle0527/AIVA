# Converters

AIVA 多語言轉換器插件包。

## 子目錄

| 目錄 | 說明 |
|------|------|
| [converters/](converters/README.md) | 格式轉換器 (SARIF, Task, DOCX) |
| [core/](core/README.md) | 代碼生成核心引擎 |
| [templates/](templates/README.md) | Jinja2 多語言模板 |

## 檔案

| 檔案 | 說明 |
|------|------|
| `__init__.py` | 包初始化 |
| `requirements.txt` | Python 依賴 |

## 功能概覽

- 🔄 SARIF 安全報告轉換
- ⚙️ AST 任務序列轉換
- 📄 Word 轉 Markdown
- 🎯 多語言 Schema 代碼生成 (Python/Go/Rust/TypeScript)

## 安裝

```bash
pip install -r requirements.txt
```

---
*Parent: [_dev_tools](../README.md)*
