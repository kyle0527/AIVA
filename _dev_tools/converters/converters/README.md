# Converters - 格式轉換器

格式轉換器模組，提供 SARIF、Task、DOCX 等格式的轉換功能。

## 檔案

| 檔案 | 說明 |
|------|------|
| `sarif_converter.py` | SARIF 2.1.0 安全報告轉換器 |
| `task_converter.py` | AST 任務序列轉換器 |
| `docx_to_md_converter.py` | Word 文檔轉 Markdown |
| `__init__.py` | 模組初始化 |

## 功能

### SARIF Converter
將掃描結果轉換為 SARIF 2.1.0 標準格式，用於安全工具整合。

### Task Converter
將 AST 分析結果轉換為可執行的任務序列。

### DOCX to MD
將 Word 文檔批量轉換為 Markdown 格式。

---
*Parent: [converters](../README.md)*
