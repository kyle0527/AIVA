# Jinja2 Templates

跨語言代碼生成模板目錄。

## 子目錄

| 目錄 | 說明 |
|------|------|
| [go/](go/README.md) | Go struct 模板 |
| [python/](python/README.md) | Pydantic 模型模板 |
| [rust/](rust/README.md) | Rust struct + Serde 模板 |
| [typescript/](typescript/README.md) | TypeScript interface 模板 |

## 用途

配合 `core/schema_codegen_tool.py` 使用，從 YAML Schema 自動生成多語言類型定義。

## 模板語法

使用 Jinja2 模板語法，支援：
- 變數替換：`{{ variable }}`
- 控制流：`{% for item in items %}`
- 過濾器：`{{ name | capitalize }}`

---
*Parent: [converters](../README.md)*
