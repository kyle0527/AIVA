# Core - 代碼生成核心

跨語言 Schema 代碼生成核心引擎。

## 檔案

| 檔案 | 說明 | 行數 |
|------|------|------|
| `schema_codegen_tool.py` | 多語言 Schema 生成器 | 1585 |
| `typescript_generator.py` | TypeScript 專用生成器 | 500+ |
| `cross_language_validator.py` | 跨語言一致性驗證器 | 300+ |
| `cross_language_interface.py` | 跨語言介面抽象層 | - |
| `schema_validator.py` | Schema 格式驗證器 | - |

## 核心功能

### Schema Codegen Tool
從 `core_schema_sot.yaml` 自動生成：
- Python (Pydantic v2)
- Go (structs)
- Rust (Serde)
- TypeScript (interfaces)

### 使用方式
```bash
python schema_codegen_tool.py --generate-all
python schema_codegen_tool.py --lang python --validate
```

---
*Parent: [converters](../README.md)*
