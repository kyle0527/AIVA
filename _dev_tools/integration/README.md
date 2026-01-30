# Integration

Rust/Python 跨語言整合插件 (Poetry/PyPI 套件)。

## 子目錄

| 目錄 | 說明 |
|------|------|
| `aiva-contracts-tooling/` | JSON Schema & TypeScript 型別匯出 CLI |
| `aiva-enums-plugin/` | 列舉型別集中管理 |
| `aiva-go-plugin/` | Go 語言整合插件 |
| `aiva-schemas-plugin/` | Schema 模型集中管理 |

## 各插件簡介

### aiva-contracts-tooling
從 `aiva_schemas_plugin` 自動匯出：
- JSON Schema
- TypeScript `.d.ts` 型別

```bash
aiva-contracts export-jsonschema --out ./schemas/aiva_schemas.json
aiva-contracts gen-ts --json ./schemas/aiva_schemas.json --out ./schemas/aiva_schemas.d.ts
```

### aiva-enums-plugin
集中管理所有列舉型別 (enums)：
- Python 端轉接 `aiva_common.enums`
- 可生成 TypeScript `enums.ts`

### aiva-schemas-plugin
將 `schemas.py` 依賴集中到單一入口：
- `from aiva_schemas_plugin import ...`
- 支援後續分離為獨立套件

### aiva-go-plugin
Go 語言整合支援。

## 安裝

```bash
pip install -e ./aiva-schemas-plugin
pip install -e ./aiva-enums-plugin
pip install -e ./aiva-contracts-tooling
```

---
*Parent: [_dev_tools](../README.md)*
