# External Tools - 外部第三方工具

## 📑 目錄
- [概述](#概述)
- [包含的工具](#包含的工具)
  - [NoSQLMap](#nosqlmap)
  - [sqlmap](#sqlmap)
  - [sql-injection-payload-list](#sql-injection-payload-list)
- [重要說明](#重要說明)
  - [不修改外部工具](#不修改外部工具)
  - [代碼質量檢查](#代碼質量檢查)
  - [集成方式](#集成方式)
- [授權信息](#授權信息)
- [更新外部工具](#更新外部工具)
- [使用無問題分析](#使用無問題分析)
- [相關文檔](#相關文檔)

---

## 概述

這個目錄包含從外部來源集成的第三方安全工具。所有工具均通過子進程執行，不導入其 Python 代碼，確保代碼隔離和系統穩定性。

---

## 包含的工具

### NoSQLMap
- **來源**: https://github.com/codingo/NoSQLMap
- **用途**: NoSQL 注入檢測
- **語言**: Python 2
- **狀態**: 外部工具，保持原始代碼不修改

### sqlmap
- **來源**: https://github.com/sqlmapproject/sqlmap
- **用途**: SQL 注入自動化檢測和利用
- **語言**: Python 3
- **狀態**: 外部工具，保持原始代碼不修改

### sql-injection-payload-list
- **來源**: https://github.com/payloadbox/sql-injection-payload-list
- **用途**: SQL 注入 payload 庫
- **狀態**: 外部資源，保持原始內容不修改

---

## 重要說明

### 不修改外部工具
依照業界最佳實踐和項目規範：
- ✅ **保持原樣**: 外部工具維持原始狀態
- ✅ **隔離使用**: 通過包裝層（hackingtool_engine.py）調用
- ✅ **忽略警告**: 外部工具的代碼風格警告已配置忽略

### 代碼質量檢查
這些外部工具已從以下檢查中排除：
- Ruff (pyproject.toml)
- MyPy (pyproject.toml)
- Pylint (.pylintrc)

### 集成方式
```python
# 通過 hackingtool_engine.py 包裝調用
from engines.hackingtool_engine import HackingToolDetectionEngine

engine = HackingToolDetectionEngine()
results = await engine.detect(task, client)
---

##

## 📝 授權信息

所有外部工具遵循其原始授權條款：
- **NoSQLMap**: BSD 許可證
- **sqlmap**: GPL v2 許可證
- **sql-injection-payload-list**: MIT 許可證

---

## 更新外部工具的 LICENSE 文件。

## 🔄 更新

如需更新外部工具：
1. 從官方倉庫獲取最新版本
2. 替換對應目錄
3. 測試 hackingtool_engine.py 的集成是否正常
## 使用無問題分析

### 錯誤狀態
- **功能性錯誤**: 0 個 ✅
- **代碼風格警告**: ~1800 個（全部來自 NoSQLMap Python 2 代碼）
- **影響使用**: 無影響 ✅

### 為什麼無影響？

1. **子進程執行**：工具通過命令行調用，不導入 Python 代碼
2. **代碼隔離**：外部工具與 AIVA 完全隔離
3. **配置忽略**：已在 pyproject.toml 和 .pylintrc 中排除

詳細分析請參閱：[USAGE_AND_ISSUES.md](USAGE_AND_ISSUES.md)

---

## 相關文檔

- [上層文檔：function_sqli README](../README.md)
- [使用分析詳情：USAGE_AND_ISSUES.md](USAGE_AND_ISSUES.md)
- [檢測引擎：engines](../engines/README.md)
- [工具整合：integration_tools](../integration_tools/README.md)

---

**維護準則**: 外部工具 = 黑盒組件，只通過標準接口集成，不修改內部代碼。

**更新日期**: 2025年12月12日

---

**維護準則**: 外部工具 = 黑盒組件，只通過標準接口集成，不修改內部代碼。
