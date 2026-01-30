# Development

開發分析工具集 - AST 解析與流程圖生成。

## Python 工具

| 檔案 | 說明 | 行數 |
|------|------|------|
| `aiva_flow_analyzer.py` | 流程組圖分析器 | 1433 |
| `aiva_flow_classifier_final.py` | 流程分類器 | 770 |
| `py2mermaid.py` | Python → Mermaid 轉換 | 521 |
| `analyze_classification.py` | 分類結果分析 | 70 |
| `analyze_discrepancies.py` | 差異分析 | 85 |
| `analyze_function_details.py` | 函數詳情分析 | 110 |
| `analyze_v8_correctness.py` | V8 正確性分析 | 175 |
| `check_flow_counts.py` | Flow 計數檢查 | 80 |
| `compare_versions.py` | 版本比較 | 150 |
| `detailed_v8_analysis.py` | V8 詳細分析 | 165 |
| `verify_analyzer_outputs.py` | 分析器輸出驗證 | 115 |

## 多語言 Mermaid 工具

| 檔案 | 語言 | 說明 |
|------|------|------|
| `py2mermaid.py` | Python | Python AST → Mermaid |
| `ts2mermaid.ts` | TypeScript | TS AST → Mermaid |
| `go2mermaid.go` | Go | Go AST → Mermaid |
| `rs2mermaid.rs` | Rust | Rust AST → Mermaid |
| `generate_navigation_map.ts` | TypeScript | 導航圖生成 |

## 測試

| 檔案 | 說明 |
|------|------|
| `test_ast_tools.ps1` | AST 工具測試腳本 |

## 核心功能

### Flow Analyzer
分析 Python 代碼結構，生成 Mermaid 流程圖並智能組合。

### Mermaid 轉換器
支援多語言代碼轉換為 Mermaid 圖表：
- 類別圖
- 流程圖
- 序列圖

---
*Parent: [common](../README.md)*
