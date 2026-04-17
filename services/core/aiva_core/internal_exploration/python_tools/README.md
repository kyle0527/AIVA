# Python Tools

Python Tools 模組，處理底層邏輯與具體實作細節。

此處檔案通常包含具體的類別與函式實作，提供最詳細的說明。

## 📄 檔案詳細資訊 (Files Details)

### `aiva_flow_analysis_v3.json`
**說明**: 無特定描述。


### `aiva_flow_analyzer.py`
**說明**: aiva_flow_analyzer.py (v3.0 - Refactored)

**類別 (Classes)**:
- `Node` - 代表流程圖中的一個節點
- `Graph` - 代表一個函數或模組的完整流程圖
- `ParameterExtractor` - 專門負責從 AST 提取函數參數與型別資訊
- `FlowBuilder` - 遍歷 AST 並構建 Graph。
- `FlowStitcher` - 全域註冊表與連接器
- `AIVAFlowAnalyzer` - AIVA 流程分析器 - 提供統一的分析接口
**函式 (Functions)**:
- `analyze_and_generate()` - 執行分析並生成結果
