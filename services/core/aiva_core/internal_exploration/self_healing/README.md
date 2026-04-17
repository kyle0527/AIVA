# Self-Healing 自我診斷模組

> **版本**: v2.2.0  
> **最後更新**: 2026-01-07  
> **狀態**: ✅ 生產就緒  
> **檔案數**: 8 個 Python 模組  
> **代碼行數**: 3,711 行  
> **核心理念**: 找出系統中「定義了但尚未被連接使用」的輸出接口，而非將「重要的核心模組」誤判為瓶頸

---

## 🎯 設計哲學

### 與 python_tools 的對比

| 模組 | 目的 | 工作方式 |
|------|------|----------|
| **python_tools** | 找到能夠連結的並接起來 | 基於類型匹配自動建立數據流連接 |
| **self_healing** | 找出尚未接起來的輸出入接口 | 檢測已定義但未被使用的函數和接口 |

**關鍵區別**: 
- python_tools 是「連接器」- 主動建立連接
- self_healing 是「診斷器」- 被動發現缺失連接

### 正確的分析邏輯

```python
# ❌ 錯誤: 高扇出 = 瓶頸
if fan_out > average * 2:
    issue = "瓶頸節點"  # 會誤判重要模組

# ✅ 正確: 潛在輸出 vs 實際連接
if potential_exports >= 5 and actual_connections < potential_exports * 0.4:
    issue = "未完整連接的輸出接口"  # 真正的問題
```

**示例**:
- `rl_models.py`: 定義 24 個函數，被使用 187 次 (779%) → ✅ 重要核心模組
- `ai_controller.py`: 定義 40 個函數，被使用 3 次 (8%) → ⚠️ 需要檢查

---

## 📄 檔案詳細資訊 (Files Details)

### `analyze_connection_recommendations.py`
**說明**: 連接建議分析器 - 分析哪些函數應該連接以及影響評估

**類別 (Classes)**:
- `FunctionInfo` - 函數信息
- `ConnectionRecommendation` - 連接建議
- `ConnectionRecommendationAnalyzer` - 連接建議分析器
**函式 (Functions)**:
- `main()` - 主函數

### `analyze_dataflow_breakpoints.py`
**說明**: 數據流斷點分析器

**類別 (Classes)**:
- `BreakpointIssue` - 數據流斷點問題
- `DataFlowBreakpointAnalyzer` - 數據流斷點分析器
**函式 (Functions)**:
- `main()`

### `analyze_missing_function_connections.py`
**說明**: 缺失函數連接分析器

**類別 (Classes)**:
- `FunctionSignature` - 函數簽名
- `MissingConnection` - 缺失的連接
- `MissingConnectionAnalyzer` - 缺失連接分析器
**函式 (Functions)**:
- `main()`

### `analyze_results.py`
**說明**: 分析結果深度評估腳本

**函式 (Functions)**:
- `load_analysis_data()` - 載入分析數據
- `print_basic_statistics()` - 列印基礎統計資訊
- `build_connection_graph()` - 構建連接圖
- `calculate_connection_stats()` - 計算每個腳本的連接統計
- `classify_connection_status()` - 分類連接狀態
- `group_by_status()` - 按狀態分組
- `print_connection_analysis()` - 列印連接完整度分析
- `print_typical_cases()` - 列印典型案例分析
- `print_over_connected_modules()` - 列印過度連接的模組
- `print_under_connected_modules()` - 列印連接不足的模組
- `print_isolated_modules()` - 列印完全孤立的模組
- `verify_design_principles()` - 驗證設計理念
- `generate_improvement_suggestions()` - 生成改進建議
- `print_improvement_suggestions()` - 列印改進建議
- `export_detailed_data()` - 導出詳細數據
- `analyze_report_quality()` - 分析報告質量

### `core_analyzer.py`
**說明**: 核心分析器 - 統一入口

**類別 (Classes)**:
- `AnalysisReport` - 統一的分析報告
- `CoreAnalyzer` - 核心分析器 - 統一入口
**函式 (Functions)**:
- `classify_script_type()` - 識別腳本類型，避免將工具腳本誤判為孤立模組
- `main()` - 命令行入口

### `practical_analyzer.py`
**說明**: 實用錯誤分析器 - 分級展示，確保重要問題不遺漏

**類別 (Classes)**:
- `Issue` - 問題
- `PracticalAnalyzer` - 實用分析器
**函式 (Functions)**:
- `main()`

### `run_analysis.py`
**說明**: Self-Healing 分析執行腳本 (修復優化版)

**類別 (Classes)**:
- `AnalysisRunner` - 分析執行器 - 封裝各個分析器的調用邏輯
**函式 (Functions)**:
- `main()` - 主函數

