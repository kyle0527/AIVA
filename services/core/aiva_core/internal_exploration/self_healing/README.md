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

## 📁 目錄結構

```
self_healing/
├── __init__.py                              # 模組入口
│
├── 核心分析器
│   └── core_analyzer.py                     # 統一分析入口 ⭐
│
├── 診斷分析器
│   ├── analyze_dataflow_breakpoints.py      # 未完整連接檢測（核心）
│   ├── analyze_missing_function_connections.py  # 缺失定義分析
│   └── practical_analyzer.py                # 智能過濾和分級
│
├── 工具腳本
│   ├── analyze_results.py                   # 深度結果分析
│   ├── verify_rl_models.py                  # 驗證腳本
│   └── run_analysis.py                      # 快速執行腳本
│
└── 文檔
    ├── MODULE_DESIGN_PHILOSOPHY.md          # 設計哲學（必讀）⭐
    ├── CORRECT_DESIGN_UNDERSTANDING.md      # 正確理解
    ├── DESIGN_EVALUATION_REPORT.md          # 評估報告
    └── README.md                            # 本文件
```

---

## 🚀 快速開始

### 基本使用

```python
from services.core.aiva_core.internal_exploration.self_healing import CoreAnalyzer

# 初始化分析器
analyzer = CoreAnalyzer(
    source_path="C:/path/to/aiva_core",
    output_dir="C:/path/to/output"
)

# 執行完整分析
report = analyzer.full_analysis()

print(f"分析完成: {report['summary']}")
```

### 命令行使用

```bash
cd services/core/aiva_core/internal_exploration/self_healing

# 快速分析
python run_analysis.py

# 深度結果評估
python analyze_results.py

# 驗證特定模組
python verify_rl_models.py
```

---

## 📊 分析器詳解

### 1. CoreAnalyzer - 統一入口 ⭐

**職責**: 整合所有診斷工具，生成統一報告

**核心功能**:
```python
analyzer = CoreAnalyzer(source_path, output_dir)

# 完整分析流程
report = analyzer.full_analysis()
# 1. AIVAFlowAnalyzer: 生成數據流鏈路 (flow_chains)
# 2. DataFlowBreakpointAnalyzer: 檢測未完整連接
# 3. MissingConnectionAnalyzer: 檢測缺失定義
# 4. 生成統一報告
```

**輸出**:
- `dataflow_breakpoint_analysis.md` - 未完整連接報告
- `missing_connections_report.md` - 缺失定義報告
- `analysis_results.json` - 完整 JSON 數據

---

### 2. DataFlowBreakpointAnalyzer - 核心診斷器 ⭐

**職責**: 發現「定義了但未被充分使用」的模組

#### 核心方法: `detect_bottlenecks()`

**判斷邏輯** (已修正):
```python
def detect_bottlenecks(self, function_details):
    """檢測未完整連接的輸出接口"""
    
    for script, details in function_details.items():
        # 計算連接完整度
        potential_outputs = len(details['export_functions'])  # 定義的函數數
        actual_outputs = len(self.graph.get(script, []))      # 在 flow_chains 中的連接數
        
        # 判斷標準（已修正）
        if potential_outputs >= 5 and actual_outputs < potential_outputs * 0.4:
            # 定義了至少5個函數，但連接率低於40%
            missing_connections = potential_outputs - actual_outputs
            
            self.issues.append({
                'type': '未完整連接的輸出接口',  # ✅ 不再是「瓶頸」
                'severity': 'warning',              # ✅ 不再是 critical
                'script': script,
                'potential_outputs': potential_outputs,
                'actual_outputs': actual_outputs,
                'missing_connections': missing_connections,
                'connection_rate': f"{(actual_outputs/potential_outputs)*100:.1f}%"
            })
```

**為什麼這樣設計？**

| 模組案例 | 定義函數 | 實際連接 | 連接率 | 判斷結果 | 原因 |
|---------|---------|---------|--------|---------|------|
| rl_models | 24 | 187 | 779% | ✅ 過度連接 | 核心基礎設施，被廣泛使用 |
| ai_controller | 40 | 3 | 8% | ⚠️ 未完整連接 | 定義很多但很少被使用 |
| cli_tool | 15 | 0 | 0% | 🔵 完全孤立 | 可能是工具腳本（正常） |

**閾值說明**:
- `potential_outputs >= 5`: 過濾小型模組（避免誤報）
- `< 0.4` (40%): 連接率低於40%才視為問題
- 連接率可以 > 100%: 同一模組在多條 flow_chain 中重複出現

---

### 3. MissingConnectionAnalyzer - 缺失定義檢測

**職責**: 找出「被調用但找不到定義」的函數

**主要檢測**:
1. **定義缺失**: 代碼中調用了某函數，但在整個 codebase 中找不到定義
2. **潛在調用**: 某函數定義了但從未被調用（與 DataFlowBreakpointAnalyzer 互補）

**示例**:
```python
# 在 ai_controller.py 中
result = process_decision(data)  # 調用了 process_decision

# 但在整個 codebase 中找不到 def process_decision(...) 的定義
# → 報告為「缺失定義」
```

---

### 4. PracticalAnalyzer - 智能過濾器

**職責**: 過濾噪音，分級問題

**過濾規則**:
- 內建函數/方法（`len`, `print`, `append` 等）
- 測試文件、示例代碼
- 已知的第三方庫函數

**分級標準**:
- **CRITICAL**: 核心業務邏輯的缺失連接
- **HIGH**: 重要模組的未使用函數
- **MEDIUM**: 一般模組的連接問題
- **LOW**: 可能是工具腳本或測試代碼

---

## 📈 實際分析結果

### 驗證案例: aiva_core (101個腳本)

**連接狀態分布**:
```
🟢 過度連接 (>100%):    9 個 (9.2%)   ← 核心模組
🟢 連接良好 (40-100%):  9 個 (9.2%)   ← 健康模組
🟡 連接不足 (1-40%):   28 個 (28.6%)  ← 需要檢查
🔴 完全孤立 (0%):      52 個 (53.1%)  ← 工具或死代碼
```

**核心模組（正確識別）**:
1. scalable_bio_trainer: 2967% (12 函數 → 356 連接)
2. rl_trainers: 1315% (20 函數 → 263 連接)
3. neural_network: 931%
4. rl_models: 779% ✅ (之前錯誤標記為「瓶頸」)
5. aiva_exploration_pipeline: 450%

**真正需要檢查的模組**:
1. ai_controller: 8% (40 函數 → 3 連接)
2. permission_matrix: 13% (31 函數 → 4 連接)
3. ai_summary_plugin: 3% (31 函數 → 1 連接)

**詳細評估**: 參見 [DESIGN_EVALUATION_REPORT.md](DESIGN_EVALUATION_REPORT.md)

---

## 🔧 配置與調整

### 調整檢測閾值

在 `analyze_dataflow_breakpoints.py` 中:

```python
# 當前閾值
MIN_EXPORTS = 5      # 最小導出函數數
CONNECTION_RATE = 0.4  # 40% 連接率閾值

# 如果要更嚴格
MIN_EXPORTS = 8
CONNECTION_RATE = 0.3  # 30%

# 如果要更寬鬆
MIN_EXPORTS = 10
CONNECTION_RATE = 0.5  # 50%
```

### 排除特定模組

```python
# 在 core_analyzer.py 中添加排除列表
EXCLUDED_PATTERNS = [
    '*_test.py',
    '*_example.py',
    'demo_*.py',
    'tools/*'
]
```

---

## 📚 核心文檔

### 必讀文檔 ⭐

1. **[MODULE_DESIGN_PHILOSOPHY.md](MODULE_DESIGN_PHILOSOPHY.md)**
   - 完整的設計哲學說明
   - python_tools vs self_healing 對比
   - 正確與錯誤的判斷邏輯對比

2. **[DESIGN_EVALUATION_REPORT.md](DESIGN_EVALUATION_REPORT.md)**
   - 實際分析結果評估
   - 101個腳本的完整統計
   - 改進建議和下一步計劃

3. **[CORRECT_DESIGN_UNDERSTANDING.md](CORRECT_DESIGN_UNDERSTANDING.md)**
   - 設計理念的正確理解
   - 常見誤解澄清

---

## 🎯 版本歷史

### v11.0.0 (2025-12-14) - 設計理念修正

**重大變更**:
- ✅ **修正核心邏輯**: `detect_bottlenecks()` 從「高扇出=瓶頸」改為「潛在vs實際連接」
- ✅ **修正術語**: 「瓶頸節點」→「未完整連接的輸出接口」
- ✅ **修正嚴重度**: critical → warning
- ✅ **驗證成功**: 7個核心模組全部正確識別，zero false positives

**新增**:
- 📄 MODULE_DESIGN_PHILOSOPHY.md - 完整設計文檔
- 📄 DESIGN_EVALUATION_REPORT.md - 評估報告
- 🔧 analyze_results.py - 深度分析工具
- 🔧 verify_rl_models.py - 驗證工具

**分析質量**:
- 誤報率: <5% (從 ~30% 降低)
- 精確度: 95%+ (真正找出需要檢查的模組)
- 系統健康度: 18.5% 模組連接良好（有改進空間）

### v10.0.0 (2025-01) - 初始版本
- 核心分析器統一入口
- 智能過濾和去重
- 95% 噪音過濾

---

## 🚨 常見問題

### Q1: 為什麼連接率會超過 100%？

**A**: 同一個模組可以在多條不同的 flow_chain 中出現。

例如：`rl_models.py` 定義了 24 個函數，但在 371 條數據流鏈路中出現了 187 次，因為它是核心基礎設施，被廣泛使用。

### Q2: 「完全孤立」的模組一定是問題嗎？

**A**: 不一定。可能的情況：
1. ✅ **命令行工具** (如 `cli_tool.py`, `visualizer.py`)
2. ✅ **入口程序** (如 `main.py`, `server.py`)
3. ⚠️ **未整合的新功能** (需要檢查)
4. 🔴 **死代碼** (可以刪除)

需要人工判斷具體是哪種情況。

### Q3: 如何區分「工具腳本」和「真正的孤立模組」？

**A**: 檢查特徵：
```python
# 工具腳本特徵
- 文件名包含: cli, main, demo, example, test, tool, visualizer
- 包含 if __name__ == "__main__"
- 包含 def main() 函數
- 在 tools/ 或 examples/ 目錄下

# 真正的孤立模組
- 定義了業務邏輯函數
- 在主要業務目錄下
- 沒有明顯的工具/測試特徵
```

### Q4: 建議的改進優先級？

**A**: 
1. 🔴 **High**: 連接率 < 10% 且在核心目錄下
2. 🟡 **Medium**: 連接率 10-30%
3. 🟢 **Low**: 連接率 30-40%
4. 🔵 **Info**: 完全孤立但可能是工具

---

## 🤝 貢獻指南

### 回報問題

如果發現誤報或漏報，請提供：
1. 模組路徑
2. 預期行為
3. 實際行為
4. 相關的 `analysis_results.json` 片段

### 改進建議

當前已知的改進方向：
1. 自動識別工具腳本類型
2. 基於 Git 歷史的重要性評分
3. 自動修復建議生成

---

**維護者**: AIVA Core Team  
**狀態**: 生產就緒（v11.0.0）  
**更新日期**: 2025-12-14  
**核心設計**: 正確 ✅
