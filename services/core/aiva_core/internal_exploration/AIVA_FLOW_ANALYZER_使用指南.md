# AIVA Flow Analyzer 使用指南

## 📋 概述

**AIVA Flow Analyzer** 是一個基於「流程圖 = 數據流」理念設計的代碼分析工具。

### 🎯 核心設計理念

> **「流程圖跟數據流本質上是一樣的東西」**

傳統觀點將它們分開：
- 流程圖：控制流（程序執行順序）
- 數據流圖：數據傳遞路徑

**AIVA 的創新視角：**
- 函數調用 = 數據傳遞
- 條件分支 = 數據決策  
- 循環 = 數據迭代
- **控制流就是數據流！**

### ✨ 工具特色

1. **產圖階段**：完整移植 py2mermaid 核心邏輯
   - 為每個 Python 檔案生成獨立流程圖
   - AST 級別的精確分析
   - 支援函數、類、控制結構

2. **組圖階段**：智能頭尾匹配
   - 基於真實 import 關係
   - 自動識別數據流介面
   - 組合完整的跨檔案數據流

3. **避免虛假連結**
   - 不猜測函數調用
   - 只使用真實依賴關係
   - 準確反映系統架構

---

## 🚀 快速開始

### 基本用法

```bash
cd services/core/aiva_core/internal_exploration
python aiva_flow_analyzer.py --target <目標> --depth <深度> --max-paths <路徑數> --output <輸出目錄>
```

### 參數說明

| 參數 | 簡寫 | 默認值 | 說明 |
|------|------|--------|------|
| `--target` | `-t` | `core` | 分析目標：`all`（全部）、`core`（核心）、或指定絕對路徑 |
| `--depth` | `-d` | `3` | 數據流追蹤深度（控制組圖的鏈路長度） |
| `--max-paths` | `-mp` | `10` | 每個入口點的最大路徑數 |
| `--output` | `-o` | `./aiva_flow_analysis` | 輸出目錄 |
| `--verbose` | `-v` | `False` | 顯示詳細處理過程 |

---

## 📚 使用範例

### 範例 1：分析 AI 核心模組

```bash
# 分析 aiva_core 目錄，生成完整數據流
python aiva_flow_analyzer.py \
  --target "C:\D\fold7\AIVA-git\services\core\aiva_core" \
  --depth 3 \
  --max-paths 20 \
  --output "C:\D\fold7\AIVA-git\flow_analysis_ai_core" \
  --verbose
```

**預期輸出：**
- 處理 100+ 個 Python 檔案
- 生成數百個組合流程圖
- 建立完整的數據流地圖

### 範例 2：快速分析核心功能

```bash
# 使用默認設置快速分析
python aiva_flow_analyzer.py --target core --verbose
```

### 範例 3：深度追蹤特定模組

```bash
# 追蹤更深的數據流鏈路
python aiva_flow_analyzer.py \
  --target "C:\D\fold7\AIVA-git\services\scan" \
  --depth 5 \
  --max-paths 50
```

---

## 📊 輸出結果說明

### 輸出目錄結構

```
flow_analysis_ai_core/
├── analysis_results.json          # JSON 格式的完整分析結果
├── data_flow_summary.md           # 數據流摘要報告
├── data_flow_chain_1_*.md         # 組合流程圖 1
├── data_flow_chain_2_*.md         # 組合流程圖 2
└── ...                            # 更多組合流程圖
```

### 關鍵輸出檔案

#### 1. `data_flow_summary.md` - 數據流摘要

包含：
- 處理檔案統計
- 真實連接列表（import 關係）
- 數據源頭節點
- 整體數據流概況

#### 2. `data_flow_chain_*.md` - 組合流程圖

每個檔案包含：
- **Mermaid 圖表**：視覺化的數據流路徑
- **腳本序列**：按順序列出經過的模組
- **檔案路徑**：每個模組的完整路徑

範例：
```markdown
# data_flow_chain_1 數據流鏈路 1

\`\`\`mermaid
graph TD
    initial_surface[initial_surface]
    exploit_orchestrator[exploit_orchestrator]
    initial_surface --> exploit_orchestrator
\`\`\`

## 腳本序列
1. initial_surface
2. exploit_orchestrator

## 檔案路徑
- c:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\initial_surface.py
- c:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_orchestrator.py
```

#### 3. `analysis_results.json` - 完整結果

包含所有原始分析數據，可用於：
- 自動化處理
- 二次分析
- 工具集成

---

## 🔍 分析流程詳解

### 第一階段：產圖（生成個別流程圖）

1. **掃描目標目錄**
   - 遞迴查找所有 `.py` 檔案
   - 過濾測試檔案和特殊目錄

2. **AST 分析**
   - 使用 py2mermaid 核心邏輯
   - 解析每個檔案的抽象語法樹
   - 識別函數、類、控制結構

3. **生成流程圖**
   - 為每個函數生成獨立圖
   - 記錄 import 語句（數據流入口）
   - 記錄函數調用（數據流出口）

**輸出範例：**
```
✅ 產圖階段完成，共處理 118 個檔案
```

### 第二階段：組圖（智能頭尾匹配）

1. **建立介面索引**
   - 提取每個模組的 import（頭部）
   - 提取每個模組的 export（尾部）

2. **識別真實連接**
   - 匹配 import 和 export 關係
   - 建立模組間的依賴圖

3. **生成數據流鏈路**
   - 從源頭節點開始
   - 深度優先搜索
   - 組合完整的數據流路徑

**輸出範例：**
```
🔗 找到 194 個真實連接
🎯 找到 4 個數據源頭節點
📊 生成 268 條獨立數據流路徑
```

---

## 🎨 進階應用

### 1. 系統架構分析

使用數據流摘要快速了解系統架構：

```bash
# 分析後查看摘要
cat flow_analysis_ai_core/data_flow_summary.md
```

**關鍵信息：**
- 哪些模組是核心（被最多模組依賴）
- 數據流的起點和終點
- 模組間的依賴關係

### 2. 重構規劃

識別需要重構的模組：

```python
# 分析 analysis_results.json
import json

with open('flow_analysis_ai_core/analysis_results.json') as f:
    data = json.load(f)

# 找出被依賴最多的模組（可能需要拆分）
# 找出孤立的模組（可能需要整合）
```

### 3. 影響分析

評估修改某個模組的影響範圍：

1. 找出包含該模組的所有數據流鏈路
2. 查看有哪些下游模組會受影響
3. 規劃測試範圍

---

## 💡 最佳實踐

### 1. 選擇合適的深度

| 深度 | 適用場景 | 輸出規模 |
|------|---------|---------|
| 1-2 | 快速概覽 | 小 |
| 3-4 | 日常分析（推薦） | 中 |
| 5+ | 深度追蹤 | 大 |

### 2. 控制輸出規模

```bash
# 大型專案建議分批分析
python aiva_flow_analyzer.py --target core --depth 2
python aiva_flow_analyzer.py --target scan --depth 2
python aiva_flow_analyzer.py --target features --depth 2
```

### 3. 定期更新分析

```bash
# 建議在重大代碼變更後重新分析
git pull
python aiva_flow_analyzer.py --target all --depth 3
```

---

## 🐛 常見問題

### Q1: 為什麼組合流程圖數量這麼多？

**A:** 這是正常的！每條獨立的數據流路徑都會生成一個圖。例如：
- 118 個檔案
- 194 個連接
- 可組合出 268 條不同的數據流路徑

這反映了系統的真實複雜度。

### Q2: 如何快速找到特定功能的數據流？

**A:** 使用檔名搜索：

```bash
# 在 Windows PowerShell
Get-ChildItem flow_analysis_ai_core/*.md | Select-String "關鍵字"

# 在 Linux/Mac
grep -r "關鍵字" flow_analysis_ai_core/*.md
```

### Q3: 分析速度慢怎麼辦？

**A:** 
1. 降低深度：`--depth 2`
2. 減少路徑數：`--max-paths 5`
3. 指定特定目錄而非 `all`

---

## 🔧 技術細節

### 核心技術棧

- **AST 解析**：Python `ast` 模組
- **圖形生成**：Mermaid 語法
- **數據結構**：`dataclass` + `defaultdict`
- **演算法**：深度優先搜索（DFS）

### 代碼組織

```python
# 核心類
class Node:           # 流程圖節點
class Graph:          # 流程圖
class Builder:        # AST → Graph 轉換器
class DataFlowStitcher:  # 數據流拼接器
class AIVAFlowAnalyzer:  # 主分析器
```

### 性能優化

- ✅ 使用 `defaultdict` 建立高效索引
- ✅ 深度限制避免無限遞迴
- ✅ 路徑數限制控制輸出規模
- ✅ 批次處理檔案

---

## 📖 延伸閱讀

### 相關概念

1. **Dataflow Programming**
   - 數據流程式設計範式
   - 關注數據流動而非控制流

2. **Call Graph Analysis**
   - 函數調用圖分析
   - 靜態程序分析技術

3. **Dependency Graph**
   - 依賴圖分析
   - 模組間依賴關係

### 工具比較

| 工具 | 優勢 | 劣勢 |
|------|------|------|
| **AIVA Flow Analyzer** | 產圖+組圖一體化，真實依賴 | 僅支援 Python |
| pycallgraph | 動態調用圖 | 需要執行程序 |
| pydeps | 模組依賴圖 | 無函數級細節 |
| py2mermaid | 函數流程圖 | 無跨檔案組合 |

**AIVA 的優勢：** 結合了 py2mermaid 的精確產圖 + 智能組圖，填補了工具鏈的空白！

---

## 🎯 總結

AIVA Flow Analyzer 實現了「流程圖 = 數據流」的創新理念：

✅ **簡單**：分析圖形介面而非複雜的函數調用  
✅ **準確**：基於真實 import，無虛假連結  
✅ **清晰**：Mermaid 圖表直觀展示  
✅ **實用**：真實反映系統架構  

這是一個**獨特且強大**的代碼分析工具！

---

## 📞 支援

如有問題或建議，請聯繫 AIVA 開發團隊。

**版本：** 1.0  
**更新日期：** 2025-12-06
