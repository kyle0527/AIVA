# Internal Exploration 實際狀況分析報告

**分析日期**: 2025年12月6日  
**目錄**: `services/core/aiva_core/internal_exploration/`  
**狀態**: ⚠️ 部分完成，存在多個未整合檔案

> **⚠️ 2026-01-09 狀態更新**  
> 本報告撰寫於 2025-12-06，以下項目已於後續版本修復：
> - ✅ **AI 整合已完成**: `internal_loop_connector.py` (2036 行) 已被 20+ 模組引用
> - ✅ **決策查詢已實現**: `EnhancedDecisionAgent` 可查詢 RAG 知識庫
> - ✅ **BioNeuronDecisionController 整合**: 透過 InternalLoopConnector 可使用能力數據
> - 最新狀態請參考 `services/README.md` (v7.1-stable)

---

## 📊 目錄結構實況

### 檔案清單與狀態

| 檔案名 | 行數 | 最後修改 | 天數 | 狀態 | 說明 |
|--------|------|---------|------|------|------|
| `__init__.py` | 29 | 2025-11-16 | 20天 | ✅ 正常 | 只匯出 2 個組件 |
| `module_explorer.py` | 199 | 2025-11-16 | 19天 | ✅ 生產就緒 | 模組掃描器 |
| `language_extractors.py` | 545 | 2025-11-16 | 19天 | ✅ 生產就緒 | 多語言提取器 |
| `models.py` | 129 | 2025-11-29 | 7天 | ✅ 正常 | SQLAlchemy 數據模型 |
| `capability_registry.py` | 514 | 2025-11-29 | 7天 | ✅ 正常 | 能力註冊中心 |
| `capability_analyzer.py` | 1042 | 2025-12-06 | **今天** | 🔄 整合中 | 整合 py2mermaid |
| `capability_analyzer_old.py` | 539 | 2025-12-06 | **今天** | ⚠️ 舊版本 | 應刪除 |
| `aiva_flow_analyzer.py` | 1157 | 2025-12-06 | **今天** | 🚨 **有錯誤** | 流程組圖分析器 |
| `chart_category_analyzer.py` | 278 | 2025-12-06 | **今天** | ⚠️ 實驗性 | 圖表分類器 |

### 子目錄

```
internal_exploration/
├── __pycache__/                      # 編譯緩存
├── aiva_flow_analysis/               # ✅ 流程分析結果（11個檔案）
│   ├── analysis_results.json         # 總結報告
│   └── *.md                          # 個別流程圖
└── flow_analysis_results/            # ⚠️ 空目錄（僅2個JSON）
    ├── analysis_summary.json         # 30個檔案分析
    └── category_analysis.json        # 分類指南
```

---

## 🔍 核心問題分析

### 1. **aiva_flow_analyzer.py 存在編譯錯誤** 🚨

**檢測到的問題**：
```python
# 1. 認知複雜度過高 (24 > 15)
def _analyze_script_head_tail(self, node: ScriptNode, graph, script_path: str):
    # 24 的認知複雜度，需要重構

# 2. 未使用的參數
def _analyze_script_head_tail(self, node: ScriptNode, graph, script_path: str):
    # 'graph' 參數從未被使用

# 3. 可合併的 if 語句
if node_ast.module:
    if some_condition:  # 應該合併

# 4. 集合構造器應改用推導式
'unique_sources': len(set(s for s, _ in self.real_connections))  # 應改用 {s for s, ...}

# 5. 不可達代碼
return self.results  # 這行永遠執行不到
```

**影響**：
- ❌ 代碼品質不符合 SonarQube 標準
- ⚠️ 可能存在邏輯錯誤
- 🐛 不可達代碼表示流程控制問題

---

### 2. **功能重複與未整合** ⚠️

#### **capability_analyzer.py vs capability_analyzer_old.py**

```python
# capability_analyzer.py (1042 行) - 2025-12-06 修改
# ✅ 整合了 py2mermaid 代碼
# ✅ 完整的 AST 分析
# ✅ 支援流程圖生成

# capability_analyzer_old.py (539 行) - 2025-12-06 修改  
# ⚠️ 舊版本，功能被新版取代
# ❌ 應該刪除或歸檔
```

**建議**：刪除 `capability_analyzer_old.py`

---

### 3. **分析結果不完整** ⚠️

#### **aiva_flow_analysis/** (完整)
```json
{
  "total_files_processed": 1620,
  "total_graphs": 1613,
  "total_stitched_sequences": 3181,
  "entry_functions": ["main", "get_config"]
}
```
✅ 包含 11 個 Markdown 流程圖
✅ 完整的分析結果

#### **flow_analysis_results/** (不完整)
```json
{
  "total_files_analyzed": 30,
  "total_individual_charts": 411,
  "total_combined_charts": 0  // ❌ 沒有組合圖！
}
```
⚠️ 只分析了 30 個檔案
⚠️ 沒有生成組合圖
⚠️ 分類指南是空的（categories: {}）

---

### 4. **__init__.py 匯出不完整** ⚠️

```python
# 當前只匯出 2 個組件
from .module_explorer import ModuleExplorer
from .capability_analyzer import CapabilityAnalyzer

__all__ = [
    "ModuleExplorer",
    "CapabilityAnalyzer",
]

# ❌ 缺失的組件：
# - CapabilityRegistry (capability_registry.py 存在)
# - AivaFlowAnalyzer (aiva_flow_analyzer.py 存在但有錯)
# - ChartCategoryAnalyzer (chart_category_analyzer.py 存在)
```

---

## 📋 實際完成度評估

### ✅ **已完成且可用**

| 組件 | 檔案 | 狀態 | 功能 |
|------|------|------|------|
| **模組探索** | module_explorer.py | ✅ 100% | 掃描五大模組 |
| **語言提取** | language_extractors.py | ✅ 100% | Python/Go/Rust/TS 提取 |
| **數據模型** | models.py | ✅ 100% | SQLAlchemy ORM |
| **能力註冊** | capability_registry.py | ✅ 100% | PostgreSQL 存儲 |

### 🔄 **部分完成**

| 組件 | 檔案 | 狀態 | 問題 |
|------|------|------|------|
| **能力分析** | capability_analyzer.py | 🔄 90% | 整合 py2mermaid，但未測試 |
| **流程分析** | aiva_flow_analyzer.py | 🔄 70% | 有編譯錯誤，需重構 |
| **圖表分類** | chart_category_analyzer.py | 🔄 50% | 功能存在但未整合 |

### ❌ **未完成**

| 組件 | 狀態 | 說明 |
|------|------|------|
| **知識圖譜** | ❌ 0% | 文檔提到但未實現 |
| **自我診斷** | ❌ 0% | 文檔提到但未實現 |
| **AST 分析器** | ❌ 0% | 文檔提到但未實現（功能在 capability_analyzer 中） |

---

## 🎯 真實能力清單

### 當前可用功能

#### 1. **內閉環核心流程** ✅
```python
from services.core.aiva_core.internal_exploration import (
    ModuleExplorer,
    CapabilityAnalyzer
)

# ✅ 可執行
explorer = ModuleExplorer()
analyzer = CapabilityAnalyzer()
modules = await explorer.explore_all_modules()
capabilities = await analyzer.analyze_capabilities(modules)
```

**功能**：
- ✅ 掃描 4 大模組（core/scan/features/integration）
- ✅ 支援 5 種語言（Python/Go/Rust/TypeScript/JavaScript）
- ✅ 提取能力函數信息
- ✅ 存儲到 PostgreSQL

#### 2. **流程圖生成** 🔄（有錯誤）
```python
# ⚠️ 目前有編譯錯誤，需修復
from .aiva_flow_analyzer import AivaFlowAnalyzer

analyzer = AivaFlowAnalyzer()
results = analyzer.analyze_directory(target="core", depth=3)
# 已生成 3,181 個流程序列
```

**功能**：
- 🔄 生成單檔案流程圖（已完成）
- 🔄 跨檔案流程拼接（已完成）
- 🔄 流程序列索引（已完成）
- ❌ 代碼品質不達標（需重構）

#### 3. **能力註冊中心** ✅
```python
from .capability_registry import CapabilityRegistry

registry = CapabilityRegistry(db_session)
registry.register_capabilities(capabilities)
# ✅ PostgreSQL 存儲
# ✅ 版本控制
# ✅ 變更追蹤
```

---

## 🚨 關鍵問題總結

### P0 - 阻塞性問題

1. **aiva_flow_analyzer.py 編譯錯誤** 🚨
   - 9 個 SonarQube 錯誤
   - 認知複雜度過高（需重構 2 個函數）
   - 存在不可達代碼（邏輯錯誤）

2. **舊檔案未清理** ⚠️
   - `capability_analyzer_old.py` 應刪除
   - 造成混淆和維護負擔

### P1 - 功能缺失

3. **流程圖未分類** ⚠️
   - 3,181 個流程序列未分類
   - `category_analysis.json` 為空（categories: {}）
   - 分類邏輯存在但未執行

4. **未整合到 AI 決策** ❌
   - 流程圖數據無法被 AI 查詢
   - 未向量化存儲到 RAG
   - BioNeuronDecisionController 無法使用

### P2 - 架構問題

5. **__init__.py 不完整** ⚠️
   - 只匯出 2 個組件（應有 5+ 個）
   - 新增組件未加入匯出列表
   - 外部無法導入完整功能

---

## 📝 修復優先級建議

### Phase 1: 修復編譯錯誤（1 天）

#### Task 1.1: 修復 aiva_flow_analyzer.py
```python
# 1. 降低認知複雜度
def _analyze_script_head_tail(self, node: ScriptNode, script_path: str):
    # 移除未使用的 graph 參數
    # 拆分為多個小函數

# 2. 修復集合構造器
'unique_sources': len({s for s, _ in self.real_connections})

# 3. 移除不可達代碼
# 分析流程控制，確保所有代碼可達
```

#### Task 1.2: 清理舊檔案
```bash
# 刪除或移到 _archive
mv capability_analyzer_old.py _archive/
```

#### Task 1.3: 更新 __init__.py
```python
from .module_explorer import ModuleExplorer
from .capability_analyzer import CapabilityAnalyzer
from .capability_registry import CapabilityRegistry
# from .aiva_flow_analyzer import AivaFlowAnalyzer  # 修復後再加入

__all__ = [
    "ModuleExplorer",
    "CapabilityAnalyzer", 
    "CapabilityRegistry",
]
```

---

### Phase 2: 完成流程圖分類（2-3 天）

#### Task 2.1: 執行流程分類
```python
# chart_category_analyzer.py 已有邏輯，需執行
from .chart_category_analyzer import analyze_chart_categories

categories, details = analyze_chart_categories()
# 輸出到 flow_analysis_results/category_analysis.json
```

#### Task 2.2: 生成組合圖
```python
# 生成 total_combined_charts（目前為 0）
# 將 3,181 個序列組合成完整流程圖
```

---

### Phase 3: AI 整合（1 週）

#### Task 3.1: 向量化流程圖
```python
# 將 3,181 個流程序列存入 RAG
from cognitive_core.internal_loop_connector import InternalLoopConnector

connector = InternalLoopConnector()
await connector.sync_flows_to_rag(flow_sequences)
```

#### Task 3.2: AI 決策使用
```python
# BioNeuronDecisionController 查詢流程圖
query = "如何掃描 SQL 注入"
flows = await connector.query_capabilities(query)
# 返回完整的執行流程
```

---

## 📊 總結

### 當前狀態
- ✅ **核心功能完成**: 60% (module_explorer, capability_analyzer, capability_registry)
- 🔄 **流程分析部分完成**: 70% (已生成 3,181 序列，但有錯誤)
- ❌ **AI 整合未開始**: 0%

### 實際可用
- ✅ 內閉環基礎流程（掃描 → 分析 → 存儲）
- ✅ PostgreSQL 能力註冊中心
- 🔄 流程圖生成（需修復錯誤）
- ❌ AI 決策使用流程圖

### 下一步行動
1. **立即修復** aiva_flow_analyzer.py 編譯錯誤
2. **清理** capability_analyzer_old.py
3. **執行** 流程圖分類（chart_category_analyzer.py）
4. **整合** 流程圖到 RAG 知識庫

---

**報告生成時間**: 2025-12-06  
**分析工具**: VS Code + Pylance + SonarQube  
**數據來源**: 實際檔案掃描 + 錯誤檢測 + JSON 分析結果
