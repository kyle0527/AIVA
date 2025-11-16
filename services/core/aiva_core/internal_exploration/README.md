# Internal Exploration - 對內探索模組

**導航**: [← 返回 AIVA Core](../README.md) | [📖 重構計劃](../REFACTORING_PLAN.md)

## 📋 目錄

- [概述](#概述)
- [核心職責](#核心職責)
- [目錄結構](#目錄結構)
- [核心組件說明](#核心組件說明)
- [閉環機制](#閉環機制)
- [設計理念](#設計理念)
- [使用範例](#使用範例)
- [遷移狀態](#遷移狀態)

---

## 📋 概述

> **🎯 定位**: AI 的「自我認知」能力  
> **✅ 狀態**: 系統就緒，測試通過  
> **🧪 測試狀態**: 階段 8 測試覆蓋 (ModuleExplorer, CapabilityAnalyzer)  
> **🔄 最後更新**: 2025年11月16日

**Internal Exploration** 是 AIVA Core 的對內探索模組,負責掃描和分析 AIVA 系統自身的五大模組 (ai_core, attack_engine, scan_engine, integration, features),構建全專案知識圖譜,實現 AI 自我認知能力。

### 🎯 核心職責

- ✅ **模組探索**: 掃描 AIVA 五大模組的代碼結構
- ✅ **能力分析**: 識別 `@register_capability` 標記的能力函數
- ✅ **AST 解析**: 將 Python 代碼解析為抽象語法樹
- ✅ **知識圖譜**: 構建系統能力的知識圖譜和依賴關係
- ✅ **自我診斷**: 檢測系統「路不通」的地方和潛在問題

---

## 📂 目錄結構

```
internal_exploration/
├── module_explorer.py            # 模組探索器 (已實現)
├── capability_analyzer.py        # 能力分析器 (已實現)  
├── __init__.py                   # 模組入口 (已實現)
└── README.md                     # 本文檔

總計: 3 個 Python 檔案 (簡約設計)
```

---

## 🎨 核心組件說明

### 1️⃣ ModuleExplorer (模組探索器)

**職責**: 掃描五大模組的文件結構和代碼組織

**使用範例**:
```python
from aiva_core.internal_exploration import ModuleExplorer

explorer = ModuleExplorer()
modules_info = await explorer.explore_all_modules()
# 返回: {'ai_core': {...}, 'attack_engine': {...}, ...}
```

---

### 2️⃣ CapabilityAnalyzer (能力分析器)

**職責**: 識別和分析系統能力函數

**使用範例**:
```python
from aiva_core.internal_exploration import CapabilityAnalyzer

analyzer = CapabilityAnalyzer()
capabilities = await analyzer.analyze_capabilities(modules_info)
# 返回: [{'name': 'sql_injection', 'module': 'attack_engine', ...}, ...]
```

---

### 3️⃣ ASTCodeAnalyzer (AST 解析器)

**職責**: 將 Python 代碼轉換為 AST 並提取語義信息

**使用範例**:
```python
from aiva_core.internal_exploration import ASTCodeAnalyzer

analyzer = ASTCodeAnalyzer()
ast_info = analyzer.parse_file("path/to/file.py")
# 返回: {'classes': [...], 'functions': [...], 'imports': [...]}
```

---

### 4️⃣ KnowledgeGraph (知識圖譜)

**職責**: 構建系統能力的知識圖譜和依賴關係

**使用範例**:
```python
from aiva_core.internal_exploration import KnowledgeGraph

graph = KnowledgeGraph()
capability_graph = graph.build_graph(capabilities)
# 返回: NetworkX 圖對象,包含節點和邊
```

---

### 5️⃣ SelfDiagnostics (自我診斷)

**職責**: 診斷系統健康狀況,找出潛在問題

**使用範例**:
```python
from aiva_core.internal_exploration import SelfDiagnostics

diagnostics = SelfDiagnostics()
issues = diagnostics.find_broken_paths(capability_graph)
# 返回: [{'issue': 'missing_dependency', 'details': {...}}, ...]
```

---

## 🔗 與內部閉環的關係

Internal Exploration 是 **內部閉環** 的核心組件:

```
內部閉環數據流:
    ModuleExplorer (探索五大模組)
        ↓
    CapabilityAnalyzer (分析能力)
        ↓
    ASTCodeAnalyzer (解析代碼結構)
        ↓
    KnowledgeGraph (構建知識圖譜)
        ↓
    InternalLoopConnector (cognitive_core)
        ↓
    RAGEngine (cognitive_core.rag)
        ↓
    AI 決策時使用這些知識
```

---

## 🚀 快速開始

### 完整探索流程

```python
from aiva_core.internal_exploration import (
    ModuleExplorer,
    CapabilityAnalyzer,
    ASTCodeAnalyzer,
    KnowledgeGraph,
    SelfDiagnostics
)

# 1. 探索所有模組
explorer = ModuleExplorer()
modules_info = await explorer.explore_all_modules()

# 2. 分析能力
analyzer = CapabilityAnalyzer()
capabilities = await analyzer.analyze_capabilities(modules_info)

# 3. 解析代碼
ast_analyzer = ASTCodeAnalyzer()
for module_path in modules_info:
    ast_info = ast_analyzer.parse_file(module_path)

# 4. 構建知識圖譜
graph_builder = KnowledgeGraph()
capability_graph = graph_builder.build_graph(capabilities)

# 5. 自我診斷
diagnostics = SelfDiagnostics()
health_report = diagnostics.diagnose(capability_graph)

print(f"發現 {len(capabilities)} 個能力")
print(f"知識圖譜節點數: {len(capability_graph.nodes)}")
print(f"健康狀況: {health_report['status']}")
```

---

## 🔧 開發指南

### 遵循 AIVA Common 規範

本模組遵循 [`services/aiva_common`](../../../aiva_common/README.md) 的標準規範:

- ✅ 使用標準枚舉和數據結構
- ✅ 完整的類型標註
- ✅ 詳細的文檔字串

### 能力標記規範

系統能力應使用 `@register_capability` 裝飾器標記:

```python
from aiva_core.core_capabilities import register_capability

@register_capability(
    name="sql_injection_scanner",
    category="security_testing",
    input_schema=SQLInjectionInput,
    output_schema=ScanResult
)
async def scan_sql_injection(target: str) -> dict:
    """掃描 SQL 注入漏洞"""
    # 實現邏輯
    pass
```

---

## 📊 探索統計

| 探索對象 | 預期數量 | 當前狀態 |
|---------|---------|---------|
| **五大模組** | 5個 | 🚧 待探索 |
| **能力函數** | 100+ 個 | 🚧 待分析 |
| **知識圖譜節點** | 500+ 個 | 🚧 待構建 |
| **依賴關係邊** | 1000+ 條 | 🚧 待識別 |

---

## 🧪 測試

```bash
# 運行單元測試
pytest tests/test_internal_exploration/

# 測試模組探索
python -m aiva_core.internal_exploration.module_explorer

# 測試能力分析
python -m aiva_core.internal_exploration.capability_analyzer
```

---

## 📚 相關文檔

- [AIVA Core 重構計劃](../REFACTORING_PLAN.md)
- [AI 自我優化雙重閉環設計](../../../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)
- [認知核心模組](../cognitive_core/README.md)

---

**📝 文檔版本**: v1.0  
**🔄 最後更新**: 2025年11月15日  
**👥 維護者**: AIVA Core 開發團隊

---

## ⚠️ 重要提醒

本模組目前處於架構搭建階段 (🚧)。組件將在後續階段逐步創建和遷移。
