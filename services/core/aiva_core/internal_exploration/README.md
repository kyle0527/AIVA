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

- ✅ **多語言掃描**: 支援 Python, Go, Rust, TypeScript, JavaScript (5 種語言)
- ✅ **模組探索**: 掃描 AIVA 六大模組的代碼結構 (380+ 文件)
- ✅ **能力分析**: 識別各語言的能力函數 (576+ 能力)
  - Python: AST 解析 `@register_capability` 裝飾器
  - Go: 正則提取 `func [A-Z]...` 導出函數
  - Rust: 正則提取 `pub fn` 公開函數
  - TypeScript: 正則提取 `export function`
- ✅ **統一接口**: 所有語言返回一致的能力元數據格式
- ✅ **知識圖譜**: 為內部閉環提供系統自我認知數據

---

## 📂 目錄結構

```
internal_exploration/
├── module_explorer.py            # 模組探索器 (199 行) ✅ 多語言文件掃描
├── capability_analyzer.py        # 能力分析器 (351 行) ✅ 多語言能力提取
├── language_extractors.py        # 語言提取器 (387 行) ✅ Go/Rust/TS 支援
├── __init__.py                   # 模組入口 (已實現)
└── README.md                     # 本文檔

總計: 3 個核心文件，937 行代碼
支援語言: Python, Go, Rust, TypeScript, JavaScript
```

---

## 🎨 核心組件說明

### 1️⃣ ModuleExplorer (模組探索器) - 199 行

**職責**: 掃描 AIVA 六大模組的多語言文件結構

**支援文件類型**:
- `*.py` - Python 文件
- `*.go` - Go 文件
- `*.rs` - Rust 文件
- `*.ts` - TypeScript 文件
- `*.js` - JavaScript 文件

**實際掃描結果** (2025-11-16):
- 總模組: 4 個 (core/aiva_core, scan, features, integration)
- 總文件: 380 個
- Python: 320 個 | Go: 27 個 | Rust: 7 個 | TS: 18 個 | JS: 8 個

**使用範例**:
```python
from aiva_core.internal_exploration import ModuleExplorer

explorer = ModuleExplorer()
modules_info = await explorer.explore_all_modules()

# 查看掃描統計
for module, data in modules_info.items():
    stats = data["stats"]
    print(f"{module}: {stats['total_files']} 個文件")
    print(f"  語言分布: {stats['by_language']}")
```

---

### 2️⃣ CapabilityAnalyzer (能力分析器) - 351 行

**職責**: 從多語言代碼中提取能力函數元數據

**分析策略**:
| 語言 | 方法 | 識別規則 | 精確度 |
|------|------|---------|--------|
| Python | AST 解析 | `@capability` 裝飾器 + 公開異步函數 | ⭐⭐⭐⭐⭐ |
| Go | 正則匹配 | `func [A-Z]...` (大寫開頭=導出) | ⭐⭐⭐⭐ |
| Rust | 正則匹配 | `pub fn` (公開函數) | ⭐⭐⭐ |
| TypeScript | 正則匹配 | `export function`, `export const` | ⭐⭐⭐⭐ |
| JavaScript | 正則匹配 | 同 TypeScript | ⭐⭐⭐⭐ |

**實際提取結果** (2025-11-16):
- 總能力: 576 個
- Python: 410 個 | Go: 88 個 | TypeScript: 78 個

**使用範例**:
```python
from aiva_core.internal_exploration import CapabilityAnalyzer

analyzer = CapabilityAnalyzer()
capabilities = await analyzer.analyze_capabilities(modules_info)

# 按語言分組
by_lang = {}
for cap in capabilities:
    lang = cap.get("language", "python")
    by_lang.setdefault(lang, []).append(cap)

print(f"發現 {len(capabilities)} 個能力")
for lang, caps in by_lang.items():
    print(f"  {lang}: {len(caps)} 個")
```

**返回格式** (統一接口):
```python
{
    "name": "function_name",           # 函數名稱
    "language": "python|go|rust|...",  # 語言類型
    "module": "core/aiva_core",        # 所屬模組
    "file_path": "/path/to/file.py",   # 文件路徑
    "parameters": [                    # 參數列表
        {"name": "param1", "type": "str"},
        {"name": "param2", "type": "int"}
    ],
    "return_type": "dict | None",      # 返回類型
    "description": "功能描述",          # 從註釋提取
    "is_async": true,                  # 是否異步 (Python)
    "is_exported": true,               # 是否導出 (Go)
    "line_number": 123                 # 行號
}
```

---

### 3️⃣ LanguageExtractor (語言提取器) - 387 行

**職責**: 為非 Python 語言提供正則表達式提取器

**類層級**:
```
LanguageExtractor (抽象基類)
  ├── GoExtractor          # Go 語言提取器
  ├── RustExtractor        # Rust 語言提取器
  └── TypeScriptExtractor  # TS/JS 共用提取器
```

**工廠函數**:
```python
from aiva_core.internal_exploration.language_extractors import get_extractor

# 獲取對應語言的提取器
go_extractor = get_extractor("go")
rust_extractor = get_extractor("rust")
ts_extractor = get_extractor("typescript")

# 使用提取器
with open("scanner.go") as f:
    content = f.read()
capabilities = go_extractor.extract_capabilities(content, "scanner.go")
```

**擴展新語言**:
```python
# 1. 創建新提取器類
class JavaExtractor(LanguageExtractor):
    FUNCTION_PATTERN = re.compile(r'public\s+\w+\s+(\w+)\s*\(')
    
    def extract_capabilities(self, content, file_path):
        # 實現提取邏輯
        pass

# 2. 註冊到工廠函數
def get_extractor(language):
    extractors = {
        "go": GoExtractor(),
        "rust": RustExtractor(),
        "java": JavaExtractor(),  # ← 添加新語言
    }
    return extractors.get(language.lower())
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

**當前系統掃描結果** (2025-11-16):

| 指標 | 數量 | 狀態 |
|------|------|------|
| **掃描模組** | 4 個 | ✅ core, scan, features, integration |
| **掃描文件** | 380 個 | ✅ 5 種語言 |
| **提取能力** | 576 個 | ✅ Python + Go + TS |
| **Python 能力** | 410 個 | ✅ AST 解析 |
| **Go 能力** | 88 個 | ✅ 正則提取 |
| **TypeScript 能力** | 78 個 | ✅ 正則提取 |
| **Rust 能力** | 0 個 | ⚠️ 實現方法未提取 (已知限制) |
| **語言覆蓋率** | 100% | ✅ 所有文件類型已掃描 |

**效能提升**:
- 相比僅 Python: 文件掃描 +18.75% (320→380)
- 相比僅 Python: 能力發現 +40.49% (410→576)
- 語言支援: 1→5 種 (+400%)

---

## 🧪 測試

```bash
# 多語言能力分析整合測試
python test_multi_language_analysis.py

# 預期輸出:
# ================================================================================
# 🚀 多語言能力分析整合測試
# ================================================================================
# 
# 📂 掃描模組文件...
# ✅ 掃描完成:
#    - 總文件: 380
#    - python: 320 個
#    - go: 27 個
#    - rust: 7 個
#    - typescript: 18 個
#    - javascript: 8 個
# 
# 🔍 提取能力...
# ✅ 提取完成:
#    - 總能力: 576
#    - python: 410 個
#    - go: 88 個
#    - typescript: 78 個
# 
# ✅ 驗證結果:
#    ✅ 多語言掃描
#    ✅ Python 提取
#    ✅ Go 提取
#    ✅ TypeScript 提取
# 
# ================================================================================
# ✅ 測試完成!
# ================================================================================

# 單元測試
pytest tests/test_capability_analyzer_multi_lang.py -v
```

**測試覆蓋**:
- ✅ 多語言文件掃描
- ✅ 語言檢測邏輯
- ✅ Python AST 提取
- ✅ Go 正則提取
- ✅ Rust 正則提取
- ✅ TypeScript 正則提取
- ✅ 統一數據格式驗證

---

## 📚 相關文檔

- [多語言分析完整指南](../../../../MULTI_LANGUAGE_ANALYSIS_COMPLETE_GUIDE.md) - 實施步驟和擴展指南
- [多語言整合報告](../../../../MULTI_LANGUAGE_ANALYSIS_INTEGRATION_REPORT.md) - 整合結果和效益分析
- [AIVA Core 重構計劃](../REFACTORING_PLAN.md)
- [AI 自我優化雙重閉環設計](../../../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)
- [認知核心模組](../cognitive_core/README.md)

---

**📝 文檔版本**: v2.0 (多語言整合版)  
**🔄 最後更新**: 2025年11月16日  
**👥 維護者**: AIVA Core 開發團隊

---

## ✅ 系統狀態

本模組已完成多語言整合,處於**生產就緒**狀態:

- ✅ 支援 5 種語言 (Python, Go, Rust, TypeScript, JavaScript)
- ✅ 已測試驗證 (test_multi_language_analysis.py)
- ✅ 統一數據接口
- ✅ 可擴展架構 (支援新增語言)
- ✅ 完整文檔覆蓋
