# AI 自我認知與優化系統執行計劃

**提案日期**: 2025-11-13  
**目標**: 讓 AI 探索和分析 AIVA 五大模組,建立自我認知能力,為自我優化奠定基礎  
**範圍**: 內部自省 (Core, Common, Features, Integration, Scan)  
**狀態**: 🚧 Phase 1 已完成 - ModuleExplorer 已實現並測試通過

---

## 📊 現況分析

### ✅ **已存在的組件**

| 組件 | 位置 | 功能 | 狀態 |
|------|------|------|------|
| **CapabilityRegistry** | `integration/capability/registry.py` | 能力註冊與發現 | ✅ 完整 |
| **CodeAnalyzer** | `core/ai_engine/tools/code_analyzer.py` | AST 分析、複雜度計算 | ✅ 完整但僅分析 |
| **AIAnalysisEngine** | `core/ai_analysis/analysis_engine.py` | AI 驅動代碼分析 | ✅ 完整 |
| **EnhancedDecisionAgent** | `core/decision/enhanced_decision_agent.py` | 決策與工具選擇 | ✅ 完整 |
| **RAGEngine** | `core/rag/rag_engine.py` | 知識檢索增強 | ✅ 完整 |
| **ExperienceManager** | `aiva_common/ai/experience_manager.py` | 經驗學習 | ✅ 完整 |
| **PlanExecutor** | `core/execution/plan_executor.py` | 計劃執行 | ✅ 完整 |

### ❌ **缺失的組件**

| 需求 | 現狀 | 缺口 |
|------|------|------|
| **內部模組探索** | CapabilityRegistry 需要手動註冊 | 缺少自動掃描五大模組的功能 |
| **自我認知分析** | CodeAnalyzer 可分析但無整合視圖 | 缺少模組依賴、能力映射、架構理解 |
| **能力評估** | 無評估機制 | 缺少效能分析、成功率統計、瓶頸識別 |
| **自我優化建議** | ExperienceManager 記錄但不分析 | 缺少基於數據的優化建議生成 |
| **RAG 內部知識** | KnowledgeBase 無內部文檔 | 缺少模組文檔、API 說明、最佳實踐 |

---

## 🎯 整合方案設計

### **核心理念: AI 自我認知循環**

```
AI 啟動 → 掃描五大模組 → 分析能力與依賴 → 建立知識圖譜 
→ 評估效能瓶頸 → 生成優化建議 → 學習執行結果 → 更新認知
```

### **五大模組結構**

```
services/
├── core/          (AI 大腦與決策引擎)
├── aiva_common/   (共享數據模型與枚舉)
├── features/      (7 大功能模組)
├── integration/   (企業整合與能力註冊)
└── scan/          (統一掃描引擎)
```

### **新增組件架構**

```
┌─────────────────────────────────────────────────────────┐
│  SelfAwarenessEngine (自我認知引擎)                     │
│  統一入口,協調探索、分析、評估、優化                     │
└──────────────────┬──────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    │              │              │              │
┌───▼─────┐  ┌────▼──────┐  ┌───▼──────┐  ┌───▼──────┐
│ 模組    │  │ 能力      │  │ 效能     │  │ 優化     │
│Explorer │  │ Analyzer  │  │ Evaluator│  │ Advisor  │
│探索器   │  │ 分析器    │  │ 評估器   │  │ 顧問     │
└────┬────┘  └─────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │              │
     └─────────────┴──────────────┴──────────────┘
                   │
        ┌──────────▼──────────┐
        │  RAG Engine         │
        │  存儲內部知識文檔    │
        └─────────────────────┘
```

---

## 📝 詳細執行計劃

### **Phase 1: 模組探索器 (ModuleExplorer)** ✅ **已完成**

**目標**: 自動掃描五大模組,發現所有能力、API、依賴關係

**完成狀態**: 
- ✅ **文件已創建**: `services/core/aiva_core/ai_engine/module_explorer.py` (620行)
- ✅ **測試已通過**: 11個測試用例全部通過 
- ✅ **問題已修復**: 
  - 修復了 `ModuleName.FEATURES` 枚舉缺失問題
  - 修復了文件編碼讀取問題 (支援 UTF-8/GBK/Latin1)
  - 修復了相對導入問題

**核心功能**:
1. **自動模組掃描**
   - 遍歷 services/ 目錄結構
   - 識別 Python/TypeScript/Rust/Go 代碼
   - 提取類、函數、接口定義

2. **能力發現與註冊**
   - AST 分析找 @register_capability 裝飾器
   - 自動發現 CLI 入口點
   - 識別 API endpoints (FastAPI routes)
   - 註冊到 CapabilityRegistry

3. **依賴關係映射**
   - 導入分析 (import statements)
   - 調用鏈追蹤 (function calls)
   - 服務間通訊 (MQ messages)
   - 數據模型引用

4. **架構圖譜生成**
   - 模組依賴樹
   - 能力調用圖
   - 數據流向圖

**代碼結構**:
```python
class ModuleExplorer:
    def __init__(self, services_root: Path, capability_registry: CapabilityRegistry):
        self.services_root = services_root
        self.registry = capability_registry
        self.modules = {
            "core": services_root / "core",
            "common": services_root / "aiva_common",
            "features": services_root / "features",
            "integration": services_root / "integration",
            "scan": services_root / "scan"
        }
    
    async def explore_all_modules(self) -> dict:
        """探索所有五大模組"""
        results = {}
        
        for module_name, module_path in self.modules.items():
            logger.info(f"🔍 探索模組: {module_name}")
            
            # 1. 掃描目錄結構
            structure = self._scan_directory_structure(module_path)
            
            # 2. 分析代碼找能力
            capabilities = await self._discover_capabilities(module_path)
            
            # 3. 分析依賴關係
            dependencies = self._analyze_dependencies(module_path)
            
            # 4. 生成模組報告
            results[module_name] = {
                "structure": structure,
                "capabilities": capabilities,
                "dependencies": dependencies,
                "stats": self._generate_stats(module_path)
            }
            
            # 5. 註冊到能力中心
            await self._register_to_capability_registry(capabilities)
        
        return results
    
    def _discover_capabilities(self, module_path: Path) -> list:
        """發現模組中的所有能力"""
        capabilities = []
        
        for py_file in module_path.rglob("*.py"):
            try:
                code = py_file.read_text(encoding="utf-8")
                tree = ast.parse(code)
                
                # 查找裝飾器
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # 檢查是否有 @register_capability
                        for decorator in node.decorator_list:
                            if self._is_capability_decorator(decorator):
                                cap = self._extract_capability_info(
                                    node, py_file, code
                                )
                                capabilities.append(cap)
                    
                    elif isinstance(node, ast.ClassDef):
                        # CLI 工具類
                        if self._is_cli_tool_class(node):
                            cap = self._extract_cli_capability(
                                node, py_file, code
                            )
                            capabilities.append(cap)
            
            except Exception as e:
                logger.warning(f"無法分析 {py_file}: {e}")
        
        return capabilities
    
    def _analyze_dependencies(self, module_path: Path) -> dict:
        """分析依賴關係"""
        dependencies = {
            "internal": set(),  # 內部模組依賴
            "external": set(),  # 外部庫依賴
            "calls": [],        # 函數調用關係
            "data_flow": []     # 數據流動
        }
        
        for py_file in module_path.rglob("*.py"):
            try:
                code = py_file.read_text(encoding="utf-8")
                tree = ast.parse(code)
                
                # 分析 import
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if "services." in alias.name:
                                dependencies["internal"].add(alias.name)
                            else:
                                dependencies["external"].add(alias.name)
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and "services." in node.module:
                            dependencies["internal"].add(node.module)
                        elif node.module:
                            dependencies["external"].add(node.module)
                    
                    # 分析函數調用
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            dependencies["calls"].append({
                                "caller": py_file.stem,
                                "callee": self._extract_call_name(node)
                            })
            
            except Exception as e:
                logger.warning(f"依賴分析失敗 {py_file}: {e}")
        
        # 轉換 set 為 list
        dependencies["internal"] = list(dependencies["internal"])
        dependencies["external"] = list(dependencies["external"])
        
        return dependencies
```

---

### **Phase 2: 能力分析器 (CapabilityAnalyzer)** 📊

**目標**: 深度分析已發現的能力,理解其功能、參數、輸出

**新建文件**: `services/core/aiva_core/ai_engine/capability_analyzer.py`

**核心功能**:
1. **能力語義分析**
   - 使用 AI 理解函數功能 (基於文檔字串)
   - 參數類型推斷
   - 返回值分析
   - 副作用識別

2. **能力分類**
   - 按功能分類 (掃描/分析/利用/報告)
   - 按風險等級分類 (安全/中等/危險)
   - 按使用頻率排序

3. **能力關聯**
   - 識別相似能力
   - 建立能力鏈 (A的輸出是B的輸入)
   - 推薦組合使用

4. **生成能力文檔**
   - 自動生成 Markdown 文檔
   - CLI 使用示例
   - API 調用示例

**代碼結構**:
```python
class CapabilityAnalyzer:
    def __init__(self, ai_engine, rag_engine):
        self.ai = ai_engine
        self.rag = rag_engine
    
    async def analyze_capability(self, capability: dict) -> dict:
        """深度分析單個能力"""
        
        # 1. AI 語義理解
        semantic_analysis = await self.ai.understand_function(
            code=capability["source_code"],
            docstring=capability["docstring"],
            signature=capability["signature"]
        )
        
        # 2. 參數分析
        parameters = self._analyze_parameters(capability)
        
        # 3. 風險評估
        risk_level = self._assess_risk_level(capability, semantic_analysis)
        
        # 4. 生成使用示例
        examples = await self._generate_examples(capability)
        
        # 5. 查找相關能力
        related = await self._find_related_capabilities(capability)
        
        return {
            "capability_id": capability["id"],
            "semantic_understanding": semantic_analysis,
            "parameters": parameters,
            "risk_level": risk_level,
            "examples": examples,
            "related_capabilities": related,
            "documentation": self._generate_documentation(capability)
        }
    
    async def classify_all_capabilities(self, capabilities: list) -> dict:
        """分類所有能力"""
        classifications = {
            "by_function": {},
            "by_risk": {},
            "by_module": {},
            "by_language": {}
        }
        
        for cap in capabilities:
            # 功能分類
            function_type = await self._classify_by_function(cap)
            if function_type not in classifications["by_function"]:
                classifications["by_function"][function_type] = []
            classifications["by_function"][function_type].append(cap)
            
            # 風險分類
            risk = self._assess_risk_level(cap, {})
            if risk not in classifications["by_risk"]:
                classifications["by_risk"][risk] = []
            classifications["by_risk"][risk].append(cap)
        
        return classifications
```

---

### **Phase 3: 效能評估器 (PerformanceEvaluator)** ⚡

**目標**: 評估每個能力的效能、成功率、瓶頸

**新建文件**: `services/core/aiva_core/ai_engine/performance_evaluator.py`

**核心功能**:
1. **主動探測**
   - 定期 ping 各能力端點
   - 健康檢查 (health check)
   - 響應時間測量

2. **被動監控**
   - 從 ExperienceManager 讀取歷史數據
   - 統計成功率、失敗率
   - 平均執行時間

3. **瓶頸識別**
   - 找出最慢的能力
   - 分析依賴鏈中的瓶頸
   - 資源消耗分析

4. **生成記分卡**
   - 為每個能力打分
   - 可用性、可靠性、效能指標
   - 推薦優化方向

**代碼結構**:
```python
class PerformanceEvaluator:
    def __init__(self, capability_registry, experience_manager):
        self.registry = capability_registry
        self.experience = experience_manager
    
    async def evaluate_all_capabilities(self) -> dict:
        """評估所有能力的效能"""
        results = {}
        
        capabilities = await self.registry.list_capabilities()
        
        for cap in capabilities:
            scorecard = await self._generate_scorecard(cap)
            results[cap["id"]] = scorecard
        
        # 識別瓶頸
        bottlenecks = self._identify_bottlenecks(results)
        
        return {
            "scorecards": results,
            "bottlenecks": bottlenecks,
            "summary": self._generate_summary(results)
        }
    
    async def _generate_scorecard(self, capability: dict) -> dict:
        """生成能力記分卡"""
        
        # 1. 主動探測
        probe_result = await self._probe_capability(capability)
        
        # 2. 歷史數據分析
        history = await self.experience.get_capability_history(
            capability["id"],
            days=30
        )
        
        # 3. 計算指標
        metrics = {
            "availability": probe_result["available"],
            "success_rate": self._calculate_success_rate(history),
            "avg_latency_ms": self._calculate_avg_latency(history),
            "p95_latency_ms": self._calculate_p95_latency(history),
            "reliability_score": self._calculate_reliability(history)
        }
        
        return metrics
```

---

### **Phase 4: 優化顧問 (OptimizationAdvisor)** 🔧

**目標**: 基於分析結果,生成優化建議

**新建文件**: `services/core/aiva_core/ai_engine/optimization_advisor.py`

**核心功能**:
1. **自動化建議生成**
   - 基於瓶頸生成優化建議
   - 依賴優化 (減少不必要的依賴)
   - 參數調優建議

2. **代碼改進建議**
   - 複雜度過高的函數重構建議
   - 性能優化建議 (緩存、並行)
   - 安全改進建議

3. **架構優化建議**
   - 服務拆分/合併建議
   - 通訊方式優化 (同步→異步)
   - 數據模型優化

4. **生成優化計劃**
   - 優先級排序
   - 預期效果評估
   - 實施步驟

**代碼結構**:
```python
class OptimizationAdvisor:
    def __init__(self, evaluator, ai_engine):
        self.evaluator = evaluator
        self.ai = ai_engine
    
    async def generate_optimization_plan(
        self, 
        scorecards: dict,
        bottlenecks: list
    ) -> dict:
        """生成完整優化計劃"""
        
        recommendations = []
        
        # 1. 處理瓶頸
        for bottleneck in bottlenecks:
            recs = await self._optimize_bottleneck(bottleneck)
            recommendations.extend(recs)
        
        # 2. 處理低分能力
        low_score_caps = [
            cap_id for cap_id, scorecard in scorecards.items()
            if scorecard["reliability_score"] < 0.7
        ]
        
        for cap_id in low_score_caps:
            recs = await self._improve_capability(cap_id, scorecards[cap_id])
            recommendations.extend(recs)
        
        # 3. 架構優化建議
        arch_recs = await self._analyze_architecture()
        recommendations.extend(arch_recs)
        
        # 4. 排序優先級
        sorted_recs = self._prioritize_recommendations(recommendations)
        
        return {
            "recommendations": sorted_recs,
            "summary": self._generate_summary(sorted_recs),
            "estimated_impact": self._estimate_impact(sorted_recs)
        }
```

---

### **Phase 5: RAG 知識整合** 📚

**目標**: 將內部文檔、最佳實踐存入 RAG 供 AI 查詢

**修改文件**: `services/core/aiva_core/rag/rag_engine.py`

**新建文件**: `services/core/aiva_core/rag/internal_knowledge_indexer.py`

**核心功能**:
1. **內部文檔索引**
   - 掃描 services/*/README.md
   - 掃描 services/*/docs/*.md
   - 提取 Python docstrings

2. **最佳實踐庫**
   - 代碼模式 (設計模式)
   - API 使用示例
   - 錯誤處理模式

3. **經驗知識庫**
   - 從 ExperienceManager 提取成功案例
   - 失敗案例與修復方案
   - 效能優化案例

**代碼結構**:
```python
class InternalKnowledgeIndexer:
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.services_root = Path("services")
    
    async def index_all_internal_docs(self):
        """索引所有內部文檔"""
        
        # 1. 索引 README 文件
        for readme in self.services_root.rglob("README.md"):
            content = readme.read_text(encoding="utf-8")
            await self.rag.add_document(
                content=content,
                metadata={
                    "type": "README",
                    "module": self._get_module_name(readme),
                    "path": str(readme)
                }
            )
        
        # 2. 索引 API 文檔
        for doc in self.services_root.rglob("docs/*.md"):
            content = doc.read_text(encoding="utf-8")
            await self.rag.add_document(
                content=content,
                metadata={
                    "type": "API_DOC",
                    "module": self._get_module_name(doc),
                    "path": str(doc)
                }
            )
        
        # 3. 索引代碼文檔字串
        for py_file in self.services_root.rglob("*.py"):
            docstrings = self._extract_docstrings(py_file)
            for docstring in docstrings:
                await self.rag.add_document(
                    content=docstring["content"],
                    metadata={
                        "type": "DOCSTRING",
                        "function": docstring["name"],
                        "file": str(py_file)
                    }
                )
```

---

## 📂 文件清單與代碼量估算

| 階段 | 文件路徑 | 新增/修改 | 預估代碼行數 |
|------|----------|-----------|--------------|
| **Phase 1** | `services/core/aiva_core/ai_engine/module_explorer.py` | 新增 | ~400 行 |
| **Phase 2** | `services/core/aiva_core/ai_engine/capability_analyzer.py` | 新增 | ~350 行 |
| **Phase 3** | `services/core/aiva_core/ai_engine/performance_evaluator.py` | 新增 | ~300 行 |
| **Phase 4** | `services/core/aiva_core/ai_engine/optimization_advisor.py` | 新增 | ~250 行 |
| **Phase 5** | `services/core/aiva_core/rag/internal_knowledge_indexer.py` | 新增 | ~200 行 |
| **整合** | `services/core/aiva_core/ai_engine/self_awareness_engine.py` | 新增 | ~300 行 |
| **增強** | `services/integration/capability/registry.py` | 修改 | +150 行 |
| **CLI** | `services/core/aiva_core/cli/self_awareness_commands.py` | 新增 | ~100 行 |
| **測試** | `services/core/tests/test_self_awareness.py` | 新增 | ~200 行 |
| **文檔** | `services/core/docs/SELF_AWARENESS_GUIDE.md` | 新增 | ~150 行 |

**總計**: ~2,400 行代碼

---

## 🧪 測試計劃

### **單元測試** (Unit Tests)

**測試文件**: `services/core/tests/test_self_awareness.py`

```python
import pytest
from aiva_core.ai_engine.module_explorer import ModuleExplorer
from aiva_core.ai_engine.capability_analyzer import CapabilityAnalyzer

class TestModuleExplorer:
    @pytest.mark.asyncio
    async def test_discover_capabilities_in_core(self):
        """測試能否發現 Core 模組的能力"""
        explorer = ModuleExplorer(services_root=Path("services"))
        
        results = await explorer.explore_all_modules()
        
        assert "core" in results
        assert len(results["core"]["capabilities"]) > 0
        assert "AIAnalysisEngine" in str(results["core"]["capabilities"])
    
    @pytest.mark.asyncio
    async def test_analyze_dependencies(self):
        """測試依賴關係分析"""
        explorer = ModuleExplorer(services_root=Path("services"))
        
        deps = explorer._analyze_dependencies(Path("services/core"))
        
        assert "aiva_common" in deps["internal"]
        assert len(deps["external"]) > 0

class TestCapabilityAnalyzer:
    @pytest.mark.asyncio
    async def test_classify_capability(self):
        """測試能力分類"""
        analyzer = CapabilityAnalyzer(ai_engine=mock_ai, rag=mock_rag)
        
        cap = {
            "id": "test_cap",
            "name": "test_function",
            "source_code": "def test(): pass"
        }
        
        result = await analyzer.analyze_capability(cap)
        
        assert "risk_level" in result
        assert "parameters" in result
```

### **集成測試** (Integration Tests)

```python
class TestSelfAwarenessIntegration:
    @pytest.mark.asyncio
    async def test_full_self_awareness_cycle(self):
        """測試完整自我認知循環"""
        engine = SelfAwarenessEngine()
        
        # 1. 探索
        results = await engine.explore_all_modules()
        assert len(results) == 5  # 五大模組
        
        # 2. 分析
        analysis = await engine.analyze_capabilities(results)
        assert "classifications" in analysis
        
        # 3. 評估
        evaluation = await engine.evaluate_performance()
        assert "scorecards" in evaluation
        
        # 4. 優化建議
        advice = await engine.generate_optimization_plan(evaluation)
        assert len(advice["recommendations"]) > 0
```

---

## ⚠️ 風險評估與緩解措施

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| **效能影響** | 中 | 中 | 探索過程在後台異步執行,不阻塞主流程 |
| **掃描遺漏** | 低 | 中 | 多種發現方式組合 (AST + 裝飾器 + 導入分析) |
| **依賴循環** | 低 | 高 | 依賴圖中檢測循環,報警提示 |
| **AI 誤判** | 中 | 低 | 人工審核優化建議,不自動執行 |
| **RAG 容量** | 低 | 低 | 設置文檔大小上限,定期清理舊數據 |

---

## ⏱️ 時間安排

| 階段 | 預估時間 | 產出 |
|------|----------|------|
| **Phase 1: ModuleExplorer** | 1 天 | 模組探索器 + 測試 |
| **Phase 2: CapabilityAnalyzer** | 0.5 天 | 能力分析器 + 測試 |
| **Phase 3: PerformanceEvaluator** | 0.5 天 | 效能評估器 + 測試 |
| **Phase 4: OptimizationAdvisor** | 0.5 天 | 優化顧問 + 測試 |
| **Phase 5: RAG Integration** | 0.5 天 | 知識索引器 + 測試 |
| **整合與測試** | 0.5 天 | SelfAwarenessEngine + 集成測試 |
| **文檔撰寫** | 0.5 天 | 使用指南 + API 文檔 |

**總計**: 約 **4 天**

---

## 🎯 關鍵決策點

執行前需確認以下問題:

### **決策點 1: CapabilityRegistry 增強方案**
- **選項 A**: 修改現有 CapabilityRegistry,添加自動發現功能
- **選項 B**: 新建 SelfAwarenessEngine,通過 API 調用 CapabilityRegistry
- **建議**: 選項 B (低耦合,易測試)

### **決策點 2: 探索頻率**
- **選項 A**: 啟動時探索一次
- **選項 B**: 定期探索 (每小時)
- **選項 C**: 手動觸發探索
- **建議**: 啟動時 + 手動觸發 (避免效能影響)

### **決策點 3: AI 語義分析範圍**
- **選項 A**: 僅分析有 docstring 的函數
- **選項 B**: 所有函數都用 AI 分析
- **建議**: 選項 A (成本控制)

### **決策點 4: 優化建議執行方式**
- **選項 A**: 僅生成建議,人工審核
- **選項 B**: 自動執行低風險優化 (如添加索引)
- **建議**: 選項 A (安全第一)

### **決策點 5: RAG 知識範圍**
- **選項 A**: 僅索引 README 和 API 文檔
- **選項 B**: 包含所有 docstrings
- **選項 C**: 包含代碼示例和最佳實踐
- **建議**: 選項 C (最全面)

---

## 📝 執行檢查清單

執行前請確認:

- [ ] 已了解五大模組結構 (Core, Common, Features, Integration, Scan)
- [ ] 已確認 CapabilityRegistry 現況 (761 行,SQLite)
- [ ] 已確認 AI 模型權重可用 (aiva_5M_weights.pth, 20MB)
- [ ] 已確認 RAGEngine 可用 (360 行)
- [ ] 已確認效能要求 (探索過程不超過 10 秒)
- [ ] 已確認風險可接受 (僅建議,不自動執行優化)
- [ ] 已確認時間安排 (4 天可接受)

---

## 🚀 執行流程

### **Step 1: 環境準備**
```powershell
# 1. 確認依賴
cd C:\D\fold7\AIVA-git\services\core
pip install -r requirements.txt

# 2. 確認權重文件
ls ../../weights/aiva_5M_weights.pth

# 3. 確認數據庫
ls ../../data/capabilities.db
```

### **Step 2: Phase 1 實施**
```powershell
# 創建 ModuleExplorer
New-Item -Path "aiva_core/ai_engine/module_explorer.py" -ItemType File

# 編寫代碼
code aiva_core/ai_engine/module_explorer.py

# 運行測試
pytest tests/test_module_explorer.py -v
```

### **Step 3: 依次實施 Phase 2-5**
(按上述時間安排逐步完成)

### **Step 4: 整合測試**
```powershell
# 運行完整測試套件
pytest tests/test_self_awareness.py -v

# 手動測試 CLI
python -m aiva_core.cli self-awareness explore --all
python -m aiva_core.cli self-awareness analyze
python -m aiva_core.cli self-awareness evaluate
python -m aiva_core.cli self-awareness optimize
```

### **Step 5: 文檔補充**
```powershell
# 生成 API 文檔
sphinx-build -b html docs/source docs/build

# 撰寫使用指南
code services/core/docs/SELF_AWARENESS_GUIDE.md
```

---

## ✅ 驗收標準

完成後應達到以下標準:

1. **功能性**
   - [ ] 能夠掃描五大模組,發現至少 80% 的能力
   - [ ] 能夠正確分析依賴關係 (無循環依賴)
   - [ ] 能夠生成效能評估報告
   - [ ] 能夠提供至少 3 條優化建議

2. **效能**
   - [ ] 完整探索時間 < 10 秒
   - [ ] 記憶體佔用 < 500MB
   - [ ] 不影響主流程響應速度

3. **可用性**
   - [ ] CLI 指令易用 (單條指令完成)
   - [ ] 輸出格式清晰 (JSON/Markdown)
   - [ ] 錯誤提示友好

4. **可維護性**
   - [ ] 代碼覆蓋率 > 80%
   - [ ] 文檔完整 (API + 使用指南)
   - [ ] 符合現有代碼風格

---

## 🎓 後續擴展方向

完成基礎自我認知後,可進一步擴展:

1. **自動化優化執行**
   - 低風險優化自動執行 (如添加日誌)
   - 高風險優化生成 PR 供審核

2. **能力市場**
   - 發布能力到內部市場
   - 其他模組訂閱能力更新

3. **智能路由**
   - 基於能力評分自動選擇最佳實現
   - 降級策略 (主能力失敗時切換備用)

4. **預測性維護**
   - 預測能力何時會失敗
   - 提前觸發優化或替換

---

## 📌 下一步行動

**請您審核並回覆:**

1. ✅ 是否同意這個整體方案?
2. 🎯 選擇執行模式 (分階段 or 一次性)?
3. 🔧 確認關鍵決策點的選項 (決策點 1-5)
4. 🚀 批准開始執行

**我會在獲得您的確認後開始實施!**

---

**提案內容**: AI 自我認知與優化系統  
**核心目標**: 讓 AI 理解自己的五大模組能力,建立自我修復優化基礎  
**待審核**: 等待用戶確認  
**預計完成**: 4 個工作日

---

## 📋 附錄: 五大模組概覽

### **1. Core (核心模組)**
- AI 引擎 (AIAnalysisEngine, BioNeuronMasterController)
- 決策引擎 (EnhancedDecisionAgent)
- RAG 系統 (RAGEngine)
- 工具系統 (CodeAnalyzer 等)

### **2. Common (共享模組)**
- 數據模型 (models/)
- 枚舉定義 (enums/)
- AI 輔助 (ExperienceManager)

### **3. Features (功能模組)**
- 7 大功能模組 (scan, exploit, report 等)
- 各模組獨立運行
- 通過 MQ 通訊

### **4. Integration (整合模組)**
- CapabilityRegistry (能力註冊中心)
- API Gateway
- 企業集成

### **5. Scan (掃描模組)**
- 統一掃描引擎
- 多種掃描器整合
- 結果標準化
```

---

### **Phase 4: 優化顧問 (OptimizationAdvisor)** 🔧

**目標**: 基於分析結果,生成優化建議

**新建文件**: `services/core/aiva_core/ai_engine/optimization_advisor.py`

**核心功能**:
1. **自動化建議生成**
   - 基於瓶頸生成優化建議
   - 依賴優化 (減少不必要的依賴)
   - 參數調優建議

2. **代碼改進建議**
   - 複雜度過高的函數重構建議
   - 性能優化建議 (緩存、並行)
   - 安全改進建議

3. **架構優化建議**
   - 服務拆分/合併建議
   - 通訊方式優化 (同步→異步)
   - 數據模型優化

4. **生成優化計劃**
   - 優先級排序
   - 預期效果評估
   - 實施步驟

---

## 📋 檔案清單

### **新建檔案** (共 5 個)

| 檔案路徑 | 行數估計 | 功能 |
|---------|---------|------|
| `ai_engine/code_explorer.py` | ~400 | 代碼探索器 |
| `ai_engine/attack_generator.py` | ~500 | 攻擊策略生成器 |
| `ai_engine/self_healer.py` | ~300 | 自我修復優化器 |
| `ai_engine/intelligent_security_engine.py` | ~600 | 統一入口引擎 |
| `rag/knowledge_initializer.py` | ~800 | RAG 知識庫初始化 |

**總計**: ~2,600 行新代碼

### **修改檔案** (共 3 個)

| 檔案路徑 | 修改內容 |
|---------|---------|
| `ai_engine/tools/code_analyzer.py` | 添加漏洞探測方法 |
| `rag/knowledge_base.py` | 添加攻擊模式查詢方法 |
| `decision/enhanced_decision_agent.py` | 添加攻擊工具選擇邏輯 |

---

## 🔄 執行步驟

### **建議分階段執行**

#### **第 1 天: Phase 1 - 代碼探索器**
- [ ] 創建 `code_explorer.py`
- [ ] 實現靜態探索功能
- [ ] 實現動態探測功能
- [ ] 編寫測試用例

#### **第 2 天: Phase 2 - 攻擊生成器**
- [ ] 創建 `attack_generator.py`
- [ ] 實現攻擊鏈生成
- [ ] 實現 CLI 指令生成
- [ ] 編寫測試用例

#### **第 3 天: Phase 3 - 自我修復器**
- [ ] 創建 `self_healer.py`
- [ ] 實現失敗分析
- [ ] 實現自動修復
- [ ] 編寫測試用例

#### **第 4 天: Phase 4 - RAG 知識庫**
- [ ] 創建 `knowledge_initializer.py`
- [ ] 初始化 SQL 注入知識
- [ ] 初始化 XSS 知識
- [ ] 初始化修復模式

#### **第 5 天: Phase 5 - 統一入口**
- [ ] 創建 `intelligent_security_engine.py`
- [ ] 整合所有組件
- [ ] 編寫端到端測試
- [ ] 編寫使用文檔

---

## 🧪 測試計劃

### **單元測試**
```python
# test_code_explorer.py
async def test_explore_sql_injection():
    explorer = CodeExplorer(...)
    result = await explorer.explore_target(
        target_code="SELECT * FROM users WHERE id = " + request.args.get('id'),
        target_type="python_web"
    )
    assert "sql_injection" in result["confirmed"]

# test_attack_generator.py
async def test_generate_sqlmap_command():
    generator = AttackStrategyGenerator(...)
    commands = generator.generate_cli_commands({
        "steps": [{
            "tool": "sqlmap",
            "target": "http://example.com?id=1",
            "params": {"technique": "U"}
        }]
    })
    assert "sqlmap -u http://example.com?id=1" in commands[0]
```

### **整合測試**
```python
# test_full_flow.py
async def test_full_security_assessment():
    engine = IntelligentSecurityEngine()
    result = await engine.full_security_assessment(
        target="http://testsite.com"
    )
    assert result["exploration"]["confirmed"]
    assert result["attack_plan"]["attack_plan"]
    assert result["cli_commands"]
```

---

## ⚠️ 風險評估

| 風險 | 影響 | 緩解措施 |
|------|------|---------|
| **代碼量大** | 開發時間長 | 分階段執行,每階段獨立可用 |
| **AI 模型未訓練** | 決策不準確 | 先用規則引擎,逐步引入 AI |
| **RAG 知識庫空** | 無參考資料 | Phase 4 預置核心知識 |
| **整合複雜** | 組件衝突 | 先測試各組件,最後整合 |

---

## 💰 成本效益分析

### **投入**
- 開發時間: 5 天
- 新代碼: ~2,600 行
- 測試: ~500 行

### **產出**
- ✅ AI 可分析代碼內部結構
- ✅ AI 可探索漏洞攻擊面
- ✅ AI 可生成攻擊策略
- ✅ AI 可產生可執行 CLI 指令
- ✅ AI 可自我修復優化
- ✅ RAG 提供攻擊和修復參考

---

## 🎯 關鍵決策點

### **請確認以下問題:**

1. **是否接受分階段執行?**  
   - [ ] 是 - 建議從 Phase 1 開始
   - [ ] 否 - 需要一次性完成

2. **AI 決策引擎使用方式?**  
   - [ ] 優先使用現有 5M 神經網絡 (可能不準確)
   - [ ] 先用規則引擎,逐步訓練 AI (推薦)
   - [ ] 混合模式 (規則 + AI)

3. **RAG 知識庫範圍?**  
   - [ ] 僅 SQL 注入 + XSS (最小集)
   - [ ] 包含 OWASP Top 10 (推薦)
   - [ ] 包含所有常見漏洞 (最大集)

4. **CLI 指令安全級別?**  
   - [ ] 僅生成,不執行 (最安全)
   - [ ] 沙箱模式執行 (推薦)
   - [ ] 直接執行 (高風險)

5. **測試環境需求?**  
   - [ ] 使用 DVWA 等靶場測試 (推薦)
   - [ ] 使用模擬環境
   - [ ] 跳過實際測試

---

## 📌 下一步行動

**請您審核並回覆:**

1. ✅ 是否同意這個整體方案?
2. 🎯 選擇執行模式 (分階段 or 一次性)?
3. 🔧 確認關鍵決策點的選項
4. 🚀 批准開始執行

**我會在獲得您的確認後開始實施!**

---

**提案人**: GitHub Copilot  
**待審核**: 等待用戶確認  
**預計完成**: 5 個工作日 (分階段執行)
