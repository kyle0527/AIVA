# Cognitive Core - AI 認知核心

## 📑 目錄

- [📋 目錄](#-目錄)
- [📋 概述](#-概述)
  - [🎯 核心職責](#-核心職責)
- [📂 目錄結構](#-目錄結構)
- [🎨 核心組件說明](#-核心組件說明)
  - [1️⃣ Neural (神經網路核心) - [📖 詳細文檔](./neural/README.md)](#1-neural-神經網路核心----詳細文檔neuralreadmemd)
  - [2️⃣ RAG (檢索增強生成) - [📖 詳細文檔](./rag/README.md)](#2-rag-檢索增強生成----詳細文檔ragreadmemd)
  - [3️⃣ Decision (決策支援) - [📖 詳細文檔](./decision/README.md)](#3-decision-決策支援----詳細文檔decisionreadmemd)
  - [4️⃣ Anti-Hallucination (反幻覺) - [📖 詳細文檔](./anti_hallucination/README.md)](#4-anti-hallucination-反幻覺----詳細文檔anti_hallucinationreadmemd)
  - [5️⃣ NLG System (自然語言生成)](#5-nlg-system-自然語言生成)
- [🔗 閉環連接器](#-閉環連接器)
  - [InternalLoopConnector (內部閉環連接器)](#internalloopconnector-內部閉環連接器)
  - [ExternalLoopConnector (外部閉環連接器)](#externalloopconnector-外部閉環連接器)
- [🚀 快速開始](#-快速開始)
  - [基本使用](#基本使用)
  - [整合閉環連接器](#整合閉環連接器)
- [🔧 開發指南](#-開發指南)
  - [🛠️ aiva_common 修復規範](#-aiva_common-修復規範)
  - [遵循 AIVA Common 規範](#遵循-aiva-common-規範)
  - [新增組件指南](#新增組件指南)
- [📊 性能指標](#-性能指標)
- [🧪 測試](#-測試)
- [📚 相關文檔](#-相關文檔)
- [⚠️ 重要提醒](#-重要提醒)

---

**導航**: [← 返回 AIVA Core](../README.md) | [📖 重構計劃](../REFACTORING_PLAN.md)

## 📋 目錄

- [概述](#概述)
- [核心職責](#核心職責)
- [目錄結構](#目錄結構)
- [核心組件說明](#核心組件說明)
- [閉環連接器](#閉環連接器)
- [設計理念](#設計理念)
- [使用範例](#使用範例)
- [開發規範](#開發規範)
- [遷移狀態](#遷移狀態)

---

## 📋 概述

> **🎯 定位**: AIVA 的「大腦」,負責思考和決策  
> **✅ 狀態**: 系統就緒，測試通過  
> **🧪 測試狀態**: 階段 4 測試 100% 通過 (4/4 組件)  
> **🔄 最後更新**: 2025年11月27日

**Cognitive Core** 是 AIVA Core 的 AI 認知核心模組,整合神經網路推理、RAG 知識增強、決策支援和反幻覺機制,實現 AI 自我優化雙重閉環的核心決策功能。

### 🎯 核心職責

- ✅ **神經網路推理**: 執行 5M 參數 BioNeuron 神經網路推理
- ✅ **RAG 知識管理**: 管理統一知識庫 (包含對內和對外知識)
- ✅ **決策支援**: 提供增強的決策代理和推理能力
- ✅ **反幻覺機制**: 確保 AI 輸出的可靠性和準確性
- ✅ **內部閉環連接**: 通過 `InternalLoopConnector` 將探索結果灌入 RAG
- ✅ **外部閉環連接**: 通過 `ExternalLoopConnector` 將偏差報告灌入學習系統

---

## 📂 目錄結構

```
cognitive_core/
├── 📁 neural/                     # 神經網路核心 (7 檔案)
│   ├── __init__.py
│   ├── real_neural_core.py        # ✅ 500萬參數神經網路核心
│   ├── real_bio_net_adapter.py    # ✅ 生物神經網路適配器
│   ├── bio_neuron_master.py       # ✅ BioNeuronRAGAgent 主控系統
│   ├── ai_model_manager.py        # ✅ AI 模型統一管理器
│   ├── neural_network.py          # ✅ 神經網路基礎架構
│   └── weight_manager.py          # ✅ 權重管理系統
│
├── 📁 rag/                        # RAG 增強系統 (6 檔案)
│   ├── __init__.py
│   ├── rag_engine.py              # ✅ RAG 核心引擎
│   ├── knowledge_base.py          # ✅ 統一知識庫管理
│   ├── unified_vector_store.py    # ✅ 統一向量存儲
│   ├── vector_store.py            # ✅ 向量存儲接口
│   ├── postgresql_vector_store.py # ✅ PostgreSQL 向量存儲
│   └── demo_rag_integration.py    # 🔧 RAG 整合示範
│
├── 📁 decision/                   # 決策支援 (3 檔案)
│   ├── __init__.py
│   ├── enhanced_decision_agent.py # ✅ 增強決策代理
│   └── skill_graph.py             # ✅ 技能圖譜和能力關係映射
│
├── 📁 anti_hallucination/         # 反幻覺模組 (2 檔案)
│   ├── __init__.py
│   └── anti_hallucination_module.py # ✅ 反幻覺檢查模組
│
├── nlg_system.py                  # ✅ 自然語言生成系統 (440行)
├── __init__.py                    # 模組入口
└── README.md                      # 本文檔

總計: 23 個 Python 檔案
```

---

## 🎨 核心組件說明

### 1️⃣ Neural (神經網路核心) - [📖 詳細文檔](./neural/README.md)

**職責**: 提供生物啟發的神經網路推理和模型管理能力

**主要組件**:

#### `real_neural_core.py` - 神經網路核心
- **功能**: 500萬參數 BioNeuron 神經網路
- **特性**: 生物啟發架構、高效推理
- **代碼量**: ~800 行

#### `real_bio_net_adapter.py` - 生物網路適配器
- **功能**: 將生物神經網路連接到 AIVA 系統
- **特性**: 適配層、介面轉換
- **代碼量**: ~600 行

#### `bio_neuron_master.py` - 主控系統
- **功能**: BioNeuronRAGAgent 主控制器
- **支援模式**: UI Mode / AI Mode / Chat Mode
- **架構**: 三模式統一調度系統
- **代碼量**: 1462 行

#### `ai_model_manager.py` - 模型管理器
- **功能**: 統一管理所有 AI 模型和訓練系統
- **職責**: 模型載入、訓練協調、版本管理
- **整合**: 連接 external_learning 訓練系統
- **代碼量**: 735 行

#### `neural_network.py` - 神經網路基礎
- **功能**: 神經網路基礎架構和通用層
- **特性**: 可復用的網路組件

#### `weight_manager.py` - 權重管理
- **功能**: 模型權重的載入、儲存和版本管理
- **特性**: 完整性檢查、安全序列化、錯誤容錯
- **代碼量**: 453 行

**使用範例**:
```python
from aiva_core.cognitive_core.neural import BioNeuronMaster, AIModelManager

# 初始化主控系統
master = BioNeuronMaster(mode="ai")  # UI/AI/Chat 模式
result = await master.process_request(query)

# AI 模型管理
model_manager = AIModelManager()
model = await model_manager.load_model("bioneuron-v1")
```

---

### 2️⃣ RAG (檢索增強生成) - [📖 詳細文檔](./rag/README.md)

**職責**: 提供知識檢索和上下文增強能力

**主要組件**:

#### `rag_engine.py` - RAG 核心引擎
- **功能**: 檢索增強生成的核心實現
- **特性**: 多源檢索、相關性排序、上下文融合

#### `knowledge_base.py` - 知識庫管理
- **功能**: 統一知識庫管理和檢索
- **來源**: 對內探索知識 + 對外學習知識

#### `unified_vector_store.py` - 統一向量存儲
- **功能**: 統一的向量存儲抽象層
- **支援**: 多種後端（內存、PostgreSQL）

#### `vector_store.py` - 向量存儲接口
- **功能**: 向量存儲的標準接口定義

#### `postgresql_vector_store.py` - PostgreSQL 後端
- **功能**: 基於 PostgreSQL + pgvector 的向量存儲
- **特性**: 持久化、高性能檢索

**使用範例**:
```python
from aiva_core.cognitive_core.rag import RAGEngine, KnowledgeBase

# 初始化 RAG
rag = RAGEngine()
kb = KnowledgeBase()

# 檢索增強
enhanced_context = await rag.retrieve_and_enhance(
    query="如何執行 SQL 注入測試",
    context={"target": "https://example.com"}
)
```

---

### 3️⃣ Decision (決策支援) - [📖 詳細文檔](./decision/README.md)

**職責**: 提供增強的 AI 決策和技能圖譜能力

**主要組件**:

#### `enhanced_decision_agent.py` - 增強決策代理
- **功能**: AI 增強的決策引擎
- **特性**: 上下文感知、多約束優化

#### `skill_graph.py` - 技能圖譜
- **功能**: 能力關係映射和依賴分析
- **特性**: NetworkX 圖結構、能力推薦
- **代碼量**: 649 行
- **用途**: 
  - 構建系統能力依賴圖
  - 智能推薦相關能力
  - 決策路徑優化

**使用範例**:
```python
from aiva_core.cognitive_core.decision import EnhancedDecisionAgent, SkillGraph

# 決策代理
agent = EnhancedDecisionAgent()
decision = await agent.make_decision(task, constraints)

# 技能圖譜
skill_graph = SkillGraph()
await skill_graph.build_from_registry()
related_skills = skill_graph.find_related("sql_injection")
```

---

### 4️⃣ Anti-Hallucination (反幻覺) - [📖 詳細文檔](./anti_hallucination/README.md)

**職責**: 驗證 AI 輸出的可靠性，防止幻覺

**主要組件**:

#### `anti_hallucination_module.py` - 反幻覺檢查
- **功能**: 驗證 AI 輸出與知識源的一致性
- **檢查項目**:
  - 事實準確性驗證
  - 知識源交叉檢查
  - 置信度評分
  - 不確定性標記

**使用範例**:
```python
from aiva_core.cognitive_core.anti_hallucination import AntiHallucinationModule

checker = AntiHallucinationModule()
validation = checker.validate(
    ai_output="建議使用 SQL 注入測試",
    source_knowledge=knowledge_base
)

if validation.is_reliable:
    print(f"可信度: {validation.confidence}%")
else:
    print(f"警告: {validation.issues}")
```

---

### 5️⃣ NLG System (自然語言生成)

**職責**: 生成高品質的中文回應，無需外部 LLM

**檔案**: `nlg_system.py` (440 行)

**主要功能**:
- **模板化回應生成**: 基於規則的專業回應
- **上下文分析**: 理解請求類型和生成適當回應
- **個性化設定**: 專業、有幫助、簡潔、技術導向
- **回應類型**:
  - 任務完成報告
  - 錯誤處理說明
  - 分析結果呈現
  - 建議和推薦

**使用範例**:
```python
from aiva_core.cognitive_core import AIVANaturalLanguageGenerator

nlg = AIVANaturalLanguageGenerator()
response = nlg.generate_response(
    response_type="task_completion",
    context={
        "action": "SQL注入測試",
        "result_detail": "發現3個漏洞",
        "confidence": 95
    }
)
# 輸出: "✅ 任務完成！SQL注入測試已成功執行，發現3個漏洞。"
```

---

## 🔗 閉環連接器

### InternalLoopConnector (內部閉環連接器)

**功能**: 連接內部探索結果 → RAG 知識庫

```python
from aiva_core.cognitive_core import InternalLoopConnector

connector = InternalLoopConnector()
result = await connector.sync_capabilities_to_rag()
```

**數據流**:
```
internal_exploration (模組探索)
    ↓
capability_analyzer (能力分析)
    ↓
knowledge_graph (知識圖譜)
    ↓
InternalLoopConnector
    ↓
cognitive_core.rag (RAG 知識庫)
```

---

### ExternalLoopConnector (外部閉環連接器)

**功能**: 連接執行偏差 → 學習系統

```python
from aiva_core.cognitive_core import ExternalLoopConnector

connector = ExternalLoopConnector()
result = await connector.process_execution_result(ast_plan, actual_trace)
```

**數據流**:
```
core_capabilities (攻擊執行)
    ↓
external_learning.tracing (追蹤記錄)
    ↓
external_learning.analysis (偏差分析)
    ↓
ExternalLoopConnector
    ↓
external_learning.learning (學習優化)
```

---

## 🚀 快速開始

### 基本使用

```python
from aiva_core.cognitive_core import CognitiveCoreOrchestrator

# 初始化認知核心
core = CognitiveCoreOrchestrator()

# 執行 AI 推理決策
result = await core.reason_and_decide(
    task="分析目標系統漏洞",
    context={"target": "example.com"},
    use_rag=True,
    check_hallucination=True
)

print(f"決策: {result['decision']}")
print(f"信心度: {result['confidence']}")
print(f"推理過程: {result['reasoning']}")
```

### 整合閉環連接器

```python
from aiva_core.cognitive_core import InternalLoopConnector, ExternalLoopConnector

# 內部閉環: 同步能力到 RAG
internal_connector = InternalLoopConnector()
await internal_connector.sync_capabilities_to_rag()

# 外部閉環: 處理執行反饋
external_connector = ExternalLoopConnector()
await external_connector.process_execution_result(plan, trace)
```

---

## 🔧 開發指南

### 🛠️ aiva_common 修復規範

> **核心原則**: 本模組作為 AIVA 系統的組成部分，必須嚴格遵循 [`services/aiva_common`](../../../aiva_common/README.md#-開發指南) 的修復規範與最佳實踐。

**完整規範文檔**: [aiva_common/README.md - 開發指南](../../../aiva_common/README.md#-開發指南)

#### 📌 核心原則 (摘要)

**1️⃣ 四層優先級**:
- 國際標準 (CVSS, CVE, SARIF) > 語言標準 > aiva_common > 模組專屬

**2️⃣ 禁止重複定義**:
```python
# ❌ 禁止
class Severity(str, Enum): pass  # aiva_common 已定義！

# ✅ 正確
from aiva_common import Severity, Confidence, TaskStatus
```

**3️⃣ 模組專屬枚舉判斷**:
- ✅ 僅模組內部使用
- ✅ 與業務邏輯強綁定
- ✅ aiva_common 無類似定義

📖 **詳細規範**: [aiva_common 修復規範完整文檔](../../../aiva_common/README.md#-開發規範與最佳實踐)

---

### 遵循 AIVA Common 規範

本模組遵循 [`services/aiva_common`](../../../aiva_common/README.md) 的標準規範:

- ✅ 使用 `aiva_common.enums` 的標準枚舉
- ✅ 使用 `aiva_common.schemas` 的數據結構
- ✅ 遵循 PEP 8 和 PEP 484 規範
- ✅ 完整的類型標註和文檔字串

### 新增組件指南

```python
from aiva_common import Severity, Confidence  # 使用標準枚舉
from pydantic import BaseModel, Field

class CognitiveResult(BaseModel):
    """認知處理結果"""
    decision: str = Field(..., description="AI 決策結果")
    confidence: Confidence = Field(..., description="信心度")
    reasoning: str = Field(..., description="推理過程")
```

---

## 📊 性能指標

| 指標 | 當前值 | 目標值 | 狀態 |
|------|--------|--------|------|
| **神經網路推理延遲** | 120ms | 100ms | 🟡 優化中 |
| **RAG 檢索延遲** | 85ms | 80ms | 🟡 優化中 |
| **決策準確率** | 89% | 95% | 🟡 優化中 |
| **反幻覺檢測率** | 94% | 98% | ✅ 良好 |

---

## 🧪 測試

```bash
# 運行單元測試
pytest tests/test_cognitive_core/

# 運行集成測試
pytest tests/integration/test_cognitive_core_integration.py

# 性能基準測試
python benchmarks/benchmark_cognitive_core.py
```

---

## 📚 相關文檔

- [AIVA Core 重構計劃](../REFACTORING_PLAN.md)
- [AI 自我優化雙重閉環設計](../../../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)
- [AIVA Common 規範](../../../aiva_common/README.md)

---

**📝 文檔版本**: v1.0  
**🔄 最後更新**: 2025年11月27日  
**👥 維護者**: AIVA Core 開發團隊

---

## ⚠️ 重要提醒

本模組目前處於架構搭建階段 (🚧)。原有組件將在後續階段遷移到對應的子目錄中。請參考 [REFACTORING_PLAN.md](../REFACTORING_PLAN.md) 了解詳細的遷移計劃。
