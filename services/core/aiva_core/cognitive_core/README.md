# 🧠 Cognitive Core - 認知核心

> **版本**: v2.1.2  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2025-12-01  
> **角色**: AIVA 的 AI 認知能力核心

**導航**: [← 返回 AIVA Core](../README.md)

---

## 📋 目錄

- [模組概述](#-模組概述)
- [子系統架構](#-子系統架構)
  - [Neural - 神經網路核心](#1-neural---神經網路核心)
  - [Decision - 決策支援系統](#2-decision---決策支援系統)
  - [RAG - 檢索增強生成](#3-rag---檢索增強生成)
  - [Anti-Hallucination - 反幻覺模組](#4-anti-hallucination---反幻覺模組)
- [整合使用](#-整合使用)
- [性能指標](#-性能指標)

---

## 🎯 模組概述

Cognitive Core 是 AIVA 的認知智能核心，整合了神經網路推理、智能決策、知識檢索和可靠性驗證四大子系統，提供完整的 AI 認知能力。

**核心職責**：
- 🧠 **神經網路推理** - 500萬參數 BioNeuron 模型的推理和管理
- 🎯 **智能決策** - AI 增強的決策支援和技能圖譜
- 🔍 **知識檢索** - RAG 系統提供上下文增強
- 🛡️ **可靠性保障** - 反幻覺機制確保輸出準確性

---

## 🏗️ 子系統架構

### 1. Neural - 神經網路核心

**位置**: `cognitive_core/neural/`

**核心組件**：
- `real_neural_core.py` - 500萬參數神經網路核心（800+ 行）
- `bio_neuron_master.py` - 三模式主控系統（UI/AI/Chat）（600+ 行）
- `ai_model_manager.py` - 統一 AI 模型管理器（400+ 行）
- `weight_manager.py` - 權重持久化和版本控制（300+ 行）
- `real_bio_net_adapter.py` - 生物網路適配器（200+ 行）
- `neural_network.py` - 神經網路基礎類（150+ 行）

**主要功能**：
```python
from aiva_core.cognitive_core.neural import RealNeuralCore, BioNeuronMaster

# 神經網路推理
neural_core = RealNeuralCore(use_5m_model=True)
neural_core.load_weights()
output = neural_core.forward(input_tensor)

# 主控系統（三模式）
master = BioNeuronMaster(mode="ai")  # ui/ai/chat
result = await master.process_request(request)
```

**特性**：
- ✅ 500萬參數生物啟發架構
- ✅ 支援 PyTorch 訓練和推理
- ✅ 權重自動持久化和版本控制
- ✅ GPU/CPU 自動切換
- ✅ 三模式統一調度（UI/AI/Chat）

---

### 2. Decision - 決策支援系統

**位置**: `cognitive_core/decision/`

**核心組件**：
- `enhanced_decision_agent.py` - AI 增強決策代理（400+ 行）
- `skill_graph.py` - 技能圖譜和關係映射（300+ 行）

**主要功能**：
```python
from aiva_core.cognitive_core.decision import EnhancedDecisionAgent, SkillGraph

# 技能圖譜
skill_graph = SkillGraph()
skill_graph.add_skill("SQL注入", category="Web安全", prerequisites=["HTTP基礎"])
recommendations = skill_graph.recommend_next_skills(completed_skills)

# AI 決策
agent = EnhancedDecisionAgent(neural_core)
decision = await agent.make_decision(context, constraints)
```

**特性**：
- ✅ 上下文感知的智能決策
- ✅ 技能依賴關係和推薦
- ✅ 多約束優化決策
- ✅ 可解釋的決策過程

---

### 3. RAG - 檢索增強生成

**位置**: `cognitive_core/rag/`

**核心組件**：
- `rag_engine.py` - RAG 核心引擎（500+ 行）
- `knowledge_base.py` - 知識庫管理（400+ 行）
- `unified_vector_store.py` - 統一向量存儲接口（300+ 行）
- `postgresql_vector_store.py` - PostgreSQL 向量後端（250+ 行）

**主要功能**：
```python
from aiva_core.cognitive_core.rag import RAGEngine, KnowledgeBase

# 初始化 RAG
rag = RAGEngine(
    knowledge_base=KnowledgeBase(),
    vector_store_type="postgresql"  # or "memory"
)

# 檢索增強
context = await rag.retrieve(query, top_k=5)
enhanced_prompt = rag.enhance_prompt(prompt, context)
```

**特性**：
- ✅ 高效向量相似度搜索
- ✅ 支援內存和 PostgreSQL 後端
- ✅ 整合內部探索和外部學習知識
- ✅ 自動上下文增強

---

### 4. Anti-Hallucination - 反幻覺模組

**位置**: `cognitive_core/anti_hallucination/`

**核心組件**：
- `anti_hallucination_module.py` - 反幻覺檢查（350+ 行）

**主要功能**：
```python
from aiva_core.cognitive_core.anti_hallucination import AntiHallucinationModule

# 反幻覺驗證
validator = AntiHallucinationModule(knowledge_base)
result = await validator.validate_output(
    output=ai_response,
    context=context,
    threshold=0.7
)

if result.is_reliable:
    return result.validated_output
else:
    logger.warning(f"Low confidence: {result.confidence_score}")
```

**驗證機制**：
- ✅ 事實準確性驗證（與知識源交叉檢查）
- ✅ 多知識源交叉驗證
- ✅ 邏輯連貫性檢查
- ✅ 置信度評分和不確定性標記

---

## 🔗 整合使用

### 完整認知流程

```python
from aiva_core.cognitive_core import (
    RealNeuralCore, 
    RAGEngine, 
    EnhancedDecisionAgent,
    AntiHallucinationModule
)

# 1. 初始化所有組件
neural_core = RealNeuralCore(use_5m_model=True)
neural_core.load_weights()

rag = RAGEngine(vector_store_type="postgresql")
decision_agent = EnhancedDecisionAgent(neural_core)
validator = AntiHallucinationModule(rag.knowledge_base)

# 2. RAG 檢索增強
context = await rag.retrieve(user_query, top_k=5)
enhanced_prompt = rag.enhance_prompt(user_query, context)

# 3. 神經網路推理
neural_output = neural_core.forward(enhanced_prompt)

# 4. AI 決策
decision = await decision_agent.make_decision(
    context={"output": neural_output, "constraints": constraints}
)

# 5. 反幻覺驗證
validated = await validator.validate_output(
    output=decision.action,
    context=context
)

# 6. 返回可靠結果
if validated.is_reliable:
    return validated.validated_output
```

---

## 📊 性能指標

### 神經網路性能
- **模型大小**: 500萬參數（~20MB）
- **推理速度**: ~50ms/batch (GPU), ~200ms/batch (CPU)
- **內存佔用**: ~150MB (模型) + ~50MB (運行時)

### RAG 檢索性能
- **向量維度**: 768 (BERT-base)
- **檢索速度**: <10ms (內存), <50ms (PostgreSQL)
- **知識庫容量**: 10萬+ 文檔

### 決策性能
- **決策延遲**: ~30ms (簡單), ~200ms (複雜約束)
- **技能圖譜**: 100+ 技能節點，500+ 關係邊

### 反幻覺性能
- **驗證速度**: ~100ms/輸出
- **準確率**: >95% (事實驗證)
- **誤判率**: <3%

---

## 🔗 相關模組

- [Task Planning](../task_planning/README.md) - 使用認知能力進行任務規劃
- [External Learning](../external_learning/README.md) - 為 RAG 提供外部知識
- [Core Capabilities](../core_capabilities/README.md) - 調用認知能力執行具體任務

---

**最後更新**: 2025-12-01 | **維護者**: AIVA Team
