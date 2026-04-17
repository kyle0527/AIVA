# Neural 神經網路核心模組

> **路徑**: `cognitive_core/neural`  
> **狀態**: ✅ 正常 | **Python 文件數**: 5 | **最後更新**: 2026-04-05

## 概述

提供生物啟發的神經網路推理能力，包含 500 萬參數的 BioNeuron 核心。使用真實的 PyTorch 神經網路進行決策，支援權重持久化和訓練。

## 📄 檔案詳細資訊 (Files Details)

### `aiva_embedding.py`
**說明**: AIVA 自研 Embedding 層

**類別 (Classes)**:
- `AIVAEmbedding` - AIVA 自主可控的 Embedding 模型
**函式 (Functions)**:
- `SentenceTransformer()` - AIVA 版本的 SentenceTransformer

### `real_bio_net_adapter.py`
**說明**: 真實AI核心的向後相容適配器

**類別 (Classes)**:
- `RealScalableBioNet` - 真實的ScalableBioNet - 向後相容的AI核心替換
- `RealBioNeuronRAGAgent` - 真實AI的RAG代理 - 向後相容的適配器
**函式 (Functions)**:
- `create_real_scalable_bionet()` - 創建真實的ScalableBioNet實例 - 工廠函數
- `create_real_rag_agent()` - 創建真實的RAG代理實例 - 工廠函數

### `real_neural_core.py`
**說明**: 真實的神經網路核心 - 替換AIVA的假AI實現

**類別 (Classes)**:
- `RealAICore` - 真實的AI核心 - 使用5M特化神經網路模型
- `RealDecisionEngine` - 真實的決策引擎 - 支援5M特化神經網路
**函式 (Functions)**:
- `create_real_ai_replacement()` - 創建真實的AI來替換AIVA的假AI

### `weight_manager.py`
**說明**: 真實AI權重管理系統

**類別 (Classes)**:
- `WeightMetadata` - 權重檔案元數據
- `AIWeightManager` - AI權重管理器
**函式 (Functions)**:
- `get_weight_manager()` - 獲取全局權重管理器實例
- `initialize_weight_manager()` - 初始化全局權重管理器

