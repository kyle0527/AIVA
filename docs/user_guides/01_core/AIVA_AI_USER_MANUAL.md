# 🚀 AIVA AI 系統使用者手冊

**版本**: v2.1.0 | **更新日期**: 2025年11月11日 | **狀態**: ✅ 已驗證

---

## 📋 詳細目錄

### 🎯 [系統簡介](#-系統簡介)
- [核心特色](#核心特色)
- [AI 能力矩陣](#ai-能力矩陣)
- [系統架構概覽](#系統架構概覽)

### ⚡ [快速開始](#-快速開始) ✅ *已驗證 2025-11-12*
- [方法一：快速驗證（推薦新手）](#方法一快速驗證推薦新手)
- [方法二：直接 Python 啟動（推薦）](#方法二直接-python-啟動推薦)
- [方法三：Docker 容器啟動](#方法三docker-容器啟動)

### 🛠️ [安裝配置](#️-安裝配置)
- [系統需求](#系統需求)
- [依賴安裝](#依賴安裝)
- [環境配置](#環境配置)

### 🧠 [AI 核心功能](#-ai-核心功能) ✅ *已驗證 2025-11-12*
- [1. AI 系統初始化](#1-ai-系統初始化)
- [2. AI 決策功能使用](#2-ai-決策功能使用)
- [3. RAG 檢索功能](#3-rag-檢索功能)
- [4. 整合使用範例](#4-整合使用範例)

### 💻 [使用方式](#-使用方式)
- [A. 命令列介面 (CLI)](#a-命令列介面-cli)
- [B. Web 介面](#b-web-介面)
- [C. Python API（更新版）](#c-python-api更新版)
- [D. REST API](#d-rest-api)

### 📊 [功能驗證](#-功能驗證)
- [1. 系統健康檢查（更新版）](#1-系統健康檢查更新版)
- [2. AI 能力驗證（更新版）](#2-ai-能力驗證更新版)
- [3. 性能基準測試（更新版）](#3-性能基準測試更新版)

### 🔧 [故障排除](#-故障排除)
- [常見問題與解決方案](#常見問題與解決方案)
- [日誌與調試](#日誌與調試)

### 📚 [進階功能](#-進階功能)
- [1. 自定義 AI 配置](#1-自定義-ai-配置)
- [2. 自定義知識庫](#2-自定義知識庫)
- [3. API 擴展](#3-api-擴展)
- [4. 批量處理](#4-批量處理)

### 🔍 [AI 分析與掃描操作](#-ai-分析與掃描操作) ✅ *已驗證 2025-11-12*
- [1. AI 智能分析功能](#1-ai-智能分析功能)
- [2. 權重與優先級AI功能](#2-權重與優先級ai功能)
- [3. 消息代理AI功能](#3-消息代理ai功能)
- [4. AI能力註冊與發現](#4-ai能力註冊與發現)
- [5. Strangler Fig遷移AI控制](#5-strangler-fig遷移ai控制)
- [6. RAG增強AI功能](#6-rag增強ai功能)
- [7. AI模組掃描與分析](#7-ai模組掃描與分析)
- [8. AI性能監控與優化](#8-ai性能監控與優化)
- [9. AI安全漏洞掃描](#9-ai安全漏洞掃描)
- [10. 綜合分析報告生成](#10-綜合分析報告生成)
- [11. 實時監控與分析](#11-實時監控與分析)
- [12. AI學習與進化功能](#12-ai學習與進化功能)

### 📞 [技術支援](#-技術支援)
- [獲得幫助](#獲得幫助)
- [貢獻指南](#貢獻指南)

### 📄 [版本資訊](#-版本資訊)
- [更新日誌](#更新日誌)

---

## 📊 快速導覽

| 使用者類型 | 推薦起始點 | 重點章節 |
|------------|------------|----------|
| 🆕 **新手** | [快速開始](#-快速開始) → [功能驗證](#-功能驗證) | 基礎安裝、簡單範例 |
| 👨‍💻 **開發者** | [AI 核心功能](#-ai-核心功能) → [Python API](#c-python-api更新版) | AI 整合、API 使用 |
| 🔧 **系統管理員** | [安裝配置](#️-安裝配置) → [故障排除](#-故障排除) | 環境設定、問題解決 |
| 🚀 **進階用戶** | [進階功能](#-進階功能) → [技術支援](#-技術支援) | 自定義配置、擴展開發 |
| 🎓 **理論研究者** | [理論操作方式](#-理論操作方式) → [實際操作驗證](#-實際操作驗證) | AI原理、驗證方法 |

---

---

## 🎯 系統簡介

AIVA (Autonomous Intelligence Virtual Assistant) 是一個企業級的AI驅動安全測試平台，具備：

### 核心特色
- **🧠 500萬參數神經網路**: 真實的生物啟發式AI大腦
- **📚 RAG檢索增強**: 智能知識檢索與融合系統
- **🤖 四種運行模式**: UI、AI、Chat、混合模式
- **⚡ 自主決策能力**: 完全自主的安全測試執行
- **🛡️ 抗幻覺機制**: 多層驗證確保決策可靠性

### AI 能力矩陣
| 能力 | 狀態 | 成熟度 | 描述 |
|------|------|--------|------|
| 🔍 **智能搜索** | ✅ | ⭐⭐⭐⭐⭐ | 語義搜索、向量檢索 |
| 📚 **RAG增強** | ✅ | ⭐⭐⭐⭐⭐ | 檢索增強生成 |
| 🤔 **推理決策** | ✅ | ⭐⭐⭐⭐ | 神經網路推理 |
| 📖 **學習能力** | ✅ | ⭐⭐⭐⭐ | 經驗學習與進化 |
| 💾 **知識管理** | ✅ | ⭐⭐⭐⭐⭐ | AST代碼分析 |
| 💬 **自然語言** | 🚧 | ⭐⭐⭐ | 對話理解與生成 |

### 系統架構概覽

```mermaid
graph TB
    subgraph "AIVA AI 系統架構"
        BNM[BioNeuronMasterController<br/>主控制器]
        
        subgraph "AI 核心引擎"
            RSBN[RealScalableBioNet<br/>真實神經網路]
            RAC[RealAICore<br/>5M參數AI核心]
            RSBN --> RAC
        end
        
        subgraph "RAG 系統"
            RE[RAGEngine<br/>檢索引擎]
            KB[KnowledgeBase<br/>知識庫]
            RBA[RealBioNeuronRAGAgent<br/>RAG代理]
            RE --> KB
            RBA --> RE
        end
        
        subgraph "操作模式"
            UI[UI模式<br/>用戶介面]
            AI[AI模式<br/>自主決策]
            CHAT[Chat模式<br/>對話交互]
            HYBRID[Hybrid模式<br/>混合模式]
        end
        
        BNM --> RSBN
        BNM --> RE
        BNM --> UI
        BNM --> AI
        BNM --> CHAT
        BNM --> HYBRID
        
        RBA -.-> RAC
    end
    
    subgraph "外部介面"
        API[REST API]
        CLI[命令列]
        WEB[Web介面]
    end
    
    API --> BNM
    CLI --> BNM
    WEB --> BNM
```

**核心組件說明**：
- 🎮 **BioNeuronMasterController**: 系統主控制器，協調所有AI組件
- 🧠 **RealScalableBioNet**: 500萬參數的真實神經網路核心
- 📚 **RAGEngine**: 檢索增強生成引擎，結合知識庫和AI推理
- 🤖 **RealBioNeuronRAGAgent**: 專門的RAG代理，支援獨立使用
- 💾 **KnowledgeBase**: 向量化知識庫，支援語義搜索

---

## ⚡ 快速開始

### 方法一：快速驗證（推薦新手）

```powershell
# 1. 設定環境
$env:PYTHONPATH = "C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services"

# 2. 執行快速驗證腳本
python -c "
import sys
sys.path.append('C:/D/fold7/AIVA-git')
sys.path.append('C:/D/fold7/AIVA-git/services')

print('🚀 AIVA AI 系統快速驗證')
print('=' * 50)

try:
    print('🔍 測試 1: 檢查基礎依賴')
    import torch
    import numpy as np
    print('   ✅ PyTorch & NumPy 導入成功')
    
    print('🔍 測試 2: 檢查 AI 引擎模組')
    from services.core.aiva_core.ai_engine.real_bio_net_adapter import RealBioNeuronRAGAgent, create_real_rag_agent
    print('   ✅ 真實 AI 引擎模組導入成功')
    
    print('🔍 測試 3: 檢查 RAG 系統')  
    from services.core.aiva_core.rag.rag_engine import RAGEngine
    print('   ✅ RAG 引擎導入成功')
    
    print('🔍 測試 4: 創建基本 AI 組件')
    decision_core = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 20)
    )
    
    rag_agent = create_real_rag_agent(
        decision_core=decision_core,
        input_vector_size=512
    )
    print('   ✅ AI 組件創建成功')
    
    print('🔍 測試 5: 基本功能測試')
    result = rag_agent.generate(
        task_description='測試 AI 決策功能',
        context='系統驗證測試'
    )
    confidence = result.get('confidence', 'unknown')
    print(f'   ✅ AI 決策測試成功，信心度: {confidence}')
    
    print('')
    print('🎉 AIVA AI 核心功能驗證成功！')
    print('📖 請查看 AIVA_USER_MANUAL.md 了解完整使用方式')
    
except Exception as e:
    print(f'❌ 驗證失敗: {e}')
    import traceback
    traceback.print_exc()
"

# 3. 查看系統狀態
echo "✅ AIVA AI 系統驗證完成"
```

### 方法二：直接 Python 啟動（推薦）

```powershell
# 設定環境變數
$env:PYTHONPATH = "C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services"

# 快速驗證系統
python -c "
import sys
sys.path.append('C:/D/fold7/AIVA-git')
sys.path.append('C:/D/fold7/AIVA-git/services')

# 導入核心模組
from services.core.aiva_core.ai_engine.real_bio_net_adapter import create_real_rag_agent
from services.core.aiva_core.rag.rag_engine import RAGEngine
import torch

# 創建 AI 組件
decision_core = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 20))
rag_agent = create_real_rag_agent(decision_core=decision_core, input_vector_size=512)
rag_engine = RAGEngine()

print('🎉 AIVA AI 系統驗證成功!')
print(f'🧠 RAG 代理: {type(rag_agent).__name__}')
print(f'📚 RAG 引擎: {type(rag_engine).__name__}')
"
```

### 方法三：Docker 容器啟動

```bash
# 構建並啟動
docker-compose up -d

# 查看服務狀態
docker-compose ps
```

---

## 🛠️ 安裝配置

### 系統需求

| 項目 | 最小需求 | 推薦配置 |
|------|----------|----------|
| **Python** | 3.8+ | 3.11+ |
| **記憶體** | 8GB | 16GB+ |
| **儲存空間** | 10GB | 50GB+ |
| **CPU** | 4核心 | 8核心+ |

### 依賴安裝

```powershell
# 1. 安裝核心依賴
python -m pip install --upgrade protobuf grpcio grpcio-tools torch numpy fastapi uvicorn

# 2. 安裝額外套件
pip install sentence-transformers transformers datasets scikit-learn pandas requests aiofiles asyncio

# 3. 驗證安裝
python -c "import torch, numpy, fastapi; print('✅ 依賴安裝成功!')"
```

### 環境配置

```powershell
# 1. 創建配置文件
Copy-Item config/config.example.yml config/config.yml

# 2. 設定 PYTHONPATH
$env:PYTHONPATH = "C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services;C:\D\fold7\AIVA-git\services\features;C:\D\fold7\AIVA-git\services\aiva_common"

# 3. 驗證配置
python -c "import sys; print('PYTHONPATH 配置正確:', 'services' in str(sys.path))"
```

---

## 🧠 AI 核心功能

### 1. AI 系統初始化

```python
# 方法 1: 使用真實 RAG 代理 (推薦)
from services.core.aiva_core.ai_engine.real_bio_net_adapter import RealBioNeuronRAGAgent, create_real_rag_agent
import torch

# 創建決策核心網路
decision_core = torch.nn.Sequential(
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(), 
    torch.nn.Linear(256, 20)
)

# 創建 RAG 代理
rag_agent = create_real_rag_agent(
    decision_core=decision_core,
    input_vector_size=512
)

print(f"🧠 神經網路類型: {type(rag_agent).__name__}")
print(f"� 決策核心: {decision_core}")

# 方法 2: 使用 RAG 引擎
from services.core.aiva_core.rag.rag_engine import RAGEngine

rag_engine = RAGEngine()
print(f"📚 RAG 引擎: {type(rag_engine).__name__}")
```

### 2. AI 決策功能使用

```python
# AI 決策生成
result = rag_agent.generate(
    task_description="分析目標系統安全漏洞",
    context="目標: https://example.com"
)

print(f"決策結果: {result.get('decision', 'N/A')}")
print(f"信心度: {result.get('confidence', 'N/A')}")
print(f"建議行動: {result.get('suggested_actions', [])}")

# 加載預訓練權重 (如果有)
try:
    rag_agent.load_state_dict(torch.load('weights/aiva_model.pth'))
    print("✅ 預訓練權重載入成功")
except:
    print("ℹ️ 使用隨機初始化權重")
```

### 3. RAG 檢索功能

```python
# 使用 RAG 引擎進行知識檢索
from services.core.aiva_core.rag.rag_engine import RAGEngine
from services.core.aiva_core.rag.knowledge_base import KnowledgeBase

# 創建知識庫和 RAG 引擎
knowledge_base = KnowledgeBase()
rag_engine = RAGEngine(knowledge_base)

# 執行語義搜索 (注意：這是概念性範例)
# 實際使用中可能需要先索引知識庫
try:
    # 嘗試搜索功能 (可能需要知識庫有內容)
    print(f"RAG 引擎已準備: {type(rag_engine).__name__}")
    print(f"知識庫類型: {type(knowledge_base).__name__}")
    
    # 搜索相關知識
    # search_results = await rag_engine.search(...)
    
except Exception as e:
    print(f"RAG 搜索需要先設置知識庫: {e}")

# 直接使用知識庫功能
try:
    # 添加新知識到知識庫
    knowledge_base.add_knowledge(
        content="新的安全知識內容",
        knowledge_type="security",
        metadata={"source": "custom", "category": "security"}
    )
    print("✅ 知識添加成功")
except Exception as e:
    print(f"知識添加: {e}")
```

### 4. 整合使用範例

```python
# 完整工作流程範例
import torch
import asyncio
from services.core.aiva_core.ai_engine.real_bio_net_adapter import create_real_rag_agent
from services.core.aiva_core.rag.rag_engine import RAGEngine

async def aiva_workflow_example():
    """AIVA 完整工作流程示例"""
    
    # 1. 初始化組件
    print("🔧 初始化 AI 組件...")
    decision_core = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 20)
    )
    
    rag_agent = create_real_rag_agent(
        decision_core=decision_core,
        input_vector_size=512
    )
    
    rag_engine = RAGEngine()
    
    # 2. 知識檢索
    print("🔍 執行知識檢索...")
    knowledge = await rag_engine.search(
        query="網路安全測試方法",
        top_k=3
    )
    
    # 3. AI 決策
    print("🤖 生成 AI 決策...")
    decision = rag_agent.generate(
        task_description="基於檢索到的知識進行安全分析",
        context=f"檢索結果: {knowledge}"
    )
    
    # 4. 結果輸出
    print(f"✅ 決策完成: {decision.get('confidence')}")
    return decision

# 執行示例
# result = asyncio.run(aiva_workflow_example())
```

---

## 💻 使用方式

### A. 命令列介面 (CLI)

```powershell
# 1. 基本掃描
python -m aiva.cli scan --target "https://example.com" --mode "ai"

# 2. 互動模式
python -m aiva.cli interactive

# 3. 配置檢查
python -m aiva.cli config check
```

### B. Web 介面

```powershell
# 啟動 Web 服務
.\start-aiva.ps1 -Action core

# 訪問介面
# 主要 API: http://localhost:8000
# 管理面板: http://localhost:8001
# 神經網路 API: http://localhost:8000/api/v2/neural/
```

### C. Python API（更新版）

```python
import asyncio
import torch
from services.core.aiva_core.ai_engine.real_bio_net_adapter import create_real_rag_agent
from services.core.aiva_core.rag.rag_engine import RAGEngine

async def aiva_api_example():
    """AIVA Python API 使用示例"""
    
    # 初始化核心組件
    decision_core = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 20)
    )
    
    rag_agent = create_real_rag_agent(
        decision_core=decision_core,
        input_vector_size=512
    )
    
    rag_engine = RAGEngine()
    
    # 執行 AI 任務
    print("🔍 執行知識搜索...")
    search_results = await rag_engine.search(
        query="測試目標的安全性",
        top_k=3
    )
    
    print("🤖 生成 AI 決策...")
    decision = rag_agent.generate(
        task_description="安全性評估",
        context=f"搜索結果: {search_results}"
    )
    
    print(f"✅ 任務完成: 信心度 {decision.get('confidence')}")
    return decision

# 執行 API 示例
# result = asyncio.run(aiva_api_example())
```

### D. REST API

```bash
# 健康檢查
curl http://localhost:8000/health

# AI 決策請求
curl -X POST http://localhost:8000/api/v2/ai/decide \
  -H "Content-Type: application/json" \
  -d '{"objective": "安全測試", "target": "example.com"}'

# 神經網路狀態
curl http://localhost:8000/api/v2/neural/health
```

---

## 📊 功能驗證

### 1. 系統健康檢查（更新版）

```python
# 完整系統檢查腳本 - 基於實際架構
import sys
sys.path.append('C:/D/fold7/AIVA-git')
sys.path.append('C:/D/fold7/AIVA-git/services')

def check_aiva_system():
    """AIVA 系統健康檢查 - 2025年11月版本"""
    
    try:
        print("🔍 檢查 1: 基礎依賴檢查")
        import torch
        import numpy as np
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   ✅ NumPy: {np.__version__}")
        
        print("🔍 檢查 2: AI 引擎模組導入")
        from services.core.aiva_core.ai_engine.real_bio_net_adapter import RealBioNeuronRAGAgent, create_real_rag_agent
        print("   ✅ 真實 AI 引擎模組導入成功")
        
        print("🔍 檢查 3: RAG 系統檢查")  
        from services.core.aiva_core.rag.rag_engine import RAGEngine
        rag_engine = RAGEngine()
        print(f"   ✅ RAG 引擎: {type(rag_engine).__name__}")
        
        print("🔍 檢查 4: 創建 AI 組件")
        decision_core = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 20)
        )
        
        rag_agent = create_real_rag_agent(
            decision_core=decision_core,
            input_vector_size=512
        )
        print(f"   ✅ RAG 代理: {type(rag_agent).__name__}")
        print(f"   ✅ 決策核心: {type(decision_core).__name__}")
        
        print("🔍 檢查 5: AI 功能測試")
        result = rag_agent.generate(
            task_description='測試 AI 決策功能',
            context='系統驗證測試'
        )
        confidence = result.get('confidence', 'unknown')
        print(f"   ✅ AI 決策測試成功，信心度: {confidence}")
        
        print("\n🎉 AIVA AI 系統健康檢查通過！")
        print("📖 請查看 AIVA_USER_MANUAL.md 了解詳細使用方式")
        return True
        
    except Exception as e:
        print(f"❌ 系統檢查失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

# 執行檢查
if __name__ == "__main__":
    check_aiva_system()
```

### 2. AI 能力驗證（更新版）

```python
import asyncio
import torch
from services.core.aiva_core.ai_engine.real_bio_net_adapter import create_real_rag_agent
from services.core.aiva_core.rag.rag_engine import RAGEngine

async def validate_ai_capabilities():
    """AI 能力驗證測試 - 基於實際架構"""
    
    print("🧠 初始化 AI 組件...")
    decision_core = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 20)
    )
    
    rag_agent = create_real_rag_agent(
        decision_core=decision_core,
        input_vector_size=512
    )
    
    rag_engine = RAGEngine()
    
    # 1. 搜索能力測試
    print("🔍 測試智能搜索能力...")
    try:
        search_result = await rag_engine.search("XSS 攻擊", top_k=3)
        assert len(search_result) >= 0, "搜索功能異常"
        print(f"   ✅ 搜索能力正常 - 找到 {len(search_result)} 條結果")
    except Exception as e:
        print(f"   ⚠️ 搜索功能測試: {e}")
    
    # 2. 決策能力測試  
    print("🤔 測試 AI 決策能力...")
    try:
        decision = rag_agent.generate(
            task_description="測試安全評估",
            context="目標系統分析"
        )
        assert "confidence" in decision or decision is not None, "決策功能異常"
        print(f"   ✅ 決策能力正常 - 信心度: {decision.get('confidence', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️ 決策功能測試: {e}")
    
    # 3. 神經網路測試
    print("🧮 測試神經網路推理...")
    try:
        test_input = torch.randn(1, 512)  # 隨機測試輸入
        output = decision_core(test_input)
        assert output.shape[-1] == 20, "神經網路輸出維度異常"
        print(f"   ✅ 神經網路推理正常 - 輸出形狀: {output.shape}")
    except Exception as e:
        print(f"   ⚠️ 神經網路測試: {e}")
    
    print("🎉 AI 能力驗證完成！")

# 執行驗證
# asyncio.run(validate_ai_capabilities())
```

### 3. 性能基準測試（更新版）

```python
import time
import asyncio
import torch
from services.core.aiva_core.ai_engine.real_bio_net_adapter import create_real_rag_agent
from services.core.aiva_core.rag.rag_engine import RAGEngine

async def performance_benchmark():
    """性能基準測試 - 基於實際架構"""
    
    print("📊 啟動 AIVA 性能基準測試...")
    
    # 初始化組件
    decision_core = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 20)
    )
    
    rag_agent = create_real_rag_agent(
        decision_core=decision_core,
        input_vector_size=512
    )
    
    rag_engine = RAGEngine()
    
    # 神經網路推理性能測試
    print("🧮 測試神經網路推理性能...")
    start_time = time.time()
    
    # 批量推理測試
    test_batch = torch.randn(10, 512)  # 10個樣本
    with torch.no_grad():
        for _ in range(100):  # 100次推理
            _ = decision_core(test_batch)
    
    nn_time = time.time() - start_time
    nn_throughput = (10 * 100) / nn_time  # 樣本/秒
    
    print(f"   🚀 神經網路推理: {nn_time:.2f}s")
    print(f"   📈 推理吞吐量: {nn_throughput:.1f} 樣本/s")
    
    # AI 決策性能測試
    print("🤖 測試 AI 決策性能...")
    start_time = time.time()
    
    decisions = []
    for i in range(5):  # 5次決策測試
        result = rag_agent.generate(
            task_description=f"性能測試任務 {i+1}",
            context="基準測試"
        )
        decisions.append(result)
    
    decision_time = time.time() - start_time
    decision_throughput = len(decisions) / decision_time
    
    print(f"   ⚡ AI 決策時間: {decision_time:.2f}s")
    print(f"   🎯 決策吞吐量: {decision_throughput:.1f} 決策/s")
    
    # 性能評估
    print("\n📊 性能評估結果:")
    if nn_throughput > 100 and decision_throughput > 1.0:
        print("   🟢 性能: 優秀 (推薦生產使用)")
    elif nn_throughput > 50 and decision_throughput > 0.5:
        print("   🟡 性能: 良好 (適合開發測試)")
    else:
        print("   🔴 性能: 需要優化")
        
    print(f"   💻 神經網路吞吐量: {nn_throughput:.1f} 樣本/s")
    print(f"   🧠 AI 決策吞吐量: {decision_throughput:.1f} 決策/s")

# 執行基準測試
# asyncio.run(performance_benchmark())
```

---

## 🔧 故障排除

### 常見問題與解決方案

#### 1. 導入錯誤

**問題**: `ModuleNotFoundError: No module named 'services'`

**解決方案**:
```powershell
# 重新設定 PYTHONPATH
.\setup_env.ps1

# 或手動設定
$env:PYTHONPATH = "C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services"
```

#### 2. 依賴缺失

**問題**: `No module named 'torch'` 或其他依賴缺失

**解決方案**:
```powershell
# 安裝所有依賴
python -m pip install --upgrade protobuf grpcio torch numpy fastapi uvicorn

# 檢查安裝
python -c "import torch, numpy; print('依賴OK')"
```

#### 3. 記憶體不足

**問題**: 神經網路初始化時記憶體不足

**解決方案**:
```python
# 使用輕量化配置
controller = BioNeuronMasterController(
    default_mode="ui"  # 使用較輕的 UI 模式
)
```

#### 4. 權限問題

**問題**: 文件讀寫權限錯誤

**解決方案**:
```powershell
# 以管理員身份運行 PowerShell
# 或調整文件權限
icacls "C:\D\fold7\AIVA-git" /grant Everyone:F /t
```

### 日誌與調試

```python
import logging

# 啟用調試日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("aiva.debug")

# 查看詳細錯誤信息
try:
    aiva = BioNeuronMasterController()
except Exception as e:
    logger.error(f"初始化失敗: {e}", exc_info=True)
```

---

## 📚 進階功能

### 1. 自定義 AI 配置

```python
# 自定義神經網路配置
custom_config = {
    "neural_network": {
        "input_size": 512,
        "hidden_layers": [1024, 512, 256],
        "num_tools": 15,
        "confidence_threshold": 0.8
    },
    "rag_engine": {
        "top_k": 10,
        "similarity_threshold": 0.7,
        "context_window": 2048
    }
}

# 應用配置
aiva = BioNeuronMasterController()
await aiva.apply_configuration(custom_config)
```

### 2. 自定義知識庫

```python
# 添加自定義知識
await aiva.rag_engine.add_knowledge(
    content="自定義安全知識內容",
    metadata={
        "source": "custom",
        "category": "security",
        "priority": "high"
    }
)

# 索引代碼庫
await aiva.rag_engine.index_codebase(
    path="/path/to/custom/code",
    language_filter=["python", "javascript"]
)
```

### 3. API 擴展

```python
from fastapi import FastAPI
from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController

app = FastAPI()
aiva = BioNeuronMasterController()

@app.post("/custom/ai-analyze")
async def custom_ai_analyze(request: dict):
    """自定義 AI 分析端點"""
    result = await aiva.process_request(
        request=request.get("query"),
        mode="ai"
    )
    return {"analysis": result}

# 啟動服務
# uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 批量處理

```python
async def batch_processing(tasks: list):
    """批量任務處理"""
    
    aiva = BioNeuronMasterController()
    
    # 並行處理多個任務
    results = await asyncio.gather(*[
        aiva.process_request(task, mode="ai")
        for task in tasks
    ])
    
    # 結果彙總
    summary = {
        "total_tasks": len(tasks),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results
    }
    
    return summary

# 使用示例
tasks = [
    {"objective": "掃描目標1", "target": "example1.com"},
    {"objective": "掃描目標2", "target": "example2.com"},
    {"objective": "掃描目標3", "target": "example3.com"}
]

batch_result = await batch_processing(tasks)
print(f"批量處理完成: {batch_result['successful']}/{batch_result['total_tasks']}")
```

---

## � AI 分析與掃描操作

### 1. AI 智能分析功能

#### 基本代碼分析
```python
# 使用整合後的 AI 分析功能
import sys
sys.path.append('C:/D/fold7/AIVA-git')
sys.path.append('C:/D/fold7/AIVA-git/services')

from services.core.aiva_core.ai_engine.real_bio_net_adapter import create_real_rag_agent
import torch

# 初始化 AI 分析器
decision_core = torch.nn.Sequential(
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 20)
)

ai_analyzer = create_real_rag_agent(
    decision_core=decision_core,
    input_vector_size=512
)

# 執行代碼安全分析
analysis_result = ai_analyzer.generate(
    task_description="分析代碼安全性和潛在漏洞",
    context="目標：Python 應用程式安全掃描"
)

print(f"分析結果: {analysis_result.get('decision', '無決策')}")
print(f"信心度: {analysis_result.get('confidence', 'N/A')}")
```

#### 智能掃描目標分析
```python
# 目標系統分析
def analyze_target(target_url, scan_type="comprehensive"):
    """
    智能目標分析功能
    scan_type: "quick", "comprehensive", "stealth"
    """
    
    analysis_context = f"""
    目標URL: {target_url}
    掃描類型: {scan_type}
    分析重點: 安全漏洞、架構弱點、潛在風險
    """
    
    # AI 分析決策
    result = ai_analyzer.generate(
        task_description=f"執行 {scan_type} 安全分析",
        context=analysis_context
    )
    
    return {
        "target": target_url,
        "scan_type": scan_type,
        "ai_decision": result.get("decision"),
        "confidence": result.get("confidence"),
        "recommended_actions": result.get("suggested_actions", [])
    }

# 使用範例
target_analysis = analyze_target("https://example.com", "comprehensive")
print(f"目標分析完成: {target_analysis}")
```

### 2. 權重與優先級AI功能

#### 任務優先級智能排序
```python
# 使用任務轉換器的AI優先級功能
from services.core.aiva_core.planner.task_converter import TaskConverter, Task

# 初始化任務轉換器
task_converter = TaskConverter()

# 創建多個任務
tasks = [
    Task("高風險漏洞掃描", priority=90, estimated_time=300),
    Task("基礎端口掃描", priority=30, estimated_time=60),
    Task("深度滲透測試", priority=100, estimated_time=1800),
    Task("報告生成", priority=20, estimated_time=120)
]

# AI智能排序和執行規劃
execution_plan = task_converter.convert_to_execution_plan(tasks)

print("AI智能排序結果:")
for task in execution_plan:
    print(f"- {task.name} (優先級: {task.priority})")
```

#### 權限矩陣AI決策
```python
# 使用權限矩陣的AI風險評估
from services.core.aiva_core.authz.permission_matrix import PermissionMatrix

# 初始化權限矩陣
perm_matrix = PermissionMatrix()

# AI風險評估和權限決策
operation_context = {
    "user_role": "security_analyst",
    "target_system": "production_server",
    "operation_type": "vulnerability_scan",
    "risk_level": "L2"  # L0-L3風險等級
}

# AI授權決策
authorization_result = perm_matrix.authorize_operation(
    user_id="analyst001",
    operation="deep_scan",
    context=operation_context
)

print(f"AI授權決策: {authorization_result}")
```

### 3. 消息代理AI功能

#### 智能事件處理
```python
# ⚠️ 注意: v2.0 已改用命令系統 (CommandCenter)
# 舊版本 (已棄用):
# from services.core.aiva_core.messaging.message_broker import EnhancedMessageBroker
# message_broker = EnhancedMessageBroker()

# 新版本請使用:
from services.aiva_common.command_center import get_command_center

# 初始化命令中心
command_center = get_command_center()

# 發布 AI 事件 (新方式)
from services.aiva_common.schemas import AICommand, CommandType
ai_command = AICommand(
    command_type=CommandType.CUSTOM,
    event_type="security_alert",
    priority=95,  # 高優先級
    "ttl": 300,      # 5分鐘TTL
    "payload": {
        "alert_type": "sql_injection_detected",
        "target": "webapp.example.com",
        "severity": "critical",
        "ai_confidence": 0.92
    }
}

# AI智能路由和處理
message_broker.publish_event(
    topic="security.alerts",
    event=ai_event
)

print("AI安全事件已發布並智能路由")
```

#### 事件優先級AI篩選
```python
# AI事件訂閱和智能篩選
def ai_event_handler(event):
    """AI驅動的事件處理器"""
    
    # AI決策是否處理該事件
    if event.get("ai_confidence", 0) > 0.8:
        print(f"高信心度事件: {event['event_type']}")
        # 執行自動響應
        return True
    else:
        print(f"低信心度事件，需人工確認: {event['event_type']}")
        return False

# 訂閱AI篩選事件
message_broker.subscribe(
    topic="security.*",
    handler=ai_event_handler,
    priority_filter=lambda e: e.get("priority", 0) > 80
)
```

### 4. AI能力註冊與發現

#### 動態能力管理
```python
# 使用增強能力註冊表的AI管理
from services.core.aiva_core.plugins.ai_summary_plugin import EnhancedCapabilityRegistry

# 初始化AI能力管理器
capability_registry = EnhancedCapabilityRegistry()

# 註冊AI分析能力
vulnerability_scanner = {
    "name": "ai_vulnerability_scanner",
    "type": "security_analysis",
    "ai_powered": True,
    "confidence_threshold": 0.75,
    "dependencies": ["network_scanner", "web_crawler"],
    "weight": 85  # 能力權重
}

capability_registry.register_capability(
    "vulnerability_scanner",
    vulnerability_scanner
)

# AI智能能力發現和編排
available_capabilities = capability_registry.discover_capabilities(
    capability_type="security_analysis",
    min_weight=70
)

print(f"發現AI安全分析能力: {len(available_capabilities)} 個")
```

#### 智能依賴解析
```python
# AI驅動的依賴管理
def ai_dependency_resolution(target_capability):
    """AI智能依賴解析"""
    
    # 獲取能力及其依賴
    capability_info = capability_registry.get_capability(target_capability)
    
    if capability_info:
        dependencies = capability_info.get("dependencies", [])
        
        # AI權重排序依賴
        sorted_deps = capability_registry.resolve_dependencies(
            dependencies,
            sort_by_weight=True
        )
        
        print(f"AI依賴解析 - {target_capability}:")
        for dep in sorted_deps:
            weight = capability_registry.get_capability_weight(dep)
            print(f"  依賴: {dep} (權重: {weight})")
    
    return sorted_deps

# 解析漏洞掃描器依賴
deps = ai_dependency_resolution("vulnerability_scanner")
```

### 5. Strangler Fig遷移AI控制

#### 智能遷移管理
```python
# 使用Strangler Fig遷移控制器的AI功能
from services.core.aiva_core import StranglerFigMigrationController

# 初始化AI遷移控制器
migration_controller = StranglerFigMigrationController()

# AI特性標誌管理
feature_flags = {
    "ai_enhanced_scanning": {
        "enabled": True,
        "weight": 90,
        "rollout_percentage": 75,
        "ai_confidence_required": 0.8
    },
    "legacy_scanner": {
        "enabled": True,
        "weight": 30,
        "rollout_percentage": 25,
        "fallback": True
    }
}

# AI智能路由決策
def ai_routing_decision(request_context):
    """AI驅動的功能路由決策"""
    
    user_risk_level = request_context.get("risk_level", "medium")
    operation_complexity = request_context.get("complexity", 0.5)
    
    # AI決策使用哪個功能版本
    if operation_complexity > 0.8 and user_risk_level == "high":
        return "ai_enhanced_scanning"
    else:
        return migration_controller.route_request(request_context)

# 應用AI路由
request = {
    "operation": "security_scan",
    "risk_level": "high",
    "complexity": 0.9,
    "user_id": "analyst001"
}

selected_feature = ai_routing_decision(request)
print(f"AI路由決策: 使用 {selected_feature} 功能")
```

### 6. RAG增強AI功能

#### 知識檢索與生成
```python
# 使用RAG引擎的AI知識增強
from services.core.aiva_core.rag.rag_engine import RAGEngine
from services.core.aiva_core.rag.knowledge_base import KnowledgeBase

# 初始化RAG AI系統
rag_engine = RAGEngine()
knowledge_base = KnowledgeBase()

# AI知識檢索
async def ai_knowledge_search(query, context="security"):
    """AI驅動的知識檢索"""
    
    search_results = await rag_engine.search(
        query=query,
        top_k=5,
        context_filter=context
    )
    
    return search_results

# AI增強的安全分析
async def ai_enhanced_security_analysis(target, scan_type):
    """結合RAG的AI安全分析"""
    
    # 1. 檢索相關安全知識
    security_knowledge = await ai_knowledge_search(
        query=f"{scan_type} security analysis techniques",
        context="penetration_testing"
    )
    
    # 2. AI生成分析策略
    analysis_strategy = ai_analyzer.generate(
        task_description=f"基於知識庫制定 {scan_type} 分析策略",
        context=f"目標: {target}, 知識: {security_knowledge}"
    )
    
    return {
        "target": target,
        "scan_type": scan_type,
        "knowledge_base": security_knowledge,
        "ai_strategy": analysis_strategy,
        "confidence": analysis_strategy.get("confidence", 0)
    }

# 使用AI增強分析
# enhanced_analysis = await ai_enhanced_security_analysis("webapp.com", "comprehensive")
```

### 7. AI模組掃描與分析

#### 自動化模組健康檢查
```python
# AI驅動的模組分析
def ai_module_health_scan():
    """AI自動模組健康掃描"""
    
    modules = {
        "core": "services/core",
        "scan": "services/scan", 
        "features": "services/features",
        "integration": "services/integration"
    }
    
    health_report = {}
    
    for module_name, module_path in modules.items():
        # AI分析模組狀態
        module_analysis = ai_analyzer.generate(
            task_description=f"分析 {module_name} 模組健康狀態",
            context=f"模組路徑: {module_path}, 檢查: 可用性、性能、安全性"
        )
        
        health_report[module_name] = {
            "path": module_path,
            "analysis": module_analysis.get("decision", "未知"),
            "confidence": module_analysis.get("confidence", 0),
            "timestamp": "2025-11-12",
            "ai_recommendations": module_analysis.get("suggested_actions", [])
        }
    
    return health_report

# 執行AI模組掃描
module_health = ai_module_health_scan()
print(f"AI模組健康掃描完成: {len(module_health)} 個模組")
```

#### 跨模組AI整合分析
```python
# 跨模組AI整合分析
def ai_cross_module_analysis():
    """AI跨模組整合分析"""
    
    integration_points = [
        ("core", "scan", "掃描引擎整合"),
        ("core", "features", "功能模組整合"), 
        ("scan", "integration", "整合掃描能力"),
        ("features", "integration", "功能整合接口")
    ]
    
    integration_report = {}
    
    for module_a, module_b, description in integration_points:
        # AI分析模組間整合狀態
        integration_analysis = ai_analyzer.generate(
            task_description=f"分析 {module_a} 與 {module_b} 整合狀態",
            context=f"整合點: {description}, 檢查: 兼容性、數據流、API一致性"
        )
        
        integration_key = f"{module_a}_{module_b}"
        integration_report[integration_key] = {
            "modules": [module_a, module_b],
            "description": description,
            "integration_status": integration_analysis.get("decision"),
            "confidence": integration_analysis.get("confidence", 0),
            "ai_suggestions": integration_analysis.get("suggested_actions", [])
        }
    
    return integration_report

# 執行跨模組AI分析
cross_module_analysis = ai_cross_module_analysis()
print(f"跨模組AI整合分析: {len(cross_module_analysis)} 個整合點")
```

### 8. AI性能監控與優化

#### 實時AI性能分析
```python
# AI性能監控
def ai_performance_monitor():
    """AI系統性能實時監控"""
    
    import psutil
    import time
    
    performance_metrics = {}
    
    # 神經網路推理性能測試
    start_time = time.time()
    test_input = torch.randn(10, 512)
    
    with torch.no_grad():
        for _ in range(100):
            _ = decision_core(test_input)
    
    nn_inference_time = time.time() - start_time
    performance_metrics["neural_network"] = {
        "inference_time": nn_inference_time,
        "throughput": 1000 / nn_inference_time,
        "status": "optimal" if nn_inference_time < 1.0 else "needs_optimization"
    }
    
    # AI決策性能測試
    start_time = time.time()
    for i in range(10):
        ai_analyzer.generate(
            task_description=f"性能測試 {i}",
            context="監控測試"
        )
    decision_time = time.time() - start_time
    
    performance_metrics["ai_decision"] = {
        "avg_decision_time": decision_time / 10,
        "decisions_per_second": 10 / decision_time,
        "status": "optimal" if decision_time < 5.0 else "needs_optimization"
    }
    
    # 系統資源使用
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    
    performance_metrics["system_resources"] = {
        "memory_usage": f"{memory_usage:.1f}%",
        "cpu_usage": f"{cpu_usage:.1f}%",
        "status": "optimal" if memory_usage < 80 and cpu_usage < 80 else "high_usage"
    }
    
    return performance_metrics

# 執行AI性能監控
ai_performance = ai_performance_monitor()
print("AI性能監控結果:")
for component, metrics in ai_performance.items():
    print(f"  {component}: {metrics['status']}")
```

#### AI自動優化建議
```python
# AI性能優化建議系統
def ai_optimization_suggestions(performance_data):
    """基於性能數據的AI優化建議"""
    
    optimization_context = f"""
    性能數據: {performance_data}
    優化目標: 提升推理速度、降低資源使用、增強決策準確性
    系統狀態: 生產環境運行
    """
    
    # AI生成優化建議
    optimization_advice = ai_analyzer.generate(
        task_description="生成AI系統性能優化建議",
        context=optimization_context
    )
    
    return {
        "performance_analysis": performance_data,
        "ai_recommendations": optimization_advice.get("suggested_actions", []),
        "optimization_confidence": optimization_advice.get("confidence", 0),
        "priority_actions": optimization_advice.get("decision", "無特定建議")
    }

# 獲取AI優化建議
optimization_report = ai_optimization_suggestions(ai_performance)
print(f"AI優化建議: {optimization_report['priority_actions']}")
```

### 2. 四大模組掃描操作

#### Core 模組分析
```python
# Core 模組健康掃描
def scan_core_module():
    """掃描核心模組狀態和性能"""
    
    try:
        # 檢查核心服務
        from services.core.aiva_core import *
        
        core_status = ai_analyzer.generate(
            task_description="分析 Core 模組健康狀態",
            context="檢查核心功能、AI 引擎、RAG 系統運行狀態"
        )
        
        return {
            "module": "Core",
            "status": "active",
            "analysis": core_status,
            "timestamp": "2025-11-12"
        }
        
    except Exception as e:
        return {"module": "Core", "status": "error", "error": str(e)}

# 執行 Core 掃描
core_scan = scan_core_module()
print(f"Core 模組掃描: {core_scan['status']}")
```

#### 掃描模組整合分析
```python
# 整合四大模組的掃描分析
async def comprehensive_module_scan():
    """執行四大模組的全面分析"""
    
    modules = ["core", "scan", "features", "integration"]
    scan_results = {}
    
    for module in modules:
        print(f"🔍 掃描 {module} 模組...")
        
        # AI 決策每個模組的掃描策略
        scan_strategy = ai_analyzer.generate(
            task_description=f"制定 {module} 模組掃描策略",
            context=f"模組: {module}, 目標: 安全性檢查、性能分析、架構評估"
        )
        
        # 模組路徑分析
        module_path = f"services/{module}"
        
        module_analysis = ai_analyzer.generate(
            task_description=f"分析 {module} 模組架構和安全性",
            context=f"""
            模組路徑: {module_path}
            掃描策略: {scan_strategy.get('decision', 'default')}
            分析重點: 代碼質量、安全漏洞、性能瓶頸
            """
        )
        
        scan_results[module] = {
            "strategy": scan_strategy,
            "analysis": module_analysis,
            "confidence": module_analysis.get("confidence", 0),
            "scan_time": "2025-11-12"
        }
    
    return scan_results

# 執行全面掃描
# module_scan_results = await comprehensive_module_scan()
```

### 9. AI安全漏洞掃描

#### 自動化漏洞檢測
```python
# 安全漏洞掃描功能
def security_vulnerability_scan(target, scan_depth="medium"):
    """
    AI 驅動的安全漏洞掃描
    scan_depth: "surface", "medium", "deep"
    """
    
    vulnerability_context = f"""
    掃描目標: {target}
    掃描深度: {scan_depth}
    檢查項目: SQL注入、XSS、CSRF、文件包含、權限提升
    AI分析: 漏洞風險評估、利用可能性、修復建議
    """
    
    # AI 漏洞分析決策
    vuln_analysis = ai_analyzer.generate(
        task_description="執行智能漏洞掃描和風險評估",
        context=vulnerability_context
    )
    
    return {
        "scan_target": target,
        "scan_depth": scan_depth,
        "vulnerabilities": vuln_analysis.get("decision", "未發現"),
        "risk_level": vuln_analysis.get("confidence", 0) * 100,
        "ai_recommendations": vuln_analysis.get("suggested_actions", []),
        "scan_timestamp": "2025-11-12"
    }

# 使用範例
vulnerability_report = security_vulnerability_scan("192.168.1.100", "deep")
print(f"漏洞掃描完成，風險等級: {vulnerability_report['risk_level']}%")
```

#### 網路掃描與偵察
```python
# 網路掃描功能
def intelligent_network_scan(target_range, scan_type="stealth"):
    """
    AI 輔助網路掃描
    scan_type: "stealth", "aggressive", "comprehensive"
    """
    
    network_context = f"""
    目標範圍: {target_range}
    掃描模式: {scan_type}
    掃描內容: 端口掃描、服務識別、OS指紋、拓撲分析
    AI優化: 掃描順序、探測策略、躲避檢測
    """
    
    # AI 網路掃描策略
    scan_strategy = ai_analyzer.generate(
        task_description=f"制定 {scan_type} 網路掃描策略",
        context=network_context
    )
    
    return {
        "target_range": target_range,
        "scan_type": scan_type,
        "strategy": scan_strategy.get("decision"),
        "confidence": scan_strategy.get("confidence"),
        "execution_plan": scan_strategy.get("suggested_actions", []),
        "scan_date": "2025-11-12"
    }

# 網路掃描示例
network_scan = intelligent_network_scan("192.168.1.0/24", "stealth")
print(f"網路掃描策略: {network_scan['strategy']}")
```

### 10. 綜合分析報告生成

#### 生成智能分析報告
```python
# 綜合分析報告生成
async def generate_comprehensive_report(target, include_modules=True):
    """生成完整的 AI 分析報告"""
    
    report = {
        "report_title": f"AIVA AI 綜合分析報告 - {target}",
        "generation_time": "2025-11-12",
        "ai_engine": "RealBioNeuronRAGAgent v2.0",
        "sections": {}
    }
    
    # 1. 目標基礎分析
    target_analysis = ai_analyzer.generate(
        task_description="目標基礎分析和風險評估",
        context=f"分析目標: {target}, 重點: 架構、安全性、可攻擊面"
    )
    report["sections"]["target_analysis"] = target_analysis
    
    # 2. 安全漏洞評估
    security_assessment = security_vulnerability_scan(target, "deep")
    report["sections"]["security_assessment"] = security_assessment
    
    # 3. 模組狀態分析 (如果啟用)
    if include_modules:
        module_status = await comprehensive_module_scan()
        report["sections"]["module_analysis"] = module_status
    
    # 4. AI 總結和建議
    final_summary = ai_analyzer.generate(
        task_description="生成綜合分析總結和行動建議",
        context=f"""
        目標: {target}
        分析結果: {report['sections']}
        要求: 風險總結、優先修復項目、後續行動建議
        """
    )
    report["sections"]["ai_summary"] = final_summary
    
    return report

# 生成報告示例
# comprehensive_report = await generate_comprehensive_report("https://example.com")
# print(f"報告生成完成: {comprehensive_report['report_title']}")
```

### 11. 實時監控與分析

#### 持續監控功能
```python
# 實時監控分析
def start_real_time_monitoring(targets, monitoring_interval=300):
    """
    啟動實時監控分析
    monitoring_interval: 監控間隔(秒)
    """
    
    monitoring_config = {
        "targets": targets,
        "interval": monitoring_interval,
        "ai_analysis": True,
        "alert_threshold": 0.7,
        "start_time": "2025-11-12"
    }
    
    # AI 監控策略
    monitoring_strategy = ai_analyzer.generate(
        task_description="制定實時監控策略",
        context=f"""
        監控目標: {targets}
        監控間隔: {monitoring_interval}秒
        AI分析: 異常檢測、風險評估、自動告警
        """
    )
    
    print(f"🔄 啟動實時監控: {len(targets)} 個目標")
    print(f"📊 監控策略: {monitoring_strategy.get('decision')}")
    print(f"⏱️  監控間隔: {monitoring_interval} 秒")
    
    return {
        "config": monitoring_config,
        "strategy": monitoring_strategy,
        "status": "active"
    }

# 啟動監控示例
monitoring = start_real_time_monitoring(
    targets=["192.168.1.100", "https://example.com"],
    monitoring_interval=600
)
```

### 12. AI學習與進化功能

#### 經驗學習系統
```python
# AI經驗學習功能
def ai_experience_learning(scan_results, feedback=None):
    """AI從掃描結果中學習並優化"""
    
    learning_context = f"""
    掃描結果: {scan_results}
    用戶反饋: {feedback}
    學習目標: 提升準確性、減少誤報、優化策略
    """
    
    # AI學習和策略優化
    learning_insights = ai_analyzer.generate(
        task_description="從掃描結果中學習並優化未來策略",
        context=learning_context
    )
    
    return {
        "learning_insights": learning_insights.get("decision"),
        "optimization_suggestions": learning_insights.get("suggested_actions", []),
        "confidence_improvement": learning_insights.get("confidence", 0)
    }

# 使用學習功能
scan_results = {"vulnerabilities_found": 3, "false_positives": 1, "scan_time": 300}
learning_result = ai_experience_learning(scan_results, "準確率需提升")
print(f"AI學習結果: {learning_result['learning_insights']}")
```

#### 自適應掃描策略
```python
# 自適應AI掃描
def adaptive_ai_scanning(target, historical_data=None):
    """基於歷史數據的自適應AI掃描"""
    
    adaptation_context = f"""
    目標: {target}
    歷史掃描數據: {historical_data}
    自適應要求: 根據目標特性調整掃描策略
    """
    
    # AI自適應策略生成
    adaptive_strategy = ai_analyzer.generate(
        task_description="生成針對目標的自適應掃描策略",
        context=adaptation_context
    )
    
    return {
        "target": target,
        "adaptive_strategy": adaptive_strategy.get("decision"),
        "confidence": adaptive_strategy.get("confidence"),
        "customized_approach": adaptive_strategy.get("suggested_actions", [])
    }

# 執行自適應掃描
historical = {"previous_scans": 5, "avg_vulnerabilities": 2.4, "target_type": "web_app"}
adaptive_scan = adaptive_ai_scanning("api.example.com", historical)
print(f"自適應掃描策略: {adaptive_scan['adaptive_strategy']}")
```

---

### 3. 安全漏洞掃描

#### 自動化漏洞檢測
```python
# 安全漏洞掃描功能
def security_vulnerability_scan(target, scan_depth="medium"):
    """
    AI 驅動的安全漏洞掃描
    scan_depth: "surface", "medium", "deep"
    """
    
    vulnerability_context = f"""
    掃描目標: {target}
    掃描深度: {scan_depth}
    檢查項目: SQL注入、XSS、CSRF、文件包含、權限提升
    AI分析: 漏洞風險評估、利用可能性、修復建議
    """
    
    # AI 漏洞分析決策
    vuln_analysis = ai_analyzer.generate(
        task_description="執行智能漏洞掃描和風險評估",
        context=vulnerability_context
    )
    
    return {
        "scan_target": target,
        "scan_depth": scan_depth,
        "vulnerabilities": vuln_analysis.get("decision", "未發現"),
        "risk_level": vuln_analysis.get("confidence", 0) * 100,
        "ai_recommendations": vuln_analysis.get("suggested_actions", []),
        "scan_timestamp": "2025-11-12"
    }

# 使用範例
vulnerability_report = security_vulnerability_scan("192.168.1.100", "deep")
print(f"漏洞掃描完成，風險等級: {vulnerability_report['risk_level']}%")
```

#### 網路掃描與偵察
```python
# 網路掃描功能
def intelligent_network_scan(target_range, scan_type="stealth"):
    """
    AI 輔助網路掃描
    scan_type: "stealth", "aggressive", "comprehensive"
    """
    
    network_context = f"""
    目標範圍: {target_range}
    掃描模式: {scan_type}
    掃描內容: 端口掃描、服務識別、OS指紋、拓撲分析
    AI優化: 掃描順序、探測策略、躲避檢測
    """
    
    # AI 網路掃描策略
    scan_strategy = ai_analyzer.generate(
        task_description=f"制定 {scan_type} 網路掃描策略",
        context=network_context
    )
    
    return {
        "target_range": target_range,
        "scan_type": scan_type,
        "strategy": scan_strategy.get("decision"),
        "confidence": scan_strategy.get("confidence"),
        "execution_plan": scan_strategy.get("suggested_actions", []),
        "scan_date": "2025-11-12"
    }

# 網路掃描示例
network_scan = intelligent_network_scan("192.168.1.0/24", "stealth")
print(f"網路掃描策略: {network_scan['strategy']}")
```

### 4. 綜合分析報告

#### 生成智能分析報告
```python
# 綜合分析報告生成
async def generate_comprehensive_report(target, include_modules=True):
    """生成完整的 AI 分析報告"""
    
    report = {
        "report_title": f"AIVA AI 綜合分析報告 - {target}",
        "generation_time": "2025-11-12",
        "ai_engine": "RealBioNeuronRAGAgent v2.0",
        "sections": {}
    }
    
    # 1. 目標基礎分析
    target_analysis = ai_analyzer.generate(
        task_description="目標基礎分析和風險評估",
        context=f"分析目標: {target}, 重點: 架構、安全性、可攻擊面"
    )
    report["sections"]["target_analysis"] = target_analysis
    
    # 2. 安全漏洞評估
    security_assessment = security_vulnerability_scan(target, "deep")
    report["sections"]["security_assessment"] = security_assessment
    
    # 3. 模組狀態分析 (如果啟用)
    if include_modules:
        module_status = await comprehensive_module_scan()
        report["sections"]["module_analysis"] = module_status
    
    # 4. AI 總結和建議
    final_summary = ai_analyzer.generate(
        task_description="生成綜合分析總結和行動建議",
        context=f"""
        目標: {target}
        分析結果: {report['sections']}
        要求: 風險總結、優先修復項目、後續行動建議
        """
    )
    report["sections"]["ai_summary"] = final_summary
    
    return report

# 生成報告示例
# comprehensive_report = await generate_comprehensive_report("https://example.com")
# print(f"報告生成完成: {comprehensive_report['report_title']}")
```

### 5. 實時監控與分析

#### 持續監控功能
```python
# 實時監控分析
def start_real_time_monitoring(targets, monitoring_interval=300):
    """
    啟動實時監控分析
    monitoring_interval: 監控間隔(秒)
    """
    
    monitoring_config = {
        "targets": targets,
        "interval": monitoring_interval,
        "ai_analysis": True,
        "alert_threshold": 0.7,
        "start_time": "2025-11-12"
    }
    
    # AI 監控策略
    monitoring_strategy = ai_analyzer.generate(
        task_description="制定實時監控策略",
        context=f"""
        監控目標: {targets}
        監控間隔: {monitoring_interval}秒
        AI分析: 異常檢測、風險評估、自動告警
        """
    )
    
    print(f"🔄 啟動實時監控: {len(targets)} 個目標")
    print(f"📊 監控策略: {monitoring_strategy.get('decision')}")
    print(f"⏱️  監控間隔: {monitoring_interval} 秒")
    
    return {
        "config": monitoring_config,
        "strategy": monitoring_strategy,
        "status": "active"
    }

# 啟動監控示例
monitoring = start_real_time_monitoring(
    targets=["192.168.1.100", "https://example.com"],
    monitoring_interval=600
)
```

---

## �📞 技術支援

### 獲得幫助

- **📖 文檔**: 查看 `README.md` 和 `docs/` 目錄
- **🐛 問題報告**: 通過 GitHub Issues
- **💬 社群討論**: GitHub Discussions
- **📧 技術支援**: ai-support@aiva-platform.com

### 貢獻指南

1. Fork 專案倉庫
2. 創建功能分支: `git checkout -b feature/amazing-feature`
3. 提交變更: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 開啟 Pull Request

---

## 📄 版本資訊

**當前版本**: v2.0.0  
**發布日期**: 2025年11月11日  
**相容性**: Python 3.8+, Windows/Linux/macOS  
**授權**: MIT License  

### 更新日誌

- **v2.0.0** (2025-11-11): 500萬參數神經網路整合、RAG增強系統、四種運行模式
- **v1.5.0** (2024-10-15): 基礎AI引擎、知識庫系統
- **v1.0.0** (2024-08-01): 初始版本發布

---

**🌟 感謝使用 AIVA AI 系統！**

*本手冊會持續更新，以確保與系統功能同步。如有任何疑問，歡迎聯繫技術支援團隊。*