# AIVA Core 依賴完整分析報告

## 📑 目錄

- [⚠️ 當前環境狀態](#-當前環境狀態)
- [📊 依賴總覽](#-依賴總覽)
- [🔴 重度依賴（>500MB）](#-重度依賴500mb)
  - [1. torch (2+ GB)](#1-torch-2-gb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [有無時的差異](#有無時的差異)
    - [替代方案](#替代方案)
    - [推薦配置](#推薦配置)
  - [2. transformers (1+ GB)](#2-transformers-1-gb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [有無時的差異](#有無時的差異)
    - [替代方案](#替代方案)
  - [3. sentence-transformers (500 MB)](#3-sentence-transformers-500-mb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [有無時的差異](#有無時的差異)
    - [替代方案](#替代方案)
  - [4. spacy (500 MB)](#4-spacy-500-mb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [有無時的差異](#有無時的差異)
    - [替代方案](#替代方案)
- [🟡 中度依賴（50-500MB）](#-中度依賴50-500mb)
  - [5. scikit-learn (200 MB)](#5-scikit-learn-200-mb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [有無時的差異](#有無時的差異)
    - [替代方案](#替代方案)
  - [6. pandas (100 MB)](#6-pandas-100-mb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [替代方案](#替代方案)
  - [7. numpy (50 MB)](#7-numpy-50-mb)
    - [必須性](#必須性)
    - [說明](#說明)
  - [8. nltk (50 MB)](#8-nltk-50-mb)
    - [必須性](#必須性)
    - [替代方案](#替代方案)
- [🟢 輕量依賴（<50MB）](#-輕量依賴50mb)
  - [9. fastapi (10 MB)](#9-fastapi-10-mb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [有無時的差異](#有無時的差異)
    - [替代方案](#替代方案)
  - [10. pydantic (10 MB)](#10-pydantic-10-mb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [替代方案](#替代方案)
  - [11. loguru (5 MB)](#11-loguru-5-mb)
    - [必須性](#必須性)
    - [實際用途](#實際用途)
    - [替代方案](#替代方案)
  - [12-19. 其他輕量依賴](#12-19-其他輕量依賴)
    - [neo4j, psycopg2, redis (Database)](#neo4j-psycopg2-redis-database)
    - [openai (AI/ML)](#openai-aiml)
    - [requests, aiohttp (Web)](#requests-aiohttp-web)
    - [cryptography (Security)](#cryptography-security)
- [📋 依賴分層配置方案](#-依賴分層配置方案)
  - [Tier 1: 核心依賴（CLI驗證）](#tier-1-核心依賴cli驗證)
  - [Tier 2: Web服務](#tier-2-web服務)
  - [Tier 3: AI能力](#tier-3-ai能力)
  - [Tier 4: 完整功能](#tier-4-完整功能)
- [🎯 使用建議](#-使用建議)
  - [場景 1: CLI驗證/參數檢查](#場景-1-cli驗證參數檢查)
  - [場景 2: Web API 服務（無AI）](#場景-2-web-api-服務無ai)
  - [場景 3: AI輔助決策](#場景-3-ai輔助決策)
  - [場景 4: 完整功能](#場景-4-完整功能)
- [📊 性能對比](#-性能對比)
- [🚀 遷移路徑](#-遷移路徑)
  - [從完整依賴遷移到分層依賴](#從完整依賴遷移到分層依賴)
- [安裝](#安裝)
  - [最小安裝（CLI驗證）](#最小安裝cli驗證)
  - [AI能力安裝](#ai能力安裝)
  - [完整安裝](#完整安裝)
- [💰 成本效益分析](#-成本效益分析)
  - [開發環境（建議完整安裝）](#開發環境建議完整安裝)
  - [CI/CD 環境（建議最小安裝）](#cicd-環境建議最小安裝)
  - [生產環境（按需安裝）](#生產環境按需安裝)
- [✅ 總結與建議](#-總結與建議)
  - [立即行動](#立即行動)
  - [短期優化（1週內）](#短期優化1週內)
  - [中期優化（1月內）](#中期優化1月內)
  - [長期優化（3月內）](#長期優化3月內)

---


**生成時間**: 2026-01-09  
**分析範圍**: services/core/aiva_core  
**總依賴數**: 19個主要依賴包

---

## ⚠️ 當前環境狀態

**✅ 全局環境已安裝完整依賴集（2026-01-09 驗證）**

所有 19 個依賴包已安裝在全局 Python 環境：
- Tier 1 (minimal): pydantic 2.12.5, loguru 0.7.3, numpy 2.3.4 ✅
- Tier 2 (web): fastapi 0.122.0, uvicorn 0.38.0, requests 2.32.3 ✅
- Tier 3 (ai): torch 2.9.1, sentence-transformers 5.1.1 ✅
- Tier 4 (full): transformers 4.57.1, spacy 3.8.11, scikit-learn 1.7.2, pandas 2.3.3, aiohttp 3.13.0, openai 2.3.0, neo4j 6.0.2, redis 6.4.0, cryptography 46.0.2, nltk 3.9.2 ✅

**無需額外安裝**，可直接使用所有功能。以下分析供優化參考。

---

## 📊 依賴總覽

| 依賴包 | 類別 | 大小 | 運行時 | CLI驗證 | 使用場景 |
|--------|------|------|--------|---------|----------|
| torch | AI/ML | 2+ GB | 必須 | 非必須 | AI能力 |
| transformers | AI/ML | 1+ GB | 必須 | 非必須 | NLP能力 |
| sentence-transformers | AI/ML | 500 MB | 必須 | 非必須 | RAG能力 |
| spacy | NLP | 500 MB | 可選 | 非必須 | 高級NLP |
| scikit-learn | ML | 200 MB | 可選 | 非必須 | 傳統ML |
| pandas | Data | 100 MB | 可選 | 非必須 | 數據分析 |
| numpy | Data | 50 MB | 必須 | 必須 | torch依賴 |
| nltk | NLP | 50 MB | 可選 | 非必須 | 基礎NLP |
| neo4j | Database | 50 MB | 可選 | 非必須 | 圖數據庫 |
| cryptography | Security | 20 MB | 可選 | 非必須 | 加密 |
| fastapi | Web | 10 MB | 必須 | 非必須 | Web服務 |
| uvicorn | Web | 10 MB | 必須 | 非必須 | ASGI服務器 |
| pydantic | Utils | 10 MB | 必須 | 必須 | 數據驗證 |
| requests | Web | 10 MB | 常用 | 非必須 | HTTP客戶端 |
| aiohttp | Web | 10 MB | 可選 | 非必須 | 異步HTTP |
| psycopg2 | Database | 10 MB | 可選 | 非必須 | PostgreSQL |
| openai | AI/ML | 10 MB | 可選 | 非必須 | OpenAI API |
| loguru | Utils | 5 MB | 必須 | 可選 | 日誌 |
| redis | Database | 5 MB | 可選 | 非必須 | 緩存/MQ |

**總計**: ~4.5 GB（完整安裝）

---

## 🔴 重度依賴（>500MB）

### 1. torch (2+ GB)

#### 必須性
- **運行時**: ✅ 必須（AI能力核心）
- **CLI驗證**: ❌ 非必須
- **使用範圍**: 12個文件

#### 實際用途
```python
# 使用場景
services/core/aiva_core/cognitive_core/learning_system/learning/rl_models.py
  └─ DQNNetwork: 強化學習決策網絡（11,876參數）
  └─ 攻擊策略智能選擇

services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py
  └─ 增強決策代理
  └─ AI驅動的攻擊鏈規劃

services/core/aiva_core/cognitive_core/learning_system/learning/continuous_learning.py
  └─ 持續學習引擎
  └─ 從攻擊結果學習優化策略
```

#### 有無時的差異

**有 torch 時**:
```python
# 可以執行 AI 能力
from services.core.aiva_core.cognitive_core.learning_system.learning.rl_models import DQNNetwork
dqn = DQNNetwork(10, 4)
action = dqn(state).argmax()  # AI 決策

# 實際輸出
✅ DQN網絡創建成功: 11,876 個參數
✅ 決策測試: 輸入狀態 → 最優動作=3 (Q=0.215)
```

**無 torch 時**:
```python
# 無法導入，報錯
ModuleNotFoundError: No module named 'torch'

# 影響範圍
❌ 所有 AI 內部能力失效（3個）
❌ 強化學習決策失效
❌ 神經網絡推理失效
✅ 非 AI 能力仍可正常使用（158個）
```

#### 替代方案

**方案 A: ONNX Runtime（推薦）**
```bash
# 優勢
- 體積: ~50 MB（比 torch 小 40倍）
- 速度: 推理速度更快
- 跨平台: 更好的部署支持

# 限制
- 只能推理，不能訓練
- 需要預先將模型轉換為 ONNX 格式

# 實施步驟
pip install onnxruntime

# 1. 訓練時（開發環境）使用 torch
torch_model = DQNNetwork(10, 4)
torch.onnx.export(torch_model, ...)

# 2. 部署時（生產環境）使用 onnxruntime
import onnxruntime as ort
session = ort.InferenceSession("dqn.onnx")
output = session.run(None, {"input": state})
```

**方案 B: TensorFlow Lite**
```bash
# 優勢
- 體積: ~10 MB
- 專為邊緣設備優化
- Google 官方支持

# 限制
- 功能有限
- 需要模型轉換

pip install tensorflow-lite
```

**方案 C: 純 NumPy 實現**
```python
# 優勢
- 體積: 50 MB（numpy）
- 無額外依賴
- 推理速度可接受

# 限制
- 手動實現神經網絡
- 不支持複雜模型
- 維護成本高

# 示例
import numpy as np

class SimpleDQN:
    def __init__(self):
        self.w1 = np.random.randn(10, 64)
        self.w2 = np.random.randn(64, 4)
    
    def predict(self, state):
        h = np.maximum(0, state @ self.w1)  # ReLU
        return h @ self.w2
```

**方案 D: 雲端 API**
```python
# 優勢
- 本地無需安裝 torch
- 可使用更強大的模型
- 彈性擴展

# 限制
- 需要網絡連接
- 增加延遲
- API 成本

# 示例
import requests

def predict_action(state):
    response = requests.post(
        "https://ai-api.aiva.com/predict",
        json={"state": state.tolist()}
    )
    return response.json()["action"]
```

#### 推薦配置

**開發環境**:
```bash
# 安裝完整 torch（支持訓練和推理）
pip install torch>=2.0.0
```

**生產環境（輕量部署）**:
```bash
# 使用 ONNX Runtime（只推理）
pip install onnxruntime
# 節省 ~2 GB 空間
```

**CLI驗證環境**:
```bash
# 不安裝 torch
# 使用 Mock 或跳過 AI 能力測試
```

---

### 2. transformers (1+ GB)

#### 必須性
- **運行時**: ✅ 必須（NLP能力）
- **CLI驗證**: ❌ 非必須
- **使用範圍**: 未統計（間接依賴）

#### 實際用途
```python
# 使用場景
- 載入預訓練語言模型（BERT, GPT等）
- Token化和文本編碼
- 文本生成和理解
```

#### 有無時的差異

**有 transformers 時**:
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "AIVA AI security assistant"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # 語義向量

# 功能
✅ 語義理解
✅ 文本分類
✅ 命名實體識別
```

**無 transformers 時**:
```python
# 無法使用預訓練模型
❌ 高級 NLP 功能失效
❌ 語義理解能力下降
✅ 基礎文本處理仍可用（使用 nltk）
```

#### 替代方案

**方案 A: sentence-transformers（已安裝）**
```python
# 優勢
- 專注於句子嵌入
- 體積較小（500 MB）
- 接口簡單

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["text1", "text2"])
```

**方案 B: spaCy（已安裝）**
```python
# 優勢
- 工業級 NLP
- 速度快
- 不依賴 transformers

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("AIVA security scanner")
# 詞性標註、實體識別等
```

**方案 C: OpenAI API**
```python
# 優勢
- 無需本地模型
- 使用最先進模型
- 零維護

import openai
response = openai.Embedding.create(
    input="AIVA AI assistant",
    model="text-embedding-ada-002"
)
embeddings = response['data'][0]['embedding']
```

**方案 D: 輕量級模型**
```python
# 使用蒸餾模型（DistilBERT）
# 速度提升 60%，體積減少 40%

from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
```

---

### 3. sentence-transformers (500 MB)

#### 必須性
- **運行時**: ✅ 必須（RAG能力）
- **CLI驗證**: ❌ 非必須
- **使用範圍**: 3個文件

#### 實際用途
```python
# 使用場景
services/core/aiva_core/cognitive_core/rag/unified_vector_store.py
  └─ 文檔向量化
  └─ 語義搜索

services/core/aiva_core/cognitive_core/rag/vector_store.py
  └─ 向量存儲
  └─ 相似度檢索

services/core/aiva_core/cognitive_core/neural/real_neural_core.py
  └─ 神經網絡核心
  └─ 語義理解
```

#### 有無時的差異

**有 sentence-transformers 時**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["SQL injection detected", "XSS vulnerability found"]
embeddings = model.encode(docs)

# 查詢
query = "web security issue"
query_embedding = model.encode([query])
similarity = cosine_similarity(query_embedding, embeddings)

# 功能
✅ 語義搜索（理解意圖）
✅ 文檔相似度
✅ RAG 知識檢索
```

**無 sentence-transformers 時**:
```python
# 降級為關鍵字搜索
def keyword_search(query, docs):
    keywords = query.lower().split()
    results = [doc for doc in docs if any(kw in doc.lower() for kw in keywords)]
    return results

# 差異
❌ 無法理解語義（"web security" 無法匹配 "XSS vulnerability"）
❌ RAG 能力大幅下降
✅ 基礎關鍵字搜索仍可用
```

#### 替代方案

**方案 A: OpenAI Embeddings API**
```python
# 優勢
- 無需本地模型
- 質量更高
- 自動更新

import openai

def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return [data['embedding'] for data in response['data']]

# 成本
- ~$0.0001 / 1K tokens
- 100萬次查詢 ≈ $100
```

**方案 B: 自建輕量模型**
```python
# 使用更小的模型
from sentence_transformers import SentenceTransformer

# 原: all-MiniLM-L6-v2 (80 MB)
# 改: paraphrase-MiniLM-L3-v2 (40 MB)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# 質量下降 5-10%，速度提升 2倍
```

**方案 C: TF-IDF + BM25**
```python
# 優勢
- 體積極小
- 速度極快
- 無 GPU 需求

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(docs)

# 限制
❌ 無法理解語義
✅ 適合關鍵字匹配場景
```

---

### 4. spacy (500 MB)

#### 必須性
- **運行時**: ⚠️ 可選（高級NLP）
- **CLI驗證**: ❌ 非必須
- **使用範圍**: 未統計

#### 實際用途
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Detected SQL injection in user input")

# 功能
for token in doc:
    print(token.text, token.pos_, token.dep_)

# 實體識別
for ent in doc.ents:
    print(ent.text, ent.label_)
```

#### 有無時的差異

**有 spacy 時**:
```python
✅ 詞性標註
✅ 依存句法分析
✅ 命名實體識別
✅ 詞向量
✅ 文本相似度
```

**無 spacy 時**:
```python
# 使用 nltk 替代基礎功能
import nltk

# 分詞
tokens = nltk.word_tokenize(text)

# 詞性標註
pos_tags = nltk.pos_tag(tokens)

# 差異
❌ 精度下降
❌ 速度較慢
✅ 基礎功能仍可用
```

#### 替代方案

**方案 A: nltk（已安裝，50 MB）**
```python
import nltk

# 基礎 NLP
tokens = nltk.word_tokenize(text)
pos = nltk.pos_tag(tokens)
ner = nltk.ne_chunk(pos)

# 適合輕量場景
```

**方案 B: stanza（Stanford NLP）**
```python
import stanza

nlp = stanza.Pipeline('en')
doc = nlp(text)

# 精度接近 spacy
# 速度較慢
```

**方案 C: 不安裝（跳過高級 NLP）**
```python
# AIVA 的核心功能不依賴高級 NLP
# 可以完全不安裝 spacy
```

---

## 🟡 中度依賴（50-500MB）

### 5. scikit-learn (200 MB)

#### 必須性
- **運行時**: ⚠️ 可選（傳統ML）
- **CLI驗證**: ❌ 非必須
- **使用範圍**: 1個文件

#### 實際用途
```python
# 使用場景
services/core/aiva_core/cognitive_core/learning_system/learning/model_trainer.py
  └─ 傳統機器學習算法
  └─ 分類器訓練（SVM, Random Forest等）
```

#### 有無時的差異

**有 scikit-learn 時**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

✅ 豐富的算法選擇
✅ 完善的工具鏈
✅ 成熟穩定
```

**無 scikit-learn 時**:
```python
# 影響
❌ 傳統 ML 算法不可用
✅ 深度學習能力不受影響（torch）
✅ 非 ML 能力正常

# 如果只使用深度學習，可不安裝
```

#### 替代方案

**方案 A: 只使用深度學習**
```python
# 用 torch 替代傳統 ML
# 現代趨勢：深度學習替代傳統算法
```

**方案 B: XGBoost/LightGBM**
```python
# 專注於樹模型
import xgboost as xgb
import lightgbm as lgb

# 體積更小，速度更快
```

---

### 6. pandas (100 MB)

#### 必須性
- **運行時**: ⚠️ 可選（數據分析）
- **CLI驗證**: ❌ 非必須
- **使用範圍**: 未統計

#### 實際用途
```python
import pandas as pd

# 數據處理
df = pd.read_csv("attack_logs.csv")
analysis = df.groupby('attack_type').size()
```

#### 替代方案

**方案 A: 標準庫 csv**
```python
import csv

# 基礎 CSV 處理
with open('file.csv') as f:
    reader = csv.DictReader(f)
    data = list(reader)
```

**方案 B: Polars（更快）**
```python
import polars as pl

df = pl.read_csv("file.csv")
# 速度提升 5-10倍
```

---

### 7. numpy (50 MB)

#### 必須性
- **運行時**: ✅ 必須（torch依賴）
- **CLI驗證**: ✅ 必須
- **使用範圍**: 間接依賴

#### 說明
```python
# numpy 是 torch 的核心依賴
# 無法單獨移除
# CLI 驗證場景也建議保留（體積小，用途廣）
```

---

### 8. nltk (50 MB)

#### 必須性
- **運行時**: ⚠️ 可選（基礎NLP）
- **CLI驗證**: ❌ 非必須

#### 替代方案
- 使用 spacy（更強大）
- 使用 Python 標準庫（基礎文本處理）

---

## 🟢 輕量依賴（<50MB）

### 9. fastapi (10 MB)

#### 必須性
- **運行時**: ✅ 必須（Web服務）
- **CLI驗證**: ❌ 非必須
- **使用範圍**: 4個文件

#### 實際用途
```python
# 使用場景
services/core/aiva_core/service_backbone/api/app.py
  └─ API 服務主入口

services/core/aiva_core/cognitive_core/ai_capability_query.py
  └─ AI 能力查詢接口
```

#### 有無時的差異

**有 fastapi 時**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/scan")
async def scan(target: str):
    return {"status": "scanning", "target": target}

✅ 提供 Web API
✅ 遠程調用
✅ 多客戶端支持
```

**無 fastapi 時**:
```python
# 只能本地使用
from services.core.aiva_core.core_capabilities import scanner

result = scanner.scan(target)

❌ 無 Web 接口
❌ 無法遠程調用
✅ 本地功能正常
```

#### 替代方案

**方案 A: Flask（更輕量）**
```python
from flask import Flask

app = Flask(__name__)

@app.route('/api/scan')
def scan():
    return {"status": "ok"}

# 體積: ~5 MB
# 功能: 較簡單
```

**方案 B: 純 CLI（無 Web）**
```bash
# 直接使用命令行
python -m services.core.aiva_core.cli scan --target example.com

# 適合本地使用場景
```

---

### 10. pydantic (10 MB)

#### 必須性
- **運行時**: ✅ 必須
- **CLI驗證**: ✅ 必須
- **使用範圍**: 3個文件（核心數據結構）

#### 實際用途
```python
from pydantic import BaseModel, Field

class ScanConfig(BaseModel):
    target: str
    timeout: int = Field(default=30, ge=1)
    threads: int = Field(default=10, ge=1, le=100)

# 自動驗證
config = ScanConfig(target="example.com", timeout=60)
# config.timeout = -1  # ValidationError

✅ 數據驗證
✅ 類型檢查
✅ 自動文檔生成
```

#### 替代方案

**無合適替代**
```python
# pydantic 是現代 Python 項目的標準
# 體積小，功能強大
# 不建議移除
```

---

### 11. loguru (5 MB)

#### 必須性
- **運行時**: ✅ 建議（生產環境）
- **CLI驗證**: ⚠️ 可選
- **使用範圍**: 廣泛使用

#### 實際用途
```python
from loguru import logger

logger.info("Scan started: {target}", target="example.com")
logger.error("Scan failed: {error}", error=str(e))

✅ 結構化日誌
✅ 自動輪轉
✅ 異常追蹤
```

#### 替代方案

**方案 A: 標準庫 logging**
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Scan started")

# 功能較基礎
# 配置複雜
```

**方案 B: 不使用日誌（不推薦）**
```python
print("Scan started")

# 適合 CLI 驗證
# 不適合生產環境
```

---

### 12-19. 其他輕量依賴

#### neo4j, psycopg2, redis (Database)
- **必須性**: ⚠️ 可選（按需）
- **替代**: 使用 SQLite（標準庫）

#### openai (AI/ML)
- **必須性**: ⚠️ 可選
- **替代**: 使用本地模型

#### requests, aiohttp (Web)
- **必須性**: ⚠️ 常用
- **替代**: urllib（標準庫）

#### cryptography (Security)
- **必須性**: ⚠️ 可選
- **替代**: hashlib（標準庫，功能有限）

---

## 📋 依賴分層配置方案

### Tier 1: 核心依賴（CLI驗證）
```txt
# requirements-minimal.txt
pydantic>=2.0.0      # 數據驗證（必須）
loguru>=0.7.0        # 日誌（建議）
numpy>=1.24.0        # 數值計算（必須）

# 總計: ~65 MB
# 啟動時間: <1秒
```

### Tier 2: Web服務
```txt
# requirements-web.txt
-r requirements-minimal.txt

fastapi>=0.104.0
uvicorn[standard]>=0.24.0
requests>=2.31.0

# 總計: ~95 MB
# 啟動時間: ~2秒
```

### Tier 3: AI能力
```txt
# requirements-ai.txt
-r requirements-web.txt

torch>=2.0.0
sentence-transformers>=2.2.0

# 總計: ~2.6 GB
# 啟動時間: ~10秒
```

### Tier 4: 完整功能
```txt
# requirements-full.txt
-r requirements-ai.txt

transformers>=4.30.0
spacy>=3.6.0
scikit-learn>=1.3.0
pandas>=2.0.0

# 總計: ~4.5 GB
# 啟動時間: ~15秒
```

---

## 🎯 使用建議

### 場景 1: CLI驗證/參數檢查
```bash
pip install -r requirements-minimal.txt
```
- 體積: 65 MB
- 啟動: <1秒
- 功能: 參數驗證、基礎檢查

### 場景 2: Web API 服務（無AI）
```bash
pip install -r requirements-web.txt
```
- 體積: 95 MB
- 啟動: ~2秒
- 功能: REST API、攻擊工具調用

### 場景 3: AI輔助決策
```bash
pip install -r requirements-ai.txt
```
- 體積: 2.6 GB
- 啟動: ~10秒
- 功能: AI決策、強化學習、RAG

### 場景 4: 完整功能
```bash
pip install -r requirements-full.txt
```
- 體積: 4.5 GB
- 啟動: ~15秒
- 功能: 全部能力

---

## 📊 性能對比

| 配置 | 依賴數 | 磁盤占用 | 內存占用 | 啟動時間 | AI能力 | Web服務 |
|------|--------|----------|----------|----------|--------|---------|
| Minimal | 3 | 65 MB | 50 MB | <1s | ❌ | ❌ |
| Web | 6 | 95 MB | 100 MB | ~2s | ❌ | ✅ |
| AI | 8 | 2.6 GB | 1 GB | ~10s | ✅ | ✅ |
| Full | 19 | 4.5 GB | 2 GB | ~15s | ✅ | ✅ |

---

## 🚀 遷移路徑

### 從完整依賴遷移到分層依賴

**步驟 1: 創建分層 requirements**
```bash
cd services/core
mkdir requirements
cat > requirements/minimal.txt << EOF
pydantic>=2.0.0
loguru>=0.7.0
numpy>=1.24.0
EOF
```

**步驟 2: 修改 __init__.py 支持延遲導入**
```python
# services/core/aiva_core/__init__.py

# 原始（立即導入）
# from .cognitive_core.learning_system.learning.rl_models import DQNNetwork

# 改為（延遲導入）
def get_dqn_network():
    """延遲導入 DQN，避免 CLI 驗證時載入 torch"""
    try:
        from .cognitive_core.learning_system.learning.rl_models import DQNNetwork
        return DQNNetwork
    except ImportError:
        raise RuntimeError("DQN requires torch. Install: pip install torch>=2.0.0")
```

**步驟 3: 添加環境變數控制**
```python
import os

AIVA_MODE = os.getenv("AIVA_MODE", "full")  # minimal, web, ai, full

if AIVA_MODE == "full":
    # 載入所有依賴
    pass
elif AIVA_MODE == "ai":
    # 載入 AI 依賴
    pass
elif AIVA_MODE == "web":
    # 載入 Web 依賴
    pass
else:  # minimal
    # 只載入核心依賴
    pass
```

**步驟 4: 更新文檔**
```markdown
# README.md

## 安裝

### 最小安裝（CLI驗證）
pip install -r requirements/minimal.txt

### AI能力安裝
pip install -r requirements/ai.txt

### 完整安裝
pip install -r requirements/full.txt
```

---

## 💰 成本效益分析

### 開發環境（建議完整安裝）
- **優勢**: 功能完整，開發方便
- **成本**: 4.5 GB磁盤，2 GB內存
- **配置**: `requirements-full.txt`

### CI/CD 環境（建議最小安裝）
- **優勢**: 快速構建，節省資源
- **成本**: 65 MB磁盤，50 MB內存
- **配置**: `requirements-minimal.txt`
- **節省**: 98.5% 磁盤，97.5% 內存，93% 時間

### 生產環境（按需安裝）
- **Web API**: `requirements-web.txt`
- **AI服務**: `requirements-ai.txt`
- **完整服務**: `requirements-full.txt`

---

## ✅ 總結與建議

### 立即行動
1. ✅ **修復 orchestrator 引用**（已完成）
2. ⚠️ **創建分層 requirements 文件**
3. ⚠️ **實施延遲導入機制**
4. ⚠️ **添加環境變數控制**

### 短期優化（1週內）
- 為 CLI 驗證創建最小依賴配置
- 測試各層級配置的功能完整性
- 更新部署文檔

### 中期優化（1月內）
- 評估 ONNX Runtime 替代 torch（生產環境）
- 優化 __init__.py 導入結構
- 實施依賴注入模式

### 長期優化（3月內）
- 微服務化：AI能力獨立部署
- 容器化：不同鏡像對應不同依賴層級
- 性能監控：跟蹤依賴加載時間

---

**維護者**: AIVA Team  
**最後更新**: 2026-01-09  
**版本**: 1.0.0
