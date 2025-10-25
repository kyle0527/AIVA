# 依賴評估報告 (Dependency Assessment Report)

## 執行日期: 2025-10-25

## 一、現有 requirements.txt 分析

### 已安裝的核心依賴 (✅):
- FastAPI, Uvicorn (Web 框架) ✅
- Pydantic v2 (資料驗證) ✅
- SQLAlchemy, Alembic (資料庫 ORM) ✅
- aio-pika (RabbitMQ 客戶端) ✅
- httpx (HTTP 客戶端) ✅
- beautifulsoup4, lxml (HTML 解析) ✅
- Redis, Neo4j (資料儲存) ✅
- Cryptography (加密) ✅

### 缺失的關鍵依賴:

#### 1. JWT 處理 (下載檔案 JWTConfusionWorker.py 需要)
```python
PyJWT>=2.8.0  # JWT 編碼/解碼
```

#### 2. OAST/SSRF 檢測 (SmartSSRFDetector, SSRFWorker 需要)
```python
requests>=2.31.0  # HTTP 請求 (已有 httpx,但部分程式碼使用 requests)
pika>=1.3.0  # RabbitMQ 同步客戶端 (Worker 使用)
```

#### 3. 網路掃描 (NetworkScanner 需要)
```python
# 已涵蓋在 Python 標準庫 (socket, threading)
```

#### 4. 掃描模組增強 (掃描模組建議.txt)
```python
# Deep API 掃描
openapi-spec-validator>=0.6.0  # OpenAPI 驗證
prance>=23.6.0  # OpenAPI 解析
python-graphql-client>=0.4.3  # GraphQL 客戶端

# EASM 增強
aiodns>=3.0.0  # 異步 DNS 解析
boto3>=1.28.0  # AWS SDK (bucket 掃描)
azure-storage-blob>=12.16.0  # Azure Storage
google-cloud-storage>=2.10.0  # GCP Storage

# Supply Chain 掃描
python-hcl2>=4.3.0  # Terraform 解析
pyyaml>=6.0.0  # YAML 解析 (K8s)

# Mobile App 掃描
androguard>=3.4.0  # APK 分析

# AI-driven 掃描
scikit-learn>=1.3.0  # 機器學習
nltk>=3.8.0  # 自然語言處理
```

#### 5. AI/RAG 相關 (依賴確認.txt 提到,如果啟用 AI Commander)
```python
# AI/RAG 核心
sentence-transformers>=2.2.2  # 語義搜索
chromadb>=0.4.0  # 向量資料庫
faiss-cpu>=1.7.4  # 向量相似度搜索
torch>=2.0.0  # 深度學習框架
transformers>=4.30.0  # Hugging Face transformers
langchain>=0.0.300  # LLM 應用框架

# 可選: OpenAI API
openai>=1.0.0  # OpenAI API 客戶端
```

---

## 二、建議的依賴新增策略

### 階段 1: 立即新增 (整合下載檔案必需)
**優先級: P0**

```python
PyJWT>=2.8.0
requests>=2.31.0
pika>=1.3.0
```

**理由**: 這些是整合 JWTConfusionWorker, SSRFWorker, SQLiOOBDetectionEngine 的必要依賴。

### 階段 2: 掃描增強 (中期規劃)
**優先級: P1**

```python
# API 掃描
openapi-spec-validator>=0.6.0
prance>=23.6.0
python-graphql-client>=0.4.3

# EASM 基礎
aiodns>=3.0.0

# Supply Chain
python-hcl2>=4.3.0
pyyaml>=6.0.0

# AI 輔助掃描
scikit-learn>=1.3.0
nltk>=3.8.0
```

**理由**: 提升掃描模組能力,根據掃描模組建議.txt 的建議。

### 階段 3: 雲端整合 (可選)
**優先級: P2**

```python
boto3>=1.28.0
azure-storage-blob>=12.16.0
google-cloud-storage>=2.10.0
```

**理由**: 需要 AWS/Azure/GCP 憑證才能使用,可作為可選依賴。

### 階段 4: AI/RAG (可選)
**優先級: P3**

```python
sentence-transformers>=2.2.2
chromadb>=0.4.0
faiss-cpu>=1.7.4
torch>=2.0.0
transformers>=4.30.0
langchain>=0.0.300
openai>=1.0.0
```

**理由**: 
- 體積龐大 (torch 約 2GB)
- 需要額外的系統配置 (CUDA 等)
- 僅在啟用 AI Commander RAG 功能時需要

---

## 三、版本相容性檢查

### Python 版本要求
- **當前**: Python 3.8+ (根據現有 requirements.txt)
- **建議**: Python 3.11+ (依賴確認.txt 建議)
- **相容性**: 所有建議的依賴都支援 Python 3.11+

### 依賴衝突檢查

#### 潛在衝突 1: requests vs httpx
- **現有**: httpx>=0.27.0
- **新增**: requests>=2.31.0
- **衝突**: ❌ 無衝突 (可並存,但建議統一為 httpx)
- **建議**: 重構下載檔案中的 requests 使用為 httpx

#### 潛在衝突 2: aio-pika vs pika
- **現有**: aio-pika>=9.4.0 (異步)
- **新增**: pika>=1.3.0 (同步)
- **衝突**: ❌ 無衝突 (可並存)
- **說明**: Worker 使用同步 pika,其他服務使用異步 aio-pika

#### 潛在衝突 3: pyyaml 版本
- **新增**: pyyaml>=6.0.0
- **現有依賴**: 未明確列出
- **衝突**: ❌ 無衝突

---

## 四、更新後的 requirements.txt

### 建議的完整 requirements.txt

```python
# ==================== Core Framework ====================
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0

# ==================== Message Queue ====================
aio-pika>=9.4.0  # Async RabbitMQ client
pika>=1.3.0  # Sync RabbitMQ client (for Workers)

# ==================== HTTP Clients ====================
httpx>=0.27.0  # Async HTTP client (preferred)
requests>=2.31.0  # Sync HTTP client (legacy Workers)

# ==================== Web Scraping ====================
beautifulsoup4>=4.12.2
lxml>=5.0.0
playwright>=1.40.0  # Headless browser (optional)

# ==================== Logging ====================
structlog>=24.1.0

# ==================== Data Storage ====================
redis>=5.0.0
neo4j>=5.23.0

# ==================== Database (SQL) ====================
sqlalchemy>=2.0.31
asyncpg>=0.29.0
psycopg2-binary>=2.9.0
alembic>=1.13.2

# ==================== Configuration ====================
python-dotenv>=1.0.1
orjson>=3.10.0

# ==================== Resilience ====================
tenacity>=8.3.0
aiofiles>=23.2.1

# ==================== System Monitoring ====================
psutil>=5.9.6

# ==================== gRPC (Cross-Language) ====================
grpcio>=1.60.0
grpcio-tools>=1.60.0
protobuf>=4.25.0

# ==================== Security & Authentication ====================
PyJWT>=2.8.0  # JWT handling
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
cryptography>=42.0.0

# ==================== API Scanning (P1) ====================
openapi-spec-validator>=0.6.0  # OpenAPI validation
prance>=23.6.0  # OpenAPI parser
python-graphql-client>=0.4.3  # GraphQL client

# ==================== EASM (P1) ====================
aiodns>=3.0.0  # Async DNS resolution

# ==================== Supply Chain Scanning (P1) ====================
python-hcl2>=4.3.0  # Terraform parser
pyyaml>=6.0.0  # YAML parser (Kubernetes)

# ==================== AI-Assisted Scanning (P1) ====================
scikit-learn>=1.3.0  # Machine learning
nltk>=3.8.0  # Natural language processing

# ==================== Cloud Integration (P2 - Optional) ====================
# boto3>=1.28.0  # AWS SDK
# azure-storage-blob>=12.16.0  # Azure Storage
# google-cloud-storage>=2.10.0  # GCP Storage

# ==================== Mobile App Scanning (P2 - Optional) ====================
# androguard>=3.4.0  # APK analysis

# ==================== AI/RAG (P3 - Optional) ====================
# sentence-transformers>=2.2.2  # Semantic search
# chromadb>=0.4.0  # Vector database
# faiss-cpu>=1.7.4  # Vector similarity search
# torch>=2.0.0  # Deep learning framework
# transformers>=4.30.0  # Hugging Face transformers
# langchain>=0.0.300  # LLM application framework
# openai>=1.0.0  # OpenAI API client

# ==================== Development Tools ====================
pytest>=8.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.23.0
black>=24.0.0
ruff>=0.3.0
mypy>=1.8.0
pre-commit>=3.6.0

# ==================== Type Stubs ====================
types-requests>=2.31.0
types-pyyaml>=6.0.0
```

---

## 五、安裝命令

### 階段 1: 立即安裝 (P0)
```bash
pip install PyJWT>=2.8.0 requests>=2.31.0 pika>=1.3.0
```

### 階段 2: 掃描增強 (P1)
```bash
pip install openapi-spec-validator>=0.6.0 prance>=23.6.0 python-graphql-client>=0.4.3 aiodns>=3.0.0 python-hcl2>=4.3.0 pyyaml>=6.0.0 scikit-learn>=1.3.0 nltk>=3.8.0
```

### 階段 3: 完整安裝 (包含所有)
```bash
pip install -r requirements.txt
```

---

## 六、測試建議

### 1. 依賴安裝測試
```bash
# 測試 P0 依賴
python -c "import jwt; import requests; import pika; print('✅ P0 依賴安裝成功')"

# 測試 P1 依賴
python -c "import openapi_spec_validator; import prance; import aiodns; import hcl2; import yaml; import sklearn; import nltk; print('✅ P1 依賴安裝成功')"
```

### 2. 功能測試
```bash
# 測試 JWT Worker
python -c "from services.features.jwt_confusion.worker import JwtConfusionDetector; print('✅ JWT Worker 可用')"

# 測試 SSRF Worker
python -c "from services.features.function_ssrf.worker import SSRFWorker; print('✅ SSRF Worker 可用')"
```

---

## 七、後續建議

### 1. 使用 extras_require (推薦)
在 pyproject.toml 中定義可選依賴:

```toml
[project.optional-dependencies]
ai = ["sentence-transformers>=2.2.2", "chromadb>=0.4.0", "faiss-cpu>=1.7.4", "torch>=2.0.0", "transformers>=4.30.0", "langchain>=0.0.300"]
cloud = ["boto3>=1.28.0", "azure-storage-blob>=12.16.0", "google-cloud-storage>=2.10.0"]
mobile = ["androguard>=3.4.0"]
scan-enhanced = ["openapi-spec-validator>=0.6.0", "prance>=23.6.0", "python-graphql-client>=0.4.3", "aiodns>=3.0.0", "python-hcl2>=4.3.0", "scikit-learn>=1.3.0", "nltk>=3.8.0"]
```

安裝方式:
```bash
pip install -e ".[scan-enhanced]"  # 安裝掃描增強
pip install -e ".[ai]"  # 安裝 AI/RAG
```

### 2. 統一 HTTP 客戶端
- **當前**: requests (同步) + httpx (異步)
- **建議**: 重構所有同步 requests 調用為 httpx (統一介面)
- **優點**: 減少依賴,代碼更一致

### 3. 依賴鎖定
使用 pip-tools 或 poetry 鎖定精確版本:
```bash
pip install pip-tools
pip-compile requirements.in -o requirements.txt
```

---

## 八、總結

### 立即行動
✅ 新增 PyJWT, requests, pika (P0)
✅ 更新 requirements.txt

### 短期規劃
⬜ 安裝掃描增強依賴 (P1)
⬜ 測試新整合的 Worker

### 中期規劃
⬜ 評估 AI/RAG 功能需求
⬜ 重構 requests → httpx
⬜ 建立 extras_require 結構

### 風險評估
- **低風險**: P0/P1 依賴 (成熟穩定)
- **中風險**: AI/RAG 依賴 (體積大,需額外配置)
- **緩解**: 使用可選依賴,按需安裝

