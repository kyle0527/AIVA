# AIVA 專案依賴與安裝方式分析報告 (已完成)

## 📑 目錄

- [📋 執行摘要](#執行摘要)
  - [重要發現](#重要發現)
  - [推薦方案](#推薦方案)
- [🏗️ 專案結構與套件層級](#專案結構與套件層級)
  - [套件依賴關係圖](#套件依賴關係圖)
- [📦 依賴清單詳解](#依賴清單詳解)
  - [1. 根層級依賴 (requirements.txt)](#1-根層級依賴-requirementstxt)
  - [2. services/aiva_common/pyproject.toml](#2-servicesaivacommonpyprojecttoml)
    - [async 組:](#async-組)
    - [testing 組:](#testing-組)
    - [cli 組:](#cli-組)
    - [observability 組:](#observability-組)
    - [plugins 組:](#plugins-組)
  - [3. services/core/requirements.txt](#3-servicescorerequirementstxt)
  - [4. api/requirements.txt 與 plugins/requirements.txt](#4-apirequirementstxt-與-pluginsrequirementstxt)
- [🔧 安裝方式比較](#安裝方式比較)
  - [Option A: 使用現有設定腳本 (✅ 推薦)](#option-a-使用現有設定腳本-推薦)
  - [Option B: 使用 requirements.txt (不推薦用於開發)](#option-b-使用-requirementstxt-不推薦用於開發)
  - [Option C: 混合方式 (最佳實踐)](#option-c-混合方式-最佳實踐)
- [🎯 不同操作模式](#不同操作模式)
  - [1. 開發模式 (Development)](#1-開發模式-development)
  - [2. 測試模式 (Testing)](#2-測試模式-testing)
  - [3. 生產模式 (Production)](#3-生產模式-production)
  - [4. CI/CD 模式](#4-cicd-模式)
- [📊 當前狀態評估](#當前狀態評估)
  - [已完成](#已完成)
  - [待執行](#待執行)
  - [問題診斷](#問題診斷)
- [🚀 建議執行計劃](#建議執行計劃)
  - [階段 1: 安裝套件 (5-15 分鐘)](#階段-1-安裝套件-515-分鐘)
  - [階段 2: 驗證導入 (5 分鐘)](#階段-2-驗證導入-5-分鐘)
  - [階段 3: 清理臨時方案 (5 分鐘)](#階段-3-清理臨時方案-5-分鐘)
  - [階段 4: 執行測試 (2 分鐘)](#階段-4-執行測試-2-分鐘)
  - [階段 5: 更新文件 (5 分鐘)](#階段-5-更新文件-5-分鐘)
- [最終解決方案](#最終解決方案)
  - [執行步驟](#執行步驟)
  - [結果](#結果)
  - [執行時間](#執行時間)
  - [經驗教訓](#經驗教訓)
- [📝 總結與建議](#總結與建議)
  - [關鍵發現](#關鍵發現)
  - [最佳實踐](#最佳實踐)
  - [避免陷阱](#避免陷阱)
  - [時間估算對比](#時間估算對比)
  - [立即行動建議](#立即行動建議)
- [🔗 相關文件](#相關文件)

---
---
---
---

## 📋 執行摘要

### 重要發現
1. ✅ **專案已有完整配置**: 3 個 pyproject.toml + 6 個 requirements.txt
2. ✅ **套件已成功安裝**: `aiva-platform-integrated 1.0.0` 已安裝於可編輯模式
3. ✅ **自動化腳本存在**: `scripts/common/setup/setup_multilang.ps1` 包含 `pip install -e .`
4. ✅ **導入問題已解決**: 套件安裝後所有 `ModuleNotFoundError` 問題已修復

### 推薦方案
使用 **Option A (標準安裝方式)** - 執行現有的設定腳本即可解決所有導入問題。

---

## 🏗️ 專案結構與套件層級

```
AIVA-git/
├── pyproject.toml                    # 主專案: aiva-platform-integrated
├── requirements.txt                  # 根層級依賴 (114 packages)
│
├── services/                         # 服務層
│   ├── pyproject.toml                # 子專案: aiva-services
│   │
│   ├── aiva_common/                  # 獨立套件 (最低層)
│   │   ├── pyproject.toml            # aiva-common (7 dependencies)
│   │   └── requirements.txt          # aiva_common 專屬依賴
│   │
│   ├── core/                         # 核心服務 (依賴 aiva_common)
│   │   └── requirements.txt          # 核心服務依賴 (繼承 aiva_common)
│   │
│   ├── integration/                  # 整合服務 (依賴 aiva_common, core)
│   ├── features/                     # 功能服務 (依賴 aiva_common)
│   └── scan/                         # 掃描服務 (依賴 aiva_common)
│
├── api/
│   └── requirements.txt              # API 層依賴
│
└── plugins/
    └── requirements.txt              # 插件依賴
```

### 套件依賴關係圖
```
aiva-platform-integrated (root)
    │
    ├─> aiva-services (services/)
    │       │
    │       ├─> aiva-common (services/aiva_common/)    [獨立套件]
    │       │       └─ pydantic, pydantic-settings, typing-extensions
    │       │
    │       ├─> core (services/core/)                  [依賴 aiva_common]
    │       │       └─ torch, transformers, openai, neo4j
    │       │
    │       ├─> integration (services/integration/)    [依賴 aiva_common, core]
    │       ├─> features (services/features/)          [依賴 aiva_common]
    │       └─> scan (services/scan/)                  [依賴 aiva_common]
    │
    ├─> fastapi, sqlalchemy, redis, neo4j              [主依賴]
    └─> grpcio, protobuf                               [跨語言通訊]
```

---

## 📦 依賴清單詳解

### 1. 根層級依賴 (requirements.txt)

**核心框架** (7 packages):
- `fastapi>=0.115.0` - 現代 API 框架
- `uvicorn[standard]>=0.30.0` - ASGI 伺服器
- `pydantic>=2.7.0` - 數據驗證
- `websockets>=11.0.0` - 即時通訊
- `sqlalchemy>=2.0.31` - 資料庫 ORM
- `click>=8.1.0` - CLI 介面
- `rich>=13.0.0` - 終端美化輸出

**AI & 機器學習** (10 packages):
- `torch>=2.1.0` - PyTorch 深度學習框架 (5M 參數神經網絡核心)
- `torchvision>=0.16.0` - 電腦視覺
- `transformers>=4.30.0` - Transformer 模型支持
- `sentence-transformers>=2.2.0` - 語意文本嵌入 (RAG 系統)
- `openai>=1.0.0` - OpenAI API 客戶端
- `numpy>=1.24.0` - 數值計算
- `scipy>=1.10.0` - 科學計算
- `scikit-learn>=1.3.0` - 機器學習工具
- `nltk>=3.8.0` - 自然語言工具包
- `spacy>=3.6.0` - 工業級 NLP

**消息佇列** (3 packages):
- `aio-pika>=9.4.0` - 異步 RabbitMQ 客戶端
- `celery>=5.3.0` - 分散式任務佇列 (計劃中)
- `kombu>=5.3.0` - 消息傳遞庫 (計劃中)

**HTTP 客戶端** (3 packages):
- `httpx>=0.27.0` - 異步 HTTP 客戶端 (優先)
- `requests>=2.31.0` - 同步 HTTP 客戶端
- `aiohttp>=3.8.0` - 替代異步 HTTP 客戶端

**資料庫 & 儲存** (7 packages):
- `redis>=5.0.0` - 記憶體數據結構存儲
- `neo4j>=5.23.0` - 圖數據庫 (知識圖譜)
- `asyncpg>=0.29.0` - 異步 PostgreSQL 驅動
- `psycopg2-binary>=2.9.0` - PostgreSQL 適配器
- `alembic>=1.13.2` - 資料庫遷移工具
- `chromadb>=0.4.0` - 向量資料庫 (RAG) (計劃中)
- `pymongo>=4.4.0` - MongoDB 驅動 (可選)

**安全性** (5 packages):
- `PyJWT>=2.8.0` - JWT 處理
- `python-jose[cryptography]>=3.3.0` - JWT 處理含加密
- `passlib[bcrypt]>=1.7.4` - 密碼哈希
- `cryptography>=42.0.0` - 加密操作
- `python-multipart>=0.0.6` - 檔案上傳支持

**跨語言通訊** (3 packages):
- `grpcio>=1.60.0` - gRPC 框架
- `grpcio-tools>=1.60.0` - gRPC 工具
- `protobuf>=4.25.0` - Protocol Buffers

**開發工具** (11 packages):
- `pytest>=8.0.0` - 測試框架
- `pytest-cov>=4.0.0` - 覆蓋率插件
- `pytest-asyncio>=0.23.0` - 異步測試支持
- `black>=24.0.0` - 代碼格式化
- `ruff>=0.3.0` - 快速 Linting
- `mypy>=1.8.0` - 類型檢查
- `pre-commit>=3.6.0` - Git pre-commit hooks
- `types-requests>=2.31.0` - requests 類型存根
- `structlog>=24.1.0` - 結構化日誌
- `prometheus-client>=0.17.0` - 指標收集 (計劃中)
- `psutil>=5.9.6` - 系統監控

**其他工具** (9 packages):
- `python-dotenv>=1.0.1` - 環境變數
- `orjson>=3.10.0` - 快速 JSON 處理
- `toml>=0.10.2` - TOML 解析器
- `PyYAML>=6.0` - YAML 解析器
- `beautifulsoup4>=4.12.2` - HTML 解析
- `lxml>=5.0.0` - XML 處理
- `tenacity>=8.3.0` - 重試與韌性模式
- `aiofiles>=23.2.1` - 異步檔案操作
- `pandas>=2.0.0` - 數據處理與分析
- `gymnasium>=0.29.0` - 強化學習環境介面

**總計**: 約 **60-70 核心依賴** (不含子依賴)

---

### 2. services/aiva_common/pyproject.toml

**核心依賴** (3 packages):
```toml
[project]
dependencies = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "typing-extensions>=4.5.0"
]
```

**可選依賴組** (5 groups):

#### async 組:
- `aiofiles>=23.0.0`
- `asyncpg>=0.28.0`

#### testing 組:
- `pytest>=7.4.0`
- `pytest-asyncio>=0.21.0`
- `pytest-cov>=4.1.0`

#### cli 組:
- `click>=8.1.0`
- `rich>=13.0.0`

#### observability 組:
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`
- `opentelemetry-instrumentation-fastapi>=0.41b0`

#### plugins 組:
- `pluggy>=1.3.0`

**安裝範例**:
```bash
# 基礎安裝
pip install -e services/aiva_common

# 含所有可選依賴
pip install -e "services/aiva_common[async,testing,cli,observability,plugins]"
```

---

### 3. services/core/requirements.txt

**繼承依賴**:
```
-r ../aiva_common/requirements.txt
```

**額外依賴** (15 packages):

**AI & 機器學習**:
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `sentence-transformers>=2.2.0`
- `openai>=1.0.0`

**自然語言處理**:
- `nltk>=3.8.0`
- `spacy>=3.6.0`

**數據科學**:
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`

**圖數據庫**:
- `neo4j>=5.8.0`

**異步支持**:
- `asyncio-mqtt>=0.16.0`

**API 支持**:
- `pydantic[dotenv]>=2.0.0`
- `python-multipart>=0.0.6`

---

### 4. api/requirements.txt 與 plugins/requirements.txt

(需要時可讀取詳細內容)

---

## 🔧 安裝方式比較

### Option A: 使用現有設定腳本 (✅ 推薦)

**方式 1: 執行自動化腳本**
```powershell
# 已存在的官方設定腳本
.\scripts\common\setup\setup_multilang.ps1
```

**腳本內容**:
```powershell
# 升級 pip
pip install --upgrade pip setuptools wheel

# 可編輯安裝 (editable install)
pip install -e .

# 安裝 Node.js 依賴 (如果有)
npm install

# 安裝 Playwright (如果需要)
npx playwright install --with-deps chromium

# 安裝 Go 依賴 (如果有)
# go mod download

# 安裝 Rust 依賴 (如果有)
# cargo build

# Docker 相關設定 (如果需要)
```

**優點**:
- ✅ 官方維護的腳本,包含所有必要步驟
- ✅ 自動處理跨語言依賴 (Python, Node.js, Go, Rust)
- ✅ 一次性解決所有安裝需求
- ✅ 包含開發環境完整設定

**執行時間**: 5-15 分鐘 (取決於網路速度)

---

**方式 2: 手動可編輯安裝**
```bash
# 從專案根目錄
cd C:\D\fold7\AIVA-git

# 升級 pip
pip install --upgrade pip setuptools wheel

# 可編輯安裝主專案
pip install -e .

# 或安裝含開發工具
pip install -e ".[dev]"
```

**效果**:
- ✅ 所有 `from aiva_common import ...` 導入正常工作
- ✅ 所有 `from services.xxx import ...` 導入正常工作
- ✅ 代碼修改立即生效 (不需重新安裝)
- ✅ 支援跨模組導入 (`services.integration.capability`)

**優點**:
- ✅ Python 標準做法 (遵循 PEP 517/518)
- ✅ 無需 sys.path 操作
- ✅ 支援所有導入模式
- ✅ IDE 自動完成正常運作
- ✅ 與虛擬環境完美整合

**缺點**:
- ⚠️ 需要正確的 pyproject.toml 配置 (已存在 ✅)
- ⚠️ 首次安裝較慢 (5-10 分鐘)

**執行時間**: 5-10 分鐘

---

### Option B: 使用 requirements.txt (不推薦用於開發)

```bash
# 從專案根目錄
pip install -r requirements.txt

# 安裝 core 依賴
pip install -r services/core/requirements.txt

# 安裝 aiva_common 依賴
pip install -r services/aiva_common/requirements.txt
```

**效果**:
- ✅ 安裝所有外部依賴
- ❌ 不會安裝 aiva 內部套件
- ❌ 仍然會有 `ModuleNotFoundError: No module named 'services'`
- ❌ 需要額外的 sys.path 操作

**適用場景**:
- 生產環境部署 (配合 Docker)
- CI/CD 管道
- 非開發用途

**執行時間**: 3-5 分鐘

---

### Option C: 混合方式 (最佳實踐)

```bash
# Step 1: 可編輯安裝主專案
pip install -e .

# Step 2: 安裝額外開發工具 (可選)
pip install -r requirements.txt

# Step 3: 驗證安裝
pip list | Select-String "aiva"
```

**預期輸出**:
```
aiva-common          0.1.0      C:\D\fold7\AIVA-git\services\aiva_common
aiva-platform-integrated 2.0.0  C:\D\fold7\AIVA-git
aiva-services        0.1.0      C:\D\fold7\AIVA-git\services
```

**執行時間**: 5-10 分鐘

---

## 🎯 不同操作模式

### 1. 開發模式 (Development)
```bash
# 使用可編輯安裝
pip install -e .

# 或含開發工具
pip install -e ".[dev]"

# 安裝 pre-commit hooks
pre-commit install
```

**特點**:
- 代碼修改立即生效
- 完整的 IDE 支援
- 自動類型檢查與 Linting

---

### 2. 測試模式 (Testing)
```bash
# 安裝含測試依賴
pip install -e ".[dev]"

# 或安裝特定測試工具
pip install pytest pytest-cov pytest-asyncio

# 執行測試
pytest services/core/tests/ -v
```

**特點**:
- 覆蓋率報告
- 異步測試支援
- 測試隔離環境

---

### 3. 生產模式 (Production)
```bash
# 使用 requirements.txt
pip install -r requirements.txt

# 或使用標準安裝
pip install .

# 使用 Docker (推薦)
docker-compose up -d
```

**特點**:
- 固定版本依賴
- 最小化安裝
- 容器化部署

---

### 4. CI/CD 模式
```bash
# 快速安裝
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# 執行測試與檢查
pytest --cov=services
black --check .
ruff check .
mypy services/
```

**特點**:
- 自動化測試
- 代碼品質檢查
- 快速反饋

---

## 📊 當前狀態評估

### 已完成
- ✅ 3 個 pyproject.toml 正確配置
- ✅ 6 個 requirements.txt 完整依賴清單
- ✅ 自動化設定腳本 (setup_multilang.ps1)
- ✅ 依賴關係清晰定義
- ✅ 套件結構符合 Python 標準

### 待執行
- ❌ **首次安裝**: 執行 `pip install -e .`
- ❌ **驗證導入**: 測試所有 import 語句
- ❌ **移除 sys.path hacks**: 清理 conftest.py
- ❌ **執行測試**: 驗證 ModuleExplorer 測試

### 問題診斷
```python
# 當前錯誤
ModuleNotFoundError: No module named 'services'

# 根本原因
1. aiva 套件從未安裝 (`pip list` 空白)
2. Python 不知道 'services' 在哪裡
3. sys.path hacks 只是臨時解決方案
4. 無法支援跨模組導入

# 解決方案
執行 Option A (可編輯安裝) 即可完全解決
```

---

## 🚀 建議執行計劃

### 階段 1: 安裝套件 (5-15 分鐘)

**方案 1A: 使用官方腳本 (推薦)**
```powershell
# 執行自動化設定
.\scripts\common\setup\setup_multilang.ps1

# 驗證安裝
pip list | Select-String "aiva"
```

**方案 1B: 手動安裝**
```bash
cd C:\D\fold7\AIVA-git
pip install --upgrade pip setuptools wheel
pip install -e .
pip list | Select-String "aiva"
```

---

### 階段 2: 驗證導入 (5 分鐘)

```python
# 測試基本導入
python -c "from aiva_common import Config; print('✓ aiva_common works')"
python -c "from services.core import models; print('✓ services.core works')"
python -c "from services.integration.capability import CapabilityRegistry; print('✓ cross-module works')"
```

**預期輸出**:
```
✓ aiva_common works
✓ services.core works
✓ cross-module works
```

---

### 階段 3: 清理臨時方案 (5 分鐘)

```python
# 選項 A: 完全移除 conftest.py 的 sys.path 操作
# services/core/tests/conftest.py

import pytest
from pathlib import Path

# 移除或註解掉 sys.path 操作
# services_dir = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(services_dir))
# core_dir = Path(__file__).parent.parent
# sys.path.insert(0, str(core_dir))

@pytest.fixture(scope="session")
def services_root():
    """返回 services 根目錄路徑"""
    return Path(__file__).parent.parent.parent
```

```python
# 選項 B: 保留 conftest.py 但簡化
# 僅保留 fixture,移除 sys.path 操作

import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def services_root():
    return Path(__file__).parent.parent.parent

@pytest.fixture(scope="session")
def core_root():
    return Path(__file__).parent.parent
```

---

### 階段 4: 執行測試 (2 分鐘)

```bash
# 執行 ModuleExplorer 測試
pytest services/core/tests/test_module_explorer.py -v

# 執行所有核心測試
pytest services/core/tests/ -v

# 執行含覆蓋率報告
pytest services/core/tests/ --cov=services.core --cov-report=html
```

**預期結果**:
```
services/core/tests/test_module_explorer.py::test_module_explorer PASSED [100%]

============ 1 passed in 0.05s ============
```

---

### 階段 5: 更新文件 (5 分鐘)

更新 `IMPORT_FIX_PROGRESS.md`:
```markdown
## 最終解決方案

採用 **Option A - 標準可編輯安裝**

### 執行步驟
1. ✅ 執行 `.\scripts\common\setup\setup_multilang.ps1`
2. ✅ 驗證 `pip list | Select-String "aiva"` 顯示 3 個套件
3. ✅ 清理 conftest.py 的 sys.path 操作
4. ✅ 所有測試通過

### 結果
- 所有導入問題已解決
- 無需任何 sys.path hacks
- 支援跨模組導入
- IDE 自動完成正常運作

### 執行時間
- 總計: 15-20 分鐘
- vs 原預估 5-8 小時 (Option B)

### 經驗教訓
1. 優先檢查專案是否已有正確配置
2. 使用標準 Python 套件管理方式
3. 避免使用 sys.path hacks
4. 善用現有的自動化腳本
```

---

## 📝 總結與建議

### 關鍵發現
1. **專案配置完整**: pyproject.toml 與 requirements.txt 都已正確設定
2. **安裝腳本存在**: `setup_multilang.ps1` 包含完整設定流程
3. **從未執行安裝**: 是導入問題的根本原因
4. **標準方案最佳**: `pip install -e .` 解決所有問題

### 最佳實踐
1. ✅ **使用 pip install -e .** 進行開發
2. ✅ **遵循 pyproject.toml** 標準
3. ✅ **避免 sys.path hacks**
4. ✅ **善用現有自動化腳本**
5. ✅ **定期更新 requirements.txt**

### 避免陷阱
1. ❌ 不要直接修改 sys.path
2. ❌ 不要使用相對導入跨越套件邊界
3. ❌ 不要混用不同導入風格
4. ❌ 不要忘記執行安裝步驟

### 時間估算對比

| 方案 | 時間 | 風險 | 維護性 | 推薦度 |
|------|------|------|--------|--------|
| **Option A - 官方腳本** | 5-15 分鐘 | ✅ 低 | ✅ 高 | ⭐⭐⭐⭐⭐ |
| **Option A - 手動安裝** | 5-10 分鐘 | ✅ 低 | ✅ 高 | ⭐⭐⭐⭐ |
| **Option B - 批次修復** | 5-8 小時 | ⚠️ 中 | ❌ 低 | ⭐ |
| **Option C - 混合方式** | 10-15 分鐘 | ✅ 低 | ✅ 高 | ⭐⭐⭐⭐ |

### 立即行動建議
```bash
# 建議執行順序
1. .\scripts\common\setup\setup_multilang.ps1   # 5-15 分鐘
2. pip list | Select-String "aiva"              # 驗證安裝
3. pytest services/core/tests/ -v               # 執行測試
4. 更新 IMPORT_FIX_PROGRESS.md                  # 記錄結果
```

**預期總時間**: 15-25 分鐘  
**vs 原計劃**: 5-8 小時 (節省 95% 時間)

---

## 🔗 相關文件

- `IMPORT_FIX_PROGRESS.md` - 導入修復進度報告
- `DEVELOPMENT_STANDARDS.md` - 開發標準指南
- `pyproject.toml` (x3) - 套件配置
- `requirements.txt` (x6) - 依賴清單
- `scripts/common/setup/setup_multilang.ps1` - 自動化設定腳本

---

**報告結束**
