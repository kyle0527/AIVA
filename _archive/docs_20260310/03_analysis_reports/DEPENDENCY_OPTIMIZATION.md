# AIVA Core 依賴優化指南

## 📑 目錄

- [⚠️ 優化狀態 (2026-01-09)](#-優化狀態-2026-01-09)
- [📋 歷史問題分析 (已解決)](#-歷史問題分析-已解決)
  - [1. 導入鏈過長導致的阻塞](#1-導入鏈過長導致的阻塞)
  - [2. 重度依賴清單](#2-重度依賴清單)
    - [AI/ML 依賴（驗證CLI時非必需）](#aiml-依賴驗證cli時非必需)
    - [數據庫依賴（驗證CLI時非必需）](#數據庫依賴驗證cli時非必需)
    - [Web 框架依賴（驗證CLI時非必需）](#web-框架依賴驗證cli時非必需)
  - [3. 建議的優化方案](#3-建議的優化方案)
    - [方案A：延遲導入（Lazy Import）✅ 推薦](#方案a延遲導入lazy-import-推薦)
    - [方案B：創建獨立的CLI工具包](#方案b創建獨立的cli工具包)
    - [方案C：修復缺失模組](#方案c修復缺失模組)
- [推薦實施順序](#推薦實施順序)
  - [階段1：快速修復（立即可用）](#階段1快速修復立即可用)
  - [階段2：結構優化（中期）](#階段2結構優化中期)
  - [階段3：依賴分層（長期）](#階段3依賴分層長期)
- [當前可用的繞過方案](#當前可用的繞過方案)
  - [方法1：直接導入目標模組（已驗證 ✅）](#方法1直接導入目標模組已驗證-)
  - [方法2：Mock 缺失模組](#方法2mock-缺失模組)
- [性能對比](#性能對比)
  - [當前完整導入](#當前完整導入)
  - [優化後（方案A）](#優化後方案a)
  - [獨立CLI驗證器（方案B）](#獨立cli驗證器方案b)
- [行動計劃](#行動計劃)
  - [立即執行（今天）](#立即執行今天)
  - [本週完成](#本週完成)
  - [下週完成](#下週完成)

---


## ⚠️ 優化狀態 (2026-01-09)

**✅ 優化已完成 - v4.1.1**

- ✅ orchestrator 模組引用已修復
- ✅ 分層依賴配置已建立 (minimal/web/ai/full)
- ✅ 當前全局環境已安裝完整依賴集
- ✅ 性能提升：CLI 啟動 15s→<1s (93%)

**使用場景**：
- 本文檔供**歷史參考**和**新環境部署**使用
- 當前環境無需額外優化，直接使用即可
- 詳見 [DEPENDENCY_ANALYSIS.md](./DEPENDENCY_ANALYSIS.md) 完整分析

---

## 📋 歷史問題分析 (已解決)

### 1. 導入鏈過長導致的阻塞
```python
# 問題路徑：
services.core.aiva_core.__init__.py (Line 566-567)
└─> service_backbone.context_manager
    └─> service_backbone.coordination.core_service_coordinator
        └─> task_planning.planner.execution_planner
            └─> task_planning.planner.orchestrator  ❌ 模組不存在
```

**阻塞原因**：
- `__init__.py` 在模組載入時立即導入所有組件
- 即使只需要單一功能（如 `internal_loop_connector`），也會觸發完整導入鏈
- 缺少的 `orchestrator` 模組導致整個導入失敗

### 2. 重度依賴清單

#### AI/ML 依賴（驗證CLI時非必需）
```
torch>=2.0.0                    # 2+ GB，載入慢
transformers>=4.30.0            # 1+ GB，載入慢
sentence-transformers>=2.2.0
openai>=1.0.0
nltk>=3.8.0
spacy>=3.6.0
scikit-learn>=1.3.0
```

#### 數據庫依賴（驗證CLI時非必需）
```
neo4j>=5.8.0                    # 圖數據庫
psycopg2-binary>=2.9.7          # PostgreSQL
redis>=5.0.0
```

#### Web 框架依賴（驗證CLI時非必需）
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
```

### 3. 建議的優化方案

#### 方案A：延遲導入（Lazy Import）✅ 推薦
將 `__init__.py` 中的全局導入改為函數內導入：

```python
# 修改前（當前）：
# services/core/aiva_core/__init__.py
from .service_backbone.context_manager import ContextManager  # 立即導入

# 修改後：
def get_context_manager():
    """延遲導入，僅在需要時載入"""
    from .service_backbone.context_manager import ContextManager
    return ContextManager()
```

**優點**：
- 不需要修改 requirements.txt
- 只載入實際使用的模組
- 向後兼容，不影響現有代碼

**實施位置**：
1. `services/core/aiva_core/__init__.py` (Line 566-578)
2. `services/core/aiva_core/service_backbone/__init__.py`
3. `services/core/aiva_core/task_planning/__init__.py`

#### 方案B：創建獨立的CLI工具包
創建一個不依賴完整 aiva_core 的獨立CLI驗證工具：

```bash
services/core/aiva_core/internal_exploration/python_tools/
├── standalone_cli_validator.py  # 獨立驗證器
├── minimal_requirements.txt     # 最小依賴
└── README_CLI_VALIDATION.md     # 使用說明
```

**最小依賴**：
```txt
# minimal_requirements.txt
pydantic>=2.0.0        # 數據驗證
python-dateutil>=2.8.0 # 時間處理
pathlib                # 路徑操作（標準庫）
json                   # JSON處理（標準庫）
```

#### 方案C：修復缺失模組
補全缺失的 `orchestrator` 模組：

```bash
services/core/aiva_core/task_planning/planner/
├── __init__.py
├── execution_planner.py
└── orchestrator.py  # ❌ 缺失，需要創建或移除引用
```

## 推薦實施順序

### 階段1：快速修復（立即可用）
1. ✅ 修復缺失的 `orchestrator` 引用
   - 選項A：移除 `__init__.py` 中的導入語句
   - 選項B：創建空的 `orchestrator.py` 占位符

2. ✅ 創建獨立CLI驗證腳本
   ```bash
   cd services/core/aiva_core/internal_exploration/python_tools
   python standalone_cli_validator.py --flow 51  # 測試 Flow 51
   ```

### 階段2：結構優化（中期）
1. 將 `__init__.py` 改為延遲導入
2. 添加環境變數控制：
   ```python
   # 環境變數：AIVA_MINIMAL_MODE=true
   import os
   if os.getenv("AIVA_MINIMAL_MODE") != "true":
       from .service_backbone import ...  # 完整導入
   ```

### 階段3：依賴分層（長期）
創建分層的 requirements 文件：
```
requirements/
├── base.txt           # 核心依賴（pydantic, loguru）
├── cli.txt            # CLI工具依賴
├── ai.txt             # AI/ML依賴
├── web.txt            # Web服務依賴
└── full.txt           # 完整依賴（引用所有）
```

## 當前可用的繞過方案

### 方法1：直接導入目標模組（已驗證 ✅）
```python
# 繞過 aiva_core.__init__.py
import sys
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接導入目標類，不經過 __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "internal_loop_connector",
    project_root / "services/core/aiva_core/cognitive_core/internal_loop_connector.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 使用
classifier = module.CapabilityScopeClassifier()
scope, visibility = classifier.classify_scope("path/to/file.py")
```

**優點**：
- 立即可用，無需修改源碼
- 已在 `demo_ai_standalone.py` 中驗證成功

### 方法2：Mock 缺失模組
```python
# 在導入前 mock 缺失模組
import sys
from unittest.mock import MagicMock

# Mock 缺失模組
sys.modules['services.core.aiva_core.task_planning.planner.orchestrator'] = MagicMock()

# 然後正常導入
from services.core.aiva_core.cognitive_core.internal_loop_connector import ...
```

## 性能對比

### 當前完整導入
```
啟動時間: ~15-30秒
內存占用: ~2-3 GB
依賴數量: 50+ packages
```

### 優化後（方案A）
```
啟動時間: ~2-5秒
內存占用: ~200-500 MB
依賴數量: 10-15 packages（核心依賴）
```

### 獨立CLI驗證器（方案B）
```
啟動時間: <1秒
內存占用: ~50-100 MB
依賴數量: 5 packages
```

## 行動計劃

### 立即執行（今天）
- [ ] 創建 `standalone_cli_validator.py`
- [ ] 創建 `minimal_requirements.txt`
- [ ] 測試 3 個 AI 內部能力的直接調用

### 本週完成
- [ ] 修復 `orchestrator` 缺失問題
- [ ] 將常用模組改為延遲導入
- [ ] 添加 `AIVA_MINIMAL_MODE` 環境變數支持

### 下週完成
- [ ] 創建分層 requirements
- [ ] 更新文檔說明快速驗證方式
- [ ] 添加 CI/CD 中的快速驗證測試

---

**維護者**: AIVA Team  
**最後更新**: 2026-01-09  
**優先級**: 🔴 高（影響開發效率）
