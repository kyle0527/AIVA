# Tests - 測試套件

**導航**: [← 返回 AIVA Core](../README.md) | [← 返回項目根目錄](../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 現有測試](#-現有測試)
  - [test_capabilities.py](#test_capabilitiespy-312-行)
  - [test_integration.py](#test_integrationpy-203-行)
- [⚠️ 測試缺口分析](#️-測試缺口分析)
- [🔧 測試框架配置](#-測試框架配置)
- [📝 測試模板](#-測試模板)
- [🚀 快速開始測試開發](#-快速開始測試開發)
- [📊 測試覆蓋率目標](#-測試覆蓋率目標)
- [🔍 測試工具推薦](#-測試工具推薦)
- [📚 相關文檔](#-相關文檔)

---

## 📋 概述

**定位**: AIVA Core 測試框架  
**狀態**: ⚠️ 測試覆蓋率極低 (< 2%)  
**文件數**: 2 個 Python 文件 (515 行)

## 📂 文件結構

```
tests/
├── test_capabilities.py (312 行) - 能力系統測試
├── test_integration.py (203 行) - 集成測試
├── __init__.py
└── README.md (本文檔)
```

## 🎯 現有測試

### test_capabilities.py (312 行)

**測試範圍**:
- 能力掃描功能
- 能力註冊和查詢
- 能力執行流程

**使用範例**:
```python
# 運行能力測試
pytest tests/test_capabilities.py -v

# 測試特定功能
pytest tests/test_capabilities.py::test_capability_scan -v
```

---

### test_integration.py (203 行)

**測試範圍**:
- 模組間集成測試
- API 端到端測試
- 基本工作流測試

**使用範例**:
```python
# 運行集成測試
pytest tests/test_integration.py -v

# 帶覆蓋率報告
pytest tests/test_integration.py --cov=aiva_core --cov-report=html
```

## ⚠️ 測試缺口分析

### 當前狀況

| 模組 | 代碼行數 | 測試行數 | 覆蓋率 | 狀態 |
|------|----------|----------|--------|------|
| **cognitive_core** | 8,135 | 0 | 0% | ❌ 無測試 |
| **core_capabilities** | 6,089 | 312 | ~5% | ⚠️ 極低 |
| **task_planning** | 5,526 | 0 | 0% | ❌ 無測試 |
| **service_backbone** | 8,051 | 203 | ~3% | ⚠️ 極低 |
| **internal_exploration** | 929 | 0 | 0% | ❌ 無測試 |
| **external_learning** | 5,726 | 0 | 0% | ❌ 無測試 |
| **ui_panel** | 2,147 | 0 | 0% | ❌ 無測試 |
| **總計** | 37,118 | 515 | **< 2%** | 🚨 緊急 |

### 優先級建議

**P0 (緊急 - 核心功能)**:
1. `cognitive_core/rag/` - RAG 系統測試
2. `core_capabilities/scanner/` - 能力掃描測試
3. `service_backbone/api/` - API 接口測試
4. `service_backbone/authz/` - 權限系統測試

**P1 (高優先級 - 關鍵功能)**:
5. `task_planning/planner/` - 任務規劃測試
6. `service_backbone/messaging/` - 消息系統測試
7. `external_learning/learning/` - 學習引擎測試

**P2 (中優先級 - 支撐功能)**:
8. `cognitive_core/memory/` - 記憶系統測試
9. `service_backbone/coordination/` - 協調器測試
10. `internal_exploration/` - 探索功能測試

## 🔧 測試框架配置

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=aiva_core
    --cov-report=html
    --cov-report=term-missing

markers =
    unit: 單元測試
    integration: 集成測試
    slow: 慢速測試
    requires_db: 需要數據庫
    requires_gpu: 需要 GPU
```

### conftest.py (建議創建)

```python
import pytest
from aiva_core import create_app

@pytest.fixture(scope="session")
def app():
    """創建測試應用"""
    app = create_app("testing")
    return app

@pytest.fixture(scope="function")
def client(app):
    """創建測試客戶端"""
    return app.test_client()

@pytest.fixture(scope="function")
def db():
    """創建測試數據庫"""
    from aiva_core.service_backbone.storage import StorageManager
    db = StorageManager.from_config({"backend": "sqlite", "database": ":memory:"})
    yield db
    db.cleanup()
```

## 📝 測試模板

### 單元測試模板

```python
import pytest
from aiva_core.module_name import TargetClass

class TestTargetClass:
    """測試 TargetClass 功能"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """測試前準備"""
        self.instance = TargetClass()
        yield
        # 清理
        
    def test_basic_functionality(self):
        """測試基本功能"""
        result = self.instance.method()
        assert result is not None
        
    def test_edge_cases(self):
        """測試邊界情況"""
        with pytest.raises(ValueError):
            self.instance.method(invalid_input)
            
    @pytest.mark.parametrize("input,expected", [
        ("input1", "output1"),
        ("input2", "output2")
    ])
    def test_multiple_cases(self, input, expected):
        """參數化測試"""
        assert self.instance.method(input) == expected
```

### 集成測試模板

```python
import pytest

class TestModuleIntegration:
    """測試模組間集成"""
    
    def test_api_workflow(self, client):
        """測試 API 工作流"""
        # 創建資源
        response = client.post("/api/resource", json={...})
        assert response.status_code == 201
        
        # 查詢資源
        response = client.get("/api/resource/1")
        assert response.status_code == 200
        
        # 更新資源
        response = client.put("/api/resource/1", json={...})
        assert response.status_code == 200
        
        # 刪除資源
        response = client.delete("/api/resource/1")
        assert response.status_code == 204
```

## 🚀 快速開始測試開發

### 1. 安裝測試依賴

```bash
pip install pytest pytest-cov pytest-mock pytest-asyncio
```

### 2. 創建測試文件

```bash
# 為每個模組創建測試
tests/
├── cognitive_core/
│   ├── test_rag.py
│   ├── test_memory.py
│   └── test_reasoning.py
├── core_capabilities/
│   ├── test_scanner.py
│   └── test_executor.py
├── service_backbone/
│   ├── test_api.py
│   └── test_authz.py
└── ...
```

### 3. 運行測試

```bash
# 運行所有測試
pytest

# 運行特定模組
pytest tests/cognitive_core/

# 帶覆蓋率報告
pytest --cov=aiva_core --cov-report=html

# 只運行單元測試
pytest -m unit

# 只運行集成測試
pytest -m integration
```

## 📊 測試覆蓋率目標

### 短期目標 (1-2 個月)

- **總體覆蓋率**: 30%
- **核心模組**: 50%+
  - cognitive_core/rag
  - core_capabilities/scanner
  - service_backbone/api

### 中期目標 (3-6 個月)

- **總體覆蓋率**: 60%
- **關鍵路徑**: 80%+

### 長期目標 (6-12 個月)

- **總體覆蓋率**: 80%+
- **核心模組**: 90%+

## 🔍 測試工具推薦

- **pytest** - 測試框架
- **pytest-cov** - 覆蓋率報告
- **pytest-mock** - Mock 工具
- **pytest-asyncio** - 異步測試
- **hypothesis** - 屬性測試
- **faker** - 測試數據生成
- **factory_boy** - 測試工廠

## 📚 相關文檔

- [AIVA Core README](../README.md)
- [測試最佳實踐](../../docs/TESTING_BEST_PRACTICES.md) (待創建)
- [CI/CD 集成](../../docs/CI_CD_INTEGRATION.md) (待創建)

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：測試中使用標準類型
from aiva_common import TaskStatus, ModuleName
import pytest

def test_task_execution():
    assert TaskStatus.SUCCESS.value == "success"

# ❌ 禁止：測試中自定義狀態
class TestStatus(str, Enum): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**狀態**: 🚨 測試覆蓋率嚴重不足,需緊急補充  
**維護者**: AIVA Core 團隊
