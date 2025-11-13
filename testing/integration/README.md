# 🔗 AIVA 集成測試

這個目錄包含 AIVA 項目的集成測試，專注於驗證多個模組間的協作和系統整體功能。

## 📁 目錄結構

```
testing/
├── integration/            # 集成測試 (本目錄)
│   ├── legacy_tests/      # 舊版測試檔案
│   ├── tools/             # 工具集成測試
│   └── [測試檔案]
├── common/                 # 通用測試工具
├── core/                   # AI 核心功能測試
├── features/               # 功能特性測試
├── performance/            # 性能測試
├── scan/                   # 掃描和滲透測試
└── README.md
```

## 📄 本目錄檔案說明

### 🌟 核心集成測試
- **`aiva_full_worker_live_test.py`** - 全功能 Worker 實戰測試
- **`comprehensive_integration_test_suite.py`** - 綜合集成測試套件
- **`aiva_system_connectivity_sop_check.py`** - 系統連接性 SOP 檢查
- **`aiva_module_status_checker.py`** - 模組狀態檢查器

### 🔧 API 和服務集成
- **`test_api.py`** - AIVA API 完整測試套件
- **`test_memory_integration.py`** - 記憶體集成測試

### 📊 數據和通信集成
- **`data_persistence_test.py`** - 數據持久化測試
- **`message_queue_test.py`** - 消息隊列集成測試
- **`test_result_database.py`** - 結果資料庫測試

### 🎯 功能模組集成測試
- **`test_basic.py`** - 基礎功能集成測試
- **`test_lifecycle.py`** - 生命週期管理測試
- **`test_function_recon.py`** - 偵察功能集成測試
- **`test_payload_generator.py`** - 有效載荷生成器集成測試
- **`test_sql_injection_tools.py`** - SQL 注入工具集成測試
- **`test_web_attack.py`** - Web 攻擊模組集成測試

### 🌐 跨語言和驗證測試
- **`test_cross_language_validation.py`** - 跨語言驗證測試

### 📋 舊版測試 (`legacy_tests/`)
- **`test_5m_integration.py`** - 5M 集成測試
- **`test_attack_plan_mapper.py`** - 攻擊計劃映射器測試
- **`test_direct_ai_core.py`** - 直接 AI 核心測試
- **`test_integration.py`** - 基礎集成測試
- **`test_real_ai_core.py`** - 真實 AI 核心測試
- **`validate_integration.py`** - 集成驗證

### 🛠️ 工具集成測試 (`tools/`)
- **`test_import.py`** - 工具導入測試

### 📄 報告檔案
- **`SYSTEM_CONNECTIVITY_REPORT.json`** - 系統連接性測試報告

## 🚀 使用方法

### 執行完整集成測試
```bash
# 執行所有集成測試
pytest testing/integration/ -v

# 執行全功能實戰測試
python testing/integration/aiva_full_worker_live_test.py

# 執行綜合集成測試套件
python testing/integration/comprehensive_integration_test_suite.py
```

### 系統連接性和狀態檢查
```bash
# 系統連接性檢查
python testing/integration/aiva_system_connectivity_sop_check.py

# 模組狀態檢查
python testing/integration/aiva_module_status_checker.py
```

### API 和服務測試
```bash
# 完整 API 測試
python testing/integration/test_api.py

# 記憶體集成測試
python testing/integration/test_memory_integration.py
```

### 數據和通信測試
```bash
# 數據持久化測試
python testing/integration/data_persistence_test.py

# 消息隊列測試
python testing/integration/message_queue_test.py

# 結果資料庫測試
python testing/integration/test_result_database.py
```

### 功能模組集成測試
```bash
# 基礎功能測試
pytest testing/integration/test_basic.py -v

# SQL 注入工具集成測試
pytest testing/integration/test_sql_injection_tools.py -v

# Web 攻擊模組測試
pytest testing/integration/test_web_attack.py -v
```

### 舊版測試執行
```bash
# 執行舊版集成測試
python testing/integration/legacy_tests/test_integration.py

# 5M 集成測試
python testing/integration/legacy_tests/test_5m_integration.py
```

## 🛠️ 核心集成測試組件

### 全功能 Worker 實戰測試 (`aiva_full_worker_live_test.py`)
**實際執行所有 Worker 模組進行真實測試：**
- **SSRF Worker** - 伺服器端請求偽造檢測
- **SQLi Worker** - SQL 注入檢測（5 引擎）
- **XSS Worker** - 跨站腳本檢測（Reflected/DOM/Blind）
- **IDOR Worker** - 不安全直接對象引用檢測
- **GraphQL AuthZ Worker** - GraphQL 權限繞過檢測

### 綜合集成測試套件 (`comprehensive_integration_test_suite.py`)
- **多模組協作測試**
- **端到端工作流程驗證**
- **系統整體性能測試**
- **錯誤處理和恢復測試**

### API 集成測試 (`test_api.py`)
- **認證和授權測試**
- **API 端點完整驗證**
- **高價值模組測試**
- **系統健康檢查**
- **錯誤響應測試**

### 系統連接性檢查 (`aiva_system_connectivity_sop_check.py`)
- **模組間連接驗證**
- **網絡通信測試**
- **服務依賴檢查**
- **SOP 合規性驗證**

### 記憶體集成測試 (`test_memory_integration.py`)
- **記憶體管理驗證**
- **數據共享測試**
- **快取機制驗證**
- **記憶體洩漏檢測**

### 數據持久化測試 (`data_persistence_test.py`)
- **資料庫連接測試**
- **數據儲存和檢索**
- **事務處理驗證**
- **備份和恢復測試**

### 消息隊列測試 (`message_queue_test.py`)
- **消息傳遞機制**
- **隊列管理**
- **異步處理驗證**
- **消息持久性測試**

## 📝 編寫集成測試

### 測試檔案命名規範
- 集成測試: `test_*.py`
- 系統測試: `*_system_*.py`
- 完整測試: `*_full_*.py`
- 檢查器: `*_checker.py`

### 集成測試結構範例
```python
#!/usr/bin/env python3
"""
集成測試模板
"""
import pytest
import asyncio
from services.core import CoreModule
from services.features import FeatureModule

class TestModuleIntegration:
    
    @pytest.fixture
    def integrated_system(self):
        """設置集成測試環境"""
        core = CoreModule()
        features = FeatureModule()
        return IntegratedSystem(core, features)
    
    def test_module_communication(self, integrated_system):
        """測試模組間通信"""
        # Arrange
        test_message = create_test_message()
        
        # Act
        result = integrated_system.process_message(test_message)
        
        # Assert
        assert result.success is True
        assert result.processed_by == "core"
        assert result.enhanced_by == "features"
    
    @pytest.mark.asyncio
    async def test_async_workflow(self, integrated_system):
        """測試異步工作流程"""
        workflow = integrated_system.create_workflow()
        result = await workflow.execute()
        
        assert result.completed is True
        assert len(result.steps) > 0
```

### Worker 集成測試範例
```python
def test_worker_integration():
    """Worker 集成測試"""
    # 設置靶場環境
    target_url = "http://localhost:3000"
    
    # 測試 SQLi Worker
    sqli_worker = SQLiWorker()
    sqli_results = sqli_worker.scan(target_url)
    
    # 測試 XSS Worker
    xss_worker = XSSWorker()
    xss_results = xss_worker.scan(target_url)
    
    # 驗證結果集成
    assert sqli_results.vulnerabilities_found >= 0
    assert xss_results.vulnerabilities_found >= 0
```

## 🔧 測試環境配置

### 環境依賴
```bash
# 核心服務
- PostgreSQL (資料庫)
- Redis (快取和消息隊列)
- RabbitMQ (消息代理)

# 靶場環境
- Juice Shop (http://localhost:3000)
- DVWA (可選)

# 外部工具
- SQLmap, Nmap, Dirb 等
```

### 環境變數設置
```bash
# 集成測試配置
export AIVA_INTEGRATION_MODE=true
export AIVA_DATABASE_URL=postgresql://test:test@localhost:5432/aiva_test
export AIVA_REDIS_URL=redis://localhost:6379/1
export AIVA_RABBITMQ_URL=amqp://guest:guest@localhost:5672/test

# 靶場配置
export AIVA_TARGET_URL=http://localhost:3000
export AIVA_TEST_TIMEOUT=300
```

### Docker 環境設置
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  juiceshop:
    image: bkimminich/juice-shop
    ports: ["3000:3000"]
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: aiva_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports: ["5432:5432"]
  
  redis:
    image: redis:6
    ports: ["6379:6379"]
```

## 📊 測試策略和覆蓋率

### 集成測試層級
1. **模組間集成** - 驗證模組界面和通信
2. **服務集成** - 測試服務間協作
3. **系統集成** - 完整系統工作流程
4. **端到端集成** - 用戶場景驗證

### 覆蓋率目標
- **API 集成**: > 90%
- **Worker 集成**: > 85%
- **數據集成**: > 95%
- **系統集成**: > 80%

### 測試矩陣
```
           | Core | Features | Scan | API
-----------|------|----------|------|-----
Core       |  ✓   |    ✓     |  ✓   |  ✓
Features   |  ✓   |    ✓     |  ✓   |  ✓  
Scan       |  ✓   |    ✓     |  ✓   |  ✓
API        |  ✓   |    ✓     |  ✓   |  ✓
```

## 🔄 持續集成和部署

### CI/CD 集成測試流程
```yaml
# GitHub Actions 範例
name: Integration Tests
on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
      redis:
        image: redis:6
      juiceshop:
        image: bkimminich/juice-shop
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install Dependencies
      run: pip install -r requirements.txt
    
    - name: Run Integration Tests
      run: |
        pytest testing/integration/ -v
        python testing/integration/aiva_full_worker_live_test.py
    
    - name: Generate Report
      run: pytest testing/integration/ --cov=services --cov-report=xml
```

### 測試報告和監控
- **測試結果報告** - JUnit XML 格式
- **覆蓋率報告** - Codecov 集成
- **性能監控** - 測試執行時間追蹤
- **失敗通知** - Slack/Email 整合

---

**目錄更新**: 2025-11-12  
**維護者**: AIVA Integration Team  
**測試重點**: 模組集成 + 系統協作 + 端到端驗證