# 🧪 AIVA 統一測試框架

現代化的測試結構，支援單元測試、整合測試和系統測試。

## 📁 目錄結構

```
testing/
├── unit/                    # 單元測試
│   ├── common/             # 通用模組測試
│   ├── core/               # 核心模組測試  
│   ├── scan/               # 掃描模組測試
│   ├── integration/        # 整合模組測試
│   └── features/           # 功能模組測試
├── integration/            # 整合測試
│   ├── api/               # API測試
│   ├── database/          # 資料庫測試
│   ├── messaging/         # 消息系統測試
│   └── workflow/          # 工作流測試
├── system/                # 系統測試
│   ├── e2e/              # 端到端測試
│   ├── performance/       # 效能測試
│   ├── security/         # 安全測試
│   └── compatibility/    # 相容性測試
├── fixtures/              # 測試數據
├── mocks/                # 模擬對象
├── utilities/            # 測試工具
├── conftest.py           # pytest配置
└── README.md
```

## 🚀 快速開始

### 安裝依賴
```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock
```

### 執行所有測試
```bash
# 從專案根目錄執行
pytest testing/ -v
```

### 執行特定類型測試
```bash
# 單元測試
pytest testing/unit/ -v

# 整合測試  
pytest testing/integration/ -v

# 系統測試
pytest testing/system/ -v
```

### 覆蓋率報告
```bash
pytest testing/ --cov=services --cov-report=html
```

## 📊 測試策略

### 🔬 單元測試 (Unit Tests)
- **目標**: 測試個別函數和類別
- **特點**: 快速執行，完全隔離
- **覆蓋率目標**: > 90%

### 🔗 整合測試 (Integration Tests)  
- **目標**: 測試模組間互動
- **特點**: 真實依賴，中等執行時間
- **覆蓋率目標**: > 70%

### 🏗️ 系統測試 (System Tests)
- **目標**: 端到端功能驗證
- **特點**: 完整工作流程，長時間執行
- **覆蓋率目標**: > 60%

## 🛠️ 測試工具

### pytest配置檔案
- `conftest.py` - 全局fixtures和配置
- `pytest.ini` - pytest設定檔

### Mock和Fixtures
- `fixtures/` - 共用測試數據
- `mocks/` - 模擬對象和服務

### 工具類
- `utilities/` - 測試輔助工具
- 資料庫助手
- API客戶端
- 檔案操作工具

## 📝 編寫測試指南

### 命名規範
- 測試檔案: `test_*.py`
- 測試類: `Test*`
- 測試方法: `test_*`

### 測試結構
```python
def test_should_do_something_when_condition():
    # Arrange - 準備
    setup_data = create_test_data()
    
    # Act - 執行
    result = function_under_test(setup_data)
    
    # Assert - 驗證
    assert result.success is True
    assert result.value == expected_value
```

### Fixtures範例
```python
@pytest.fixture
def sample_user():
    return {
        "id": "test-user-1",
        "name": "Test User",
        "email": "test@example.com"
    }
```

## 🔄 CI/CD整合

### GitHub Actions
```yaml
- name: Run Tests
  run: |
    pytest testing/ --cov=services --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

### 測試階段
1. **Lint & Format** - 代碼品質檢查
2. **Unit Tests** - 快速單元測試
3. **Integration Tests** - 整合測試
4. **System Tests** - 系統測試
5. **Coverage Report** - 覆蓋率報告

## 📊 測試報告

### 生成報告
```bash
# HTML覆蓋率報告
pytest testing/ --cov=services --cov-report=html

# XML覆蓋率報告 (CI/CD)
pytest testing/ --cov=services --cov-report=xml

# JUnit XML報告
pytest testing/ --junitxml=reports/junit.xml
```

### 報告位置
- `htmlcov/index.html` - HTML覆蓋率報告
- `coverage.xml` - XML覆蓋率報告
- `reports/junit.xml` - JUnit測試報告

---

**建立時間**: 2025-10-24  
**維護者**: QA Team  
**測試策略**: TDD + BDD + E2E