# AIVA Services 測試套件

本目錄包含所有 `services/` 模組的測試文件。

## 📁 目錄結構

```
tests/services/
├── scan/              # 掃描引擎測試
│   ├── test_all_engines.py
│   ├── test_typescript_engine.py
│   ├── test_scanner.py
│   ├── test_ssrf_server.py
│   └── test_phase_loop.py
├── core/              # 核心系統測試
│   ├── conftest.py
│   ├── test_capability_analyzer.py
│   ├── test_capability_registry.py
│   ├── test_dual_write_integration.py
│   └── test_module_explorer.py
└── features/          # 功能模組測試
    └── test_detector.py
```

## 🚀 快速開始

### 💡 核心理念

**實際執行程式本身就是最好的驗證**，不需要額外創建測試腳本。直接運行功能模組來驗證其正確性：

```bash
# ✅ 最佳實踐：直接執行實際功能
python -m services.scan.engines.python_engine.scanner --target http://example.com
python -m services.features.function_sqli.detector --url http://test.com
python -m services.core.aiva_core.cognitive_core.bioneuron_decision_controller

# ✅ 次選：必要時執行測試套件
pytest tests/services/ -v

# 執行特定模組測試
pytest tests/services/scan/ -v
pytest tests/services/core/ -v
pytest tests/services/features/ -v
```

### 執行單個測試文件

```bash
# 掃描引擎測試
pytest tests/services/scan/test_all_engines.py -v

# 核心功能測試
pytest tests/services/core/test_capability_analyzer.py -v
```

### 生成覆蓋率報告

```bash
# 生成 HTML 覆蓋率報告
pytest tests/services/ --cov=services --cov-report=html

# 生成終端覆蓋率報告（顯示缺失行）
pytest tests/services/ --cov=services --cov-report=term-missing

# 查看報告
# 打開 htmlcov/index.html
```

## 📋 測試規範

### 1. 測試文件命名

- 測試文件必須以 `test_` 開頭
- 測試函數必須以 `test_` 開頭
- 測試類必須以 `Test` 開頭

```python
# ✅ 正確
def test_scan_engine_initialization():
    pass

class TestScanEngine:
    def test_detect_vulnerability(self):
        pass

# ❌ 錯誤
def scan_engine_test():  # 不會被 pytest 識別
    pass
```

### 2. 優先實際執行，謹慎編寫測試

**最佳實踐**: 在模組中添加 `if __name__ == "__main__"` 區塊直接執行驗證：

```python
# ✅ 最佳：在模組本身添加執行入口
class PythonScanner:
    def scan(self, target: str):
        # 實際掃描邏輯
        pass

if __name__ == "__main__":
    scanner = PythonScanner()
    result = scanner.scan("http://example.com")
    print(f"掃描完成: {result}")
    # 直接執行 python scanner.py 即可驗證

# ⚠️ 次選：必要時才寫測試導入
from services.scan.engines.python_engine import PythonScanner
from services.core.aiva_core import CapabilityAnalyzer
```

### 3. 測試數據隔離

每個測試應該獨立運行，不依賴其他測試的狀態：

```python
import pytest

@pytest.fixture
def clean_database():
    """每次測試前清理數據庫"""
    db = Database()
    db.clear()
    yield db
    db.close()

def test_insert_data(clean_database):
    # 使用乾淨的數據庫
    clean_database.insert({"key": "value"})
    assert clean_database.count() == 1
```

### 4. 實際測試 vs Mock 測試

**優先使用實際組件測試**，符合 aiva_common 規範：

```python
# ✅ 優先：測試實際組件
def test_rust_engine_real():
    engine = RustInfoGatherer()
    if engine.is_available():
        result = engine.gather_info("target")
        assert result is not None
    else:
        pytest.skip("Rust engine not available")

# ⚠️ 謹慎使用：僅在無法測試實際組件時使用 Mock
from unittest.mock import patch

@patch('services.scan.external_api')
def test_with_mock(mock_api):
    mock_api.return_value = {"status": "ok"}
    # 僅用於測試外部依賴不可用的情況
```

### 5. 測試斷言清晰化

使用清晰的斷言消息：

```python
# ✅ 正確：清晰的錯誤消息
def test_vulnerability_detection():
    result = scanner.scan("http://example.com")
    assert len(result.vulnerabilities) > 0, \
        f"應該檢測到漏洞，但得到 {len(result.vulnerabilities)} 個結果"

# ❌ 錯誤：沒有錯誤消息
def test_vulnerability_detection():
    result = scanner.scan("http://example.com")
    assert len(result.vulnerabilities) > 0  # 失敗時不知道原因
```

## 🔧 測試配置

### pytest.ini

在專案根目錄的 `pytest.ini` 中配置：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    slow: 標記為慢速測試
    integration: 標記為整合測試
    unit: 標記為單元測試
```

### 環境變量

測試時使用的環境變量：

```bash
# 設置測試環境
export AIVA_ENV=test
export AIVA_LOG_LEVEL=DEBUG

# 或使用 .env.test 文件
cp .env.example .env.test
```

## 📊 持續整合

### GitHub Actions 範例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest tests/services/ --cov=services --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 🐛 故障排除

### 常見問題

1. **導入錯誤**
   ```
   ModuleNotFoundError: No module named 'services'
   ```
   解決：確保從專案根目錄執行 pytest，或設置 PYTHONPATH：
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   pytest tests/services/
   ```

2. **測試數據庫衝突**
   ```
   Database locked
   ```
   解決：使用測試專用數據庫或臨時數據庫：
   ```python
   @pytest.fixture
   def temp_db(tmp_path):
       db_path = tmp_path / "test.db"
       return Database(db_path)
   ```

3. **異步測試失敗**
   ```
   RuntimeError: Event loop is closed
   ```
   解決：安裝 pytest-asyncio 並使用正確的標記：
   ```python
   import pytest
   
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_operation()
       assert result is not None
   ```

## 📚 相關文檔

- [aiva_common 測試規範](../../services/aiva_common/README.md#測試覆蓋與規範)
- [pytest 官方文檔](https://docs.pytest.org/)
- [pytest-cov 使用指南](https://pytest-cov.readthedocs.io/)

## 🤝 貢獻

添加新測試時：

1. ✅ 將測試文件放在 `tests/services/` 對應的模組目錄下
2. ✅ 使用 `test_` 前綴命名文件和函數
3. ✅ 使用絕對導入引用生產代碼
4. ✅ 執行測試確保通過：`pytest tests/services/your_test.py -v`
5. ✅ 檢查覆蓋率：`pytest --cov=services.your_module`
6. ❌ 不要在 `services/` 目錄下創建測試文件
