# 🧪 AIVA 通用測試工具和配置

這個目錄包含 AIVA 項目的通用測試工具、共享配置和實用程式。

## 📁 目錄結構

```
testing/
├── common/                 # 通用測試工具和配置 (本目錄)
├── core/                   # 核心功能測試
├── features/               # 功能特性測試
├── integration/            # 集成測試
├── performance/            # 性能測試
├── scan/                   # 掃描和滲透測試
└── README.md
```

## 📄 本目錄檔案說明

### 測試工具
- **`api_testing.py`** - API 測試工具和實用程式
- **`module_connectivity_tester.py`** - 模組連接性測試工具
- **`security_test.py`** - 安全性測試工具

### Schema 和配置測試
- **`test_schema_codegen_converters.py`** - Schema 代碼生成和轉換器測試
- **`test_unified_storage_config.py`** - 統一存儲配置測試
- **`test_vector_storage.py`** - 向量存儲測試

### 測試配置
- **`conftest_converters.py`** - 轉換器相關的 pytest 配置和 fixtures

## 🚀 使用方法

### 執行通用測試
```bash
# 執行所有通用測試
pytest testing/common/ -v

# 執行特定測試檔案
pytest testing/common/test_schema_codegen_converters.py -v

# 執行 API 測試
python testing/common/api_testing.py

# 執行安全測試
python testing/common/security_test.py
```

### 模組連接測試
```bash
# 測試模組連接性
python testing/common/module_connectivity_tester.py
```

## 🛠️ 測試工具功能

### API 測試工具 (`api_testing.py`)
- RESTful API 端點測試
- API 響應驗證
- 認證和授權測試

### 模組連接測試器 (`module_connectivity_tester.py`)
- 模組間依賴檢查
- 服務連接性驗證
- 網絡連接測試

### 安全測試 (`security_test.py`)
- 安全漏洞檢測
- 權限驗證
- 資料安全測試

### Schema 測試
- **代碼生成測試**: 驗證從 schema 生成多語言代碼
- **存儲配置測試**: 測試統一存儲配置
- **向量存儲測試**: 驗證向量存儲功能

## 📝 編寫新測試

### 測試檔案命名規範
- 測試檔案: `test_*.py`
- 測試類: `Test*`
- 測試方法: `test_*`

### 使用共享 Fixtures
```python
# 在測試檔案中使用 conftest_converters.py 中的 fixtures
def test_my_function(converter_instance, sample_schemas):
    result = converter_instance.convert(sample_schemas["test_schema"])
    assert result is not None
```

### 添加新的通用工具
1. 創建新的測試工具檔案
2. 確保檔案包含適當的文檔字符串
3. 更新本 README 檔案
4. 如需要，更新 `conftest_converters.py` 添加新的 fixtures

## 🔧 配置說明

### Pytest 配置
- **`conftest_converters.py`** 提供轉換器相關的測試配置
- 包含預定義的 fixtures 用於常見測試場景
- 支持異步測試和 mock 配置

## 📊 測試覆蓋率

建議的測試覆蓋率目標：
- **通用工具**: > 90%
- **API 測試**: > 85%
- **Schema 測試**: > 95%
- **安全測試**: > 80%

## 🔄 持續集成

這些通用測試工具在 CI/CD 流程中的角色：
1. **前置檢查**: 模組連接性測試
2. **功能驗證**: Schema 和配置測試
3. **安全掃描**: 安全測試工具
4. **API 驗證**: API 測試套件

---

**目錄更新**: 2025-11-12  
**維護者**: AIVA QA Team  
**測試策略**: 通用工具 + 共享配置