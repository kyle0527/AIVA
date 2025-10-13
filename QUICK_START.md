# AIVA 快速開始指南

## 環境設定

### 1. 設定 Python 路徑

```batch
# Windows
cd c:\D\E\AIVA\AIVA-main
.\setup_env.bat
```

### 2. 安裝依賴

```batch
pip install sqlalchemy alembic
# 或完整安裝
pip install -e .
```

### 3. 重啟 VS Code

關閉並重新開啟 VS Code 以載入新的配置。

## 執行測試

```batch
# 設定環境並執行測試
cd c:\D\E\AIVA\AIVA-main
set PYTHONPATH=c:\D\E\AIVA\AIVA-main
python -c "from services.aiva_common.schemas import AivaMessage; print('✅ 成功')"
```

## 已知問題與解決方案

### Mypy 警告: Cannot find module "services.aiva_common"

**解決方案**: 重啟 VS Code 並確保執行了 `setup_env.bat`

### ModuleNotFoundError: sqlalchemy

**解決方案**: `pip install sqlalchemy alembic`

## 四大模組導入範例

```python
# 設定 PYTHONPATH
import sys
sys.path.insert(0, 'c:\\D\\E\\AIVA\\AIVA-main')

# Core 模組
from services.core.aiva_core.app import app as core_app

# Scan 模組
from services.scan.aiva_scan.scan_orchestrator import ScanOrchestrator

# Function 模組
from services.function.function_idor.aiva_func_idor.worker import run

# Integration 模組（需要先安裝 sqlalchemy）
from services.integration.aiva_integration.app import app as integration_app
```

## Pydantic BaseModel 使用範例

```python
from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.function.function_idor.aiva_func_idor.vertical_escalation_tester import (
    VerticalTestResult,
    PrivilegeLevel
)

# 創建實例（自動驗證）
result = VerticalTestResult(
    vulnerable=True,
    confidence=Confidence.FIRM,
    severity=Severity.HIGH,
    vulnerability_type=VulnerabilityType.IDOR,
    actual_level=PrivilegeLevel.USER,
    attempted_level=PrivilegeLevel.ADMIN
)

# JSON 序列化
json_data = result.model_dump()

# JSON 反序列化
result2 = VerticalTestResult.model_validate(json_data)
```

## 開發最佳實踐

1. ✅ 總是使用 `from services.*` 完整路徑導入
2. ✅ 使用 Pydantic BaseModel 而非 dataclass
3. ✅ 使用現代類型提示: `str | None` 而非 `Union[str, None]`
4. ✅ 執行前設定 PYTHONPATH 環境變數
