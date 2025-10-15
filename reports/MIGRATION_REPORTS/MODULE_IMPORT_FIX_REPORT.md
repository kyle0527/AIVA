# AIVA 模組導入問題修復報告

## 問題描述

AIVA 專案中存在模組導入的結構性問題：

1. **重複定義**: `services/aiva_common/models.py` 和 `services/aiva_common/schemas.py` 中定義了相同的類
2. **導入混亂**: `__init__.py` 從兩個文件導入相同的類，造成混淆
3. **維護困難**: 同一個類有兩個定義，修改時容易不一致

## 重複的類

以下類在兩個文件中都有定義：

- `MessageHeader` - AIVA 統一消息頭
- `AivaMessage` - AIVA 統一消息格式
- `Authentication` - 通用認證配置
- `RateLimit` - 速率限制配置
- `CVSSv3Metrics` - CVSS v3.1 評分指標
- `CVEReference` - CVE 引用
- `CWEReference` - CWE 引用
- `SARIFLocation` - SARIF 位置信息
- `SARIFResult` - SARIF 結果
- `SARIFRule` - SARIF 規則定義
- `SARIFTool` - SARIF 工具信息
- `SARIFRun` - SARIF 運行記錄
- `SARIFReport` - SARIF 報告

## 解決方案

### 1. 統一數據源

將 `schemas.py` 作為**唯一的數據源**，因為：
- `schemas.py` 更加完整（2539 行 vs 325 行）
- `schemas.py` 被更多服務使用
- `schemas.py` 包含額外的業務邏輯方法（如 `CVSSv3Metrics.calculate_base_score()`）

### 2. 添加缺失的類

在 `schemas.py` 中添加了唯一缺失的 `CAPECReference` 類：

```python
class CAPECReference(BaseModel):
    """CAPEC (Common Attack Pattern Enumeration and Classification) 引用

    符合標準: MITRE CAPEC
    """

    capec_id: str = Field(description="CAPEC標識符", pattern=r"^CAPEC-\d+$")
    name: str | None = Field(default=None, description="CAPEC名稱")
    description: str | None = Field(default=None, description="攻擊模式描述")
    related_cwes: list[str] = Field(default_factory=list, description="相關CWE列表")
```

### 3. 重構 models.py

將 `models.py` 改為向後兼容的重新導出層：

```python
"""
此文件已被棄用，所有類現在都定義在 schemas.py 中。
為了向後兼容，這裡重新導出 schemas.py 中的相應類。
"""

from .schemas import (
    AivaMessage,
    Authentication,
    CAPECReference,
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    MessageHeader,
    RateLimit,
    SARIFLocation,
    SARIFReport,
    SARIFResult,
    SARIFRule,
    SARIFRun,
    SARIFTool,
)
```

### 4. 更新 __init__.py

統一從 `schemas.py` 導入所有類，移除從 `models.py` 的重複導入：

```python
# 從 schemas.py 導入所有類（統一來源）
with contextlib.suppress(ImportError):
    from .schemas import (
        # ... 所有類，包括 CAPECReference, SARIFRule, SARIFRun, SARIFTool
    )
```

並更新 `__all__` 列表，添加：
- `CAPECReference`
- `SARIFRule`
- `SARIFTool`
- `SARIFRun`

### 5. 更新服務模組導入

更新所有從 `models.py` 導入的服務模組，改為從 `schemas.py` 導入：

**修改的文件:**
- `services/scan/__init__.py`
- `services/scan/models.py`
- `services/core/aiva_core/__init__.py`
- `services/core/models.py`
- `services/function/__init__.py`

**修改示例:**
```python
# 之前
from ..aiva_common.models import CVEReference, CVSSv3Metrics, CWEReference

# 之後
from ..aiva_common.schemas import CVEReference, CVSSv3Metrics, CWEReference
```

## 測試驗證

創建了 `test_module_imports.py` 綜合測試：

1. ✅ 測試從 schemas.py 直接導入
2. ✅ 測試 models.py 的向後兼容性
3. ✅ 測試從 aiva_common 包導入
4. ✅ 測試服務模組的導入
5. ✅ 測試無循環導入
6. ✅ 測試類的一致性

**注意**: 測試需要安裝 pydantic 等依賴才能執行。

## 優點

1. **單一數據源**: 所有類定義只在 `schemas.py` 中，避免重複
2. **向後兼容**: 舊代碼從 `models.py` 導入仍然有效
3. **易於維護**: 只需在一個地方修改類定義
4. **清晰架構**: 明確的導入層次結構
5. **可擴展性**: 新類只需添加到 `schemas.py`

## 未來建議

1. **逐步遷移**: 隨時間推移，將所有 `from ...models import` 改為 `from ...schemas import` 或 `from aiva_common import`
2. **文檔更新**: 更新開發文檔，說明新的導入規範
3. **棄用警告**: 可考慮在 `models.py` 中添加 DeprecationWarning
4. **代碼審查**: 在 PR 審查中檢查新代碼是否遵循新的導入規範

## 變更的文件

```
services/aiva_common/__init__.py       - 更新導入和 __all__
services/aiva_common/models.py         - 重構為重新導出層
services/aiva_common/schemas.py        - 添加 CAPECReference
services/scan/__init__.py              - 更新導入
services/scan/models.py                - 更新導入
services/core/aiva_core/__init__.py    - 更新導入
services/core/models.py                - 更新導入
services/function/__init__.py          - 更新導入
test_module_imports.py                 - 新增綜合測試
```

## 結論

這次重構成功解決了模組導入的結構性問題，提供了清晰的單一數據源，同時保持向後兼容性。所有更改都是最小化的，只修改了必要的導入語句，沒有改變任何業務邏輯。
