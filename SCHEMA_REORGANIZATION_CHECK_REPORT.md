# Schema 重組檢查報告

## 執行摘要

已完成 schema 定義的模組化重組，但發現部分類別**未被遷移**到新的 models.py 文件中。

---

## 📊 完成狀態

### ✅ 已成功遷移的模組

| 模組 | 文件 | 行數 | 狀態 |
|------|------|------|------|
| aiva_common | models.py | 248 | ✅ 完成 |
| scan | models.py | 338 | ✅ 完成 |
| function | models.py | 368 | ✅ 完成 |
| integration | models.py | 143 | ✅ 完成 |
| core | models.py | 522 | ✅ 完成 |
| core | ai_models.py | 391 | ✅ 完成 |
| **總計** | **6 文件** | **2,010 行** | **✅ 83% 完成** |

---

## ⚠️ 缺失的類別 (Critical)

以下類別在**舊 schemas.py 中存在**，但**未遷移**到新 models.py：

### 1. 掃描相關 (Scan Module 應包含)

| 類別名稱 | 行號 | 用途 | 建議遷移目標 |
|---------|------|------|-------------|
| `Authentication` | 50 | 認證信息 | `aiva_common/models.py` |
| `Fingerprints` | 124 | 指紋識別 | `scan/models.py` |
| `JavaScriptAnalysisResult` | 616 | JS分析結果 | `scan/models.py` |
| `AssetInventoryItem` | 2342 | 資產清單項 | `scan/models.py` |

**影響文件**:
- `services/scan/aiva_scan/authentication_manager.py` (使用 `Authentication`)
- `services/scan/aiva_scan/fingerprint_manager.py` (使用 `Fingerprints`)
- `services/scan/aiva_scan/info_gatherer/passive_fingerprinter.py` (使用 `Fingerprints`)
- `services/scan/aiva_scan/javascript_analyzer.py` (使用 `JavaScriptAnalysisResult`)

### 2. 基礎類別 (需要檢查)

| 類別名稱 | 可能位置 | 建議處理 |
|---------|---------|---------|
| `BaseModel` | Pydantic | 應從 `pydantic` 導入 |
| `ScanResult` | 未找到 | 需要在 schemas.py 中搜索 |
| `AssetInventory` | 未找到 | 需要在 schemas.py 中搜索 |
| `ConfigurationData` | 未找到 | 需要在 schemas.py 中搜索 |
| `IOCRecord` | 未找到 | 應在 integration/models.py |
| `RiskAssessment` | 未找到 | 應在 core/models.py |
| `SystemStatus` | 未找到 | 應在 core/models.py |
| `TargetInfo` | 未找到 | 應在 scan/models.py |
| `TechStackInfo` | 未找到 | 應在 scan/models.py |
| `ServiceInfo` | 未找到 | 應在 scan/models.py |
| `ScopeDefinition` | 未找到 | 應在 scan/models.py |
| `TestResult` | 未找到 | 應在 function/models.py |
| `ThreatIndicator` | 未找到 | 應在 integration/models.py |
| `VulnerabilityFinding` | 未找到 | 應在 scan/models.py |

---

## 🔍 詳細檢查結果

### 使用 `from services.aiva_common.schemas import` 的文件統計

共找到 **50+** 個文件仍在使用舊的導入路徑。

#### Scan 模組 (12 個文件)
```
✗ services/scan/aiva_scan/worker_refactored.py
✗ services/scan/aiva_scan/worker.py
✗ services/scan/aiva_scan/sensitive_data_scanner.py
✗ services/scan/aiva_scan/scope_manager.py
✗ services/scan/aiva_scan/scan_orchestrator.py
✗ services/scan/aiva_scan/scan_context.py
✗ services/scan/aiva_scan/javascript_analyzer.py
✗ services/scan/aiva_scan/info_gatherer/passive_fingerprinter.py
✗ services/scan/aiva_scan/fingerprint_manager.py
✗ services/scan/aiva_scan/dynamic_engine/dynamic_content_extractor.py
✗ services/scan/aiva_scan/dynamic_engine/ajax_api_handler.py
✗ services/scan/aiva_scan/core_crawling_engine/static_content_parser.py
✗ services/scan/aiva_scan/authentication_manager.py
```

#### Function 模組 (估計 30+ 個文件)
```
✗ services/function/function_xss/** (多個文件)
✗ services/function/function_ssrf/** (多個文件)
✗ services/function/function_sqli/** (多個文件)
✗ services/function/function_idor/** (可能存在)
✗ services/function/function_postex/** (可能存在)
```

#### Integration 模組 (1 個文件)
```
✗ services/integration/api_gateway/api_gateway/app.py
```

---

## 🚨 立即需要的行動

### Priority 1: 補充缺失的類別

#### 1.1 補充到 `aiva_common/models.py`

需要從 schemas.py (行 50) 遷移 `Authentication` 類：

```python
class Authentication(BaseModel):
    """認證信息"""
    auth_type: str = Field(description="認證類型")
    credentials: dict[str, Any] = Field(default_factory=dict)
    # ... (完整定義需要從 schemas.py 複製)
```

#### 1.2 補充到 `scan/models.py`

需要遷移以下類別：

1. **Fingerprints** (行 124)
```python
class Fingerprints(BaseModel):
    """技術指紋識別"""
    technologies: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    # ... (完整定義需要從 schemas.py 複製)
```

2. **JavaScriptAnalysisResult** (行 616)
```python
class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""
    # ... (完整定義需要從 schemas.py 複製)
```

3. **AssetInventoryItem** (行 2342)
```python
class AssetInventoryItem(BaseModel):
    """資產清單項"""
    # ... (完整定義需要從 schemas.py 複製)
```

4. 其他缺失類別 (需要在 schemas.py 中搜索)

---

### Priority 2: 創建向後兼容層

在 `services/aiva_common/schemas.py` 中創建重新導出：

```python
"""
AIVA Common Schemas - 向後兼容層

此文件重新導出所有已遷移到各模組 models.py 的類別。
⚠️ 此文件將逐步棄用，新代碼請直接從各模組的 models.py 導入。

遷移狀態:
- ✅ 基礎設施類 → aiva_common.models
- ✅ 掃描相關類 → scan.models
- ✅ 功能測試類 → function.models
- ✅ 集成相關類 → integration.models
- ✅ 核心業務類 → core.models
- ✅ AI 系統類 → core.ai_models
"""

# 從新位置重新導出所有類別
from .models import *  # aiva_common 基礎設施
from services.scan.models import *  # 掃描模組
from services.function.models import *  # 功能測試模組
from services.integration.models import *  # 集成模組
from services.core.models import *  # 核心業務邏輯
from services.core.ai_models import *  # AI 系統

# 明確列出所有導出的類別
__all__ = [
    # ... (完整列表)
]
```

**注意**: 這個方法可以讓現有代碼**無需修改**立即工作。

---

### Priority 3: 逐步更新導入路徑 (可選，建議後續執行)

詳見 `SCHEMA_IMPORT_MIGRATION_PLAN.md`

---

## 📋 執行檢查清單

### 立即執行 (今天)

- [ ] 從 `schemas.py` 複製缺失的類別定義
  - [ ] `Authentication` → `aiva_common/models.py`
  - [ ] `Fingerprints` → `scan/models.py`
  - [ ] `JavaScriptAnalysisResult` → `scan/models.py`
  - [ ] `AssetInventoryItem` → `scan/models.py`
- [ ] 在 `schemas.py` 中搜索其他缺失類別
  - [ ] `ScanResult`
  - [ ] `AssetInventory`
  - [ ] `ConfigurationData`
  - [ ] `IOCRecord`
  - [ ] `RiskAssessment`
  - [ ] `SystemStatus`
  - [ ] `TargetInfo`
  - [ ] `TechStackInfo`
  - [ ] `ServiceInfo`
  - [ ] `ScopeDefinition`
  - [ ] `TestResult`
  - [ ] `ThreatIndicator`
  - [ ] `VulnerabilityFinding`
- [ ] 遷移所有缺失類別到對應模組
- [ ] 更新各模組的 `__init__.py` 導出新增的類別
- [ ] 創建向後兼容層 (schemas.py 重新導出)

### 短期執行 (1-2 天)

- [ ] 測試兼容層是否正常工作
- [ ] 運行掃描模組測試
- [ ] 運行功能測試模組測試
- [ ] 修復發現的導入錯誤

### 中期執行 (1 週)

- [ ] 開始逐步更新導入路徑 (按照遷移計劃)
- [ ] 更新文檔
- [ ] 添加 deprecation warnings

---

## 📊 完整性檢查

### 原始 schemas.py 分析

需要執行以下命令檢查原始文件的所有類別：

```powershell
# 列出所有類別定義
Select-String -Path "c:\AMD\AIVA\services\aiva_common\schemas.py" -Pattern "^class \w+" |
    Select-Object LineNumber, Line |
    Format-Table -AutoSize
```

然後逐一檢查每個類別是否已遷移。

---

## 🎯 預期結果

完成上述行動後：

1. ✅ **100% 覆蓋**: 所有 schemas.py 中的類別都已遷移
2. ✅ **向後兼容**: 現有代碼無需修改可正常工作
3. ✅ **模組化**: 新代碼使用清晰的模組化導入
4. ✅ **無破壞**: 不影響現有功能

---

## 📝 建議

### 建議 1: 先完成補充，再考慮重構

**原因**: 避免破壞現有功能

**步驟**:
1. 補充所有缺失類別到對應 models.py
2. 創建兼容層確保現有代碼工作
3. 測試驗證
4. (可選) 逐步更新導入路徑

### 建議 2: 使用自動化工具

可以編寫腳本自動：
1. 掃描 schemas.py 中所有類別
2. 檢查哪些已遷移，哪些未遷移
3. 生成遷移報告

---

**生成時間**: 2025-10-15
**當前狀態**: ⚠️ **不完整 - 需要補充缺失類別**
**下一步**: 補充缺失類別並創建兼容層
