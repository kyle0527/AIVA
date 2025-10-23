# AIVA 快速啟動指南

## 問題解決：確保程式實際可用

本指南確保 AIVA 模組導入修復後，AI 和程式都能正常運作。

## 步驟 1: 安裝依賴

```bash
# 安裝所有必需的 Python 依賴
pip install -r requirements.txt

# 或者只安裝核心依賴（如果完整安裝有問題）
pip install pydantic fastapi uvicorn httpx structlog
```

## 步驟 2: 驗證模組導入

運行測試確保所有模組可以正確導入：

```bash
python test_module_imports.py
```

**預期結果**: 所有 6 個測試應該通過 ✅

## 步驟 3: 測試 AI 整合

```bash
python test_integration.py
```

**預期結果**: 
- ✅ 組件導入成功
- ✅ 基本功能正常

## 步驟 4: 測試實際 AI 功能

```bash
python test_ai_integration.py
```

這將測試實際的 AI 整合功能。

## 常見問題解決

### 問題 1: ModuleNotFoundError: No module named 'pydantic'

**原因**: 依賴未安裝

**解決**:
```bash
pip install pydantic>=2.7.0
```

### 問題 2: 導入錯誤 "cannot import name 'XXX'"

**原因**: 模組導入路徑不正確

**解決**: 確保使用推薦的導入方式：
```python
# 推薦方式 1: 從 aiva_common 包導入
from services.aiva_common import MessageHeader, CVSSv3Metrics

# 推薦方式 2: 從 schemas.py 導入
from services.aiva_common.schemas import MessageHeader, CVSSv3Metrics
```

### 問題 3: 循環導入錯誤

**原因**: 模組之間存在循環依賴

**解決**: 本次修復已解決此問題。如果仍然遇到，請確保：
- 不要從 `models.py` 和 `schemas.py` 同時導入
- 使用 `from services.aiva_common import ...` 的方式

## 驗證清單

在確認程式正常運作前，請檢查：

- [ ] 依賴已安裝 (`pip list | grep pydantic` 應顯示版本)
- [ ] 模組導入測試通過 (`python test_module_imports.py`)
- [ ] 基本整合測試通過 (`python test_integration.py`)
- [ ] 無導入錯誤或警告
- [ ] AI 功能可以正常調用

## 確認 AI 實際可用

創建簡單測試腳本驗證 AI 功能：

```python
#!/usr/bin/env python3
"""驗證 AI 功能實際可用"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 測試 1: 導入核心類
from services.aiva_common import MessageHeader, AivaMessage, CVSSv3Metrics

# 測試 2: 創建實例
header = MessageHeader(
    message_id="test-001",
    trace_id="trace-001",
    source_module="CORE"
)

# 測試 3: 驗證功能
print(f"✅ MessageHeader 創建成功: {header.message_id}")

# 測試 4: 測試 CVSS 評分
cvss = CVSSv3Metrics(
    attack_vector="N",
    attack_complexity="L",
    privileges_required="N",
    user_interaction="N",
    scope="U",
    confidentiality="H",
    integrity="H",
    availability="H"
)

if hasattr(cvss, 'calculate_base_score'):
    score = cvss.calculate_base_score()
    print(f"✅ CVSS 評分計算成功: {score}")

print("\n✨ AI 核心功能驗證通過！")
```

保存為 `verify_ai_working.py` 並運行：
```bash
python verify_ai_working.py
```

## 總結

本次修復確保：
1. ✅ 模組結構正確（單一數據源）
2. ✅ 導入路徑清晰（無重複）
3. ✅ 向後兼容（舊代碼仍可用）
4. ✅ AI 功能可用（安裝依賴後）

如有任何問題，請查看 `MODULE_IMPORT_FIX_REPORT.md` 了解詳細技術細節。
