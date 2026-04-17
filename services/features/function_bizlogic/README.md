# function_bizlogic - 業務邏輯漏洞測試模組

> **版本**: v3.0.0 | **狀態**: ✅ 模組完成 | **語言**: Python

## 模組概述

業務邏輯漏洞測試模組，透過對業務流程進行深度探測，發掘無法被傳統掃描器識別的漏洞，例如競爭條件、價格竄改、工作流程繞過等業務層面的安全問題。

### 功能清單

| 功能 | 說明 |
|------|------|
| 競爭條件測試 | 並行請求分析、餘額競爭、優惠券重用漏洞測試 |
| 價格竄改測試 | 負數價格、零元、溢位、參數竄改測試 |
| 工作流程繞過 | 步驟跳過、直接結帳、支付繞過、驗證繞過測試 |
| 獨立 CLI 工具 | `__main__.py` 支援命令列介面直接呼叫所有測試模組 |

## 架構設計

```
function_bizlogic/
├── __main__.py                   # 獨立 CLI 執行入口
├── __init__.py                   # 模組入口匯出
├── race_condition_scanner.py     # 競爭條件偵測器 (RaceConditionScanner)
├── price_manipulation_scanner.py # 價格竄改偵測器 (PriceManipulationScanner)
├── workflow_bypass_scanner.py    # 工作流程繞過偵測器 (WorkflowBypassScanner)
├── business_schemas.py           # 資料模型 (AttackSurfaceAnalysis 等)
├── finding_helper.py             # 發現結果生成輔助工具
└── integration_tools/
    └── bizlogic_tools.py         # BizLogicManager (Python 綜合入口)
```

## 執行方式

### 透過獨立 CLI 執行 (推薦)

透過 CLI 可以直接以 JSON 格式取得漏洞掃描結果，非常適合腳本或外部整合：

```bash
# 執行價格操控測試
python -m services.features.function_bizlogic price --url "https://example.com" --endpoint "/api/checkout"

# 執行競態條件測試
python -m services.features.function_bizlogic race --url "https://example.com"

# 執行流程繞過測試
python -m services.features.function_bizlogic workflow --url "https://example.com"
```

### 作為 Python 模組匯入

可以匯入掃描器或使用統一的 `BizLogicManager` 來整合入其他專案：

```python
from services.features.function_bizlogic import BizLogicManager

manager = BizLogicManager()
# 執行完整業務邏輯漏洞掃描
results = manager.comprehensive_scan(
    target_url="https://example.com",
    options={"auth_token": "Bearer token"}
)
```

## 可調用方法（內部 API）

| 類別 | 方法 | 說明 |
|------|------|------|
| `RaceConditionScanner` | `test_concurrent_requests()` | 競爭條件測試 |
| `RaceConditionScanner` | `test_balance_manipulation(...)` | 餘額競爭測試 |
| `RaceConditionScanner` | `test_coupon_reuse(...)` | 優惠券重用測試 |
| `PriceManipulationScanner` | `test_negative_price(...)` | 負數價格測試 |
| `PriceManipulationScanner` | `test_zero_price(...)` | 零元價格測試 |
| `PriceManipulationScanner` | `test_overflow_price(...)` | 溢位價格測試 |
| `WorkflowBypassScanner` | `test_step_skipping(...)` | 步驟跳過測試 |
| `WorkflowBypassScanner` | `test_payment_bypass(...)` | 支付繞過測試 |
| `BizLogicManager` | `comprehensive_scan(...)` | 綜合掃描入口 |

## 注意事項

- ⚠️ 競爭條件測試對系統負載極高，請注意請求量。
- 測試可能對目標系統造成資料不一致 (如產生實際的付款流程修改)，**強烈建議在測試環境執行**。
- 僅限授權滲透測試使用。
