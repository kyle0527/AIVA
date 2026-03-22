# function_bizlogic - 業務邏輯漏洞測試模組

> **版本**: v1.0.0 | **狀態**: ✅ 引擎完成，⬜ CLI 入口待接通 | **語言**: Python | **能力登錄**: ⬜ 待登錄

## 模組概述

業務邏輯漏洞測試模組，測試競爭條件、價格竄改、工作流程繞過等業務層面的安全問題。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 競爭條件測試 | ✅ 完成 | 並行請求分析、餘額競爭、優惠券重用 |
| 價格竄改測試 | ✅ 完成 | 負數價格、零元、溢位、參數竄改 |
| 工作流程繞過 | ✅ 完成 | 步驟跳過、直接結帳、支付繞過、驗證繞過 |
| 綜合掃描入口 | ✅ 完成 | BizLogicManager.comprehensive_scan() |
| CLI 入口接通 | ⬜ 待完成 | aiva_external_executor 尚未對應 |

## 架構

```
function_bizlogic/
├── race_condition_scanner.py     # 競爭條件（RaceConditionScanner）
├── price_manipulation_scanner.py # 價格竄改（PriceManipulationScanner）
├── workflow_bypass_scanner.py    # 工作流程繞過（WorkflowBypassScanner）
├── business_schemas.py           # 資料模型（AttackSurfaceAnalysis 等）
├── finding_helper.py             # 發現結果輔助工具
└── integration_tools/
    └── bizlogic_tools.py         # BizLogicManager（綜合入口）
```

## 執行方式

### 直接使用

```python
from services.features.function_bizlogic.race_condition_scanner import RaceConditionScanner
from services.features.function_bizlogic.price_manipulation_scanner import PriceManipulationScanner
from services.features.function_bizlogic.workflow_bypass_scanner import WorkflowBypassScanner

# 競爭條件
scanner = RaceConditionScanner(target_url="https://example.com")
results = await scanner.run_all_tests(test_endpoints=["/api/purchase", "/api/coupon"])

# 價格竄改
price = PriceManipulationScanner()
results = await price.run_all_tests("/api/checkout")

# 工作流程繞過
bypass = WorkflowBypassScanner()
results = await bypass.run_all_tests()
```

## 可調用方法（公開 API）

| 類別 | 方法 | 說明 |
|------|------|------|
| `RaceConditionScanner` | `test_concurrent_requests(endpoint, method, payload, concurrent_count)` | 競爭條件測試 |
| `RaceConditionScanner` | `test_balance_manipulation(...)` | 餘額競爭測試 |
| `RaceConditionScanner` | `test_coupon_reuse(coupon_endpoint, coupon_code)` | 優惠券重用測試 |
| `RaceConditionScanner` | `run_all_tests(test_endpoints)` | 執行所有競爭條件測試 |
| `PriceManipulationScanner` | `test_negative_price(endpoint, price_param)` | 負數價格測試 |
| `PriceManipulationScanner` | `test_zero_price(endpoint, price_param)` | 零元價格測試 |
| `PriceManipulationScanner` | `test_overflow_price(endpoint)` | 溢位價格測試 |
| `PriceManipulationScanner` | `run_all_tests(endpoint)` | 執行所有價格竄改測試 |
| `WorkflowBypassScanner` | `test_step_skipping(workflow_steps, skip_step_index)` | 步驟跳過測試 |
| `WorkflowBypassScanner` | `test_payment_bypass(order_endpoint, payment_endpoint)` | 支付繞過測試 |
| `WorkflowBypassScanner` | `run_all_tests()` | 執行所有工作流程繞過測試 |
| `BizLogicManager` | `comprehensive_scan(target_url, options)` | 綜合掃描入口 |

## 待完成工作

- 接通 `aiva_external_executor.py` 的 CLI 入口
- 將 `race_condition` / `price_manipulation` / `workflow_bypass` 新增至 `CAPABILITY_CONFIGS`

## 注意事項

- 僅限授權滲透測試使用
- 競爭條件測試可能對目標系統造成資料不一致，建議在測試環境執行
