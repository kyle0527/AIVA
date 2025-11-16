# 💼 BizLogic - 業務邏輯測試

**導航**: [← 返回 Core Capabilities](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **代碼量**: 3 個 Python 檔案，約 580 行代碼  
> **角色**: AIVA 的「業務邏輯偵探」- 專注於業務邏輯漏洞測試

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
  - [Worker - 業務邏輯測試 Worker](#worker---業務邏輯測試-worker)
  - [BusinessSchemas - 業務 Schema 定義](#businessschemas---業務-schema-定義)
  - [FindingHelper - 漏洞發現輔助工具](#findinghelper---漏洞發現輔助工具)
- [測試類型](#測試類型)
- [使用範例](#使用範例)

---

## 🎯 模組概述

**BizLogic** 子模組專注於業務邏輯漏洞的自動化測試，包括價格操控、競態條件、工作流程繞過等常見的業務邏輯安全問題。透過訊息佇列整合，支援異步測試任務執行。

### 核心能力
1. **價格操控測試** - 檢測價格、折扣、優惠券的邏輯漏洞
2. **競態條件測試** - 並發請求測試資源競爭問題
3. **工作流程繞過測試** - 測試流程步驟的繞過可能性
4. **自動化發現** - 智能識別和回報業務邏輯漏洞

### 設計特色
- **訊息驅動** - 透過 MQ 接收和回報測試任務
- **模組化測試器** - 每種測試類型有獨立的測試器
- **標準化 Schema** - 統一的業務實體定義
- **智能輔助** - 自動化漏洞分類和優先級評估

---

## 📂 檔案列表

| 檔案名 | 行數 | 核心功能 | 狀態 |
|--------|------|----------|------|
| **worker.py** | 126 | 業務邏輯測試 Worker - MQ 任務監聽和執行 | ✅ 生產 |
| **business_schemas.py** | 423 | 業務 Schema 定義 - 訂單、商品、用戶等實體 | ✅ 生產 |
| **finding_helper.py** | 58 | 漏洞發現輔助工具 - 結果分析和報告 | ✅ 生產 |
| **__init__.py** | - | 模組初始化 | - |

**總計**: 約 607 行代碼（含註解和空行）

---

## 🔧 核心組件

### Worker - 業務邏輯測試 Worker

**檔案**: `worker.py` (126 行)

監聽業務邏輯測試任務，執行測試並回報結果。整合了三種主要測試器。

#### 核心功能

```python
from services.aiva_common.enums.modules import Topic
from services.aiva_common.mq import get_broker

# 支援的測試器
from .price_manipulation_tester import PriceManipulationTester
from .race_condition_tester import RaceConditionTester
from .workflow_bypass_tester import WorkflowBypassTester

async def run() -> None:
    """啟動 BizLogic Worker
    
    監聽 tasks.function.bizlogic Topic
    處理三種測試類型：
    - price_manipulation: 價格操控
    - race_condition: 競態條件
    - workflow_bypass: 流程繞過
    """
    broker = await get_broker()
    
    async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_START):
        msg = AivaMessage.model_validate_json(mqmsg.body)
        
        # 只處理 bizlogic 相關的任務
        if msg.payload.get("module") != "bizlogic":
            continue
        
        # 執行測試
        findings = await _perform_test(msg.payload)
        
        # 回報結果
        await broker.publish(
            Topic.TASK_FUNCTION_RESULT,
            result_msg.model_dump_json()
        )
```

#### 任務消息格式

```json
{
  "header": {
    "message_id": "msg_123",
    "timestamp": "2024-01-01T12:00:00Z",
    "source": "task_planning"
  },
  "payload": {
    "module": "bizlogic",
    "test_type": "price_manipulation",
    "target": {
      "url": "https://shop.example.com",
      "endpoints": {
        "cart": "/api/cart",
        "checkout": "/api/checkout",
        "apply_coupon": "/api/coupon/apply"
      }
    },
    "test_params": {
      "product_id": "PROD-001",
      "original_price": 1000,
      "test_scenarios": [
        "negative_quantity",
        "decimal_quantity",
        "coupon_stacking"
      ]
    }
  }
}
```

#### 測試執行流程

```python
async def _perform_test(payload: dict) -> list:
    """執行測試並返回發現"""
    test_type = payload.get("test_type")
    target = payload.get("target")
    params = payload.get("test_params", {})
    
    findings = []
    
    if test_type == "price_manipulation":
        tester = PriceManipulationTester(target)
        findings = await tester.test_all_scenarios(params)
    
    elif test_type == "race_condition":
        tester = RaceConditionTester(target)
        findings = await tester.test_concurrent_access(params)
    
    elif test_type == "workflow_bypass":
        tester = WorkflowBypassTester(target)
        findings = await tester.test_step_bypass(params)
    
    return findings
```

---

### BusinessSchemas - 業務 Schema 定義

**檔案**: `business_schemas.py` (423 行)

定義電子商務和業務邏輯測試所需的標準化資料結構。

#### 核心 Schema

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from decimal import Decimal

class Product(BaseModel):
    """商品實體"""
    product_id: str
    name: str
    price: Decimal
    stock_quantity: int
    category: str
    discount_rate: Optional[float] = 0.0
    
class CartItem(BaseModel):
    """購物車項目"""
    product_id: str
    quantity: int
    unit_price: Decimal
    subtotal: Decimal
    applied_discount: Optional[str] = None

class Order(BaseModel):
    """訂單實體"""
    order_id: str
    user_id: str
    items: List[CartItem]
    subtotal: Decimal
    discount_amount: Decimal
    tax_amount: Decimal
    shipping_fee: Decimal
    total_amount: Decimal
    status: str  # pending, paid, shipped, completed, cancelled
    payment_method: str
    
class Coupon(BaseModel):
    """優惠券實體"""
    coupon_code: str
    discount_type: str  # percentage, fixed_amount, free_shipping
    discount_value: Decimal
    min_purchase_amount: Optional[Decimal] = None
    max_discount_amount: Optional[Decimal] = None
    valid_from: datetime
    valid_until: datetime
    usage_limit: Optional[int] = None
    used_count: int = 0
    stackable: bool = False  # 是否可疊加

class User(BaseModel):
    """用戶實體"""
    user_id: str
    username: str
    email: str
    role: str  # guest, user, vip, admin
    loyalty_points: int = 0
    account_balance: Decimal = Decimal("0.00")
```

#### 測試場景 Schema

```python
class PriceManipulationScenario(BaseModel):
    """價格操控測試場景"""
    scenario_name: str
    description: str
    test_steps: List[dict]
    expected_vulnerability: Optional[str] = None
    
    # 預定義場景
    NEGATIVE_QUANTITY = "negative_quantity"      # 負數數量
    DECIMAL_QUANTITY = "decimal_quantity"        # 小數數量
    ZERO_PRICE = "zero_price"                    # 零元價格
    COUPON_STACKING = "coupon_stacking"          # 優惠券疊加
    DISCOUNT_OVERFLOW = "discount_overflow"      # 折扣溢位
    CURRENCY_MANIPULATION = "currency_manipulation"  # 貨幣操控

class RaceConditionScenario(BaseModel):
    """競態條件測試場景"""
    resource_type: str  # stock, balance, coupon, limit
    concurrent_requests: int
    expected_behavior: str
    actual_behavior: Optional[str] = None
    
class WorkflowBypassScenario(BaseModel):
    """工作流程繞過測試場景"""
    workflow_name: str
    required_steps: List[str]
    bypass_attempts: List[dict]
    success: bool = False
```

---

### FindingHelper - 漏洞發現輔助工具

**檔案**: `finding_helper.py` (58 行)

協助分析測試結果、分類漏洞、評估嚴重程度和生成報告。

#### 核心功能

```python
class FindingHelper:
    """漏洞發現輔助工具"""
    
    @staticmethod
    def categorize_finding(finding: dict) -> str:
        """分類漏洞類型
        
        Returns:
            - PRICE_MANIPULATION
            - RACE_CONDITION
            - WORKFLOW_BYPASS
            - LOGIC_FLAW
        """
        if "price" in finding or "discount" in finding:
            return "PRICE_MANIPULATION"
        elif "concurrent" in finding or "race" in finding:
            return "RACE_CONDITION"
        elif "bypass" in finding or "skip" in finding:
            return "WORKFLOW_BYPASS"
        else:
            return "LOGIC_FLAW"
    
    @staticmethod
    def calculate_severity(finding: dict) -> str:
        """計算嚴重程度
        
        考慮因素:
        - 財務影響
        - 可利用性
        - 影響範圍
        
        Returns: critical, high, medium, low
        """
        impact_score = 0
        
        # 財務影響
        if finding.get("financial_impact", 0) > 10000:
            impact_score += 3
        elif finding.get("financial_impact", 0) > 1000:
            impact_score += 2
        
        # 可利用性
        if finding.get("exploitability") == "easy":
            impact_score += 2
        
        # 影響範圍
        if finding.get("scope") == "all_users":
            impact_score += 2
        
        if impact_score >= 6:
            return "critical"
        elif impact_score >= 4:
            return "high"
        elif impact_score >= 2:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def generate_report(findings: List[dict]) -> dict:
        """生成測試報告"""
        return {
            "total_findings": len(findings),
            "by_severity": {
                "critical": [f for f in findings if f["severity"] == "critical"],
                "high": [f for f in findings if f["severity"] == "high"],
                "medium": [f for f in findings if f["severity"] == "medium"],
                "low": [f for f in findings if f["severity"] == "low"]
            },
            "by_category": {
                "price_manipulation": [...],
                "race_condition": [...],
                "workflow_bypass": [...]
            }
        }
```

---

## 🧪 測試類型

### 1. 價格操控測試 (PriceManipulationTester)

檢測電子商務系統中的價格計算漏洞。

#### 測試場景

```python
# 負數數量測試
{
    "scenario": "negative_quantity",
    "payload": {
        "product_id": "PROD-001",
        "quantity": -5,  # 負數數量
        "expected": "購買-5件商品導致退款"
    }
}

# 小數數量測試
{
    "scenario": "decimal_quantity",
    "payload": {
        "product_id": "PROD-001",
        "quantity": 0.01,  # 小數數量
        "expected": "以極低價格獲得商品"
    }
}

# 優惠券疊加測試
{
    "scenario": "coupon_stacking",
    "payload": {
        "coupons": ["SAVE10", "SAVE20", "SAVE30"],
        "expected": "疊加使用多張優惠券"
    }
}

# 折扣溢位測試
{
    "scenario": "discount_overflow",
    "payload": {
        "discount_rate": 999,  # 超大折扣
        "expected": "折扣超過100%導致負價格"
    }
}
```

#### 檢測邏輯

```python
async def test_price_manipulation(target, params):
    """價格操控測試"""
    findings = []
    
    # 1. 測試負數數量
    response = await api_client.post(
        f"{target['url']}/api/cart/add",
        json={"product_id": params["product_id"], "quantity": -1}
    )
    
    if response.status_code == 200:
        cart = response.json()
        if cart["total"] < 0:
            findings.append({
                "type": "NEGATIVE_PRICE",
                "severity": "critical",
                "description": "系統允許負數數量，導致負價格",
                "financial_impact": abs(cart["total"]),
                "exploit_steps": [...]
            })
    
    # 2. 測試優惠券疊加
    for coupon1, coupon2 in itertools.combinations(coupons, 2):
        response = await api_client.post(
            f"{target['url']}/api/cart/apply-coupons",
            json={"coupons": [coupon1, coupon2]}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["total_discount"] > result["original_price"]:
                findings.append({
                    "type": "COUPON_STACKING_OVERFLOW",
                    "severity": "high",
                    "description": "可疊加多張優惠券導致超額折扣",
                    "coupons_used": [coupon1, coupon2],
                    "discount_overflow": result["total_discount"] - result["original_price"]
                })
    
    return findings
```

---

### 2. 競態條件測試 (RaceConditionTester)

檢測並發請求場景下的資源競爭問題。

#### 測試場景

```python
# 庫存競態
{
    "scenario": "stock_race_condition",
    "resource": "product_stock",
    "initial_stock": 1,
    "concurrent_purchases": 10,
    "expected": "超賣問題 - 實際賣出 > 庫存"
}

# 餘額競態
{
    "scenario": "balance_race_condition",
    "resource": "user_balance",
    "initial_balance": 100,
    "concurrent_withdrawals": 10,
    "withdrawal_amount": 50,
    "expected": "餘額變負數"
}

# 優惠券競態
{
    "scenario": "coupon_race_condition",
    "resource": "coupon_usage",
    "usage_limit": 1,
    "concurrent_uses": 5,
    "expected": "超過使用次數限制"
}
```

#### 檢測邏輯

```python
async def test_race_condition(target, params):
    """競態條件測試"""
    findings = []
    
    # 準備並發請求
    tasks = []
    for i in range(params["concurrent_requests"]):
        task = asyncio.create_task(
            api_client.post(
                f"{target['url']}/api/purchase",
                json={"product_id": params["product_id"], "quantity": 1}
            )
        )
        tasks.append(task)
    
    # 同時發送
    responses = await asyncio.gather(*tasks)
    
    # 分析結果
    successful_purchases = sum(1 for r in responses if r.status_code == 200)
    
    # 檢查庫存
    stock_response = await api_client.get(
        f"{target['url']}/api/products/{params['product_id']}/stock"
    )
    final_stock = stock_response.json()["stock"]
    
    # 計算超賣數量
    oversold = successful_purchases - (params["initial_stock"] - final_stock)
    
    if oversold > 0:
        findings.append({
            "type": "RACE_CONDITION_OVERSELLING",
            "severity": "critical",
            "description": f"庫存競態條件導致超賣 {oversold} 件",
            "initial_stock": params["initial_stock"],
            "successful_purchases": successful_purchases,
            "final_stock": final_stock,
            "oversold_quantity": oversold
        })
    
    return findings
```

---

### 3. 工作流程繞過測試 (WorkflowBypassTester)

測試是否可以跳過必要的業務流程步驟。

#### 測試場景

```python
# 支付流程繞過
{
    "workflow": "checkout_process",
    "required_steps": [
        "add_to_cart",
        "enter_shipping_info",
        "select_payment_method",
        "confirm_payment",
        "place_order"
    ],
    "bypass_attempt": "直接調用 place_order API"
}

# 認證流程繞過
{
    "workflow": "account_upgrade",
    "required_steps": [
        "login",
        "verify_email",
        "complete_profile",
        "submit_upgrade_request",
        "admin_approval"
    ],
    "bypass_attempt": "跳過 admin_approval 直接設置 VIP 角色"
}
```

#### 檢測邏輯

```python
async def test_workflow_bypass(target, params):
    """工作流程繞過測試"""
    findings = []
    workflow = params["workflow"]
    required_steps = params["required_steps"]
    
    # 嘗試跳過每個步驟
    for skip_step in required_steps:
        remaining_steps = [s for s in required_steps if s != skip_step]
        
        # 執行剩餘步驟
        result = await execute_workflow(target, remaining_steps)
        
        if result["success"]:
            findings.append({
                "type": "WORKFLOW_BYPASS",
                "severity": "high",
                "description": f"可以跳過 '{skip_step}' 步驟",
                "workflow": workflow,
                "skipped_step": skip_step,
                "impact": assess_skip_impact(skip_step)
            })
    
    # 嘗試改變步驟順序
    for permuted_order in generate_permutations(required_steps):
        if permuted_order == required_steps:
            continue
        
        result = await execute_workflow(target, permuted_order)
        
        if result["success"]:
            findings.append({
                "type": "WORKFLOW_ORDER_BYPASS",
                "severity": "medium",
                "description": "工作流程步驟順序可被改變",
                "correct_order": required_steps,
                "actual_order": permuted_order
            })
    
    return findings
```

---

## 🚀 使用範例

### 完整業務邏輯測試流程

```python
from core_capabilities.bizlogic import (
    PriceManipulationTester,
    RaceConditionTester,
    WorkflowBypassTester,
    FindingHelper
)

# 1. 初始化測試目標
target = {
    "url": "https://shop.example.com",
    "endpoints": {
        "cart": "/api/cart",
        "checkout": "/api/checkout",
        "products": "/api/products"
    }
}

# 2. 價格操控測試
price_tester = PriceManipulationTester(target)
price_findings = await price_tester.test_all_scenarios({
    "product_id": "PROD-001",
    "original_price": 1000,
    "test_scenarios": [
        "negative_quantity",
        "decimal_quantity",
        "coupon_stacking",
        "discount_overflow"
    ]
})

# 3. 競態條件測試
race_tester = RaceConditionTester(target)
race_findings = await race_tester.test_concurrent_access({
    "product_id": "PROD-001",
    "initial_stock": 10,
    "concurrent_requests": 50
})

# 4. 工作流程繞過測試
workflow_tester = WorkflowBypassTester(target)
workflow_findings = await workflow_tester.test_step_bypass({
    "workflow": "checkout_process",
    "required_steps": [
        "add_to_cart",
        "enter_shipping",
        "select_payment",
        "confirm_payment"
    ]
})

# 5. 分析和分類結果
all_findings = price_findings + race_findings + workflow_findings

for finding in all_findings:
    finding["category"] = FindingHelper.categorize_finding(finding)
    finding["severity"] = FindingHelper.calculate_severity(finding)

# 6. 生成報告
report = FindingHelper.generate_report(all_findings)

print(f"總共發現 {report['total_findings']} 個業務邏輯漏洞:")
print(f"  Critical: {len(report['by_severity']['critical'])}")
print(f"  High: {len(report['by_severity']['high'])}")
print(f"  Medium: {len(report['by_severity']['medium'])}")
print(f"  Low: {len(report['by_severity']['low'])}")
```

### 透過 MQ 提交測試任務

```python
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, MessageHeader
from services.aiva_common.enums.modules import Topic

# 構建測試任務消息
task_msg = AivaMessage(
    header=MessageHeader(
        message_id=new_id(),
        timestamp=datetime.now(),
        source="user_interface"
    ),
    payload={
        "module": "bizlogic",
        "test_type": "price_manipulation",
        "target": {
            "url": "https://shop.example.com",
            "endpoints": {
                "cart": "/api/cart",
                "checkout": "/api/checkout"
            }
        },
        "test_params": {
            "product_id": "PROD-001",
            "original_price": 1000,
            "test_scenarios": [
                "negative_quantity",
                "coupon_stacking"
            ]
        }
    }
)

# 發布測試任務
broker = await get_broker()
await broker.publish(
    Topic.TASK_FUNCTION_START,
    task_msg.model_dump_json()
)

# 監聽結果
async for result_msg in broker.subscribe(Topic.TASK_FUNCTION_RESULT):
    result = AivaMessage.model_validate_json(result_msg.body)
    findings = result.payload.get("findings", [])
    
    print(f"收到測試結果: {len(findings)} 個發現")
    for finding in findings:
        print(f"  [{finding['severity'].upper()}] {finding['type']}")
```

---

## 📊 性能指標

| 指標 | 說明 | 典型值 |
|------|------|--------|
| **單場景測試時間** | 完成單個測試場景的時間 | 1-5 秒 |
| **並發請求數** | 競態條件測試的並發請求數 | 50-100 requests |
| **工作流程測試覆蓋率** | 測試的步驟組合比例 | >90% |
| **漏洞檢測準確率** | 真實漏洞 / 報告漏洞 | >85% |
| **誤報率** | 誤報 / 總報告 | <15% |

---

## 📚 相關文檔

- [Core Capabilities 主文檔](../README.md)
- [Attack 子模組](../attack/README.md) - 攻擊執行系統
- [Analysis 子模組](../analysis/README.md) - 代碼分析
- [Service Backbone - MQ](../../service_backbone/messaging/README.md) - 訊息佇列
- [Task Planning](../../task_planning/README.md) - 任務規劃

---

**版權所有** © 2024 AIVA Project. 保留所有權利。
