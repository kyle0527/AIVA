# Payment Logic Bypass 功能比較報告

**日期**: 2025-10-25  
**對比項目**: 現有實現 vs 下載檔案  

---

## 執行摘要

### 結論
✅ **現有實現功能更完整且實用**,但下載檔案提供了 **2 個有價值的補充功能**:
1. **Race Condition 測試**框架 (現有實現未涵蓋)
2. **動態參數識別邏輯** (可補強現有實現)

---

## 詳細功能對比

### A. 現有實現優勢 (services/features/payment_logic_bypass/worker.py)

| 功能 | 現有實現 | 下載檔案 | 評分 |
|-----|---------|---------|------|
| **價格操縱檢測** | ✅ 完整實現 (0.01價格+負數價格) | ⚠️ 框架提到但未實現 | 現有勝 |
| **負數數量檢測** | ✅ 完整實現 (計算退款金額) | ⚠️ 框架提到但未實現 | 現有勝 |
| **折扣券濫用** | ✅ 完整實現 (重複使用+堆疊) | ⚠️ 框架提到但未實現 | 現有勝 |
| **小數精度攻擊** | ✅ 完整實現 (999*0.001測試) | ❌ 無 | 現有勝 |
| **證據收集** | ✅ 詳細 (步驟重現+影響分析) | ⚠️ 基礎 | 現有勝 |
| **建議提供** | ✅ 詳細修復建議 | ❌ 無 | 現有勝 |
| **架構整合** | ✅ FeatureBase + SafeHttp | ❌ Dummy 類別 | 現有勝 |

### B. 下載檔案優勢

| 功能 | 下載檔案 | 現有實現 | 評分 |
|-----|---------|---------|------|
| **Race Condition 測試** | ✅ 有框架 (confirm+cancel 並發) | ❌ **無此功能** | **下載勝** ⭐ |
| **動態參數識別** | ✅ 關鍵字匹配邏輯 | ❌ 需手動指定參數 | **下載勝** ⭐ |
| **Currency 操縱** | ✅ 有測試邏輯 (USD/EUR/XXX) | ❌ 無 | 下載勝 |
| **Status 操縱** | ✅ 有測試邏輯 (paid/complete) | ❌ 無 | 下載勝 |

---

## 可提取的有用功能

### 1. ⭐ Race Condition 測試框架

**價值**: 檢測支付確認與取消的競態條件,這是現有實現**完全缺失**的功能

**下載檔案代碼**:
```python
def test_race_condition(self, confirm_request: RequestDefinition, 
                        cancel_request: Optional[RequestDefinition] = None) -> List[Finding]:
    """
    測試競態條件:同時發送確認和取消請求
    """
    # 使用多線程同時發送請求
    threads = []
    results = {}
    
    def send_req(key, req): 
        results[key] = self.http_client.send_request(req)
    
    t1 = threading.Thread(target=send_req, args=('confirm', confirm_request))
    threads.append(t1)
    
    if cancel_request:
        t2 = threading.Thread(target=send_req, args=('cancel', cancel_request))
        threads.append(t2)
    
    for t in threads: t.start()
    for t in threads: t.join()
    
    # 分析最終狀態
    # ...
```

**建議整合**:
```python
# 添加到現有 PaymentLogicBypassWorker.run()
async def _test_race_condition(
    self,
    http: SafeHttp,
    confirm_ep: str,
    cancel_ep: str,
    order_id: str,
    headers: Dict[str, Any]
) -> Optional[Finding]:
    """測試支付確認與取消的競態條件"""
    import asyncio
    
    async def confirm():
        return await http.request("POST", confirm_ep, headers=headers, 
                                  json={"order_id": order_id, "action": "confirm"})
    
    async def cancel():
        return await http.request("POST", cancel_ep, headers=headers, 
                                  json={"order_id": order_id, "action": "cancel"})
    
    # 同時發送
    results = await asyncio.gather(confirm(), cancel())
    confirm_resp, cancel_resp = results
    
    # 檢查是否同時成功(不應該發生)
    if confirm_resp.status_code == 200 and cancel_resp.status_code == 200:
        return Finding(
            vuln_type="Payment - Race Condition",
            severity="critical",
            title="支付系統存在競態條件,可同時確認和取消訂單",
            evidence={
                "order_id": order_id,
                "confirm_status": confirm_resp.status_code,
                "cancel_status": cancel_resp.status_code,
                "both_succeeded": True
            },
            reproduction=[
                {"step": 1, "description": "同時發送確認和取消請求"},
                {"step": 2, "expect": "只有一個請求成功", 
                 "actual": "兩個請求都成功"},
                {"step": 3, "impact": "可能導致已取消訂單仍然扣款"}
            ],
            impact="攻擊者可以利用競態條件在確認支付後立即取消,但仍然收到商品",
            recommendation=(
                "1. 使用分散式鎖(如 Redis SETNX)保護訂單狀態變更\n"
                "2. 實施冪等性檢查,防止重複操作\n"
                "3. 使用樂觀鎖(版本號)檢測並發修改\n"
                "4. 確保訂單狀態機的原子性轉換"
            )
        )
    
    return None
```

---

### 2. ⭐ 動態參數識別邏輯

**價值**: 自動識別可能與支付相關的參數,減少手動配置

**下載檔案代碼**:
```python
class PaymentLogicBypassDetector:
    def __init__(self, http_client: HttpClient):
        # 參數關鍵字列表
        self.amount_param_keywords = ['amount', 'price', 'total', 'cost', 'value', 'subtotal']
        self.quantity_param_keywords = ['quantity', 'qty', 'count', 'items']
        self.currency_param_keywords = ['currency', 'curr']
        self.status_param_keywords = ['status', 'state', 'paid', 'confirmed']
        self.coupon_param_keywords = ['coupon', 'discount', 'voucher', 'promo']
    
    def test_parameter_tampering(self, base_request: RequestDefinition) -> List[Finding]:
        payment_params: List[ParameterDefinition] = []
        
        # 自動識別候選參數
        for p in base_request.parameters:
            name_lower = p.name.lower()
            if any(keyword in name_lower for keyword in self.amount_param_keywords) or \
               any(keyword in name_lower for keyword in self.quantity_param_keywords):
                payment_params.append(p)
        
        # 對識別出的參數進行測試
        for param in payment_params:
            # 測試各種惡意值
            test_values = [
                (0, "Zero value"),
                (-1, "Negative value"),
                (0.01, "Minimal value"),
                # ...
            ]
```

**建議整合**:
```python
# 添加到現有 PaymentLogicBypassWorker
class PaymentLogicBypassWorker(FeatureBase):
    
    # 新增參數關鍵字配置
    PARAM_KEYWORDS = {
        'amount': ['amount', 'price', 'total', 'cost', 'value', 'subtotal', 'sum'],
        'quantity': ['quantity', 'qty', 'count', 'items', 'num'],
        'currency': ['currency', 'curr', 'money_type'],
        'status': ['status', 'state', 'paid', 'confirmed', 'payment_status'],
        'coupon': ['coupon', 'discount', 'voucher', 'promo', 'code']
    }
    
    def _identify_payment_params(self, request_data: Dict[str, Any]) -> Dict[str, str]:
        """
        自動識別請求中的支付相關參數
        
        Returns:
            Dict mapping param_type -> param_name
            Example: {'amount': 'total_price', 'quantity': 'item_qty'}
        """
        identified = {}
        
        for param_name, param_value in request_data.items():
            name_lower = param_name.lower()
            
            for param_type, keywords in self.PARAM_KEYWORDS.items():
                if any(keyword in name_lower for keyword in keywords):
                    identified[param_type] = param_name
                    break
        
        return identified
    
    def run(self, params: Dict[str, Any]) -> FeatureResult:
        # 支援自動模式
        auto_detect = params.get("auto_detect_params", False)
        
        if auto_detect:
            # 從 request body 自動識別參數
            request_data = params.get("request_data", {})
            identified_params = self._identify_payment_params(request_data)
            
            # 使用識別出的參數覆蓋手動配置
            if 'amount' in identified_params and not params.get("product_price"):
                params["product_price"] = identified_params['amount']
            # ...
        
        # 繼續原有邏輯
        # ...
```

---

### 3. Currency 與 Status 操縱測試

**下載檔案提供的測試邏輯**:

```python
# Currency 操縱
if any(keyword in param.name.lower() for keyword in self.currency_param_keywords):
    test_values.extend([
        ("USD", "Common currency (USD)"), 
        ("EUR", "Common currency (EUR)"), 
        ("XXX", "Invalid currency code")
    ])

# Status 操縱
if any(keyword in param.name.lower() for keyword in self.status_param_keywords):
    test_values.extend([
        ("paid", "Force status 'paid'"), 
        ("complete", "Force status 'complete'"), 
        ("confirmed", "Force status 'confirmed'"), 
        (True, "Boolean True")
    ])
```

**建議整合**:
```python
# 新增方法到 PaymentLogicBypassWorker
def _test_currency_manipulation(
    self,
    http: SafeHttp,
    checkout_ep: str,
    product_id: str,
    original_currency: str,
    headers: Dict[str, Any]
) -> Optional[Finding]:
    """測試貨幣操縱漏洞"""
    try:
        # 測試切換到匯率更低的貨幣
        test_currencies = [
            ("IDR", "印尼盾(匯率極低)"),  # 1 USD ≈ 15,000 IDR
            ("VND", "越南盾(匯率極低)"),  # 1 USD ≈ 23,000 VND
            ("XXX", "無效貨幣代碼")
        ]
        
        for currency, desc in test_currencies:
            cart_data = {
                "product_id": product_id,
                "quantity": 1,
                "currency": currency
            }
            
            r = http.request("POST", checkout_ep, headers=headers, json=cart_data)
            
            if r.status_code in (200, 201):
                resp_data = r.json()
                accepted_currency = resp_data.get("currency")
                
                if accepted_currency == currency and currency != original_currency:
                    return Finding(
                        vuln_type="Payment - Currency Manipulation",
                        severity="high",
                        title=f"支付系統接受客戶端指定的貨幣代碼: {currency}",
                        evidence={
                            "original_currency": original_currency,
                            "manipulated_currency": currency,
                            "accepted": True
                        },
                        impact=f"攻擊者可能利用匯率差異以較低成本購買商品",
                        recommendation=(
                            "1. 貨幣代碼必須由伺服器端根據用戶地區或設定決定\n"
                            "2. 不接受客戶端傳遞的貨幣參數\n"
                            "3. 實施貨幣一致性檢查\n"
                            "4. 記錄所有貨幣切換操作"
                        )
                    )
    except Exception:
        pass
    
    return None

def _test_status_manipulation(
    self,
    http: SafeHttp,
    order_ep: str,
    order_id: str,
    headers: Dict[str, Any]
) -> Optional[Finding]:
    """測試訂單狀態操縱漏洞"""
    try:
        # 嘗試直接將訂單狀態改為已付款
        status_tests = [
            ("paid", "已付款"),
            ("completed", "已完成"),
            ("confirmed", "已確認"),
            (True, "布林值 True")
        ]
        
        for status, desc in status_tests:
            order_data = {
                "order_id": order_id,
                "status": status  # 客戶端控制狀態
            }
            
            r = http.request("PATCH", f"{order_ep}/{order_id}", 
                           headers=headers, json=order_data)
            
            if r.status_code in (200, 204):
                # 檢查狀態是否被更新
                r2 = http.request("GET", f"{order_ep}/{order_id}", headers=headers)
                
                if r2.status_code == 200:
                    order_info = r2.json()
                    actual_status = order_info.get("status", "").lower()
                    
                    if actual_status in ["paid", "completed", "confirmed"]:
                        return Finding(
                            vuln_type="Payment - Status Manipulation",
                            severity="critical",
                            title="訂單狀態可被客戶端直接操縱為已付款",
                            evidence={
                                "order_id": order_id,
                                "manipulated_status": status,
                                "accepted_status": actual_status,
                                "bypass_payment": True
                            },
                            impact="攻擊者可以不付款就將訂單標記為已付款並獲得商品",
                            recommendation=(
                                "1. 訂單狀態變更必須透過安全的工作流程\n"
                                "2. 付款狀態只能由支付網關回調更新\n"
                                "3. 實施狀態機驗證,防止非法狀態轉換\n"
                                "4. 使用 HMAC 簽名驗證狀態更新請求的合法性"
                            )
                        )
    except Exception:
        pass
    
    return None
```

---

## 整合建議

### Priority 0 - 立即整合 ⭐
1. **Race Condition 測試** - 現有實現完全缺失此功能
   - 在 `worker.py` 中新增 `_test_race_condition()` 方法
   - 需要 `confirm_endpoint` 和 `cancel_endpoint` 參數

2. **動態參數識別** - 提升易用性
   - 新增 `_identify_payment_params()` 方法
   - 在 `run()` 中支援 `auto_detect_params=True` 模式

### Priority 1 - 補充功能
3. **Currency 操縱測試** - 補充測試場景
   - 新增 `_test_currency_manipulation()` 方法
   - 測試 IDR/VND 等低匯率貨幣

4. **Status 操縱測試** - 補充測試場景
   - 新增 `_test_status_manipulation()` 方法
   - 測試訂單狀態直接修改

### Priority 2 - 代碼優化
5. **參數修改工具函數** - 通用工具
   - 提取 `_modify_request_parameter()` 為 aiva_common 工具
   - 支援 query/body/header/path 多種位置

---

## 實施計劃

### Step 1: 新增 Race Condition 測試
```python
# 檔案: services/features/payment_logic_bypass/worker.py
# 位置: 在 _test_precision_attack() 後面新增

async def _test_race_condition(...):
    # [實現代碼如上]
```

### Step 2: 新增動態參數識別
```python
# 檔案: services/features/payment_logic_bypass/worker.py
# 位置: 在類別開頭新增常量

class PaymentLogicBypassWorker(FeatureBase):
    PARAM_KEYWORDS = {...}  # [如上]
    
    def _identify_payment_params(...):
        # [實現代碼如上]
```

### Step 3: 更新 run() 方法
```python
def run(self, params: Dict[str, Any]) -> FeatureResult:
    # 支援自動參數識別
    auto_detect = params.get("auto_detect_params", False)
    if auto_detect:
        identified_params = self._identify_payment_params(...)
    
    # 新增 race condition 測試
    if params.get("enable_race_condition", True):
        race_result = await self._test_race_condition(...)
        if race_result:
            findings.append(race_result)
```

### Step 4: 新增 Currency/Status 測試
```python
# 在 run() 中新增測試調用
if enable_currency_manipulation:
    currency_result = self._test_currency_manipulation(...)
    
if enable_status_manipulation:
    status_result = self._test_status_manipulation(...)
```

---

## 測試計劃

### 單元測試
```python
# tests/features/test_payment_logic_bypass.py

async def test_race_condition_detection():
    """測試競態條件檢測"""
    worker = PaymentLogicBypassWorker()
    result = await worker._test_race_condition(
        http=mock_http,
        confirm_ep="http://test.com/api/confirm",
        cancel_ep="http://test.com/api/cancel",
        order_id="test123",
        headers={}
    )
    assert result is not None
    assert result.severity == "critical"

def test_dynamic_param_identification():
    """測試動態參數識別"""
    worker = PaymentLogicBypassWorker()
    request_data = {
        "total_price": 100.0,
        "item_qty": 2,
        "discount_code": "SAVE10"
    }
    identified = worker._identify_payment_params(request_data)
    assert identified['amount'] == 'total_price'
    assert identified['quantity'] == 'item_qty'
    assert identified['coupon'] == 'discount_code'
```

---

## 風險評估

### 低風險 ✅
- **動態參數識別**: 純邏輯改進,不影響現有功能
- **Currency/Status 測試**: 新增測試場景,可選啟用

### 中風險 ⚠️
- **Race Condition 測試**: 
  - 需要異步支援 (現有是同步架構)
  - **緩解**: 使用 asyncio.run() 包裝,保持介面兼容

---

## 總結

✅ **下載檔案提供 4 個有價值的增強功能**:
1. ⭐⭐⭐ Race Condition 測試 (現有缺失,高優先級)
2. ⭐⭐ 動態參數識別 (提升易用性)
3. ⭐ Currency 操縱測試 (補充場景)
4. ⭐ Status 操縱測試 (補充場景)

**建議**: 優先整合 Race Condition 測試和動態參數識別功能。

---

**報告產生時間**: 2025-10-25  
**對比方法**: 代碼功能分析、安全場景覆蓋度評估  
**下一步**: 實施 Step 1-4 整合計劃  
