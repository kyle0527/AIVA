# -*- coding: utf-8 -*-
"""
Payment Logic Bypass 攻擊檢測模組

專門檢測電商/支付系統中的業務邏輯漏洞，包括價格操縱、折扣券濫用、
數量限制繞過等高價值漏洞，這類漏洞在 Bug Bounty 中屬於 Critical 級別。
"""
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin
import random
import json
import decimal
import asyncio
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

@FeatureRegistry.register
class PaymentLogicBypassWorker(FeatureBase):
    """
    Payment Logic Bypass 檢測
    
    檢測原理：
    電商和支付系統的業務邏輯漏洞可能導致直接的經濟損失，是最高價值的漏洞類型之一。
    
    常見的業務邏輯漏洞：
    
    1. 價格操縱漏洞：
       - 客戶端控制價格參數
       - 負數金額繞過
       - 小數精度操縱
       
    2. 折扣券濫用漏洞：
       - 同一折扣券多次使用
       - 過期折扣券仍然有效
       - 折扣券堆疊使用（應該互斥）
       - 最低消費限制繞過
       
    3. 數量限制繞過：
       - 負數數量導致退款
       - 購買數量超過庫存
       - 限購數量繞過
       
    4. 運費操縱：
       - 修改運費計算參數
       - 運費為負數
       - 免運費條件繞過
       
    5. 稅金計算漏洞：
       - 修改稅率參數
       - 稅金為負數
       - 跨區域稅金繞過
       
    6. 訂單狀態操縱：
       - 未付款訂單標記為已付款
       - 已取消訂單重新啟動
       - 退款金額操縱
       
    7. 積分/餘額濫用：
       - 負數積分兌換
       - 餘額溢出
       - 積分回收繞過
       
    8. 競態條件漏洞：
       - 支付確認與取消的並發衝突
       - 庫存扣減的並發問題
       - 折扣券使用的並發濫用
       
    這些漏洞在 Bug Bounty 平台屬於 Critical/High 級別，
    因為它們可能造成直接的經濟損失。
    """
    
    name = "payment_logic_bypass"
    version = "1.1.0"
    tags = ["payment", "business-logic", "price-manipulation", "coupon-abuse", "race-condition", "critical-severity"]

    # 支付相關參數關鍵字映射
    PARAM_KEYWORDS = {
        'amount': ['amount', 'price', 'total', 'cost', 'value', 'subtotal', 'sum', 'payment'],
        'quantity': ['quantity', 'qty', 'count', 'items', 'num', 'amount_items'],
        'currency': ['currency', 'curr', 'money_type', 'denomination'],
        'status': ['status', 'state', 'paid', 'confirmed', 'payment_status', 'order_status'],
        'coupon': ['coupon', 'discount', 'voucher', 'promo', 'code', 'promo_code', 'discount_code']
    }

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 Payment Logic Bypass 檢測
        
        Args:
            params: 檢測參數
              - target (str): 目標基礎 URL
              - cart_endpoint (str): 購物車/下單端點
              - checkout_endpoint (str): 結帳端點
              - session_token (str): 已登入的 session token
              - product_id (str): 測試商品 ID
              - original_price (float): 商品原價
              - headers (dict): 額外的 HTTP 標頭
              - coupon_code (str): 測試用折扣券代碼
              
              # 原有測試開關
              - enable_price_manipulation (bool): 是否測試價格操縱，預設 True
              - enable_negative_quantity (bool): 是否測試負數數量，預設 True
              - enable_coupon_abuse (bool): 是否測試折扣券濫用，預設 True
              - enable_precision_attack (bool): 是否測試小數精度攻擊，預設 True
              
              # 新增測試開關 (v1.1.0+)
              - enable_race_condition (bool): 是否測試競態條件，預設 False
              - enable_currency_manipulation (bool): 是否測試貨幣操縱，預設 False
              - enable_status_manipulation (bool): 是否測試狀態操縱，預設 False
              - auto_detect_params (bool): 是否自動識別支付參數，預設 False
              
              # 新增測試所需參數
              - confirm_endpoint (str): 確認支付端點 (競態條件測試用)
              - cancel_endpoint (str): 取消訂單端點 (競態條件測試用)
              - order_endpoint (str): 訂單管理端點 (狀態操縱測試用)
              - order_id (str): 測試訂單 ID (競態條件/狀態操縱測試用)
              - original_currency (str): 原始貨幣代碼 (貨幣操縱測試用)，預設 "USD"
              - request_data (dict): 請求數據 (自動參數識別用)
              
        Returns:
            FeatureResult: 檢測結果，包含發現的漏洞和證據
        """
        http = SafeHttp()
        base = params.get("target", "")
        cart_ep = urljoin(base, params.get("cart_endpoint", "/api/cart/add"))
        checkout_ep = urljoin(base, params.get("checkout_endpoint", "/api/checkout"))
        session_token = params.get("session_token", "")
        product_id = params.get("product_id", "")
        original_price = float(params.get("original_price", 100.0))
        coupon_code = params.get("coupon_code", "")
        
        # 原有測試開關
        enable_price = params.get("enable_price_manipulation", True)
        enable_negative = params.get("enable_negative_quantity", True)
        enable_coupon = params.get("enable_coupon_abuse", True)
        enable_precision = params.get("enable_precision_attack", True)
        
        # 新增測試開關
        enable_race = params.get("enable_race_condition", False)
        enable_currency = params.get("enable_currency_manipulation", False)
        enable_status = params.get("enable_status_manipulation", False)
        auto_detect = params.get("auto_detect_params", False)
        
        # 新增測試參數
        confirm_ep = urljoin(base, params.get("confirm_endpoint", "/api/order/confirm")) if enable_race else ""
        cancel_ep = urljoin(base, params.get("cancel_endpoint", "/api/order/cancel")) if enable_race else ""
        order_ep = urljoin(base, params.get("order_endpoint", "/api/order")) if enable_status else ""
        order_id = params.get("order_id", "")
        original_currency = params.get("original_currency", "USD")
        
        headers = params.get("headers", {}).copy()
        if session_token:
            headers["Authorization"] = f"Bearer {session_token}"
        
        findings: List[Finding] = []
        trace = []
        
        # 自動參數識別
        if auto_detect:
            request_data = params.get("request_data", {})
            if request_data:
                identified_params = self._identify_payment_params(request_data)
                trace.append({
                    "auto_detect": True, 
                    "identified_params": identified_params
                })
                
                # 使用識別出的參數覆蓋手動配置（如果未手動指定）
                if 'amount' in identified_params and not params.get("product_price"):
                    trace.append({
                        "auto_override": "product_price", 
                        "param_name": identified_params['amount']
                    })
        
        if not all([base, cart_ep, product_id]):
            cmd = self.build_command_record(
                command="payment.logic.bypass",
                description="Payment logic bypass detection (missing params)",
                parameters={"error": "Missing required parameters"}
            )
            return FeatureResult(
                ok=False,
                feature=self.name,
                command_record=cmd,
                findings=[],
                meta={"error": "Missing required parameters: target, cart_endpoint, product_id"}
            )
        
        # 測試 1: 價格操縱漏洞
        if enable_price:
            price_result = self._test_price_manipulation(
                http, cart_ep, checkout_ep, product_id, original_price, headers
            )
            if price_result:
                findings.append(price_result)
                trace.append({"test": "price_manipulation", "result": "vulnerable"})
            else:
                trace.append({"test": "price_manipulation", "result": "secure"})
        
        # 測試 2: 負數數量漏洞
        if enable_negative:
            negative_result = self._test_negative_quantity(
                http, cart_ep, checkout_ep, product_id, original_price, headers
            )
            if negative_result:
                findings.append(negative_result)
                trace.append({"test": "negative_quantity", "result": "vulnerable"})
            else:
                trace.append({"test": "negative_quantity", "result": "secure"})
        
        # 測試 3: 折扣券濫用漏洞
        if enable_coupon and coupon_code:
            coupon_result = self._test_coupon_abuse(
                http, cart_ep, checkout_ep, product_id, original_price, coupon_code, headers
            )
            if coupon_result:
                findings.append(coupon_result)
                trace.append({"test": "coupon_abuse", "result": "vulnerable"})
            else:
                trace.append({"test": "coupon_abuse", "result": "secure"})
        
        # 測試 4: 小數精度攻擊
        if enable_precision:
            precision_result = self._test_precision_attack(
                http, cart_ep, checkout_ep, product_id, original_price, headers
            )
            if precision_result:
                findings.append(precision_result)
                trace.append({"test": "precision_attack", "result": "vulnerable"})
            else:
                trace.append({"test": "precision_attack", "result": "secure"})
        
        # 測試 5: 競態條件漏洞 (v1.1.0+)
        if enable_race and order_id and confirm_ep and cancel_ep:
            race_result = self._test_race_condition(
                http, confirm_ep, cancel_ep, order_id, headers
            )
            if race_result:
                findings.append(race_result)
                trace.append({"test": "race_condition", "result": "vulnerable"})
            else:
                trace.append({"test": "race_condition", "result": "secure"})
        elif enable_race:
            trace.append({
                "test": "race_condition", 
                "result": "skipped", 
                "reason": "Missing required parameters (order_id, confirm_endpoint, cancel_endpoint)"
            })
        
        # 測試 6: 貨幣操縱漏洞 (v1.1.0+)
        if enable_currency:
            currency_result = self._test_currency_manipulation(
                http, checkout_ep, product_id, original_currency, headers
            )
            if currency_result:
                findings.append(currency_result)
                trace.append({"test": "currency_manipulation", "result": "vulnerable"})
            else:
                trace.append({"test": "currency_manipulation", "result": "secure"})
        
        # 測試 7: 狀態操縱漏洞 (v1.1.0+)
        if enable_status and order_id and order_ep:
            status_result = self._test_status_manipulation(
                http, order_ep, order_id, headers
            )
            if status_result:
                findings.append(status_result)
                trace.append({"test": "status_manipulation", "result": "vulnerable"})
            else:
                trace.append({"test": "status_manipulation", "result": "secure"})
        elif enable_status:
            trace.append({
                "test": "status_manipulation", 
                "result": "skipped", 
                "reason": "Missing required parameters (order_id, order_endpoint)"
            })
        
        # 構建命令記錄
        cmd = self.build_command_record(
            command="payment.logic.bypass",
            description=f"Payment logic bypass detection on {base}",
            parameters={
                "cart_endpoint": cart_ep,
                "product_id": product_id,
                "original_price": original_price,
                "tests_enabled": {
                    "price_manipulation": enable_price,
                    "negative_quantity": enable_negative,
                    "coupon_abuse": enable_coupon,
                    "precision_attack": enable_precision,
                    "race_condition": enable_race,
                    "currency_manipulation": enable_currency,
                    "status_manipulation": enable_status
                },
                "auto_detect_params": auto_detect
            }
        )
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={"trace": trace, "tests_run": len(trace)}
        )
    
    def _test_price_manipulation(
        self,
        http: SafeHttp,
        cart_ep: str,
        checkout_ep: str,
        product_id: str,
        original_price: float,
        headers: Dict[str, Any]
    ) -> Optional[Finding]:
        """測試價格操縱漏洞"""
        try:
            # 嘗試將價格改為 0.01
            manipulated_price = 0.01
            
            cart_data = {
                "product_id": product_id,
                "quantity": 1,
                "price": manipulated_price  # 客戶端控制價格
            }
            
            r1 = http.request("POST", cart_ep, headers=headers, json=cart_data)
            
            if r1.status_code not in (200, 201):
                return None
            
            # 檢查響應中的價格
            try:
                resp_data = r1.json()
                accepted_price = resp_data.get("price") or resp_data.get("total") or resp_data.get("amount")
                
                # 如果伺服器接受了操縱的價格
                if accepted_price and float(accepted_price) <= manipulated_price + 0.1:
                    return Finding(
                        vuln_type="Payment - Price Manipulation",
                        severity="critical",
                        title="支付系統允許客戶端操縱商品價格",
                        evidence={
                            "product_id": product_id,
                            "original_price": original_price,
                            "manipulated_price": manipulated_price,
                            "accepted_price": accepted_price,
                            "price_difference": original_price - float(accepted_price),
                            "request": cart_data,
                            "response": resp_data
                        },
                        reproduction=[
                            {"step": 1, "request": {"method": "POST", "url": cart_ep, "body": cart_data}, "description": "發送包含操縱價格的購物車請求"},
                            {"step": 2, "expect": f"系統應該使用伺服器端的商品價格 ${original_price}", "actual": f"系統接受了客戶端提供的價格 ${manipulated_price}"},
                            {"step": 3, "impact": f"攻擊者可以用 ${manipulated_price} 購買原價 ${original_price} 的商品"}
                        ],
                        impact=f"攻擊者可以任意修改商品價格，造成嚴重經濟損失。原價 ${original_price} 的商品可以用 ${manipulated_price} 購買。",
                        recommendation=(
                            "1. 永遠不要信任客戶端提供的價格參數\n"
                            "2. 所有價格計算都必須在伺服器端進行\n"
                            "3. 使用商品 ID 從資料庫查詢當前有效價格\n"
                            "4. 實施價格完整性檢查，比對前後端價格\n"
                            "5. 記錄所有異常的價格修改嘗試並發出警報"
                        )
                    )
            except:
                pass
                
            # 嘗試負數價格
            cart_data_negative = {
                "product_id": product_id,
                "quantity": 1,
                "price": -10.0  # 負數價格
            }
            
            r2 = http.request("POST", cart_ep, headers=headers, json=cart_data_negative)
            
            if r2.status_code in (200, 201):
                try:
                    resp_data2 = r2.json()
                    accepted_price2 = resp_data2.get("price") or resp_data2.get("total")
                    
                    if accepted_price2 and float(accepted_price2) < 0:
                        return Finding(
                            vuln_type="Payment - Negative Price",
                            severity="critical",
                            title="支付系統接受負數價格，導致退款漏洞",
                            evidence={
                                "product_id": product_id,
                                "negative_price": -10.0,
                                "accepted": True,
                                "response": resp_data2
                            },
                            reproduction=[
                                {"step": 1, "description": "發送包含負數價格的請求"},
                                {"step": 2, "expect": "系統應該拒絕負數價格", "actual": "系統接受了負數價格"},
                                {"step": 3, "impact": "攻擊者可以透過負數價格獲得退款"}
                            ],
                            impact="攻擊者可以透過負數價格從系統中提取資金，造成嚴重經濟損失",
                            recommendation=(
                                "1. 嚴格驗證所有金額參數，拒絕負數\n"
                                "2. 使用無符號數據類型儲存價格\n"
                                "3. 實施金額範圍檢查（0 到合理上限）\n"
                                "4. 退款操作應該使用獨立的 API，不應該透過負數金額實現"
                            )
                        )
                except:
                    pass
        except Exception:
            pass
        
        return None
    
    def _test_negative_quantity(
        self,
        http: SafeHttp,
        cart_ep: str,
        checkout_ep: str,
        product_id: str,
        original_price: float,
        headers: Dict[str, Any]
    ) -> Optional[Finding]:
        """測試負數數量漏洞"""
        try:
            cart_data = {
                "product_id": product_id,
                "quantity": -5  # 負數數量
            }
            
            r = http.request("POST", cart_ep, headers=headers, json=cart_data)
            
            if r.status_code in (200, 201):
                try:
                    resp_data = r.json()
                    total = resp_data.get("total") or resp_data.get("amount")
                    
                    # 如果總金額為負數，說明系統接受了負數數量
                    if total and float(total) < 0:
                        return Finding(
                            vuln_type="Payment - Negative Quantity",
                            severity="critical",
                            title="購物車接受負數商品數量，導致退款漏洞",
                            evidence={
                                "product_id": product_id,
                                "quantity": -5,
                                "total_amount": total,
                                "original_price": original_price,
                                "calculated_refund": abs(float(total)),
                                "response": resp_data
                            },
                            reproduction=[
                                {"step": 1, "request": {"method": "POST", "url": cart_ep, "body": cart_data}, "description": "發送包含負數數量的購物車請求"},
                                {"step": 2, "expect": "系統應該拒絕負數數量", "actual": f"系統接受了負數數量，總金額為 ${total}"},
                                {"step": 3, "impact": "攻擊者可以透過負數數量獲得退款"}
                            ],
                            impact=f"攻擊者可以透過購買負數數量商品從系統中提取資金。每次可提取約 ${abs(float(total))}",
                            recommendation=(
                                "1. 嚴格驗證商品數量，只接受正整數\n"
                                "2. 實施數量範圍檢查（1 到合理上限）\n"
                                "3. 退貨/退款應該使用獨立的流程，不應該透過負數數量實現\n"
                                "4. 記錄所有負數數量的嘗試並發出警報"
                            )
                        )
                except:
                    pass
        except Exception:
            pass
        
        return None
    
    def _test_coupon_abuse(
        self,
        http: SafeHttp,
        cart_ep: str,
        checkout_ep: str,
        product_id: str,
        original_price: float,
        coupon_code: str,
        headers: Dict[str, Any]
    ) -> Optional[Finding]:
        """測試折扣券濫用漏洞"""
        try:
            # 第一次使用折扣券
            cart_data1 = {
                "product_id": product_id,
                "quantity": 1,
                "coupon": coupon_code
            }
            
            r1 = http.request("POST", checkout_ep, headers=headers, json=cart_data1)
            
            if r1.status_code not in (200, 201):
                return None
            
            try:
                resp1 = r1.json()
                total1 = float(resp1.get("total") or resp1.get("final_amount") or original_price)
                discount1 = original_price - total1
                
                # 如果有折扣，嘗試第二次使用相同折扣券
                if discount1 > 0:
                    r2 = http.request("POST", checkout_ep, headers=headers, json=cart_data1)
                    
                    if r2.status_code in (200, 201):
                        resp2 = r2.json()
                        total2 = float(resp2.get("total") or resp2.get("final_amount") or original_price)
                        discount2 = original_price - total2
                        
                        # 如果第二次仍然有折扣，說明折扣券可以重複使用
                        if discount2 > 0:
                            return Finding(
                                vuln_type="Payment - Coupon Reuse",
                                severity="high",
                                title="折扣券可以重複使用，導致折扣濫用",
                                evidence={
                                    "coupon_code": coupon_code,
                                    "product_id": product_id,
                                    "original_price": original_price,
                                    "first_use": {"total": total1, "discount": discount1},
                                    "second_use": {"total": total2, "discount": discount2},
                                    "total_discount": discount1 + discount2
                                },
                                reproduction=[
                                    {"step": 1, "request": {"method": "POST", "url": checkout_ep, "body": cart_data1}, "description": "第一次使用折扣券結帳"},
                                    {"step": 2, "response": f"獲得折扣 ${discount1}，最終金額 ${total1}", "description": "折扣券成功應用"},
                                    {"step": 3, "request": {"method": "POST", "url": checkout_ep, "body": cart_data1}, "description": "第二次使用相同折扣券"},
                                    {"step": 4, "expect": "第二次應該拒絕（折扣券已使用）", "actual": f"第二次仍然獲得折扣 ${discount2}"}
                                ],
                                impact=f"攻擊者可以無限次使用同一折扣券，每次可節省 ${discount1}，造成經濟損失",
                                recommendation=(
                                    "1. 在資料庫中記錄折扣券使用狀態\n"
                                    "2. 實施折扣券使用次數限制\n"
                                    "3. 為折扣券添加使用者綁定（每個用戶只能使用一次）\n"
                                    "4. 實施分散式鎖防止並發濫用\n"
                                    "5. 記錄所有折扣券使用情況並監控異常模式"
                                )
                            )
                
                # 測試折扣券堆疊
                cart_data_stack = {
                    "product_id": product_id,
                    "quantity": 1,
                    "coupons": [coupon_code, coupon_code]  # 多個相同折扣券
                }
                
                r3 = http.request("POST", checkout_ep, headers=headers, json=cart_data_stack)
                
                if r3.status_code in (200, 201):
                    resp3 = r3.json()
                    total3 = float(resp3.get("total") or original_price)
                    discount3 = original_price - total3
                    
                    # 如果折扣超過單次使用，說明折扣券可以堆疊
                    if discount3 > discount1 * 1.5:  # 容忍度
                        return Finding(
                            vuln_type="Payment - Coupon Stacking",
                            severity="high",
                            title="折扣券可以堆疊使用，導致過度折扣",
                            evidence={
                                "coupon_code": coupon_code,
                                "stacked_count": 2,
                                "single_discount": discount1,
                                "stacked_discount": discount3,
                                "excess_discount": discount3 - discount1
                            },
                            reproduction=[
                                {"step": 1, "description": "嘗試在單次結帳中使用多個相同折扣券"},
                                {"step": 2, "expect": "系統應該只應用一次折扣", "actual": f"系統應用了多次折扣，總折扣 ${discount3}"}
                            ],
                            impact=f"攻擊者可以透過堆疊折扣券獲得超額折扣，每次可額外節省 ${discount3 - discount1}",
                            recommendation=(
                                "1. 實施折扣券互斥邏輯，同一折扣券只能使用一次\n"
                                "2. 驗證折扣券組合的有效性\n"
                                "3. 設置最大折扣上限（如不超過原價的 50%）\n"
                                "4. 記錄所有折扣券組合使用情況"
                            )
                        )
            except:
                pass
        except Exception:
            pass
        
        return None
    
    def _test_precision_attack(
        self,
        http: SafeHttp,
        cart_ep: str,
        checkout_ep: str,
        product_id: str,
        original_price: float,
        headers: Dict[str, Any]
    ) -> Optional[Finding]:
        """測試小數精度攻擊"""
        try:
            # 嘗試購買大量低精度商品（如 0.001 單價）
            # 如果系統四捨五入不當，可能導致免費商品
            
            cart_data = {
                "product_id": product_id,
                "quantity": 999,
                "price": 0.001  # 極低價格
            }
            
            r = http.request("POST", cart_ep, headers=headers, json=cart_data)
            
            if r.status_code in (200, 201):
                try:
                    resp_data = r.json()
                    total = float(resp_data.get("total") or resp_data.get("amount") or 0)
                    expected_total = 0.001 * 999  # 0.999
                    
                    # 如果總金額因為精度問題被四捨五入為 0
                    if total == 0 or total < expected_total * 0.1:
                        return Finding(
                            vuln_type="Payment - Precision Attack",
                            severity="medium",
                            title="支付系統存在小數精度漏洞，可能導致免費商品",
                            evidence={
                                "product_id": product_id,
                                "unit_price": 0.001,
                                "quantity": 999,
                                "expected_total": expected_total,
                                "actual_total": total,
                                "precision_loss": expected_total - total
                            },
                            reproduction=[
                                {"step": 1, "description": "購買大量極低單價商品"},
                                {"step": 2, "expect": f"總金額應該是 ${expected_total}", "actual": f"總金額是 ${total}"},
                                {"step": 3, "impact": "攻擊者可以透過精度漏洞獲得免費或極低價格的商品"}
                            ],
                            impact="攻擊者可以利用小數精度問題以極低價格或免費獲得商品",
                            recommendation=(
                                "1. 使用 Decimal 類型而非 float 儲存金額\n"
                                "2. 實施最小單價限制（如 0.01）\n"
                                "3. 所有金額計算使用固定精度（如 2 位小數）\n"
                                "4. 在最終計算前驗證總金額合理性"
                            )
                        )
                except:
                    pass
        except Exception:
            pass
        
        return None

    def _identify_payment_params(self, request_data: Dict[str, Any]) -> Dict[str, str]:
        """
        自動識別請求中的支付相關參數
        
        Args:
            request_data: 請求數據字典
            
        Returns:
            Dict[str, str]: 映射 param_type -> param_name
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

    def _test_race_condition(
        self,
        http: SafeHttp,
        confirm_ep: str,
        cancel_ep: str,
        order_id: str,
        headers: Dict[str, Any]
    ) -> Optional[Finding]:
        """
        測試支付確認與取消的競態條件漏洞
        
        Args:
            http: HTTP 客戶端
            confirm_ep: 確認支付端點
            cancel_ep: 取消訂單端點
            order_id: 訂單 ID
            headers: HTTP 標頭
            
        Returns:
            Finding: 如果發現競態條件漏洞則返回，否則返回 None
        """
        try:
            async def async_test():
                """使用 asyncio 進行並發測試"""
                async def confirm():
                    """確認支付請求"""
                    return http.request(
                        "POST", 
                        confirm_ep, 
                        headers=headers, 
                        json={"order_id": order_id, "action": "confirm"}
                    )
                
                async def cancel():
                    """取消訂單請求"""
                    return http.request(
                        "POST", 
                        cancel_ep, 
                        headers=headers, 
                        json={"order_id": order_id, "action": "cancel"}
                    )
                
                # 同時發送確認和取消請求
                results = await asyncio.gather(
                    asyncio.to_thread(confirm),
                    asyncio.to_thread(cancel),
                    return_exceptions=True
                )
                
                return results
            
            # 執行異步測試
            results = asyncio.run(async_test())
            
            # 檢查是否有異常
            if any(isinstance(r, Exception) for r in results):
                return None
            
            confirm_resp, cancel_resp = results
            
            # 檢查是否兩個請求都成功（不應該發生）
            if confirm_resp.status_code in (200, 201) and cancel_resp.status_code in (200, 201):
                # 進一步檢查訂單最終狀態
                try:
                    confirm_data = confirm_resp.json()
                    cancel_data = cancel_resp.json()
                    
                    confirm_success = confirm_data.get("success", False) or confirm_data.get("status") == "confirmed"
                    cancel_success = cancel_data.get("success", False) or cancel_data.get("status") == "cancelled"
                    
                    if confirm_success and cancel_success:
                        return Finding(
                            vuln_type="Payment - Race Condition",
                            severity="critical",
                            title="支付系統存在競態條件，可同時確認和取消訂單",
                            evidence={
                                "order_id": order_id,
                                "confirm_status": confirm_resp.status_code,
                                "cancel_status": cancel_resp.status_code,
                                "confirm_response": confirm_data,
                                "cancel_response": cancel_data,
                                "both_succeeded": True,
                                "race_condition_detected": True
                            },
                            reproduction=[
                                {
                                    "step": 1, 
                                    "description": "創建訂單並獲得訂單 ID",
                                    "order_id": order_id
                                },
                                {
                                    "step": 2, 
                                    "description": "同時發送確認支付和取消訂單請求（使用並發）",
                                    "request_confirm": {"method": "POST", "url": confirm_ep, "body": {"order_id": order_id, "action": "confirm"}},
                                    "request_cancel": {"method": "POST", "url": cancel_ep, "body": {"order_id": order_id, "action": "cancel"}}
                                },
                                {
                                    "step": 3, 
                                    "expect": "只有一個請求成功（確認或取消）", 
                                    "actual": "兩個請求都成功"
                                },
                                {
                                    "step": 4, 
                                    "impact": "訂單可能處於不一致狀態（已確認但同時被取消）"
                                }
                            ],
                            impact=(
                                "攻擊者可以利用競態條件在確認支付後立即取消訂單，但仍然獲得商品或服務。"
                                "這可能導致：\n"
                                "1. 用戶收到商品但訂單被取消，不需付款\n"
                                "2. 用戶付款後訂單被取消，但款項未退還\n"
                                "3. 庫存管理出現不一致\n"
                                "4. 財務記錄混亂"
                            ),
                            recommendation=(
                                "1. 使用分散式鎖（如 Redis SETNX）保護訂單狀態變更\n"
                                "2. 實施冪等性檢查，防止重複操作\n"
                                "3. 使用樂觀鎖（版本號）檢測並發修改\n"
                                "4. 確保訂單狀態機的原子性轉換（confirmed <-> cancelled 不可並存）\n"
                                "5. 實施訂單狀態變更的事務處理\n"
                                "6. 記錄所有狀態變更操作並監控異常模式"
                            )
                        )
                except Exception:
                    pass
                    
        except Exception:
            pass
        
        return None

    def _test_currency_manipulation(
        self,
        http: SafeHttp,
        checkout_ep: str,
        product_id: str,
        original_currency: str,
        headers: Dict[str, Any]
    ) -> Optional[Finding]:
        """
        測試貨幣代碼操縱漏洞
        
        Args:
            http: HTTP 客戶端
            checkout_ep: 結帳端點
            product_id: 商品 ID
            original_currency: 原始貨幣代碼
            headers: HTTP 標頭
            
        Returns:
            Finding: 如果發現貨幣操縱漏洞則返回，否則返回 None
        """
        try:
            # 測試切換到匯率更低的貨幣
            test_currencies = [
                ("IDR", "印尼盾(匯率極低)", 15000),  # 1 USD ≈ 15,000 IDR
                ("VND", "越南盾(匯率極低)", 23000),  # 1 USD ≈ 23,000 VND
                ("XXX", "無效貨幣代碼", 0)
            ]
            
            for currency, desc, exchange_rate in test_currencies:
                cart_data = {
                    "product_id": product_id,
                    "quantity": 1,
                    "currency": currency
                }
                
                r = http.request("POST", checkout_ep, headers=headers, json=cart_data)
                
                if r.status_code in (200, 201):
                    try:
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
                                    "accepted": True,
                                    "response": resp_data,
                                    "exchange_rate_difference": exchange_rate
                                },
                                reproduction=[
                                    {
                                        "step": 1, 
                                        "request": {"method": "POST", "url": checkout_ep, "body": cart_data},
                                        "description": f"發送包含操縱貨幣代碼的請求（{currency}）"
                                    },
                                    {
                                        "step": 2, 
                                        "expect": f"系統應該使用伺服器端設定的貨幣 {original_currency}", 
                                        "actual": f"系統接受了客戶端提供的貨幣 {currency}"
                                    },
                                    {
                                        "step": 3, 
                                        "impact": f"攻擊者可能利用匯率差異以較低成本購買商品（{desc}）"
                                    }
                                ],
                                impact=(
                                    f"攻擊者可能利用匯率差異以較低成本購買商品。\n"
                                    f"例如：如果原價是 100 {original_currency}，攻擊者可能切換到 {currency}，"
                                    f"然後只支付 100 {currency}（實際價值遠低於 100 {original_currency}）。"
                                ),
                                recommendation=(
                                    "1. 貨幣代碼必須由伺服器端根據用戶地區或設定決定\n"
                                    "2. 不接受客戶端傳遞的貨幣參數\n"
                                    "3. 實施貨幣一致性檢查\n"
                                    "4. 記錄所有貨幣切換操作並監控異常模式\n"
                                    "5. 如果必須支援多貨幣，應該在伺服器端進行匯率轉換並驗證"
                                )
                            )
                    except Exception:
                        pass
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
        """
        測試訂單狀態操縱漏洞
        
        Args:
            http: HTTP 客戶端
            order_ep: 訂單管理端點
            order_id: 訂單 ID
            headers: HTTP 標頭
            
        Returns:
            Finding: 如果發現狀態操縱漏洞則返回，否則返回 None
        """
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
                        try:
                            order_info = r2.json()
                            actual_status = str(order_info.get("status", "")).lower()
                            
                            if actual_status in ["paid", "completed", "confirmed"]:
                                return Finding(
                                    vuln_type="Payment - Status Manipulation",
                                    severity="critical",
                                    title="訂單狀態可被客戶端直接操縱為已付款",
                                    evidence={
                                        "order_id": order_id,
                                        "manipulated_status": status,
                                        "accepted_status": actual_status,
                                        "bypass_payment": True,
                                        "patch_response": r.status_code,
                                        "get_response": order_info
                                    },
                                    reproduction=[
                                        {
                                            "step": 1, 
                                            "description": "創建未付款訂單",
                                            "order_id": order_id
                                        },
                                        {
                                            "step": 2, 
                                            "request": {"method": "PATCH", "url": f"{order_ep}/{order_id}", "body": order_data},
                                            "description": f"嘗試將訂單狀態直接改為 {status}"
                                        },
                                        {
                                            "step": 3, 
                                            "expect": "系統應該拒絕客戶端的狀態更新", 
                                            "actual": f"系統接受了狀態更新，訂單狀態變為 {actual_status}"
                                        },
                                        {
                                            "step": 4, 
                                            "impact": "攻擊者可以不付款就獲得商品"
                                        }
                                    ],
                                    impact="攻擊者可以不付款就將訂單標記為已付款並獲得商品或服務，造成嚴重經濟損失",
                                    recommendation=(
                                        "1. 訂單狀態變更必須透過安全的工作流程\n"
                                        "2. 付款狀態只能由支付網關回調更新\n"
                                        "3. 實施狀態機驗證，防止非法狀態轉換\n"
                                        "4. 使用 HMAC 簽名驗證狀態更新請求的合法性\n"
                                        "5. 客戶端不應該有權限直接修改訂單狀態\n"
                                        "6. 記錄所有狀態變更操作並監控異常模式"
                                    )
                                )
                        except Exception:
                            pass
        except Exception:
            pass
        
        return None
