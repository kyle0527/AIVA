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
       
    這些漏洞在 Bug Bounty 平台屬於 Critical/High 級別，
    因為它們可能造成直接的經濟損失。
    """
    
    name = "payment_logic_bypass"
    version = "1.0.0"
    tags = ["payment", "business-logic", "price-manipulation", "coupon-abuse", "critical-severity"]

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
              - enable_price_manipulation (bool): 是否測試價格操縱，預設 True
              - enable_negative_quantity (bool): 是否測試負數數量，預設 True
              - enable_coupon_abuse (bool): 是否測試折扣券濫用，預設 True
              - enable_precision_attack (bool): 是否測試小數精度攻擊，預設 True
              
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
        
        enable_price = params.get("enable_price_manipulation", True)
        enable_negative = params.get("enable_negative_quantity", True)
        enable_coupon = params.get("enable_coupon_abuse", True)
        enable_precision = params.get("enable_precision_attack", True)
        
        headers = params.get("headers", {}).copy()
        if session_token:
            headers["Authorization"] = f"Bearer {session_token}"
        
        findings: List[Finding] = []
        trace = []
        
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
                    "precision_attack": enable_precision
                }
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
