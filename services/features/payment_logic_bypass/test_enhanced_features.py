# -*- coding: utf-8 -*-
"""
Payment Logic Bypass Worker - Enhanced Features 單元測試

測試新增的功能:
- Race Condition 檢測
- 動態參數識別
- Currency 操縱檢測
- Status 操縱檢測
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
from .worker import PaymentLogicBypassWorker
from ..base.result_schema import Finding


class TestDynamicParameterIdentification(unittest.TestCase):
    """測試動態參數識別功能"""
    
    def setUp(self):
        self.worker = PaymentLogicBypassWorker()
    
    def test_identify_amount_params(self):
        """測試金額參數識別"""
        request_data = {
            "total_price": 100.0,
            "item_qty": 2,
            "discount_code": "SAVE10"
        }
        
        identified = self.worker._identify_payment_params(request_data)
        
        self.assertEqual(identified['amount'], 'total_price')
        self.assertEqual(identified['quantity'], 'item_qty')
        self.assertEqual(identified['coupon'], 'discount_code')
    
    def test_identify_multiple_keywords(self):
        """測試多個關鍵字匹配"""
        request_data = {
            "payment_amount": 50.0,
            "quantity": 1,
            "currency_code": "USD",
            "order_status": "pending",
            "promo_code": "SUMMER2025"
        }
        
        identified = self.worker._identify_payment_params(request_data)
        
        self.assertEqual(identified['amount'], 'payment_amount')
        self.assertEqual(identified['quantity'], 'quantity')
        self.assertEqual(identified['currency'], 'currency_code')
        self.assertEqual(identified['status'], 'order_status')
        self.assertEqual(identified['coupon'], 'promo_code')
    
    def test_no_matching_params(self):
        """測試無匹配參數的情況"""
        request_data = {
            "user_id": "123",
            "session_id": "abc"
        }
        
        identified = self.worker._identify_payment_params(request_data)
        
        self.assertEqual(len(identified), 0)


class TestRaceConditionDetection(unittest.TestCase):
    """測試競態條件檢測功能"""
    
    def setUp(self):
        self.worker = PaymentLogicBypassWorker()
    
    @patch('services.features.base.http_client.SafeHttp')
    def test_race_condition_detected(self, mock_http_class):
        """測試檢測到競態條件的情況"""
        # 設置 mock HTTP 客戶端
        mock_http = Mock()
        mock_http_class.return_value = mock_http
        
        # 模擬兩個請求都成功的情況
        confirm_resp = Mock()
        confirm_resp.status_code = 200
        confirm_resp.json.return_value = {
            "success": True,
            "status": "confirmed"
        }
        
        cancel_resp = Mock()
        cancel_resp.status_code = 200
        cancel_resp.json.return_value = {
            "success": True,
            "status": "cancelled"
        }
        
        # 模擬 request 方法返回值
        mock_http.request.side_effect = [confirm_resp, cancel_resp]
        
        result = self.worker._test_race_condition(
            mock_http,
            "https://test.com/api/confirm",
            "https://test.com/api/cancel",
            "order123",
            {}
        )
        
        # 驗證檢測到漏洞
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Finding)
        self.assertEqual(result.vuln_type, "Payment - Race Condition")
        self.assertEqual(result.severity, "critical")
    
    @patch('services.features.base.http_client.SafeHttp')
    def test_race_condition_not_detected(self, mock_http_class):
        """測試未檢測到競態條件的情況"""
        mock_http = Mock()
        mock_http_class.return_value = mock_http
        
        # 模擬只有一個請求成功
        confirm_resp = Mock()
        confirm_resp.status_code = 200
        confirm_resp.json.return_value = {"success": True, "status": "confirmed"}
        
        cancel_resp = Mock()
        cancel_resp.status_code = 400
        cancel_resp.json.return_value = {"error": "Cannot cancel confirmed order"}
        
        mock_http.request.side_effect = [confirm_resp, cancel_resp]
        
        result = self.worker._test_race_condition(
            mock_http,
            "https://test.com/api/confirm",
            "https://test.com/api/cancel",
            "order123",
            {}
        )
        
        # 應該返回 None (未檢測到漏洞)
        self.assertIsNone(result)


class TestCurrencyManipulation(unittest.TestCase):
    """測試貨幣操縱檢測功能"""
    
    def setUp(self):
        self.worker = PaymentLogicBypassWorker()
    
    @patch('services.features.base.http_client.SafeHttp')
    def test_currency_manipulation_detected(self, mock_http_class):
        """測試檢測到貨幣操縱漏洞"""
        mock_http = Mock()
        mock_http_class.return_value = mock_http
        
        # 模擬系統接受了 IDR 貨幣
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "currency": "IDR",
            "total": 100
        }
        
        mock_http.request.return_value = resp
        
        result = self.worker._test_currency_manipulation(
            mock_http,
            "https://test.com/api/checkout",
            "product123",
            "USD",
            {}
        )
        
        # 驗證檢測到漏洞
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Finding)
        self.assertEqual(result.vuln_type, "Payment - Currency Manipulation")
        self.assertEqual(result.severity, "high")
    
    @patch('services.features.base.http_client.SafeHttp')
    def test_currency_manipulation_not_detected(self, mock_http_class):
        """測試未檢測到貨幣操縱漏洞"""
        mock_http = Mock()
        mock_http_class.return_value = mock_http
        
        # 模擬系統拒絕貨幣操縱
        resp = Mock()
        resp.status_code = 400
        resp.json.return_value = {"error": "Invalid currency"}
        
        mock_http.request.return_value = resp
        
        result = self.worker._test_currency_manipulation(
            mock_http,
            "https://test.com/api/checkout",
            "product123",
            "USD",
            {}
        )
        
        # 應該返回 None
        self.assertIsNone(result)


class TestStatusManipulation(unittest.TestCase):
    """測試狀態操縱檢測功能"""
    
    def setUp(self):
        self.worker = PaymentLogicBypassWorker()
    
    @patch('services.features.base.http_client.SafeHttp')
    def test_status_manipulation_detected(self, mock_http_class):
        """測試檢測到狀態操縱漏洞"""
        mock_http = Mock()
        mock_http_class.return_value = mock_http
        
        # 模擬 PATCH 請求成功
        patch_resp = Mock()
        patch_resp.status_code = 200
        
        # 模擬 GET 請求確認狀態已改變
        get_resp = Mock()
        get_resp.status_code = 200
        get_resp.json.return_value = {
            "order_id": "order123",
            "status": "paid"
        }
        
        mock_http.request.side_effect = [patch_resp, get_resp]
        
        result = self.worker._test_status_manipulation(
            mock_http,
            "https://test.com/api/order",
            "order123",
            {}
        )
        
        # 驗證檢測到漏洞
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Finding)
        self.assertEqual(result.vuln_type, "Payment - Status Manipulation")
        self.assertEqual(result.severity, "critical")
    
    @patch('services.features.base.http_client.SafeHttp')
    def test_status_manipulation_not_detected(self, mock_http_class):
        """測試未檢測到狀態操縱漏洞"""
        mock_http = Mock()
        mock_http_class.return_value = mock_http
        
        # 模擬 PATCH 請求被拒絕
        patch_resp = Mock()
        patch_resp.status_code = 403
        
        mock_http.request.return_value = patch_resp
        
        result = self.worker._test_status_manipulation(
            mock_http,
            "https://test.com/api/order",
            "order123",
            {}
        )
        
        # 應該返回 None
        self.assertIsNone(result)


class TestIntegrationWithRun(unittest.TestCase):
    """測試新功能與 run() 方法的整合"""
    
    def setUp(self):
        self.worker = PaymentLogicBypassWorker()
    
    @patch('services.features.base.http_client.SafeHttp')
    def test_run_with_new_features_enabled(self, mock_http_class):
        """測試啟用新功能的 run() 方法"""
        mock_http = Mock()
        mock_http_class.return_value = mock_http
        
        # 設置參數
        params = {
            "target": "https://test.com",
            "product_id": "prod123",
            "enable_race_condition": True,
            "enable_currency_manipulation": True,
            "enable_status_manipulation": True,
            "order_id": "order123",
            "confirm_endpoint": "/api/confirm",
            "cancel_endpoint": "/api/cancel",
            "order_endpoint": "/api/order",
            # 禁用原有測試避免複雜度
            "enable_price_manipulation": False,
            "enable_negative_quantity": False,
            "enable_coupon_abuse": False,
            "enable_precision_attack": False
        }
        
        # 模擬所有請求失敗(無漏洞)
        mock_resp = Mock()
        mock_resp.status_code = 400
        mock_http.request.return_value = mock_resp
        
        result = self.worker.run(params)
        
        # 驗證結果結構
        self.assertEqual(result.feature, "payment_logic_bypass")
        self.assertIsNotNone(result.command_record)
        self.assertIsNotNone(result.meta)
        self.assertIn("trace", result.meta)
    
    def test_auto_detect_params(self):
        """測試自動參數識別功能整合"""
        params = {
            "target": "https://test.com",
            "product_id": "prod123",
            "auto_detect_params": True,
            "request_data": {
                "total_price": 100.0,
                "item_count": 2
            },
            # 禁用所有測試
            "enable_price_manipulation": False,
            "enable_negative_quantity": False,
            "enable_coupon_abuse": False,
            "enable_precision_attack": False
        }
        
        result = self.worker.run(params)
        
        # 驗證 trace 中包含自動識別信息
        self.assertIn("trace", result.meta)
        trace_items = result.meta["trace"]
        auto_detect_items = [t for t in trace_items if t.get("auto_detect")]
        self.assertGreater(len(auto_detect_items), 0)


if __name__ == "__main__":
    unittest.main()
