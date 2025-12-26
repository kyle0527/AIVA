"""
Python Intelligence Engine 測試套件

測試三個核心檢測器:
1. XXE Detector
2. Deserialization Detector
3. Passive Analyzer
"""

import unittest
from xxe_detector import XXEDetector, XXEFinding
from deserialization_detector_v2 import DeserializationDetector, DeserializationFinding
from passive_analyzer import PassiveAnalyzer, PassiveFinding


class TestXXEDetector(unittest.TestCase):
    """XXE 檢測器測試"""
    
    def setUp(self):
        self.detector = XXEDetector(callback_server="http://test.com")
    
    def test_payload_generation(self):
        """測試 payload 生成"""
        payloads = self.detector.payloads
        self.assertIsInstance(payloads, list)
        self.assertGreater(len(payloads), 0)
        
        # 檢查是否包含基本的 XXE payload
        has_basic_xxe = any('file:///' in p for p in payloads)
        self.assertTrue(has_basic_xxe, "Should contain basic file:// XXE payload")
    
    def test_evil_dtd_generation(self):
        """測試 evil.dtd 生成"""
        dtd = self.detector.generate_evil_dtd()
        self.assertIn('ENTITY', dtd)
        self.assertIn('file:///', dtd)
        self.assertIn(self.detector.callback_server, dtd)
    
    def test_response_analysis_file_disclosure(self):
        """測試文件洩漏檢測"""
        from unittest.mock import Mock
        
        # 模擬包含 /etc/passwd 內容的響應
        mock_response = Mock()
        mock_response.text = "root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin"
        mock_response.status_code = 200
        
        vulnerable, evidence, vuln_type = self.detector._analyze_response(mock_response, "test_payload")
        
        self.assertTrue(vulnerable)
        self.assertIn('root:', evidence.lower())
        self.assertIn('disclosure', vuln_type.lower())
    
    def test_response_analysis_parser_error(self):
        """測試解析器錯誤檢測"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.text = "SAXParseException: External entity access forbidden"
        mock_response.status_code = 500
        
        vulnerable, evidence, vuln_type = self.detector._analyze_response(mock_response, "test_payload")
        
        self.assertTrue(vulnerable)
        self.assertIn('error', vuln_type.lower())


class TestDeserializationDetector(unittest.TestCase):
    """反序列化檢測器測試"""
    
    def setUp(self):
        self.detector = DeserializationDetector(timeout=5)
    
    def test_java_payload_generation(self):
        """測試 Java payload 生成"""
        payloads = self.detector.generate_detection_payloads('java')
        
        self.assertIsInstance(payloads, dict)
        self.assertGreater(len(payloads), 0)
        
        # 檢查是否包含主要的序列化格式
        expected_types = ['java_serialized', 'jackson_jndi', 'fastjson']
        for expected in expected_types:
            self.assertIn(expected, payloads, f"Should contain {expected} payload")
    
    def test_python_payload_generation(self):
        """測試 Python payload 生成"""
        payloads = self.detector.generate_detection_payloads('python')
        
        self.assertIn('pickle', payloads)
        self.assertIn('pyyaml', payloads)
        
        # 檢查 pickle payload 是否 Base64 編碼
        import base64
        try:
            decoded = base64.b64decode(payloads['pickle'])
            self.assertIsInstance(decoded, bytes)
        except Exception as e:
            self.fail(f"Pickle payload should be valid Base64: {e}")
    
    def test_php_payload_generation(self):
        """測試 PHP payload 生成"""
        payloads = self.detector.generate_detection_payloads('php')
        
        self.assertIn('php_serialize', payloads)
        
        # PHP 序列化格式檢查
        php_payload = payloads['php_serialize'].decode()
        self.assertTrue(php_payload.startswith('O:'), "PHP payload should start with O:")
    
    def test_dotnet_payload_generation(self):
        """測試 .NET payload 生成"""
        payloads = self.detector.generate_detection_payloads('dotnet')
        
        self.assertIn('binaryformatter', payloads)
        self.assertIn('jsonnet_typenamehandling', payloads)
    
    def test_error_detection_java(self):
        """測試 Java 反序列化錯誤檢測"""
        text = "java.io.ObjectInputStream readObject failed: InvalidClassException"
        result = self.detector._check_deserialization_error(text, 'java')
        self.assertTrue(result)
    
    def test_error_detection_python(self):
        """測試 Python 反序列化錯誤檢測"""
        text = "pickle.loads failed: _pickle.UnpicklingError: invalid load key"
        result = self.detector._check_deserialization_error(text, 'python')
        self.assertTrue(result)
    
    def test_no_false_positive(self):
        """測試不應誤報"""
        text = "Normal response without deserialization errors"
        
        for language in ['java', 'python', 'php', 'dotnet']:
            result = self.detector._check_deserialization_error(text, language)
            self.assertFalse(result, f"Should not detect error for {language}")


class TestPassiveAnalyzer(unittest.TestCase):
    """被動分析器測試"""
    
    def setUp(self):
        self.analyzer = PassiveAnalyzer()
    
    def test_sensitive_data_patterns(self):
        """測試敏感數據模式"""
        self.assertIn('email', self.analyzer.sensitive_patterns)
        self.assertIn('credit_card', self.analyzer.sensitive_patterns)
        self.assertIn('ssn', self.analyzer.sensitive_patterns)
        self.assertIn('jwt', self.analyzer.sensitive_patterns)
    
    def test_email_detection(self):
        """測試郵箱檢測"""
        url = "http://example.com"
        body = "Contact us at support@example.com or admin@test.org"
        
        findings = self.analyzer._check_sensitive_data_in_body(body, url)
        
        email_findings = [f for f in findings if 'email' in f.evidence.lower()]
        self.assertGreater(len(email_findings), 0, "Should detect email addresses")
    
    def test_credit_card_detection(self):
        """測試信用卡號檢測"""
        url = "http://example.com"
        body = "Card number: 4111 1111 1111 1111"
        
        findings = self.analyzer._check_sensitive_data_in_body(body, url)
        
        cc_findings = [f for f in findings if 'credit' in f.description.lower()]
        self.assertGreater(len(cc_findings), 0, "Should detect credit card")
    
    def test_jwt_detection(self):
        """測試 JWT token 檢測"""
        url = "http://example.com"
        body = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        
        findings = self.analyzer._check_sensitive_data_in_body(body, url)
        
        jwt_findings = [f for f in findings if 'jwt' in f.evidence.lower()]
        self.assertGreater(len(jwt_findings), 0, "Should detect JWT token")
    
    def test_security_headers_missing(self):
        """測試缺失的安全頭部檢測"""
        url = "http://example.com"
        headers = {}  # 空頭部
        
        findings = self.analyzer._check_security_headers(headers, url)
        
        # 應該檢測到多個缺失的頭部
        self.assertGreater(len(findings), 0, "Should detect missing security headers")
        
        # 檢查是否包含 HSTS
        hsts_findings = [f for f in findings if 'hsts' in f.description.lower()]
        self.assertGreater(len(hsts_findings), 0, "Should detect missing HSTS")
    
    def test_security_headers_present(self):
        """測試存在的安全頭部"""
        url = "http://example.com"
        headers = {
            'strict-transport-security': 'max-age=31536000; includeSubDomains',
            'content-security-policy': "default-src 'self'",
            'x-frame-options': 'DENY',
            'x-content-type-options': 'nosniff'
        }
        
        findings = self.analyzer._check_security_headers(headers, url)
        
        # 不應該有關於這些頭部缺失的發現
        missing_findings = [f for f in findings if 'not found' in f.evidence.lower()]
        self.assertEqual(len(missing_findings), 0, "Should not report missing headers when present")
    
    def test_weak_hsts(self):
        """測試弱 HSTS 配置"""
        url = "http://example.com"
        headers = {
            'strict-transport-security': 'max-age=3600'  # 1小時,太短
        }
        
        findings = self.analyzer._check_security_headers(headers, url)
        
        hsts_findings = [f for f in findings if 'hsts' in f.description.lower() and 'short' in f.description.lower()]
        self.assertGreater(len(hsts_findings), 0, "Should detect weak HSTS max-age")
    
    def test_cookie_without_httponly(self):
        """測試缺少 HttpOnly 的 Cookie"""
        url = "http://example.com"
        set_cookie = "session=abc123; Secure; SameSite=Strict"
        
        findings = self.analyzer._analyze_set_cookie(set_cookie, url)
        
        httponly_findings = [f for f in findings if 'httponly' in f.description.lower()]
        self.assertGreater(len(httponly_findings), 0, "Should detect missing HttpOnly")
    
    def test_cookie_without_secure(self):
        """測試缺少 Secure 的 Cookie"""
        url = "http://example.com"
        set_cookie = "session=abc123; HttpOnly; SameSite=Strict"
        
        findings = self.analyzer._analyze_set_cookie(set_cookie, url)
        
        secure_findings = [f for f in findings if 'secure' in f.description.lower()]
        self.assertGreater(len(secure_findings), 0, "Should detect missing Secure")
    
    def test_stack_trace_detection(self):
        """測試堆棧跟踪檢測"""
        url = "http://example.com"
        
        # Java stack trace
        response_java = {
            'status': 500,
            'content': {
                'text': 'at com.example.MyClass.doSomething(MyClass.java:42)'
            }
        }
        findings = self.analyzer._check_error_disclosure(response_java, url)
        self.assertGreater(len(findings), 0, "Should detect Java stack trace")
        
        # Python stack trace
        response_python = {
            'status': 500,
            'content': {
                'text': 'File "/app/views.py", line 123, in process_request'
            }
        }
        findings = self.analyzer._check_error_disclosure(response_python, url)
        self.assertGreater(len(findings), 0, "Should detect Python stack trace")
    
    def test_database_error_detection(self):
        """測試數據庫錯誤檢測"""
        url = "http://example.com"
        
        response = {
            'status': 500,
            'content': {
                'text': "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version"
            }
        }
        
        findings = self.analyzer._check_error_disclosure(response, url)
        
        db_findings = [f for f in findings if 'database' in f.description.lower()]
        self.assertGreater(len(db_findings), 0, "Should detect database error")


class IntegrationTest(unittest.TestCase):
    """集成測試"""
    
    def test_xxe_detector_workflow(self):
        """測試 XXE 檢測器完整流程"""
        detector = XXEDetector()
        
        # 確保能生成 payloads
        self.assertIsInstance(detector.payloads, list)
        self.assertGreater(len(detector.payloads), 0)
        
        # 確保能生成 evil.dtd
        dtd = detector.generate_evil_dtd()
        self.assertIsInstance(dtd, str)
        self.assertGreater(len(dtd), 0)
    
    def test_deserialization_detector_workflow(self):
        """測試反序列化檢測器完整流程"""
        detector = DeserializationDetector()
        
        # 測試所有支持的語言
        for language in ['java', 'python', 'php', 'dotnet']:
            payloads = detector.generate_detection_payloads(language)
            self.assertIsInstance(payloads, dict)
            self.assertGreater(len(payloads), 0, f"Should generate payloads for {language}")
    
    def test_passive_analyzer_workflow(self):
        """測試被動分析器完整流程"""
        analyzer = PassiveAnalyzer()
        
        # 確保模式已設置
        self.assertIsInstance(analyzer.sensitive_patterns, dict)
        self.assertIsInstance(analyzer.security_headers, dict)
        
        # 測試各種檢測方法
        url = "http://example.com"
        
        # 測試 URL 檢查
        findings = analyzer._check_sensitive_data_in_url(url + "?email=test@example.com")
        self.assertGreater(len(findings), 0)
        
        # 測試頭部檢查
        findings = analyzer._check_security_headers({}, url)
        self.assertGreater(len(findings), 0)
        
        # 測試 Cookie 檢查
        findings = analyzer._analyze_set_cookie("session=abc123", url)
        self.assertGreater(len(findings), 0)


def run_tests():
    """運行所有測試"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有測試類
    suite.addTests(loader.loadTestsFromTestCase(TestXXEDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestDeserializationDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestPassiveAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTest))
    
    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印總結
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    sys.exit(0 if run_tests() else 1)
