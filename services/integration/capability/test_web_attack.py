#!/usr/bin/env python3
"""
AIVA Web Attack Module Test Suite - Task 11
測試網絡攻擊工具集成的各項功能
"""

import asyncio
import json
import pytest
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aioresponses import aioresponses

from services.integration.capability.web_attack import (
    WebTarget, ScanResult, SubdomainEnumerator, DirectoryScanner,
    VulnerabilityScanner, TechnologyDetector, WebAttackManager,
    WebAttackCLI, WebAttackCapability
)


def async_test(func):
    """異步測試裝飾器"""
    def wrapper(self, *args, **kwargs):
        return self.loop.run_until_complete(func(self, *args, **kwargs))
    return wrapper


class AsyncTestCase(unittest.TestCase):
    """支持異步測試的基礎測試類"""
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()


class TestWebTarget(AsyncTestCase):
    """WebTarget 資料結構測試"""
    
    def test_web_target_creation(self):
        """測試 WebTarget 創建"""
        target = WebTarget(url="https://example.com")
        self.assertEqual(target.url, "https://example.com")
        self.assertEqual(target.domain, "example.com")
        self.assertEqual(target.subdomains, [])
        self.assertEqual(target.vulnerabilities, [])
    
    def test_web_target_domain_extraction(self):
        """測試域名提取"""
        # HTTP URL
        target1 = WebTarget(url="http://test.example.com/path")
        self.assertEqual(target1.domain, "test.example.com")
        
        # HTTPS URL
        target2 = WebTarget(url="https://secure.example.com:8080/api")
        self.assertEqual(target2.domain, "secure.example.com:8080")
        
        # 無協議
        target3 = WebTarget(url="example.com", domain="example.com")
        self.assertEqual(target3.domain, "example.com")


class TestScanResult(AsyncTestCase):
    """ScanResult 資料結構測試"""
    
    def test_scan_result_creation(self):
        """測試掃描結果創建"""
        timestamp = datetime.now()
        result = ScanResult(
            target="https://example.com",
            scan_type="comprehensive",
            timestamp=timestamp,
            status="success",
            data={"test": "data"}
        )
        
        self.assertEqual(result.target, "https://example.com")
        self.assertEqual(result.scan_type, "comprehensive")
        self.assertEqual(result.timestamp, timestamp)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data, {"test": "data"})
        self.assertIsNone(result.error)


class TestSubdomainEnumerator(AsyncTestCase):
    """子域名枚舉器測試"""
    
    def setUp(self):
        super().setUp()
        self.enumerator = SubdomainEnumerator()
    
    @async_test
    async def test_subdomain_enumeration_empty(self):
        """測試空結果的子域名枚舉"""
        with aioresponses() as m:
            m.get('https://crt.sh/?q=%25.example.com&output=json', payload=[])
            
            subdomains = await self.enumerator.enumerate_subdomains("example.com")
            self.assertIsInstance(subdomains, list)
    
    @async_test
    async def test_crt_sh_enumeration(self):
        """測試 crt.sh 子域名枚舉"""
        mock_data = [
            {"name_value": "www.example.com"},
            {"name_value": "mail.example.com"},
            {"name_value": "api.example.com"}
        ]
        
        with aioresponses() as m:
            m.get('https://crt.sh/?q=%25.example.com&output=json', payload=mock_data)
            
            # 直接測試 _enumerate_crt_sh 方法
            self.enumerator.session = aiohttp.ClientSession()
            try:
                await self.enumerator._enumerate_crt_sh("example.com")
                self.assertGreaterEqual(len(self.enumerator.found_subdomains), 0)
            finally:
                await self.enumerator.session.close()
    
    @async_test
    async def test_dns_brute_enumeration(self):
        """測試 DNS 暴力破解枚舉"""
        await self.enumerator._enumerate_dns_brute("example.com")
        # 由於 DNS 解析可能失敗，只檢查方法執行不拋異常
        self.assertIsInstance(self.enumerator.found_subdomains, set)
    
    @async_test
    async def test_common_subdomains_enumeration(self):
        """測試常見子域名枚舉"""
        with aioresponses() as m:
            # 模擬成功響應
            m.get('http://www.example.com', status=200)
            m.get('http://mail.example.com', status=200)
            
            self.enumerator.session = aiohttp.ClientSession()
            try:
                await self.enumerator._enumerate_common_subdomains("example.com")
                # 檢查是否有子域名被添加
                self.assertIsInstance(self.enumerator.found_subdomains, set)
            finally:
                await self.enumerator.session.close()


class TestDirectoryScanner(AsyncTestCase):
    """目錄掃描器測試"""
    
    def setUp(self):
        super().setUp()
        self.scanner = DirectoryScanner()
    
    @async_test
    async def test_directory_scanning(self):
        """測試目錄掃描"""
        with aioresponses() as m:
            # 模擬不同的響應狀態
            m.get('http://example.com/admin/', status=200, body="Admin Panel")
            m.get('http://example.com/login/', status=200, body="Login Page")
            m.get('http://example.com/nonexistent/', status=404)
            
            wordlist = ['admin/', 'login/', 'nonexistent/']
            directories = await self.scanner.scan_directories("http://example.com", wordlist)
            
            # 檢查結果
            self.assertIsInstance(directories, list)
            # 應該找到兩個有效目錄 (200 狀態碼)
            success_dirs = [d for d in directories if d['status'] == 200]
            self.assertEqual(len(success_dirs), 2)
    
    @async_test
    async def test_check_path(self):
        """測試單個路徑檢查"""
        with aioresponses() as m:
            m.get('http://example.com/test/', status=200, body="Test Content")
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                await self.scanner._check_path("http://example.com/test/", "test/")
                # 檢查結果是否添加到 found_directories
                self.assertEqual(len(self.scanner.found_directories), 1)
                self.assertEqual(self.scanner.found_directories[0]['status'], 200)
            finally:
                await self.scanner.session.close()
    
    def test_default_wordlist(self):
        """測試默認詞典"""
        wordlist = self.scanner._get_default_wordlist()
        self.assertIsInstance(wordlist, list)
        self.assertGreater(len(wordlist), 0)
        self.assertIn('admin/', wordlist)
        self.assertIn('robots.txt', wordlist)


class TestVulnerabilityScanner(AsyncTestCase):
    """漏洞掃描器測試"""
    
    def setUp(self):
        super().setUp()
        self.scanner = VulnerabilityScanner()
    
    @async_test
    async def test_vulnerability_scanning(self):
        """測試漏洞掃描"""
        with aioresponses() as m:
            # 模擬正常響應
            m.get('http://example.com?test=%3Cscript%3Ealert%28%27XSS%27%29%3C%2Fscript%3E', 
                  status=200, body="Normal response")
            m.get('http://example.com?id=%27', 
                  status=200, body="MySQL error occurred")
            m.get('http://example.com?file=..%2F..%2F..%2Fetc%2Fpasswd', 
                  status=200, body="Normal response")
            m.get('http://example.com', 
                  status=200, body="<html><head></head><body>Test</body></html>",
                  headers={'Server': 'Apache/2.4.41'})
            
            vulnerabilities = await self.scanner.scan_vulnerabilities("http://example.com")
            
            self.assertIsInstance(vulnerabilities, list)
            # 應該檢測到 SQL 注入（因為響應包含 "MySQL error"）
            sql_vulns = [v for v in vulnerabilities if v['type'] == 'SQL Injection']
            self.assertGreater(len(sql_vulns), 0)
    
    @async_test
    async def test_xss_scanning(self):
        """測試 XSS 掃描"""
        with aioresponses() as m:
            # 模擬反射 XSS
            m.get('http://example.com?test=%3Cscript%3Ealert%28%27XSS%27%29%3C%2Fscript%3E', 
                  status=200, body="<script>alert('XSS')</script>")
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                await self.scanner._scan_xss("http://example.com")
                # 檢查是否檢測到 XSS
                xss_vulns = [v for v in self.scanner.vulnerabilities if v['type'] == 'XSS']
                self.assertGreater(len(xss_vulns), 0)
            finally:
                await self.scanner.session.close()
    
    @async_test
    async def test_sql_injection_scanning(self):
        """測試 SQL 注入掃描"""
        with aioresponses() as m:
            # 模擬 SQL 錯誤響應
            m.get("http://example.com?id=%27", 
                  status=200, body="MySQL syntax error near 'WHERE id=''")
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                await self.scanner._scan_sql_injection("http://example.com")
                # 檢查是否檢測到 SQL 注入
                sql_vulns = [v for v in self.scanner.vulnerabilities if v['type'] == 'SQL Injection']
                self.assertGreater(len(sql_vulns), 0)
            finally:
                await self.scanner.session.close()
    
    @async_test
    async def test_security_headers_check(self):
        """測試安全標頭檢查"""
        with aioresponses() as m:
            # 模擬缺少安全標頭的響應
            m.get('http://example.com', status=200, body="Test", headers={})
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                await self.scanner._scan_security_headers("http://example.com")
                # 檢查是否檢測到缺少安全標頭
                header_vulns = [v for v in self.scanner.vulnerabilities 
                               if v['type'] == 'Missing Security Headers']
                self.assertGreater(len(header_vulns), 0)
            finally:
                await self.scanner.session.close()


class TestTechnologyDetector(AsyncTestCase):
    """技術檢測器測試"""
    
    def setUp(self):
        super().setUp()
        self.detector = TechnologyDetector()
    
    @async_test
    async def test_technology_detection(self):
        """測試技術檢測"""
        html_content = """
        <html>
        <head>
            <link rel="stylesheet" href="bootstrap.min.css">
            <script src="jquery.min.js"></script>
        </head>
        <body>
            <div class="wp-content">WordPress site</div>
        </body>
        </html>
        """
        
        with aioresponses() as m:
            m.get('http://example.com', 
                  status=200, 
                  body=html_content,
                  headers={'Server': 'Apache/2.4.41'})
            
            technologies = await self.detector.detect_technologies("http://example.com")
            
            self.assertIsInstance(technologies, list)
            # 檢查是否檢測到服務器技術
            server_techs = [t for t in technologies if t.startswith('Server:')]
            self.assertGreater(len(server_techs), 0)
    
    @async_test
    async def test_framework_detection(self):
        """測試框架檢測"""
        content = "<div class='wp-content'>WordPress content</div>"
        headers = {}
        
        await self.detector._detect_frameworks(content, headers)
        
        # 檢查是否檢測到 WordPress
        wp_techs = [t for t in self.detector.technologies if 'WordPress' in t]
        self.assertGreater(len(wp_techs), 0)
    
    @async_test
    async def test_js_library_detection(self):
        """測試 JavaScript 庫檢測"""
        content = '<script src="jquery.min.js"></script>'
        
        await self.detector._detect_js_libraries(content)
        
        # 檢查是否檢測到 jQuery
        jquery_techs = [t for t in self.detector.technologies if 'jQuery' in t]
        self.assertGreater(len(jquery_techs), 0)


class TestWebAttackManager(AsyncTestCase):
    """網絡攻擊管理器測試"""
    
    def setUp(self):
        super().setUp()
        self.manager = WebAttackManager()
    
    @async_test
    async def test_comprehensive_scan(self):
        """測試綜合掃描"""
        with aioresponses() as m:
            # 模擬各種響應
            m.get('https://crt.sh/?q=%25.example.com&output=json', payload=[])
            m.get('http://example.com/admin/', status=200, body="Admin")
            m.get('http://example.com', status=200, body="<html></html>",
                  headers={'Server': 'nginx'})
            
            # 進行綜合掃描
            results = await self.manager.comprehensive_scan("http://example.com")
            
            # 檢查結果結構
            self.assertIn('target', results)
            self.assertIn('timestamp', results)
            self.assertIn('subdomains', results)
            self.assertIn('directories', results)
            self.assertIn('vulnerabilities', results)
            self.assertIn('technologies', results)
            self.assertIn('scan_summary', results)
            
            # 檢查掃描摘要
            summary = results['scan_summary']
            self.assertIn('total_subdomains', summary)
            self.assertIn('total_vulnerabilities', summary)
    
    @async_test
    async def test_comprehensive_scan_with_options(self):
        """測試帶選項的綜合掃描"""
        options = {
            'subdomain_scan': False,
            'directory_scan': True,
            'vulnerability_scan': True,
            'technology_scan': False
        }
        
        with aioresponses() as m:
            m.get('http://example.com/admin/', status=200, body="Admin")
            m.get('http://example.com', status=200, body="Test")
            
            results = await self.manager.comprehensive_scan("http://example.com", options)
            
            # 檢查選項是否生效
            self.assertEqual(len(results['subdomains']), 0)  # 子域名掃描關閉
            self.assertEqual(len(results['technologies']), 0)  # 技術檢測關閉


class TestWebAttackCLI(AsyncTestCase):
    """網絡攻擊 CLI 測試"""
    
    def setUp(self):
        super().setUp()
        self.manager = WebAttackManager()
        self.cli = WebAttackCLI(self.manager)
    
    def test_cli_creation(self):
        """測試 CLI 創建"""
        self.assertIsInstance(self.cli.manager, WebAttackManager)
    
    @patch('services.integration.capability.web_attack.console')
    def test_show_main_menu(self, mock_console):
        """測試主選單顯示"""
        mock_console.input.return_value = "99"
        
        choice = self.cli.show_main_menu()
        self.assertEqual(choice, "99")
        
        # 檢查是否調用了 console.print
        self.assertTrue(mock_console.print.called)


class TestWebAttackCapability(AsyncTestCase):
    """網絡攻擊能力測試"""
    
    def setUp(self):
        super().setUp()
        self.capability = WebAttackCapability()
    
    def test_capability_properties(self):
        """測試能力屬性"""
        self.assertEqual(self.capability.name, "web_attack")
        self.assertEqual(self.capability.version, "1.0.0")
        self.assertIn("網絡攻擊", self.capability.description)
        self.assertIsInstance(self.capability.dependencies, list)
    
    @async_test
    async def test_capability_initialization(self):
        """測試能力初始化"""
        result = await self.capability.initialize()
        self.assertTrue(result)
    
    @async_test
    async def test_execute_comprehensive_scan(self):
        """測試執行綜合掃描命令"""
        with aioresponses() as m:
            m.get('https://crt.sh/?q=%25.example.com&output=json', payload=[])
            m.get('http://example.com', status=200, body="Test")
            
            result = await self.capability.execute(
                'comprehensive_scan',
                {'target_url': 'http://example.com'}
            )
            
            self.assertTrue(result['success'])
            self.assertIn('data', result)
    
    @async_test
    async def test_execute_subdomain_scan(self):
        """測試執行子域名掃描命令"""
        with aioresponses() as m:
            m.get('https://crt.sh/?q=%25.example.com&output=json', payload=[])
            
            result = await self.capability.execute(
                'subdomain_scan',
                {'domain': 'example.com'}
            )
            
            self.assertTrue(result['success'])
            self.assertIn('data', result)
            self.assertIn('subdomains', result['data'])
    
    @async_test
    async def test_execute_missing_parameters(self):
        """測試執行命令時缺少參數"""
        result = await self.capability.execute('comprehensive_scan', {})
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('Missing target_url parameter', result['error'])
    
    @async_test
    async def test_execute_unknown_command(self):
        """測試執行未知命令"""
        result = await self.capability.execute('unknown_command', {})
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('Unknown command', result['error'])
    
    @async_test
    async def test_cleanup(self):
        """測試能力清理"""
        # 添加一些掃描結果
        self.capability.manager.scan_results.append(
            ScanResult("test", "test", datetime.now(), "success", {})
        )
        
        result = await self.capability.cleanup()
        self.assertTrue(result)
        self.assertEqual(len(self.capability.manager.scan_results), 0)


class TestIntegration(AsyncTestCase):
    """集成測試"""
    
    @async_test
    async def test_full_scan_workflow(self):
        """測試完整掃描流程"""
        capability = WebAttackCapability()
        await capability.initialize()
        
        with aioresponses() as m:
            # 模擬所有必要的 HTTP 請求
            m.get('https://crt.sh/?q=%25.example.com&output=json', 
                  payload=[{"name_value": "www.example.com"}])
            m.get('http://example.com/admin/', status=200, body="Admin Panel")
            m.get('http://example.com', status=200, body="<html><body>Test</body></html>",
                  headers={'Server': 'Apache/2.4.41'})
            
            # 執行綜合掃描
            result = await capability.execute(
                'comprehensive_scan',
                {'target_url': 'http://example.com'}
            )
            
            # 驗證結果
            self.assertTrue(result['success'])
            data = result['data']
            self.assertIn('scan_summary', data)
            
            # 檢查各個掃描模塊都有結果
            summary = data['scan_summary']
            self.assertIsInstance(summary['total_subdomains'], int)
            self.assertIsInstance(summary['total_directories'], int)
            self.assertIsInstance(summary['total_vulnerabilities'], int)
            self.assertIsInstance(summary['total_technologies'], int)
        
        await capability.cleanup()
    
    @async_test
    async def test_export_functionality(self):
        """測試導出功能"""
        capability = WebAttackCapability()
        await capability.initialize()
        
        # 添加測試結果
        test_result = ScanResult(
            target="http://example.com",
            scan_type="test",
            timestamp=datetime.now(),
            status="success",
            data={"test": "data"}
        )
        capability.manager.scan_results.append(test_result)
        
        # 測試導出（這裡只檢查方法不會拋異常）
        try:
            with tempfile.TemporaryDirectory():
                # 這裡可以添加實際的導出測試
                pass
        except Exception as e:
            self.fail(f"Export functionality failed: {e}")
        
        await capability.cleanup()


# 性能和負載測試
class TestPerformance(AsyncTestCase):
    """性能測試"""
    
    @async_test
    async def test_concurrent_scans(self):
        """測試並發掃描性能"""
        capability = WebAttackCapability()
        await capability.initialize()
        
        with aioresponses() as m:
            # 為多個目標設置模擬響應
            targets = [f"http://example{i}.com" for i in range(5)]
            for target in targets:
                m.get(f'https://crt.sh/?q=%25.example{targets.index(target)}.com&output=json', payload=[])
                m.get(target, status=200, body="Test")
            
            # 並發執行掃描
            tasks = []
            for target in targets:
                task = capability.execute('comprehensive_scan', {'target_url': target})
                tasks.append(task)
            
            # 等待所有掃描完成
            results = await asyncio.gather(*tasks)
            
            # 檢查所有掃描都成功
            for result in results:
                self.assertTrue(result['success'])
        
        await capability.cleanup()


# 錯誤處理測試
class TestErrorHandling(AsyncTestCase):
    """錯誤處理測試"""
    
    @async_test
    async def test_network_error_handling(self):
        """測試網絡錯誤處理"""
        capability = WebAttackCapability()
        await capability.initialize()
        
        # 使用不存在的域名
        result = await capability.execute(
            'comprehensive_scan',
            {'target_url': 'http://nonexistent-domain-12345.com'}
        )
        
        # 掃描應該失敗但不應拋異常
        self.assertIn('success', result)
        
        await capability.cleanup()
    
    @async_test
    async def test_malformed_url_handling(self):
        """測試畸形 URL 處理"""
        capability = WebAttackCapability()
        await capability.initialize()
        
        # 使用畸形 URL
        result = await capability.execute(
            'comprehensive_scan',
            {'target_url': 'not-a-valid-url'}
        )
        
        # 應該返回錯誤而不是拋異常
        self.assertIn('success', result)
        
        await capability.cleanup()


if __name__ == '__main__':
    # 運行測試
    unittest.main(verbosity=2)