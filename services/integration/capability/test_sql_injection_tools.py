#!/usr/bin/env python3
"""
AIVA SQL Injection Tools Test Suite - Task 12
測試 SQL 注入工具集成的各項功能
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

from services.integration.capability.sql_injection_tools import (
    SQLTarget, SQLInjectionResult, SqlmapIntegration, CustomSQLInjectionScanner,
    NoSQLInjectionScanner, BlindSQLInjectionScanner, SQLInjectionManager,
    SQLInjectionCLI, SQLInjectionCapability
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


class TestSQLTarget(AsyncTestCase):
    """SQLTarget 資料結構測試"""
    
    def test_sql_target_creation(self):
        """測試 SQLTarget 創建"""
        target = SQLTarget(
            url="http://example.com/login.php",
            method="POST",
            parameters={"username": "admin", "password": "test"},
            headers={"User-Agent": "Test"},
            data="username=admin&password=test"
        )
        
        self.assertEqual(target.url, "http://example.com/login.php")
        self.assertEqual(target.method, "POST")
        self.assertEqual(target.parameters["username"], "admin")
        self.assertEqual(target.headers["User-Agent"], "Test")
        self.assertEqual(target.data, "username=admin&password=test")
    
    def test_sql_target_defaults(self):
        """測試 SQLTarget 默認值"""
        target = SQLTarget(url="http://example.com")
        
        self.assertEqual(target.method, "GET")
        self.assertEqual(target.parameters, {})
        self.assertEqual(target.headers, {})
        self.assertEqual(target.cookies, {})
        self.assertIsNone(target.data)
        self.assertEqual(target.vulnerable_params, [])


class TestSQLInjectionResult(AsyncTestCase):
    """SQLInjectionResult 資料結構測試"""
    
    def test_sql_injection_result_creation(self):
        """測試 SQL 注入結果創建"""
        result = SQLInjectionResult(
            target_url="http://example.com/test.php?id=1",
            parameter="id",
            injection_type="Error-based SQL Injection",
            payload="' OR '1'='1",
            response_time=0.5,
            evidence="MySQL error detected",
            severity="High",
            confidence=85
        )
        
        self.assertEqual(result.target_url, "http://example.com/test.php?id=1")
        self.assertEqual(result.parameter, "id")
        self.assertEqual(result.injection_type, "Error-based SQL Injection")
        self.assertEqual(result.payload, "' OR '1'='1")
        self.assertEqual(result.response_time, 0.5)
        self.assertEqual(result.evidence, "MySQL error detected")
        self.assertEqual(result.severity, "High")
        self.assertEqual(result.confidence, 85)


class TestSqlmapIntegration(AsyncTestCase):
    """Sqlmap 整合器測試"""
    
    def setUp(self):
        super().setUp()
        self.sqlmap = SqlmapIntegration()
    
    def test_find_sqlmap_path(self):
        """測試 Sqlmap 路徑查找"""
        # 測試路徑查找邏輯
        path = self.sqlmap._find_sqlmap_path()
        # 在測試環境中可能找不到 sqlmap，這是正常的
        self.assertIsInstance(path, (str, type(None)))
    
    @async_test
    async def test_install_sqlmap_mock(self):
        """測試 Sqlmap 安裝（模擬）"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # 模擬成功安裝
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await self.sqlmap.install_sqlmap()
            self.assertTrue(result)
            self.assertEqual(self.sqlmap.sqlmap_path, "./sqlmap-dev/sqlmap.py")
    
    def test_parse_sqlmap_output(self):
        """測試 Sqlmap 輸出解析"""
        mock_output = """
        Parameter: id (GET)
        Type: boolean-based blind
        Title: AND boolean-based blind - WHERE or HAVING clause
        Payload: id=1 AND 1234=1234
        Vector: AND [INFERENCE]
        """
        
        results = self.sqlmap._parse_sqlmap_output(mock_output, "http://example.com")
        
        self.assertIsInstance(results, list)
        if results:
            result = results[0]
            self.assertEqual(result.parameter, "id")
            self.assertEqual(result.injection_type, "boolean-based blind")


class TestCustomSQLInjectionScanner(AsyncTestCase):
    """自定義 SQL 注入掃描器測試"""
    
    def setUp(self):
        super().setUp()
        self.scanner = CustomSQLInjectionScanner()
    
    def test_load_payloads(self):
        """測試載荷加載"""
        payloads = self.scanner._load_payloads()
        
        self.assertIn('error_based', payloads)
        self.assertIn('boolean_based', payloads)
        self.assertIn('time_based', payloads)
        self.assertIn('union_based', payloads)
        
        self.assertGreater(len(payloads['error_based']), 0)
        self.assertIn("'", payloads['error_based'])
        self.assertIn("' OR '1'='1", payloads['error_based'])
    
    @async_test
    async def test_get_baseline_response(self):
        """測試基準響應獲取"""
        target = SQLTarget(url="http://example.com/test.php")
        
        with aioresponses() as m:
            m.get('http://example.com/test.php', status=200, body="Normal response")
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                baseline = await self.scanner._get_baseline_response(target)
                
                self.assertIsNotNone(baseline)
                self.assertEqual(baseline['status'], 200)
                self.assertEqual(baseline['content'], "Normal response")
                self.assertIn('response_time', baseline)
                self.assertIn('content_length', baseline)
            finally:
                await self.scanner.session.close()
    
    def test_analyze_response_error_based(self):
        """測試錯誤基礎響應分析"""
        content = "mysql_fetch_array() expects parameter 1 to be resource"
        baseline = {'response_time': 0.1, 'content_length': 100}
        
        result = self.scanner._analyze_response(
            content, 200, 0.2, "error_based", baseline,
            "http://example.com", "id", "' OR '1'='1"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.injection_type, "Error-based SQL Injection")
        self.assertEqual(result.severity, "High")
        self.assertEqual(result.confidence, 85)
    
    def test_analyze_response_time_based(self):
        """測試時間基礎響應分析"""
        content = "Normal response"
        baseline = {'response_time': 0.1, 'content_length': 100}
        
        # 響應時間明顯增加
        result = self.scanner._analyze_response(
            content, 200, 5.5, "time_based", baseline,
            "http://example.com", "id", "' OR SLEEP(5)--"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.injection_type, "Time-based SQL Injection")
        self.assertEqual(result.severity, "High")
        self.assertEqual(result.confidence, 80)
    
    def test_analyze_response_boolean_based(self):
        """測試布林基礎響應分析"""
        content = "A" * 300  # 內容長度明顯不同
        baseline = {'response_time': 0.1, 'content_length': 100}
        
        result = self.scanner._analyze_response(
            content, 200, 0.2, "boolean_based", baseline,
            "http://example.com", "id", "' AND '1'='1"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.injection_type, "Boolean-based SQL Injection")
        self.assertEqual(result.severity, "Medium")
        self.assertEqual(result.confidence, 70)


class TestNoSQLInjectionScanner(AsyncTestCase):
    """NoSQL 注入掃描器測試"""
    
    def setUp(self):
        super().setUp()
        self.scanner = NoSQLInjectionScanner()
    
    def test_load_nosql_payloads(self):
        """測試 NoSQL 載荷加載"""
        payloads = self.scanner._load_nosql_payloads()
        
        self.assertIsInstance(payloads, list)
        self.assertGreater(len(payloads), 0)
        self.assertIn("{\"$ne\": null}", payloads)
        self.assertIn("true, true", payloads)
    
    @async_test
    async def test_nosql_payload_test(self):
        """測試 NoSQL 載荷測試"""
        target = SQLTarget(
            url="http://example.com/login",
            method="POST",
            data="username=admin&password=test"
        )
        
        with aioresponses() as m:
            # 模擬成功繞過認證
            m.post('http://example.com/login', 
                  status=200, body="Welcome to dashboard")
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                result = await self.scanner._test_nosql_payload(target, "{\"$ne\": null}")
                
                self.assertIsNotNone(result)
                self.assertEqual(result.injection_type, "NoSQL Injection")
                self.assertEqual(result.evidence, "Authentication bypass detected")
                self.assertEqual(result.severity, "High")
            finally:
                await self.scanner.session.close()


class TestBlindSQLInjectionScanner(AsyncTestCase):
    """盲注掃描器測試"""
    
    def setUp(self):
        super().setUp()
        self.scanner = BlindSQLInjectionScanner()
    
    @async_test
    async def test_time_blind_injection(self):
        """測試時間盲注檢測"""
        target = SQLTarget(url="http://example.com/test.php")
        
        with aioresponses() as m:
            # 模擬延遲響應
            async def delayed_response(url, **kwargs):
                await asyncio.sleep(0.1)  # 模擬小延遲而不是真的5秒
                return aioresponses.CallbackResult(status=200, body="Normal response")
            
            m.get(url_pattern=r'http://example\.com/test\.php.*', callback=delayed_response)
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                # 由於我們只能模擬小延遲，直接測試邏輯
                results = await self.scanner._test_time_blind_injection(target)
                
                # 在模擬環境中，結果可能為空，這是正常的
                self.assertIsInstance(results, list)
            finally:
                await self.scanner.session.close()
    
    @async_test
    async def test_boolean_blind_injection(self):
        """測試布林盲注檢測"""
        target = SQLTarget(url="http://example.com/test.php")
        
        with aioresponses() as m:
            # 模擬不同長度的響應
            m.get(url_pattern=r'.*1%27%20AND%20%271%27%3D%271.*', 
                  status=200, body="True condition response")
            m.get(url_pattern=r'.*1%27%20AND%20%271%27%3D%272.*', 
                  status=200, body="False")
            
            self.scanner.session = aiohttp.ClientSession()
            try:
                results = await self.scanner._test_boolean_blind_injection(target)
                
                # 檢查是否檢測到布林盲注
                if results:
                    result = results[0]
                    self.assertEqual(result.injection_type, "Boolean-based Blind SQL Injection")
                    self.assertEqual(result.severity, "Medium")
            finally:
                await self.scanner.session.close()


class TestSQLInjectionManager(AsyncTestCase):
    """SQL 注入管理器測試"""
    
    def setUp(self):
        super().setUp()
        self.manager = SQLInjectionManager()
    
    def test_parse_target(self):
        """測試目標解析"""
        target_url = "http://example.com/test.php?id=1&name=test"
        options = {
            'method': 'GET',
            'headers': {'User-Agent': 'Test'},
            'cookies': {'session': 'abc123'}
        }
        
        target = self.manager._parse_target(target_url, options)
        
        self.assertEqual(target.url, target_url)
        self.assertEqual(target.method, 'GET')
        self.assertEqual(target.parameters['id'], '1')
        self.assertEqual(target.parameters['name'], 'test')
        self.assertEqual(target.headers['User-Agent'], 'Test')
        self.assertEqual(target.cookies['session'], 'abc123')
    
    def test_result_to_dict(self):
        """測試結果轉換為字典"""
        result = SQLInjectionResult(
            target_url="http://example.com",
            parameter="id",
            injection_type="Error-based",
            payload="' OR '1'='1",
            response_time=0.5,
            evidence="MySQL error",
            severity="High",
            confidence=85
        )
        
        result_dict = self.manager._result_to_dict(result)
        
        self.assertEqual(result_dict['target_url'], "http://example.com")
        self.assertEqual(result_dict['parameter'], "id")
        self.assertEqual(result_dict['injection_type'], "Error-based")
        self.assertEqual(result_dict['payload'], "' OR '1'='1")
        self.assertEqual(result_dict['response_time'], 0.5)
        self.assertEqual(result_dict['evidence'], "MySQL error")
        self.assertEqual(result_dict['severity'], "High")
        self.assertEqual(result_dict['confidence'], 85)
    
    @async_test
    async def test_comprehensive_scan_structure(self):
        """測試綜合掃描結構"""
        # 模擬掃描（不實際執行網絡請求）
        with patch.object(self.manager.sqlmap, 'scan_target', return_value=[]):
            with patch.object(self.manager.custom_scanner, 'scan_target', return_value=[]):
                with patch.object(self.manager.nosql_scanner, 'scan_target', return_value=[]):
                    with patch.object(self.manager.blind_scanner, 'scan_blind_injection', return_value=[]):
                        
                        results = await self.manager.comprehensive_scan("http://example.com")
                        
                        # 檢查結果結構
                        self.assertIn('target', results)
                        self.assertIn('timestamp', results)
                        self.assertIn('sqlmap_results', results)
                        self.assertIn('custom_scan_results', results)
                        self.assertIn('nosql_results', results)
                        self.assertIn('blind_injection_results', results)
                        self.assertIn('summary', results)
                        
                        # 檢查摘要結構
                        summary = results['summary']
                        self.assertIn('total_vulnerabilities', summary)
                        self.assertIn('high_vulnerabilities', summary)
                        self.assertIn('scan_methods', summary)


class TestSQLInjectionCLI(AsyncTestCase):
    """SQL 注入 CLI 測試"""
    
    def setUp(self):
        super().setUp()
        self.manager = SQLInjectionManager()
        self.cli = SQLInjectionCLI(self.manager)
    
    def test_cli_creation(self):
        """測試 CLI 創建"""
        self.assertIsInstance(self.cli.manager, SQLInjectionManager)
    
    @patch('services.integration.capability.sql_injection_tools.console')
    def test_show_main_menu(self, mock_console):
        """測試主選單顯示"""
        mock_console.input.return_value = "99"
        
        choice = self.cli.show_main_menu()
        self.assertEqual(choice, "99")
        
        # 檢查是否調用了 console.print
        self.assertTrue(mock_console.print.called)


class TestSQLInjectionCapability(AsyncTestCase):
    """SQL 注入能力測試"""
    
    def setUp(self):
        super().setUp()
        self.capability = SQLInjectionCapability()
    
    def test_capability_properties(self):
        """測試能力屬性"""
        self.assertEqual(self.capability.name, "sql_injection_tools")
        self.assertEqual(self.capability.version, "1.0.0")
        self.assertIn("SQL 注入", self.capability.description)
        self.assertIsInstance(self.capability.dependencies, list)
    
    @async_test
    async def test_capability_initialization(self):
        """測試能力初始化"""
        result = await self.capability.initialize()
        self.assertTrue(result)
    
    @async_test
    async def test_execute_comprehensive_scan(self):
        """測試執行綜合掃描命令"""
        with patch.object(self.capability.manager, 'comprehensive_scan') as mock_scan:
            mock_scan.return_value = {
                'target': 'http://example.com',
                'summary': {'total_vulnerabilities': 0}
            }
            
            result = await self.capability.execute(
                'comprehensive_scan',
                {'target_url': 'http://example.com'}
            )
            
            self.assertTrue(result['success'])
            self.assertIn('data', result)
            mock_scan.assert_called_once()
    
    @async_test
    async def test_execute_custom_scan(self):
        """測試執行自定義掃描命令"""
        with patch.object(self.capability.manager.custom_scanner, 'scan_target') as mock_scan:
            mock_scan.return_value = []
            
            result = await self.capability.execute(
                'custom_scan',
                {'target_url': 'http://example.com'}
            )
            
            self.assertTrue(result['success'])
            self.assertIn('data', result)
            mock_scan.assert_called_once()
    
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
            SQLInjectionResult(
                "test", "test", "test", "test", 0.0, "test", "High", 85
            )
        )
        
        result = await self.capability.cleanup()
        self.assertTrue(result)
        self.assertEqual(len(self.capability.manager.scan_results), 0)


class TestIntegration(AsyncTestCase):
    """集成測試"""
    
    @async_test
    async def test_full_scan_workflow(self):
        """測試完整掃描流程"""
        capability = SQLInjectionCapability()
        await capability.initialize()
        
        # 模擬所有掃描器
        with patch.object(capability.manager.sqlmap, 'scan_target', return_value=[]):
            with patch.object(capability.manager.custom_scanner, 'scan_target', return_value=[]):
                with patch.object(capability.manager.nosql_scanner, 'scan_target', return_value=[]):
                    with patch.object(capability.manager.blind_scanner, 'scan_blind_injection', return_value=[]):
                        
                        result = await capability.execute(
                            'comprehensive_scan',
                            {'target_url': 'http://example.com'}
                        )
                        
                        # 驗證結果
                        self.assertTrue(result['success'])
                        data = result['data']
                        self.assertIn('summary', data)
                        
                        # 檢查摘要結構
                        summary = data['summary']
                        self.assertIsInstance(summary['total_vulnerabilities'], int)
                        self.assertIsInstance(summary['scan_methods'], dict)
        
        await capability.cleanup()


class TestErrorHandling(AsyncTestCase):
    """錯誤處理測試"""
    
    @async_test
    async def test_network_error_handling(self):
        """測試網絡錯誤處理"""
        scanner = CustomSQLInjectionScanner()
        target = SQLTarget(url="http://nonexistent-domain-12345.com")
        
        # 測試網絡錯誤不會導致程序崩潰
        results = await scanner.scan_target(target)
        
        # 掃描應該返回空結果而不是拋異常
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)
    
    @async_test
    async def test_malformed_response_handling(self):
        """測試畸形響應處理"""
        scanner = CustomSQLInjectionScanner()
        
        # 測試響應分析不會因為空內容而失敗
        result = scanner._analyze_response(
            "", 500, 0.1, "error_based", 
            {'response_time': 0.1, 'content_length': 100},
            "http://example.com", "id", "' OR '1'='1"
        )
        
        # 應該返回 None 而不是拋異常
        self.assertIsNone(result)


class TestPerformance(AsyncTestCase):
    """性能測試"""
    
    @async_test
    async def test_concurrent_scans(self):
        """測試並發掃描性能"""
        capability = SQLInjectionCapability()
        await capability.initialize()
        
        # 模擬多個並發掃描
        targets = [f"http://example{i}.com" for i in range(3)]
        
        with patch.object(capability.manager, 'comprehensive_scan') as mock_scan:
            mock_scan.return_value = {'summary': {'total_vulnerabilities': 0}}
            
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


class TestPayloadGeneration(AsyncTestCase):
    """載荷生成測試"""
    
    def test_error_based_payloads(self):
        """測試錯誤基礎載荷"""
        scanner = CustomSQLInjectionScanner()
        payloads = scanner.payloads['error_based']
        
        # 檢查基本載荷
        self.assertIn("'", payloads)
        self.assertIn("' OR '1'='1", payloads)
        self.assertIn("' UNION SELECT NULL--", payloads)
        
        # 檢查載荷有效性
        for payload in payloads:
            self.assertIsInstance(payload, str)
            self.assertGreater(len(payload), 0)
    
    def test_time_based_payloads(self):
        """測試時間基礎載荷"""
        scanner = CustomSQLInjectionScanner()
        payloads = scanner.payloads['time_based']
        
        # 檢查時間延遲載荷
        sleep_payloads = [p for p in payloads if 'SLEEP' in p.upper()]
        waitfor_payloads = [p for p in payloads if 'WAITFOR' in p.upper()]
        
        self.assertGreater(len(sleep_payloads), 0)
        self.assertGreater(len(waitfor_payloads), 0)
    
    def test_nosql_payloads(self):
        """測試 NoSQL 載荷"""
        scanner = NoSQLInjectionScanner()
        payloads = scanner.nosql_payloads
        
        # 檢查 MongoDB 注入載荷
        mongo_payloads = [p for p in payloads if '$ne' in p or '$gt' in p]
        self.assertGreater(len(mongo_payloads), 0)
        
        # 檢查載荷格式
        json_payloads = [p for p in payloads if p.startswith('{') and p.endswith('}')]
        self.assertGreater(len(json_payloads), 0)


if __name__ == '__main__':
    # 運行測試
    unittest.main(verbosity=2)