# -*- coding: utf-8 -*-
"""
AIVA API 測試腳本

提供 API 端點的快速測試和驗證功能。
包含認證測試、高價值模組測試、系統測試等。
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

class AIVAAPITester:
    """AIVA API 測試器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.access_token: Optional[str] = None
        self.client = httpx.AsyncClient()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def get_headers(self) -> Dict[str, str]:
        """獲取請求標頭"""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers
    
    async def test_health(self) -> bool:
        """測試健康檢查端點"""
        print("🔍 Testing health check...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Health check passed: {result.get('status')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_auth(self, username: str = "admin", password: str = "aiva-admin-2025") -> bool:
        """測試認證"""
        print(f"🔐 Testing authentication with {username}...")
        try:
            login_data = {
                "username": username,
                "password": password
            }
            response = await self.client.post(
                f"{self.base_url}/auth/login",
                json=login_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result.get("access_token")
                print(f"✅ Authentication successful for {username}")
                print(f"   Role: {result.get('user', {}).get('role')}")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    async def test_user_info(self) -> bool:
        """測試用戶信息獲取"""
        print("👤 Testing user info...")
        try:
            response = await self.client.get(
                f"{self.base_url}/auth/me",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ User info retrieved: {result.get('username')} ({result.get('role')})")
                return True
            else:
                print(f"❌ User info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ User info error: {e}")
            return False
    
    async def test_mass_assignment_scan(self) -> Optional[str]:
        """測試 Mass Assignment 掃描"""
        print("🔍 Testing Mass Assignment scan...")
        try:
            scan_data = {
                "target": "https://httpbin.org",
                "update_endpoint": "/put",
                "auth_headers": {"Authorization": "Bearer test-token"},
                "test_fields": ["admin", "role", "is_admin"]
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/security/mass-assignment",
                json=scan_data,
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                scan_id = result.get("scan_id")
                print(f"✅ Mass Assignment scan started: {scan_id}")
                print(f"   Status: {result.get('status')}")
                print(f"   Potential value: {result.get('potential_value')}")
                return scan_id
            else:
                print(f"❌ Mass Assignment scan failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Mass Assignment scan error: {e}")
            return None
    
    async def test_scan_status(self, scan_id: str) -> bool:
        """測試掃描狀態查詢"""
        print(f"📊 Testing scan status for {scan_id}...")
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/scans/{scan_id}",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Scan status retrieved: {result.get('status')}")
                if result.get('result'):
                    print(f"   Result available: {type(result.get('result'))}")
                return True
            else:
                print(f"❌ Scan status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Scan status error: {e}")
            return False
    
    async def test_scan_list(self) -> bool:
        """測試掃描列表"""
        print("📋 Testing scan list...")
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/scans",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Scan list retrieved: {result.get('total')} scans")
                return True
            else:
                print(f"❌ Scan list failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Scan list error: {e}")
            return False
    
    async def test_system_stats(self) -> bool:
        """測試系統統計"""
        print("📈 Testing system stats...")
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/admin/stats",
                headers=self.get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ System stats retrieved: {result.get('total_scans')} total scans")
                return True
            else:
                print(f"❌ System stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ System stats error: {e}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, bool]:
        """執行綜合測試"""
        print("🚀 Starting comprehensive API test...")
        print("=" * 50)
        
        results = {}
        
        # 基礎測試
        results['health_check'] = await self.test_health()
        results['authentication'] = await self.test_auth()
        
        if results['authentication']:
            results['user_info'] = await self.test_user_info()
            
            # 功能測試
            scan_id = await self.test_mass_assignment_scan()
            if scan_id:
                # 等待掃描開始
                await asyncio.sleep(2)
                results['scan_status'] = await self.test_scan_status(scan_id)
            else:
                results['scan_status'] = False
            
            results['scan_list'] = await self.test_scan_list()
            results['system_stats'] = await self.test_system_stats()
        else:
            results.update({
                'user_info': False,
                'scan_status': False,
                'scan_list': False,
                'system_stats': False
            })
        
        # 輸出結果摘要
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! API is fully operational.")
        elif passed >= total * 0.8:
            print("⚠️ Most tests passed. Minor issues detected.")
        else:
            print("❌ Multiple test failures. API may have issues.")
        
        return results

async def main():
    """主測試函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIVA API Tester')
    parser.add_argument('--host', default='localhost', help='API host (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='API port (default: 8000)')
    parser.add_argument('--username', default='admin', help='Test username (default: admin)')
    parser.add_argument('--password', default='aiva-admin-2025', help='Test password')
    parser.add_argument('--test', choices=['health', 'auth', 'scan', 'all'], default='all',
                       help='Specific test to run (default: all)')
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    async with AIVAAPITester(base_url) as tester:
        print(f"🎯 Testing AIVA API at {base_url}")
        print(f"📅 Test started at {datetime.now()}")
        print()
        
        if args.test == 'all':
            await tester.run_comprehensive_test()
        elif args.test == 'health':
            await tester.test_health()
        elif args.test == 'auth':
            await tester.test_auth(args.username, args.password)
        elif args.test == 'scan':
            if await tester.test_auth(args.username, args.password):
                scan_id = await tester.test_mass_assignment_scan()
                if scan_id:
                    await asyncio.sleep(3)
                    await tester.test_scan_status(scan_id)

if __name__ == '__main__':
    asyncio.run(main())