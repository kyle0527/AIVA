#!/usr/bin/env python3
"""
最小化診斷測試：追蹤 HTTP 請求為什麼發不出去
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.scan.engines.python_engine.core_crawling_engine.http_client_hi import HiHttpClient
from services.scan.engines.python_engine.authentication_manager import AuthenticationManager
from services.scan.engines.python_engine.header_configuration import HeaderConfiguration


async def test_http_directly():
    """直接測試 HiHttpClient 能否發送請求"""
    print("=" * 80)
    print("🔍 HTTP 請求診斷測試")
    print("=" * 80)
    
    # 步驟 1: 初始化組件
    print("\n[Step 1] 初始化 HTTP 客戶端組件...")
    auth = AuthenticationManager(None)
    headers = HeaderConfiguration(None)
    
    client = HiHttpClient(
        auth,
        headers,
        requests_per_second=10.0,
        per_host_rps=5.0,
        retries=3,
        timeout=10.0
    )
    print("   ✅ HTTP 客戶端初始化完成")
    
    # 步驟 2: 發送測試請求
    test_urls = [
        "http://localhost:3000",
        "http://example.com",
        "http://httpbin.org/get"
    ]
    
    for url in test_urls:
        print(f"\n[Step 2] 測試 URL: {url}")
        print(f"   ⏳ 發送 GET 請求...")
        
        try:
            response = await client.get(url)
            
            if response is None:
                print(f"   ❌ 返回 None - 請求失敗")
            else:
                print(f"   ✅ 成功收到響應")
                print(f"      Status: {response.status_code}")
                print(f"      Content-Type: {response.headers.get('content-type', 'N/A')}")
                print(f"      Body Size: {len(response.text)} bytes")
                print(f"      Body Preview: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ 異常: {e}")
            import traceback
            traceback.print_exc()
    
    # 步驟 3: 清理
    print(f"\n[Step 3] 清理資源...")
    await client.close()
    print("   ✅ 清理完成")


async def test_with_logging():
    """啟用詳細日誌的測試"""
    import logging
    
    # 啟用 DEBUG 日誌
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("🔍 啟用詳細日誌的測試")
    print("=" * 80)
    
    auth = AuthenticationManager(None)
    headers = HeaderConfiguration(None)
    
    client = HiHttpClient(
        auth,
        headers,
        requests_per_second=10.0,
        per_host_rps=5.0,
        retries=3,
        timeout=10.0
    )
    
    print("\n[Testing] http://localhost:3000")
    response = await client.get("http://localhost:3000")
    
    if response:
        print(f"✅ 成功: {response.status_code}")
    else:
        print(f"❌ 失敗: response = None")
    
    await client.close()


if __name__ == "__main__":
    print("開始 HTTP 請求診斷測試\n")
    
    # 測試 1: 基礎測試
    asyncio.run(test_http_directly())
    
    # 測試 2: 詳細日誌測試
    # asyncio.run(test_with_logging())
