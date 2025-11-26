#!/usr/bin/env python3
"""測試 StaticContentParser 為什麼無法解析鏈接"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from services.scan.engines.python_engine.core_crawling_engine.static_content_parser import StaticContentParser

# 測試所有四個靶場
targets = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3003",
    "http://localhost:8080",
]

parser = StaticContentParser()

for url in targets:
    print(f"\n{'='*80}")
    print(f"測試: {url}")
    print(f"{'='*80}")
    
    try:
        response = httpx.get(url, timeout=10.0)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Response size: {len(response.text)} bytes")
        
        # 解析
        assets, forms_count = parser.extract(url, response)
        print(f"\n解析結果:")
        print(f"  資產數: {len(assets)}")
        print(f"  表單數: {forms_count}")
        
        if assets:
            print(f"\n  前 5 個資產:")
            for i, asset in enumerate(assets[:5], 1):
                print(f"    [{i}] {asset.type}: {asset.value}")
        else:
            print(f"\n  ❌ 沒有解析到任何資產！")
            print(f"\n  HTML 前 500 字元:")
            print(f"  {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
