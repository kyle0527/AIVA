#!/usr/bin/env python3
"""手動打開瀏覽器測試靶場內容"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from bs4 import BeautifulSoup

targets = [
    ("http://localhost:3000", "Juice Shop 3000"),
    ("http://localhost:3001", "靶場 3001"),
    ("http://localhost:3003", "靶場 3003"),
    ("http://localhost:8080", "靶場 8080"),
]

print("=" * 80)
print("檢查所有靶場的實際 HTML 內容")
print("=" * 80)

for url, name in targets:
    print(f"\n\n{'#'*80}")
    print(f"# {name}: {url}")
    print(f"{'#'*80}")
    
    try:
        response = httpx.get(url, timeout=10.0, follow_redirects=True)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Size: {len(response.text)} bytes")
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 檢查 <a> 標籤
        links = soup.find_all('a')
        print(f"\n找到 {len(links)} 個 <a> 標籤")
        if links:
            print("前 10 個鏈接:")
            for i, a in enumerate(links[:10], 1):
                href = a.get('href', '')
                text = a.get_text(strip=True)[:50]
                print(f"  [{i}] href={href} | text={text}")
        
        # 檢查 <form> 標籤
        forms = soup.find_all('form')
        print(f"\n找到 {len(forms)} 個 <form> 標籤")
        if forms:
            print("前 5 個表單:")
            for i, form in enumerate(forms[:5], 1):
                action = form.get('action', '')
                method = form.get('method', '')
                print(f"  [{i}] action={action} method={method}")
        
        # 檢查是否為 SPA (有沒有 app-root 之類的)
        app_root = soup.find('app-root') or soup.find(id='app') or soup.find(class_='app')
        if app_root:
            print(f"\n⚠️ 這是 SPA 應用！找到: {app_root.name}")
        
        # 顯示 HTML body 前 800 字元
        print(f"\n<body> 內容前 800 字元:")
        body = soup.find('body')
        if body:
            print(body.get_text()[:800])
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
