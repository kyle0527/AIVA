#!/usr/bin/env python3
"""
測試 AIVA BioNeuronRAGAgent API 服務
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api_endpoint(endpoint, method="GET", data=None):
    """測試 API 端點"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n🔍 測試端點: {method} {endpoint}")
    print("-" * 50)
    
    response = None
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        if response:
            print(f"狀態碼: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 請求成功!")
                print(f"回應內容:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"❌ 請求失敗: {response.status_code}")
                print(f"錯誤訊息: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 連接錯誤: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析錯誤: {e}")
        if response:
            print(f"原始回應: {response.text}")


def main():
    """主測試函數"""
    print("=" * 70)
    print("   AIVA BioNeuronRAGAgent API 測試")
    print("=" * 70)
    
    # 等待服務完全啟動
    print("⏳ 等待服務啟動...")
    time.sleep(2)
    
    # 測試基本端點
    test_api_endpoint("/")
    test_api_endpoint("/health")
    
    # 測試統計端點
    test_api_endpoint("/stats")
    
    # 測試執行歷史
    test_api_endpoint("/history")
    
    # 測試 AI 代理呼叫 - 系統命令
    print("\n" + "=" * 70)
    print("   測試 AI 代理功能")
    print("=" * 70)
    
    # 測試 1: 執行系統命令
    test_data = {
        "query": "檢查 Python 版本",
        "command": "python --version"
    }
    test_api_endpoint("/invoke", "POST", test_data)
    
    # 測試 2: 程式碼分析
    test_data = {
        "query": "分析這個 API 服務檔案的程式碼結構",
        "path": "examples/demo_bio_neuron_agent.py"
    }
    test_api_endpoint("/invoke", "POST", test_data)
    
    # 測試 3: 檔案讀取
    test_data = {
        "query": "讀取 README 檔案",
        "path": "README.md"
    }
    test_api_endpoint("/invoke", "POST", test_data)
    
    # 再次檢查統計和歷史
    print("\n" + "=" * 70)
    print("   測試完成後的統計")
    print("=" * 70)
    
    test_api_endpoint("/stats")
    test_api_endpoint("/history")
    
    print("\n" + "=" * 70)
    print("   測試完成! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    main()