#!/usr/bin/env python
"""測試 AIVA 掃描功能"""

import requests
import time

def main():
    print("=== AIVA 掃描測試 ===")
    print()
    
    # 1. 健康檢查
    print("1. 健康檢查...")
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        print(f"   狀態: {r.json()['status']}")
    except Exception as e:
        print(f"   錯誤: {e}")
        return
    
    # 2. 發送掃描請求
    print()
    print("2. 發送掃描請求...")
    try:
        r = requests.post(
            "http://localhost:8000/scan",
            json={
                "target": "http://localhost:3000",
                "scan_type": "comprehensive",
                "max_depth": 2
            },
            timeout=30
        )
        result = r.json()
        scan_id = result.get("scan_id", "")
        print(f"   掃描 ID: {scan_id}")
        print(f"   狀態: {result.get('status')}")
        print(f"   訊息: {result.get('message')}")
    except Exception as e:
        print(f"   錯誤: {e}")
        return
    
    # 3. 輪詢狀態
    print()
    print("3. 監控掃描進度...")
    for i in range(6):
        time.sleep(5)
        try:
            r = requests.get(f"http://localhost:8000/status/{scan_id}", timeout=5)
            status = r.json()
            print(f"   [{i*5}s] 狀態: {status.get('status')}, 階段: {status.get('phase')}, 進度: {status.get('progress')}")
        except Exception as e:
            print(f"   [{i*5}s] 查詢錯誤: {e}")
    
    print()
    print("=== 測試完成 ===")

if __name__ == "__main__":
    main()
