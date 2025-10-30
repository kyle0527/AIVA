import requests
import json
import time
import random

print("🤖 AIVA 自動化測試組件驗證")
print("=" * 40)

class JuiceShopTester:
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.session = requests.Session()
        
    def test_user_registration(self):
        """測試用戶註冊功能"""
        print("\n📝 測試用戶註冊...")
        
        # 生成隨機用戶
        timestamp = int(time.time())
        test_user = {
            "email": f"test{timestamp}@juice-sh.op",
            "password": "Password123!",
            "passwordRepeat": "Password123!",
            "securityQuestion": {
                "id": 1,
                "question": "Your eldest siblings middle name?",
                "answer": "TestAnswer"
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/Users/",
                json=test_user,
                timeout=10
            )
            
            if response.status_code == 201:
                print("✅ 用戶註冊成功")
                return response.json()['data']
            else:
                print(f"❌ 註冊失敗: {response.status_code} - {response.text[:100]}")
                return None
        except Exception as e:
            print(f"❌ 註冊請求失敗: {e}")
            return None
    
    def test_user_login(self, email, password):
        """測試用戶登入功能"""
        print(f"\n🔐 測試用戶登入: {email}")
        
        login_data = {
            "email": email,
            "password": password
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/rest/user/login",
                json=login_data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ 用戶登入成功")
                auth_data = response.json()
                return auth_data.get('authentication', {}).get('token')
            else:
                print(f"❌ 登入失敗: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 登入請求失敗: {e}")
            return None
    
    def test_product_search(self):
        """測試產品搜索功能"""
        print("\n🔍 測試產品搜索...")
        
        search_terms = ["apple", "banana", "coffee", "juice"]
        
        for term in search_terms:
            try:
                response = self.session.get(
                    f"{self.base_url}/rest/products/search",
                    params={"q": term},
                    timeout=10
                )
                
                if response.status_code == 200:
                    results = response.json()
                    print(f"✅ 搜索 '{term}': 找到 {len(results['data'])} 個結果")
                else:
                    print(f"❌ 搜索 '{term}' 失敗: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 搜索請求失敗: {e}")
    
    def test_sql_injection_attempts(self):
        """測試各種 SQL 注入嘗試"""
        print("\n💉 測試 SQL 注入防護...")
        
        injection_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "1' AND 1=1--",
            "1' AND 1=2--"
        ]
        
        vulnerable_endpoints = [
            "/rest/user/login",
            "/rest/products/search",
            "/api/Users/"
        ]
        
        vulnerabilities_found = 0
        
        for endpoint in vulnerable_endpoints:
            print(f"\n  測試端點: {endpoint}")
            
            for payload in injection_payloads:
                try:
                    if endpoint == "/rest/user/login":
                        data = {"email": payload, "password": "test"}
                        response = self.session.post(f"{self.base_url}{endpoint}", json=data, timeout=5)
                    elif endpoint == "/rest/products/search":
                        response = self.session.get(f"{self.base_url}{endpoint}", params={"q": payload}, timeout=5)
                    else:
                        continue
                    
                    # 檢查響應是否表明漏洞
                    if (response.status_code == 500 and 
                        ("error" in response.text.lower() or "sql" in response.text.lower())):
                        vulnerabilities_found += 1
                        print(f"    🚨 疑似漏洞: {payload[:20]}... -> {response.status_code}")
                    elif response.status_code == 200 and "OR" in payload:
                        print(f"    ⚠️  可疑響應: {payload[:20]}... -> {response.status_code}")
                        
                except Exception as e:
                    pass  # 忽略連接錯誤
        
        print(f"\n📊 發現 {vulnerabilities_found} 個潛在 SQL 注入點")
        return vulnerabilities_found

# 執行測試
def main():
    tester = JuiceShopTester()
    
    # 1. 測試用戶註冊
    user_data = tester.test_user_registration()
    
    # 2. 測試用戶登入
    if user_data:
        token = tester.test_user_login(user_data['email'], "Password123!")
    
    # 3. 測試產品搜索
    tester.test_product_search()
    
    # 4. 測試 SQL 注入防護
    vuln_count = tester.test_sql_injection_attempts()
    
    print("\n" + "=" * 40)
    print("🎯 自動化測試總結:")
    print(f"✅ 基本功能測試: 完成")
    print(f"🔍 產品搜索測試: 完成") 
    print(f"💉 SQL 注入測試: 發現 {vuln_count} 個潛在漏洞")
    print(f"🏆 測試狀態: {'通過' if vuln_count > 0 else '無漏洞發現'}")

if __name__ == "__main__":
    main()