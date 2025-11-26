import requests
import socket
import threading
import time
from urllib.parse import urljoin, urlparse
import re

print("🔍 AIVA 系統探索組件驗證")
print("=" * 40)

class SystemExplorer:
    def __init__(self, target_url):
        self.target_url = target_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AIVA-Explorer/1.0'
        })
        self.discovered_endpoints = set()
        self.technologies = set()
        
    def discover_technologies(self):
        """技術棧探索"""
        print("\n⚙️ 探索技術棧...")
        
        try:
            response = self.session.get(self.target_url, timeout=10)
            
            # 檢查響應頭
            headers = response.headers
            
            # 服務器信息
            if 'Server' in headers:
                print(f"🌐 服務器: {headers['Server']}")
                self.technologies.add(f"Server: {headers['Server']}")
            
            # 框架檢測
            if 'X-Powered-By' in headers:
                print(f"⚡ 驅動技術: {headers['X-Powered-By']}")
                self.technologies.add(f"Powered-By: {headers['X-Powered-By']}")
            
            # 檢查響應內容
            content = response.text.lower()
            
            # JavaScript 框架檢測
            js_frameworks = {
                'angular': ['ng-', 'angular'],
                'react': ['react', '_react'],  
                'vue': ['vue', 'v-'],
                'jquery': ['jquery', '$'],
                'bootstrap': ['bootstrap']
            }
            
            for framework, patterns in js_frameworks.items():
                if any(pattern in content for pattern in patterns):
                    print(f"📚 前端框架: {framework.title()}")
                    self.technologies.add(f"Frontend: {framework}")
            
            # 檢查 Node.js 特徵
            if 'express' in content or '/js/' in content:
                print(f"🟢 後端技術: Node.js/Express")
                self.technologies.add("Backend: Node.js")
                
        except Exception as e:
            print(f"❌ 技術探索失敗: {e}")
    
    def port_scan(self, host, ports):
        """端口掃描"""
        print(f"\n🚪 掃描 {host} 的端口...")
        
        open_ports = []
        
        def scan_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                if result == 0:
                    open_ports.append(port)
                    print(f"✅ 端口 {port}: 開放")
                sock.close()
            except:
                pass
        
        threads = []
        for port in ports:
            t = threading.Thread(target=scan_port, args=(port,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        return sorted(open_ports)
    
    def discover_directories(self):
        """目錄探索"""
        print("\n📁 探索目錄結構...")
        
        common_dirs = [
            'admin', 'api', 'assets', 'css', 'js', 'images', 
            'uploads', 'backup', 'config', 'test', 'dev',
            'static', 'public', 'private', 'tmp', 'cache',
            'rest', 'graphql', 'websocket'
        ]
        
        found_dirs = []
        
        for directory in common_dirs:
            try:
                url = urljoin(self.target_url, f"/{directory}/")
                response = self.session.get(url, timeout=5)
                
                if response.status_code == 200:
                    print(f"✅ 發現目錄: /{directory}/")
                    found_dirs.append(directory)
                    self.discovered_endpoints.add(f"/{directory}/")
                elif response.status_code == 403:
                    print(f"🔒 受保護目錄: /{directory}/ (403 Forbidden)")
                    found_dirs.append(f"{directory} (protected)")
                    
            except:
                pass
        
        return found_dirs
    
    def discover_api_endpoints(self):
        """API 端點探索"""
        print("\n🔌 探索 API 端點...")
        
        api_endpoints = [
            '/api/users', '/api/user', '/api/login', '/api/auth',
            '/api/products', '/api/search', '/api/config',
            '/rest/user/login', '/rest/products/search',
            '/graphql', '/api/graphql', 
            '/api/v1/', '/api/v2/', '/api/docs', '/swagger'
        ]
        
        found_apis = []
        
        for endpoint in api_endpoints:
            try:
                url = urljoin(self.target_url, endpoint)
                response = self.session.get(url, timeout=5)
                
                if response.status_code in [200, 201, 400, 401, 403]:
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'json' in content_type.lower():
                        print(f"✅ API 端點: {endpoint} (JSON)")
                        found_apis.append(f"{endpoint} (JSON)")
                    elif response.status_code == 200:
                        print(f"✅ 端點: {endpoint} ({response.status_code})")
                        found_apis.append(endpoint)
                    elif response.status_code in [401, 403]:
                        print(f"🔐 受保護端點: {endpoint} ({response.status_code})")
                        found_apis.append(f"{endpoint} (protected)")
                        
                    self.discovered_endpoints.add(endpoint)
                    
            except:
                pass
        
        return found_apis
    
    def analyze_security_headers(self):
        """安全標頭分析"""
        print("\n🛡️ 分析安全標頭...")
        
        try:
            response = self.session.get(self.target_url, timeout=10)
            headers = response.headers
            
            security_headers = {
                'X-Content-Type-Options': '防止 MIME 類型嗅探',
                'X-Frame-Options': '防止點擊劫持',
                'X-XSS-Protection': 'XSS 保護',
                'Content-Security-Policy': '內容安全政策',
                'Strict-Transport-Security': 'HTTPS 強制',
                'Referrer-Policy': '引用來源政策'
            }
            
            present_headers = []
            missing_headers = []
            
            for header, description in security_headers.items():
                if header in headers:
                    print(f"✅ {header}: {headers[header]}")
                    present_headers.append(header)
                else:
                    print(f"❌ 缺少 {header} ({description})")
                    missing_headers.append(header)
            
            return present_headers, missing_headers
            
        except Exception as e:
            print(f"❌ 安全標頭分析失敗: {e}")
            return [], []

def main():
    target_url = "http://localhost:3000"
    
    explorer = SystemExplorer(target_url)
    
    print(f"🎯 目標: {target_url}")
    
    # 1. 技術棧探索
    explorer.discover_technologies()
    
    # 2. 端口掃描
    host = urlparse(target_url).hostname
    common_ports = [22, 23, 25, 53, 80, 110, 443, 993, 995, 3000, 5432, 6379]
    open_ports = explorer.port_scan(host, common_ports)
    
    # 3. 目錄探索
    found_dirs = explorer.discover_directories()
    
    # 4. API 端點探索
    found_apis = explorer.discover_api_endpoints()
    
    # 5. 安全標頭分析
    present_headers, missing_headers = explorer.analyze_security_headers()
    
    # 探索總結
    print("\n" + "=" * 40)
    print("🔍 系統探索總結:")
    print(f"⚙️  發現技術: {len(explorer.technologies)} 項")
    print(f"🚪 開放端口: {len(open_ports)} 個 - {open_ports}")
    print(f"📁 發現目錄: {len(found_dirs)} 個")
    print(f"🔌 API 端點: {len(found_apis)} 個")
    print(f"🛡️  安全標頭: {len(present_headers)}/{len(present_headers) + len(missing_headers)}")
    print(f"📊 總發現項: {len(explorer.discovered_endpoints)} 個端點")
    
    if len(missing_headers) > 0:
        print(f"⚠️  安全建議: 添加 {len(missing_headers)} 個缺失的安全標頭")

if __name__ == "__main__":
    main()