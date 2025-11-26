#!/usr/bin/env python3
"""
簡單的 SSRF 測試服務器
模擬一個存在 SSRF 漏洞的 Web 應用
"""
import http.server
import socketserver
import urllib.request
import urllib.parse
from urllib.parse import urlparse, parse_qs

class SSRFTestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # 解析請求路徑和參數
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        
        print(f"[SSRF Server] ===== 新請求 =====")
        print(f"[SSRF Server] 完整路徑: {self.path}")
        print(f"[SSRF Server] 來源IP: {self.client_address[0]}:{self.client_address[1]}")
        print(f"[SSRF Server] User-Agent: {self.headers.get('User-Agent', 'None')}")
        print(f"[SSRF Server] 查詢參數: {query_params}")
        print(f"[SSRF Server] ====================")
        
        print(f"[SSRF Server] 收到請求: {self.path}")
        
        # 檢查是否有 URL 參數 (模擬存在 SSRF 漏洞)
        if 'url' in query_params:
            target_url = query_params['url'][0]
            print(f"[SSRF Server] 嘗試訪問目標 URL: {target_url}")
            
            try:
                # 這裡模擬易受攻擊的代碼 - 直接請求用戶提供的 URL
                with urllib.request.urlopen(target_url, timeout=5) as response:
                    content = response.read()
                    
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                response_html = f"""
                <html>
                <head><title>SSRF Test Result</title></head>
                <body>
                    <h1>SSRF Test - URL Content Retrieved</h1>
                    <p>Successfully fetched content from: {target_url}</p>
                    <hr>
                    <pre>{content.decode('utf-8', errors='ignore')[:500]}...</pre>
                </body>
                </html>
                """
                self.wfile.write(response_html.encode())
                print(f"[SSRF Server] 成功從 {target_url} 獲取內容")
                
            except Exception as e:
                print(f"[SSRF Server] 錯誤: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = f"""
                <html>
                <body>
                    <h1>Error</h1>
                    <p>Failed to fetch: {target_url}</p>
                    <p>Error: {e}</p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
        else:
            # 默認頁面
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            default_html = """
            <html>
            <head><title>SSRF Test Server</title></head>
            <body>
                <h1>SSRF Test Server</h1>
                <p>This server is vulnerable to SSRF attacks.</p>
                <p>Usage: /?url=http://target-url</p>
                <form>
                    <label>URL: <input name="url" placeholder="http://example.com"></label>
                    <button type="submit">Fetch</button>
                </form>
            </body>
            </html>
            """
            self.wfile.write(default_html.encode())

if __name__ == '__main__':
    PORT = 9999
    Handler = SSRFTestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"[SSRF Server] 啟動在端口 {PORT}")
        print(f"[SSRF Server] 訪問 http://localhost:{PORT}")
        print(f"[SSRF Server] SSRF 測試: http://localhost:{PORT}/?url=http://httpbin.org/get")
        httpd.serve_forever()