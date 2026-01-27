"""
被動流量分析器
OWASP A02:2021 - Cryptographic Failures
OWASP A05:2021 - Security Misconfiguration

分析 HTTP 流量中的敏感資訊和安全配置問題
"""

import re
import json
from typing import List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

# OWASP 常量定義
OWASP_A02_2021 = 'A02:2021'
OWASP_A05_2021 = 'A05:2021'

@dataclass
class PassiveFinding:
    """被動分析發現"""
    category: str
    severity: str
    owasp: str
    description: str
    evidence: str
    remediation: str
    url: Optional[str] = None


class PassiveAnalyzer:
    """被動流量分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self._setup_patterns()
    
    def _setup_patterns(self):
        """設置檢測模式"""
        # 敏感數據模式
        self.sensitive_patterns = {
            'email': (
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'Email address exposed'
            ),
            'ssn': (
                r'\b\d{3}-\d{2}-\d{4}\b',
                'Social Security Number exposed'
            ),
            'credit_card': (
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                'Credit card number exposed'
            ),
            'api_key': (
                r'\b[A-Za-z0-9_-]{32,}\b',
                'Possible API key exposed'
            ),
            'jwt': (
                r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*',
                'JWT token exposed'
            ),
            'aws_key': (
                r'AKIA[0-9A-Z]{16}',
                'AWS Access Key ID exposed'
            ),
            'private_key': (
                r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----',
                'Private key exposed'
            ),
            'password': (
                r'password["\']?\s*[:=]\s*["\'][^"\']{3,}["\']',
                'Password in response'
            )
        }
        
        # 安全頭部
        self.security_headers = {
            'strict-transport-security': {
                'required': True,
                'min_age': 31536000,  # 1 year
                'description': 'HSTS not set or too short'
            },
            'content-security-policy': {
                'required': True,
                'description': 'CSP not set'
            },
            'x-frame-options': {
                'required': True,
                'values': ['DENY', 'SAMEORIGIN'],
                'description': 'X-Frame-Options not set'
            },
            'x-content-type-options': {
                'required': True,
                'values': ['nosniff'],
                'description': 'X-Content-Type-Options not set'
            },
            'referrer-policy': {
                'required': True,
                'values': ['no-referrer', 'same-origin', 'strict-origin'],
                'description': 'Referrer-Policy not secure'
            },
            'permissions-policy': {
                'required': False,
                'description': 'Permissions-Policy not set'
            }
        }
    
    def analyze_har(self, har_path: str) -> List[PassiveFinding]:
        """
        分析 HAR 文件
        
        Args:
            har_path: HAR 文件路徑
            
        Returns:
            發現列表
        """
        findings = []
        
        try:
            with open(har_path, 'r', encoding='utf-8') as f:
                har_data = json.load(f)
            
            entries = har_data.get('log', {}).get('entries', [])
            
            for entry in entries:
                request = entry.get('request', {})
                response = entry.get('response', {})
                url = request.get('url', '')
                
                # 分析請求
                findings.extend(self._analyze_request(request, url))
                
                # 分析響應
                findings.extend(self._analyze_response(response, url))
                
        except Exception as e:
            findings.append(PassiveFinding(
                category='Error',
                severity='Info',
                owasp='N/A',
                description=f'Failed to analyze HAR: {str(e)}',
                evidence='',
                remediation='Check HAR file format'
            ))
        
        return findings
    
    def _analyze_request(self, request: Dict, url: str) -> List[PassiveFinding]:
        """分析 HTTP 請求"""
        findings = []
        
        # 檢查 URL 中的敏感信息
        findings.extend(self._check_sensitive_data_in_url(url))
        
        # 檢查查詢參數
        parsed_url = urlparse(url)
        if parsed_url.query:
            params = parse_qs(parsed_url.query)
            findings.extend(self._check_sensitive_params(params, url))
        
        # 檢查 Cookie
        headers = {h['name'].lower(): h['value'] for h in request.get('headers', [])}
        if 'cookie' in headers:
            findings.extend(self._analyze_cookies(headers['cookie'], url))
        
        # 檢查請求體
        if 'postData' in request:
            post_data = request['postData'].get('text', '')
            findings.extend(self._check_sensitive_data_in_body(post_data, url))
        
        return findings
    
    def _analyze_response(self, response: Dict, url: str) -> List[PassiveFinding]:
        """分析 HTTP 響應"""
        findings = []
        
        # 檢查響應頭
        headers = {h['name'].lower(): h['value'] for h in response.get('headers', [])}
        findings.extend(self._check_security_headers(headers, url))
        
        # 檢查 Set-Cookie
        for header in response.get('headers', []):
            if header['name'].lower() == 'set-cookie':
                findings.extend(self._analyze_set_cookie(header['value'], url))
        
        # 檢查響應體
        content = response.get('content', {})
        if 'text' in content:
            findings.extend(self._check_sensitive_data_in_body(content['text'], url))
        
        # 檢查錯誤消息
        if response.get('status', 0) >= 400:
            findings.extend(self._check_error_disclosure(response, url))
        
        return findings
    
    def _check_sensitive_data_in_url(self, url: str) -> List[PassiveFinding]:
        """檢查 URL 中的敏感數據"""
        findings = []
        
        for pattern_name, (pattern, description) in self.sensitive_patterns.items():
            matches = re.finditer(pattern, url, re.IGNORECASE)
            for match in matches:
                findings.append(PassiveFinding(
                    category='Sensitive Data in URL',
                    severity='High',
                    owasp='A02:2021',
                    description=description,
                    evidence=f'Found in URL: {match.group()}',
                    remediation='Never include sensitive data in URLs',
                    url=url
                ))
        
        return findings
    
    def _check_sensitive_params(self, params: dict[str, List[str]], url: str) -> List[PassiveFinding]:
        """檢查敏感參數名稱"""
        findings = []
        sensitive_param_names = [
            'password', 'passwd', 'pwd', 'secret', 'token',
            'api_key', 'apikey', 'access_token', 'auth',
            'credit_card', 'cc', 'ssn', 'social_security'
        ]
        
        for param_name in params.keys():
            if any(sensitive in param_name.lower() for sensitive in sensitive_param_names):
                findings.append(PassiveFinding(
                    category='Sensitive Parameter in URL',
                    severity='High',
                    owasp='A02:2021',
                    description=f'Sensitive parameter "{param_name}" in URL',
                    evidence=f'Parameter: {param_name}',
                    remediation='Use POST method and encrypt data',
                    url=url
                ))
        
        return findings
    
    def _check_sensitive_data_in_body(self, body: str, url: str) -> List[PassiveFinding]:
        """檢查請求/響應體中的敏感數據"""
        findings = []
        
        for pattern_name, (pattern, description) in self.sensitive_patterns.items():
            matches = re.finditer(pattern, body, re.IGNORECASE)
            count = sum(1 for _ in matches)
            
            if count > 0:
                findings.append(PassiveFinding(
                    category='Sensitive Data in Body',
                    severity='Medium',
                    owasp='A02:2021',
                    description=f'{description} ({count} occurrences)',
                    evidence=f'Pattern: {pattern_name}',
                    remediation='Encrypt sensitive data and use proper access controls',
                    url=url
                ))
        
        return findings
    
    def _check_security_headers(self, headers: dict[str, str], url: str) -> List[PassiveFinding]:
        """檢查安全頭部"""
        findings = []
        
        for header_name, config in self.security_headers.items():
            header_value = headers.get(header_name, '')
            
            if config['required'] and not header_value:
                findings.append(PassiveFinding(
                    category='Missing Security Header',
                    severity='Medium',
                    owasp='A05:2021',
                    description=config['description'],
                    evidence=f'Header "{header_name}" not found',
                    remediation=f'Add {header_name} header',
                    url=url
                ))
                continue
            
            # 檢查特定值
            if 'values' in config and header_value:
                if not any(v in header_value for v in config['values']):
                    findings.append(PassiveFinding(
                        category='Weak Security Header',
                        severity='Low',
                        owasp='A05:2021',
                        description=f'{header_name} has weak value',
                        evidence=f'Current value: {header_value}',
                        remediation=f'Use one of: {", ".join(config["values"])}',
                        url=url
                    ))
            
            # 檢查 HSTS max-age
            if header_name == 'strict-transport-security' and header_value:
                max_age_match = re.search(r'max-age=(\d+)', header_value)
                if max_age_match:
                    max_age = int(max_age_match.group(1))
                    if max_age < config['min_age']:
                        findings.append(PassiveFinding(
                            category='Weak HSTS',
                            severity='Low',
                            owasp='A05:2021',
                            description='HSTS max-age too short',
                            evidence=f'Current: {max_age}s, Recommended: {config["min_age"]}s',
                            remediation='Increase max-age to at least 1 year',
                            url=url
                        ))
        
        return findings
    
    def _analyze_cookies(self, cookie_header: str, url: str) -> List[PassiveFinding]:
        """分析 Cookie 頭"""
        findings = []
        
        # 檢查是否包含敏感名稱
        sensitive_cookie_names = ['session', 'token', 'auth', 'jwt']
        
        for cookie in cookie_header.split(';'):
            cookie = cookie.strip()
            if '=' in cookie:
                name, value = cookie.split('=', 1)
                
                if any(s in name.lower() for s in sensitive_cookie_names):
                    if len(value) < 16:
                        findings.append(PassiveFinding(
                            category='Weak Session Token',
                            severity='High',
                            owasp='A02:2021',
                            description='Session token too short',
                            evidence=f'Cookie: {name}, Length: {len(value)}',
                            remediation='Use longer session tokens (>= 128 bits)',
                            url=url
                        ))
        
        return findings
    
    def _analyze_set_cookie(self, set_cookie: str, url: str) -> List[PassiveFinding]:
        """分析 Set-Cookie 頭"""
        findings = []
        
        # 提取 cookie 名稱
        cookie_name = set_cookie.split('=')[0] if '=' in set_cookie else ''
        
        # 檢查安全屬性
        attributes = set_cookie.lower()
        
        if 'httponly' not in attributes:
            findings.append(PassiveFinding(
                category='Cookie without HttpOnly',
                severity='Medium',
                owasp='A05:2021',
                description='Cookie missing HttpOnly flag',
                evidence=f'Cookie: {cookie_name}',
                remediation='Add HttpOnly flag to prevent XSS access',
                url=url
            ))
        
        if 'secure' not in attributes:
            findings.append(PassiveFinding(
                category='Cookie without Secure',
                severity='Medium',
                owasp='A02:2021',
                description='Cookie missing Secure flag',
                evidence=f'Cookie: {cookie_name}',
                remediation='Add Secure flag to enforce HTTPS',
                url=url
            ))
        
        if 'samesite' not in attributes:
            findings.append(PassiveFinding(
                category='Cookie without SameSite',
                severity='Low',
                owasp='A05:2021',
                description='Cookie missing SameSite attribute',
                evidence=f'Cookie: {cookie_name}',
                remediation='Add SameSite=Strict or Lax',
                url=url
            ))
        
        return findings
    
    def _check_error_disclosure(self, response: Dict, url: str) -> List[PassiveFinding]:
        """檢查錯誤信息洩露"""
        findings = []
        
        content = response.get('content', {}).get('text', '')
        status = response.get('status', 0)
        
        # 檢查堆棧跟踪
        stack_trace_patterns = [
            r'at\s+[\w\.<>]+\s*\([^)]+\.java:\d+\)',  # Java
            r'File\s+"[^"]+\.py",\s+line\s+\d+',  # Python
            r'at\s+[\w\.<>]+\s*\([^)]+\.cs:\d+\)',  # C#
            r'Error\s+in\s+/[\w/]+\.php\s+on\s+line\s+\d+',  # PHP
        ]
        
        for pattern in stack_trace_patterns:
            if re.search(pattern, content):
                findings.append(PassiveFinding(
                    category='Stack Trace Disclosure',
                    severity='Medium',
                    owasp='A05:2021',
                    description='Application stack trace exposed',
                    evidence=f'HTTP {status}',
                    remediation='Use generic error pages',
                    url=url
                ))
                break
        
        # 檢查數據庫錯誤
        db_error_patterns = [
            r'SQL syntax.*MySQL',
            r'Warning.*mysql_',
            r'PostgreSQL.*ERROR',
            r'Oracle.*ORA-\d+',
            r'Microsoft SQL Server.*\d+',
        ]
        
        for pattern in db_error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                findings.append(PassiveFinding(
                    category='Database Error Disclosure',
                    severity='High',
                    owasp='A05:2021',
                    description='Database error message exposed',
                    evidence=f'HTTP {status}',
                    remediation='Use generic error messages',
                    url=url
                ))
                break
        
        return findings


# 使用示例

