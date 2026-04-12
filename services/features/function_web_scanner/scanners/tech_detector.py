"""
function_web_scanner.scanners.tech_detector
技術棧檢測器

識別 Web 應用的技術棧（框架、服務器、語言等）。
"""

import re
import requests
from typing import Dict, List, Set
from dataclasses import dataclass

from aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Technology:
    """檢測到的技術"""
    name: str
    version: str = None
    category: str = None  # framework, server, cms, language, etc.
    confidence: int = 100  # 0-100
    evidence: tuple = None


class TechDetector:
    """技術棧檢測器"""
    
    # Pre-compiled regex patterns for script analysis
    _SCRIPT_SRC_PATTERN = re.compile(r'<script[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
    _JQUERY_VERSION_PATTERN = re.compile(r'jquery[.-](\d+\.[\d.]+)')
    _VUE_VERSION_PATTERN = re.compile(r'vue[.-](\d+\.[\d.]+)')
    _BOOTSTRAP_VERSION_PATTERN = re.compile(r'bootstrap[.-](\d+\.[\d.]+)')

    def __init__(self, verify_ssl: bool = True):
        """
        初始化技術檢測器

        Args:
            verify_ssl: 是否驗證 SSL 憑證 (若目標使用自簽證書請設為 False)
        """
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.verify_ssl = verify_ssl
        self._load_fingerprints()
        logger.info(f"技術檢測器初始化完成 (verify_ssl: {verify_ssl})")
    
    def detect(self, url: str) -> List[Technology]:
        """
        檢測 URL 使用的技術
        
        Args:
            url: 目標 URL
        
        Returns:
            List[Technology]: 檢測到的技術列表
        """
        technologies = set()
        
        try:
            response = self.session.get(url, timeout=10, verify=self.verify_ssl)
            
            # 1. HTTP Headers
            technologies.update(self._analyze_headers(response.headers))
            
            # 2. HTML Content
            technologies.update(self._analyze_html(response.text))
            
            # 3. Cookies
            technologies.update(self._analyze_cookies(response.cookies))
            
            # 4. Meta Tags
            technologies.update(self._analyze_meta_tags(response.text))
            
            # 5. JavaScript Files
            technologies.update(self._analyze_scripts(response.text, url))
        
        except Exception as e:
            logger.error(f"技術檢測錯誤: {e}")
        
        logger.info(f"檢測到 {len(technologies)} 項技術")
        return list(technologies)
    
    def _load_fingerprints(self):
        """加載技術指紋庫"""
        self.fingerprints = {
            'headers': {
                'X-Powered-By': {
                    'PHP': 'PHP',
                    'ASP.NET': 'ASP.NET',
                    'Express': 'Express.js'
                },
                'Server': {
                    'nginx': 'Nginx',
                    'Apache': 'Apache',
                    'Microsoft-IIS': 'IIS',
                    'cloudflare': 'Cloudflare'
                },
                'X-AspNet-Version': {
                    r'\d+\.\d+': 'ASP.NET'
                }
            },
            'html_patterns': {
                'WordPress': [
                    r'/wp-content/',
                    r'/wp-includes/',
                    r'wp-json'
                ],
                'Joomla': [
                    r'/components/com_',
                    r'Joomla!'
                ],
                'Drupal': [
                    r'Drupal',
                    r'/sites/default/'
                ],
                'Laravel': [
                    r'laravel',
                    r'csrf-token'
                ],
                'Django': [
                    r'csrfmiddlewaretoken'
                ],
                'React': [
                    r'react',
                    r'__REACT'
                ],
                'Vue.js': [
                    r'vue\.js',
                    r'v-if=',
                    r'v-for='
                ],
                'Angular': [
                    r'ng-app',
                    r'ng-controller'
                ],
                'jQuery': [
                    r'jquery'
                ],
                'Bootstrap': [
                    r'bootstrap',
                    r'class="container'
                ]
            },
            'cookies': {
                'PHPSESSID': ('PHP', 'language'),
                'ASP.NET_SessionId': ('ASP.NET', 'language'),
                'JSESSIONID': ('Java', 'language'),
                'wordpress_': ('WordPress', 'cms'),
                'wp-settings': ('WordPress', 'cms')
            }
        }
    
    def _analyze_headers(self, headers: Dict) -> Set[Technology]:
        """分析 HTTP 標頭"""
        technologies = set()
        
        for header, value in headers.items():
            if header in self.fingerprints['headers']:
                patterns = self.fingerprints['headers'][header]
                
                for pattern, tech_name in patterns.items():
                    if pattern.lower() in value.lower():
                        # Extract version if possible
                        version_match = re.search(r'(\d+\.[\d.]+)', value)
                        version = version_match.group(1) if version_match else None
                        
                        technologies.add(Technology(
                            name=tech_name,
                            version=version,
                            category='server' if header == 'Server' else 'framework',
                            confidence=100,
                            evidence=(f"{header}: {value}",)
                        ))
        
        return technologies
    
    def _analyze_html(self, html: str) -> Set[Technology]:
        """分析 HTML 內容"""
        technologies = set()
        
        for tech_name, patterns in self.fingerprints['html_patterns'].items():
            confidence = 0
            evidence = []
            
            for pattern in patterns:
                if re.search(pattern, html, re.IGNORECASE):
                    confidence += 50
                    evidence.append(f"Pattern matched: {pattern}")
            
            if confidence > 0:
                category = 'cms' if tech_name in ['WordPress', 'Joomla', 'Drupal'] else 'framework'
                
                technologies.add(Technology(
                    name=tech_name,
                    category=category,
                    confidence=min(confidence, 100),
                    evidence=tuple(evidence)
                ))
        
        return technologies
    
    def _analyze_cookies(self, cookies: requests.cookies.RequestsCookieJar) -> Set[Technology]:
        """分析 Cookies"""
        technologies = set()
        
        for cookie in cookies:
            cookie_name = cookie.name
            
            for pattern, (tech_name, category) in self.fingerprints['cookies'].items():
                if pattern in cookie_name:
                    technologies.add(Technology(
                        name=tech_name,
                        category=category,
                        confidence=100,
                        evidence=(f"Cookie: {cookie_name}",)
                    ))
        
        return technologies
    
    def _analyze_meta_tags(self, html: str) -> Set[Technology]:
        """分析 Meta 標籤"""
        technologies = set()
        
        # Extract generator meta tag
        generator_match = re.search(r'<meta\s+name=["\']generator["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        
        if generator_match:
            generator = generator_match.group(1)
            
            # Parse generator string (e.g., "WordPress 5.8")
            tech_match = re.match(r'([A-Za-z]+)\s*([\d.]+)?', generator)
            if tech_match:
                name = tech_match.group(1)
                version = tech_match.group(2)
                
                technologies.add(Technology(
                    name=name,
                    version=version,
                    category='cms',
                    confidence=100,
                    evidence=(f"Meta generator: {generator}",)
                ))
        
        return technologies
    
    def _analyze_scripts(self, html: str, base_url: str) -> Set[Technology]:
        """分析 JavaScript 文件"""
        technologies = set()
        
        # Extract script sources
        script_srcs = self._SCRIPT_SRC_PATTERN.findall(html)
        
        for src in script_srcs:
            src_lower = src.lower()
            
            # Check for common libraries
            if 'jquery' in src_lower:
                version_match = self._JQUERY_VERSION_PATTERN.search(src_lower)
                technologies.add(Technology(
                    name='jQuery',
                    version=version_match.group(1) if version_match else None,
                    category='library',
                    confidence=100,
                    evidence=(f"Script: {src}",)
                ))
            
            elif 'react' in src_lower:
                technologies.add(Technology(
                    name='React',
                    category='framework',
                    confidence=90,
                    evidence=(f"Script: {src}",)
                ))
            
            elif 'vue' in src_lower:
                version_match = self._VUE_VERSION_PATTERN.search(src_lower)
                technologies.add(Technology(
                    name='Vue.js',
                    version=version_match.group(1) if version_match else None,
                    category='framework',
                    confidence=90,
                    evidence=(f"Script: {src}",)
                ))
            
            elif 'angular' in src_lower:
                technologies.add(Technology(
                    name='Angular',
                    category='framework',
                    confidence=90,
                    evidence=(f"Script: {src}",)
                ))
            
            elif 'bootstrap' in src_lower:
                version_match = self._BOOTSTRAP_VERSION_PATTERN.search(src_lower)
                technologies.add(Technology(
                    name='Bootstrap',
                    version=version_match.group(1) if version_match else None,
                    category='ui_framework',
                    confidence=100,
                    evidence=(f"Script: {src}",)
                ))
        
        return technologies
