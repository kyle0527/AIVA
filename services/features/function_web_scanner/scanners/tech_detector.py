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


@dataclass
class Technology:
    """檢測到的技術"""
    name: str
    version: str = None
    category: str = None  # framework, server, cms, language, etc.
    confidence: int = 100  # 0-100
    evidence: List[str] = None


class TechDetector:
    """技術棧檢測器"""
    
    def __init__(self):
        """初始化技術檢測器"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._load_fingerprints()
        logger.info("技術檢測器初始化完成")
    
    def detect(self, url: str) -> List[Technology]:
        """
        檢測 URL 使用的技術
        
        Args:
            url: 目標 URL
        
        Returns:
            List[Technology]: 檢測到的技術列表
        """
        technologies = []
        
        try:
            response = self.session.get(url, timeout=10, verify=False)
            
            # 1. HTTP Headers
            technologies.extend(self._analyze_headers(response.headers))
            
            # 2. HTML Content
            technologies.extend(self._analyze_html(response.text))
            
            # 3. Cookies
            technologies.extend(self._analyze_cookies(response.cookies))
            
            # 4. Meta Tags
            technologies.extend(self._analyze_meta_tags(response.text))
            
            # 5. JavaScript Files
            technologies.extend(self._analyze_scripts(response.text, url))
        
        except Exception as e:
            logger.error(f"技術檢測錯誤: {e}")
        
        # Deduplicate
        unique_technologies = []
        for t in technologies:
            if t not in unique_technologies:
                unique_technologies.append(t)

        logger.info(f"檢測到 {len(unique_technologies)} 項技術")
        return unique_technologies
    
    def _load_fingerprints(self):
        """加載技術指紋庫"""
        # Helper to compile patterns
        def compile_patterns(patterns):
            return [re.compile(p, re.IGNORECASE) for p in patterns]

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
                'WordPress': compile_patterns([
                    r'/wp-content/',
                    r'/wp-includes/',
                    r'wp-json'
                ]),
                'Joomla': compile_patterns([
                    r'/components/com_',
                    r'Joomla!'
                ]),
                'Drupal': compile_patterns([
                    r'Drupal',
                    r'/sites/default/'
                ]),
                'Laravel': compile_patterns([
                    r'laravel',
                    r'csrf-token'
                ]),
                'Django': compile_patterns([
                    r'csrfmiddlewaretoken'
                ]),
                'React': compile_patterns([
                    r'react',
                    r'__REACT'
                ]),
                'Vue.js': compile_patterns([
                    r'vue\.js',
                    r'v-if=',
                    r'v-for='
                ]),
                'Angular': compile_patterns([
                    r'ng-app',
                    r'ng-controller'
                ]),
                'jQuery': compile_patterns([
                    r'jquery'
                ]),
                'Bootstrap': compile_patterns([
                    r'bootstrap',
                    r'class="container'
                ])
            },
            'cookies': {
                'PHPSESSID': ('PHP', 'language'),
                'ASP.NET_SessionId': ('ASP.NET', 'language'),
                'JSESSIONID': ('Java', 'language'),
                'wordpress_': ('WordPress', 'cms'),
                'wp-settings': ('WordPress', 'cms')
            }
        }
    
    def _analyze_headers(self, headers: Dict) -> List[Technology]:
        """分析 HTTP 標頭"""
        technologies = []
        
        for header, value in headers.items():
            if header in self.fingerprints['headers']:
                patterns = self.fingerprints['headers'][header]
                
                for pattern, tech_name in patterns.items():
                    if pattern.lower() in value.lower():
                        # Extract version if possible
                        version_match = re.search(r'(\d+\.[\d.]+)', value)
                        version = version_match.group(1) if version_match else None
                        
                        tech = Technology(
                            name=tech_name,
                            version=version,
                            category='server' if header == 'Server' else 'framework',
                            confidence=100,
                            evidence=[f"{header}: {value}"]
                        )
                        if tech not in technologies:
                            technologies.append(tech)
        
        return technologies
    
    def _analyze_html(self, html: str) -> List[Technology]:
        """分析 HTML 內容"""
        technologies = []
        
        for tech_name, patterns in self.fingerprints['html_patterns'].items():
            confidence = 0
            evidence = []
            
            for pattern in patterns:
                if pattern.search(html):
                    confidence += 50
                    evidence.append(f"Pattern matched: {pattern.pattern}")
            
            if confidence > 0:
                category = 'cms' if tech_name in ['WordPress', 'Joomla', 'Drupal'] else 'framework'
                
                tech = Technology(
                    name=tech_name,
                    category=category,
                    confidence=min(confidence, 100),
                    evidence=evidence
                )
                if tech not in technologies:
                    technologies.append(tech)
        
        return technologies
    
    def _analyze_cookies(self, cookies: requests.cookies.RequestsCookieJar) -> List[Technology]:
        """分析 Cookies"""
        technologies = []
        
        for cookie in cookies:
            cookie_name = cookie.name
            
            for pattern, (tech_name, category) in self.fingerprints['cookies'].items():
                if pattern in cookie_name:
                    tech = Technology(
                        name=tech_name,
                        category=category,
                        confidence=100,
                        evidence=[f"Cookie: {cookie_name}"]
                    )
                    if tech not in technologies:
                        technologies.append(tech)
        
        return technologies
    
    def _analyze_meta_tags(self, html: str) -> List[Technology]:
        """分析 Meta 標籤"""
        technologies = []
        
        # Extract generator meta tag
        generator_match = re.search(r'<meta\s+name=["\']generator["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        
        if generator_match:
            generator = generator_match.group(1)
            
            # Parse generator string (e.g., "WordPress 5.8")
            tech_match = re.match(r'([A-Za-z]+)\s*([\d.]+)?', generator)
            if tech_match:
                name = tech_match.group(1)
                version = tech_match.group(2)
                
                tech = Technology(
                    name=name,
                    version=version,
                    category='cms',
                    confidence=100,
                    evidence=[f"Meta generator: {generator}"]
                )
                if tech not in technologies:
                    technologies.append(tech)
        
        return technologies
    
    def _analyze_scripts(self, html: str, base_url: str) -> List[Technology]:
        """分析 JavaScript 文件"""
        technologies = []
        
        # Extract script sources
        script_srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
        
        for src in script_srcs:
            src_lower = src.lower()
            
            # Check for common libraries
            if 'jquery' in src_lower:
                version_match = re.search(r'jquery[.-](\d+\.[\d.]+)', src_lower)
                tech = Technology(
                    name='jQuery',
                    version=version_match.group(1) if version_match else None,
                    category='library',
                    confidence=100,
                    evidence=[f"Script: {src}"]
                )
                if tech not in technologies:
                    technologies.append(tech)
            
            elif 'react' in src_lower:
                tech = Technology(
                    name='React',
                    category='framework',
                    confidence=90,
                    evidence=[f"Script: {src}"]
                )
                if tech not in technologies:
                    technologies.append(tech)
            
            elif 'vue' in src_lower:
                version_match = re.search(r'vue[.-](\d+\.[\d.]+)', src_lower)
                tech = Technology(
                    name='Vue.js',
                    version=version_match.group(1) if version_match else None,
                    category='framework',
                    confidence=90,
                    evidence=[f"Script: {src}"]
                )
                if tech not in technologies:
                    technologies.append(tech)
            
            elif 'angular' in src_lower:
                tech = Technology(
                    name='Angular',
                    category='framework',
                    confidence=90,
                    evidence=[f"Script: {src}"]
                )
                if tech not in technologies:
                    technologies.append(tech)
            
            elif 'bootstrap' in src_lower:
                version_match = re.search(r'bootstrap[.-](\d+\.[\d.]+)', src_lower)
                tech = Technology(
                    name='Bootstrap',
                    version=version_match.group(1) if version_match else None,
                    category='ui_framework',
                    confidence=100,
                    evidence=[f"Script: {src}"]
                )
                if tech not in technologies:
                    technologies.append(tech)
        
        return technologies
