"""
function_web_scanner.scanners.web_crawler
Web 爬蟲

爬取網站結構，發現隱藏頁面和參數。
"""

import re
import requests
from urllib.parse import urljoin, urlparse, parse_qs
from typing import Set, List, Dict
from dataclasses import dataclass
from bs4 import BeautifulSoup

from aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CrawlResult:
    """爬蟲結果"""
    url: str
    status_code: int
    title: str = None
    forms: List[Dict] = None
    links: List[str] = None
    parameters: Set[str] = None


class WebCrawler:
    """Web 爬蟲"""
    
    def __init__(self, max_depth: int = 3, max_pages: int = 100, verify_ssl: bool = True):
        """
        初始化爬蟲
        
        Args:
            max_depth: 最大爬取深度
            max_pages: 最大爬取頁面數
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.verify_ssl = verify_ssl
        self.visited = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info(f"Web 爬蟲初始化 (depth: {max_depth}, pages: {max_pages})")
    
    def crawl(self, start_url: str) -> List[CrawlResult]:
        """
        開始爬取
        
        Args:
            start_url: 起始 URL
        
        Returns:
            List[CrawlResult]: 爬取結果
        """
        results = []
        to_visit = [(start_url, 0)]  # (url, depth)
        base_domain = urlparse(start_url).netloc
        
        while to_visit and len(results) < self.max_pages:
            url, depth = to_visit.pop(0)
            
            if url in self.visited or depth > self.max_depth:
                continue
            
            self.visited.add(url)
            
            try:
                result = self._crawl_page(url)
                if result:
                    results.append(result)
                    
                    # 添加新鏈接到待訪問列表
                    if result.links and depth < self.max_depth:
                        for link in result.links:
                            parsed = urlparse(link)
                            if parsed.netloc == base_domain and link not in self.visited:
                                to_visit.append((link, depth + 1))
            
            except Exception as e:
                logger.debug(f"爬取錯誤 {url}: {e}")
        
        logger.info(f"爬蟲完成，訪問 {len(results)} 個頁面")
        return results
    
    def _crawl_page(self, url: str) -> CrawlResult:
        """爬取單個頁面"""
        try:
            response = self.session.get(url, timeout=10, verify=self.verify_ssl)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取標題
            title = soup.title.string if soup.title else None
            
            # 提取表單
            forms = self._extract_forms(soup, url)
            
            # 提取鏈接
            links = self._extract_links(soup, url)
            
            # 提取參數
            parameters = self._extract_parameters(url, soup)
            
            return CrawlResult(
                url=url,
                status_code=response.status_code,
                title=title,
                forms=forms,
                links=links,
                parameters=parameters
            )
        
        except Exception as e:
            logger.debug(f"頁面爬取失敗 {url}: {e}")
            return None
    
    def _extract_forms(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """提取表單"""
        forms = []
        
        for form in soup.find_all('form'):
            form_data = {
                'action': urljoin(base_url, form.get('action', '')),
                'method': form.get('method', 'get').upper(),
                'inputs': []
            }
            
            for input_tag in form.find_all('input'):
                form_data['inputs'].append({
                    'name': input_tag.get('name'),
                    'type': input_tag.get('type', 'text'),
                    'value': input_tag.get('value', '')
                })
            
            forms.append(form_data)
        
        return forms
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """提取鏈接"""
        links = []
        
        for tag in soup.find_all(['a', 'link', 'script', 'img']):
            href = tag.get('href') or tag.get('src')
            if href:
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
        
        return links
    
    def _extract_parameters(self, url: str, soup: BeautifulSoup) -> Set[str]:
        """提取參數名稱"""
        parameters = set()
        
        # URL 參數
        parsed = urlparse(url)
        if parsed.query:
            params = parse_qs(parsed.query)
            parameters.update(params.keys())
        
        # 表單參數
        for form in soup.find_all('form'):
            for input_tag in form.find_all('input'):
                name = input_tag.get('name')
                if name:
                    parameters.add(name)
        
        return parameters
