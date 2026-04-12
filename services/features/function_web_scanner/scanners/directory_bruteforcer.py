"""
function_web_scanner.scanners.directory_bruteforcer
目錄暴力破解器

執行 Web 目錄和文件發現。
"""

import requests
import concurrent.futures
from typing import List, Dict
from dataclasses import dataclass
from urllib.parse import urljoin

from aiva_common.utils import get_logger
from aiva_common.enums.common import Severity

logger = get_logger(__name__)


@dataclass
class DirectoryResult:
    """目錄掃描結果"""
    url: str
    status_code: int
    content_length: int
    content_type: str
    redirect_url: str = None
    severity: Severity = Severity.LOW


class DirectoryBruteforcer:
    """目錄暴力破解器"""
    
    def __init__(self, wordlist_path: str = None, threads: int = 10, verify_ssl: bool = True):
        """
        初始化目錄暴力破解器
        
        Args:
            wordlist_path: 目錄字典文件路徑
            threads: 並發線程數
            verify_ssl: 是否驗證 SSL 憑證 (若目標使用自簽證書請設為 False)
        """
        self.wordlist_path = wordlist_path or self._get_default_wordlist()
        self.threads = threads
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info(f"目錄暴力破解器初始化 (threads: {threads}, verify_ssl: {verify_ssl})")
    
    def scan(self, base_url: str, extensions: List[str] = None) -> List[DirectoryResult]:
        """
        掃描目錄和文件
        
        Args:
            base_url: 目標 URL
            extensions: 文件擴展名列表 (如 ['.php', '.txt', '.bak'])
        
        Returns:
            List[DirectoryResult]: 發現的目錄和文件
        """
        extensions = extensions or []
        results = []
        
        try:
            with open(self.wordlist_path, 'r') as f:
                wordlist = [line.strip() for line in f if line.strip()]
            
            # Generate URLs to test
            urls_to_test = []
            for word in wordlist[:5000]:  # Limit for performance
                urls_to_test.append(urljoin(base_url, word))
                
                for ext in extensions:
                    urls_to_test.append(urljoin(base_url, f"{word}{ext}"))
            
            logger.info(f"測試 {len(urls_to_test)} 個 URL")
            
            # Concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                future_to_url = {executor.submit(self._test_url, url): url for url in urls_to_test}
                
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future.result()
                    if result:
                        results.append(result)
        
        except FileNotFoundError:
            logger.error(f"Wordlist not found: {self.wordlist_path}")
        except Exception as e:
            logger.error(f"目錄掃描錯誤: {e}")
        
        logger.info(f"發現 {len(results)} 個有效路徑")
        return results
    
    def _test_url(self, url: str) -> DirectoryResult:
        """測試單個 URL"""
        try:
            response = self.session.get(
                url,
                allow_redirects=False,
                timeout=5,
                    verify=self.verify_ssl
            )
            
            # Filter interesting responses
            if response.status_code in [200, 301, 302, 401, 403]:
                severity = self._determine_severity(response)
                
                return DirectoryResult(
                    url=url,
                    status_code=response.status_code,
                    content_length=len(response.content),
                    content_type=response.headers.get('Content-Type', ''),
                    redirect_url=response.headers.get('Location'),
                    severity=severity
                )
        
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            logger.debug(f"Request failed for {url}: {e}")
        
        return None
    
    def _determine_severity(self, response: requests.Response) -> Severity:
        """根據響應判斷嚴重程度"""
        url = response.url.lower()
        
        # Critical paths
        critical_paths = [
            'admin', 'phpmyadmin', 'config', 'backup',
            '.git', '.env', '.sql', 'database'
        ]
        
        if any(path in url for path in critical_paths):
            if response.status_code == 200:
                return Severity.CRITICAL
            elif response.status_code == 403:
                return Severity.HIGH
        
        # Sensitive extensions
        sensitive_exts = ['.bak', '.old', '.sql', '.zip', '.tar.gz', '.config']
        if any(url.endswith(ext) for ext in sensitive_exts) and response.status_code == 200:
            return Severity.HIGH
        
        # 401/403 indicates protected resources
        if response.status_code in [401, 403]:
            return Severity.MEDIUM
        
        return Severity.LOW
    
    def _get_default_wordlist(self) -> str:
        """獲取默認字典路徑"""
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "wordlists", "directories.txt")
