# 此文件已廢棄
# 請直接使用 services.features.function_web_scanner.integration_tools.web_tools.WebAttackManager
# 不需要額外的同步包裝層

"""
已廢棄：請使用 WebAttackManager

根據 SIMPLE_ARCHITECTURE.md 核心理念，每個模組按自己的最佳實踐組織。
function_web_scanner 已經有完整的 WebAttackManager（異步實現）。

AI Commander 可以用 asyncio.to_thread() 處理異步調度，不需要額外的同步包裝。

正確用法：
    from services.features.function_web_scanner.integration_tools.web_tools import WebAttackManager
    
    # 方式 1: 直接異步調用
    manager = WebAttackManager()
    result = await manager.comprehensive_scan(target, options)
    
    # 方式 2: AI Commander 異步調度
    result = await asyncio.to_thread(
        lambda: asyncio.run(manager.comprehensive_scan(target, options))
    )
"""

from typing import Any, Dict, Optional, List
import requests

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class WebScannerManager:
    """Web 掃描管理器 - 純同步實現"""
    
    def __init__(self):
        """初始化 Web 掃描管理器"""
        logger.info("WebScannerManager 初始化完成（同步模式）")
    
    def scan(self, target: str, options: Optional[Dict[str, Any]] = None) -> dict:
        """
        執行 Web 掃描（純同步實現）
        
        Args:
            target: 目標 URL (例如: "https://example.com")
            options: 掃描選項
                - subdomain_scan: bool - 是否執行子域名掃描 (預設: True)
                - directory_scan: bool - 是否執行目錄掃描 (預設: True)
                - tech_detect: bool - 是否執行技術檢測 (預設: True)
        
        Returns:
            dict: 掃描結果
                {
                    "success": bool,
                    "target": str,
                    "findings": list,
                    "summary": {
                        "total_subdomains": int,
                        "total_directories": int,
                        "total_technologies": int
                    }
                }
        
        Example:
            >>> manager = WebScannerManager()
            >>> result = manager.scan("https://example.com")
            >>> print(f"發現 {len(result['findings'])} 項結果")
        """
        options = options or {}
        
        try:
            logger.info(f"開始 Web 掃描（同步模式）: {target}")
            
            findings = []
            
            # 1. 子域名掃描（如果啟用）
            if options.get("subdomain_scan", True):
                subdomains = self._scan_subdomains_sync(target)
                findings.extend(subdomains)
            
            # 2. 目錄掃描（如果啟用）
            if options.get("directory_scan", True):
                directories = self._scan_directories_sync(target)
                findings.extend(directories)
            
            # 3. 技術檢測（如果啟用）
            if options.get("tech_detect", True):
                technologies = self._detect_technologies_sync(target)
                findings.extend(technologies)
            
            # 組裝結果
            summary = {
                "total_subdomains": sum(1 for f in findings if f.get("type") == "subdomain"),
                "total_directories": sum(1 for f in findings if f.get("type") == "directory"),
                "total_technologies": sum(1 for f in findings if f.get("type") == "technology")
            }
            
            return {
                "success": True,
                "target": target,
                "findings": findings,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Web 掃描失敗: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "target": target,
                "findings": [],
                "summary": {}
            }
    
    def _scan_subdomains_sync(self, target: str) -> List[dict]:
        """同步子域名掃描（內部方法）"""
        # 簡單的子域名枚舉邏輯（示範用）
        # 實際應該調用專門的子域名枚舉工具或現有檢測器
        subdomains = []
        common_prefixes = ["www", "mail", "ftp", "admin", "api", "dev", "test"]
        
        import urllib.parse
        parsed = urllib.parse.urlparse(target)
        domain = parsed.netloc or target
        
        for prefix in common_prefixes:
            subdomain = f"{prefix}.{domain}"
            try:
                # 簡單的 DNS 檢查
                import socket
                socket.gethostbyname(subdomain)
                subdomains.append({
                    "type": "subdomain",
                    "value": subdomain,
                    "status": "active"
                })
            except:
                pass
        
        return subdomains
    
    def _scan_directories_sync(self, target: str) -> List[dict]:
        """同步目錄掃描（內部方法）"""
        # 簡單的目錄掃描邏輯（示範用）
        directories = []
        common_dirs = ["/admin", "/api", "/backup", "/config", "/uploads"]
        
        for dir_path in common_dirs:
            try:
                url = f"{target.rstrip('/')}{dir_path}"
                response = requests.head(url, timeout=3, allow_redirects=False)
                if response.status_code < 400:
                    directories.append({
                        "type": "directory",
                        "path": dir_path,
                        "status_code": response.status_code,
                        "url": url
                    })
            except:
                pass
        
        return directories
    
    def _detect_technologies_sync(self, target: str) -> List[dict]:
        """同步技術檢測（內部方法）"""
        # 簡單的技術檢測邏輯（示範用）
        technologies = []
        
        try:
            response = requests.get(target, timeout=5)
            headers = response.headers
            
            # 檢測 Server
            if "Server" in headers:
                technologies.append({
                    "type": "technology",
                    "category": "server",
                    "name": headers["Server"]
                })
            
            # 檢測 X-Powered-By
            if "X-Powered-By" in headers:
                technologies.append({
                    "type": "technology",
                    "category": "framework",
                    "name": headers["X-Powered-By"]
                })
            
            # 檢測常見框架特徵
            content = response.text.lower()
            if "wordpress" in content:
                technologies.append({"type": "technology", "category": "cms", "name": "WordPress"})
            if "drupal" in content:
                technologies.append({"type": "technology", "category": "cms", "name": "Drupal"})
            
        except Exception as e:
            logger.debug(f"技術檢測請求失敗: {e}")
        
        return technologies


# 為了與其他功能模組保持一致性，提供快捷函數
def scan_target(target: str, options: Optional[Dict[str, Any]] = None) -> dict:
    """
    快捷函數：執行 Web 掃描
    
    這是一個便捷函數，用於快速執行 Web 掃描，無需手動創建 Manager 實例。
    
    Args:
        target: 目標 URL
        options: 掃描選項
    
    Returns:
        dict: 掃描結果
    
    Example:
        >>> from services.features.function_web_scanner.scanner_manager import scan_target
        >>> result = scan_target("https://example.com")
    """
    manager = WebScannerManager()
    return manager.scan(target, options)



