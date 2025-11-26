"""
Python 掃描引擎適配器

將原本混雜在 Coordinator 中的 ScanOrchestrator 調用邏輯封裝於此。
這是最直接的適配器，因為它是原生 Python 調用，無需跨語言通信。

優勢：
- 原生 asyncio 支援，性能最佳
- 可直接訪問 Python 生態系統（BeautifulSoup, Selenium 等）
- 錯誤處理和調試最簡單

適用場景：
- 靜態內容爬取（HTML, CSS, JS）
- 表單發現和參數提取
- 基礎 API 端點識別
"""

from typing import List, Dict, Any
from pydantic import HttpUrl
from services.aiva_common.schemas import ScanStartPayload, Asset
from .base import BaseScannerAdapter


class PythonAdapter(BaseScannerAdapter):
    """Python 引擎適配器
    
    封裝 ScanOrchestrator 的調用邏輯，提供統一接口。
    """
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self._orchestrator = None
    
    async def is_available(self) -> bool:
        """檢查 Python 引擎是否可用"""
        try:
            # 嘗試導入 ScanOrchestrator
            from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
            return True
        except ImportError as e:
            self.logger.warning(f"Python 引擎不可用: {e}")
            return False
    
    async def scan(
        self,
        targets: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行 Python 引擎掃描
        
        Args:
            targets: 目標 URL 列表
            options: 掃描配置
        
        Returns:
            包含 assets 和 metadata 的字典
        """
        try:
            # 延遲導入以避免循環依賴
            from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
            
            # 創建或重用 orchestrator 實例
            if self._orchestrator is None:
                self._orchestrator = ScanOrchestrator()
            
            # 轉換 URL 格式以符合 Pydantic 要求
            validated_urls = []
            for url in targets:
                if isinstance(url, str):
                    validated_urls.append(HttpUrl(url))
                else:
                    validated_urls.append(url)
            
            # 構建標準 Payload
            request = ScanStartPayload(
                scan_id=options.get("scan_id", "unknown"),
                targets=validated_urls,
                strategy=options.get("strategy", "deep"),
                custom_headers=options.get("headers", {})
            )
            
            self.logger.info(f"🐍 Python 引擎開始掃描: {len(targets)} 個目標")
            
            # 執行掃描
            result = await self._orchestrator.execute_scan(request)
            
            # 提取資產和元數據
            assets = result.assets if hasattr(result, 'assets') else []
            
            # 構建元數據
            metadata = {
                "engine": "python",
                "targets_scanned": len(targets)
            }
            
            # 如果有 summary，添加詳細統計
            if hasattr(result, 'summary') and result.summary:
                metadata.update({
                    "urls_found": result.summary.urls_found,
                    "forms_found": result.summary.forms_found,
                    "scan_duration": result.summary.scan_duration_seconds,
                    "pages_crawled": getattr(result.summary, 'pages_crawled', 0)
                })
            
            self.logger.info(
                f"✅ Python 引擎完成: {len(assets)} 個資產, "
                f"{metadata.get('scan_duration', 0):.1f}s"
            )
            
            return {
                "assets": assets,
                "metadata": metadata,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Python 引擎錯誤: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "assets": [],
                "metadata": {
                    "engine": "python",
                    "targets_scanned": len(targets)
                },
                "error": error_msg
            }
