import asyncio
import logging
from typing import List, Dict, Any, Optional
import uuid

from aiva_common.schemas import FunctionTaskPayload, FunctionTaskTarget, FunctionTaskTestConfig
from services.features.function_xss.traditional_detector import TraditionalXssDetector
from services.features.function_xss.dom_xss_detector import DomXssDetector
from services.features.function_xss.stored_detector import StoredXssDetector
from services.features.function_xss.payload_generator import XssPayloadGenerator

logger = logging.getLogger(__name__)

class XssScanner:
    """
    Unified XSS Scanner
    
    整合 Relfected, Stored, DOM XSS 檢測功能的統一入口。
    完全使用原生 Python 實現，不依賴外部 Binary 工具。
    """

    def __init__(self):
        self.payload_generator = XssPayloadGenerator()
        self.dom_detector = DomXssDetector()
        
    async def scan(
        self, 
        target_url: str, 
        scan_type: str = "comprehensive", 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        執行 XSS 掃描
        
        Args:
            target_url: 目標 URL
            scan_type: 掃描類型 (comprehensive, reflected, stored, dom)
            options: 額外選項
                - timeout: 超時時間 (默認 30)
                - method: HTTP 方法 (默認 GET)
                - params: 參數字典
                - view_url: Stored XSS 查看頁面
        
        Returns:
            掃描結果字典
        """
        options = options or {}
        timeout = float(options.get("timeout", 30))
        method = options.get("method", "GET")
        
        results = []
        
        # 1. 準備任務 Target
        # 注意: 這裡簡化了 Target 建立，實際應根據 options["params"] 迭代測試
        # 為了保持與原邏輯一致，我們先假設針對 URL Query 參數進行測試
        
        logger.info(f"開始 XSS 掃描: {target_url} [{scan_type}]")
        
        tasks_to_run = []
        
        # 根據 scan_type 決定執行哪些檢測
        if scan_type in ["comprehensive", "reflected"]:
            tasks_to_run.append(self._scan_reflected(target_url, method, timeout, options))
            
        if scan_type in ["comprehensive", "dom"]:
            tasks_to_run.append(self._scan_dom(target_url, timeout, options))
            
        if scan_type in ["comprehensive", "stored"]:
            if options.get("view_url"):
                tasks_to_run.append(self._scan_stored(target_url, method, timeout, options))
            elif scan_type == "stored":
                logger.warning("Stored XSS 掃描需要提供 'view_url' 選項")

        # 並行執行所有選擇的掃描任務
        scan_results = await asyncio.gather(*tasks_to_run)
        
        for res in scan_results:
            results.extend(res)
            
        return {
            "target": target_url,
            "scan_type": scan_type,
            "vulnerable": len(results) > 0,
            "findings_count": len(results),
            "findings": results
        }

    async def _scan_reflected(self, url: str, method: str, timeout: float, options: Dict) -> List[Dict]:
        """執行 Reflected XSS 掃描"""
        from urllib.parse import urlparse, parse_qs
        
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # 決定要測試的參數列表
        target_params = []
        if options.get("param"):
            target_params = [options.get("param")]
        else:
            target_params = list(query_params.keys())
            
        if not target_params:
            logger.info(f"目標 URL 沒有發現可測試參數: {url}")
            return []

        # 針對每個參數建立測試任務
        all_detections = []
        detector = None # Reuse detector instance if possible, but TraditionalXssDetector takes a task in __init__
        
        payloads = self.payload_generator.generate_basic_payloads()
        
        for param in target_params:
            task = FunctionTaskPayload(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                scan_id=f"scan_{uuid.uuid4().hex[:8]}",
                target=FunctionTaskTarget(
                    url=url,
                    method=method,
                    parameter_location="query",
                    parameter=param
                ),
                test_config=FunctionTaskTestConfig(timeout=timeout)
            )
            
            # 為每個參數實例化檢測器 (因為 task 綁定在實例上)
            detector = TraditionalXssDetector(task, timeout=timeout)
            detections = await detector.execute(payloads)
            all_detections.extend(detections)
        
        return [
            {
                "type": "reflected",
                "payload": d.payload,
                "parameter": d.request.url.params.get("xss_probe") or "unknown", # 修正: 這裡應該從 d.request 獲取參數名，但 TraditionalResult 沒存 param name，暫時用 request url 推斷或忽略
                "status": d.response_status,
                "evidence": d.response_text[:200],
                "severity": "high"
            }
            for d in all_detections
        ]

    async def _scan_dom(self, url: str, timeout: float, options: Dict) -> List[Dict]:
        """執行 DOM XSS 掃描 (Static Analysis)"""
        import httpx
        
        # 設置瀏覽器偽裝 Header，避免被簡單阻擋
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                
                # 使用 DomXssDetector 進行 Source-to-Sink 靜態分析
                # 這裡不依賴 headless browser，而是分析 HTML/JS 源碼
                result = self.dom_detector.analyze(
                    payload="", # 靜態分析不需要特定 payload，是分析代碼模式
                    document=resp.text
                )
                
                if result:
                    return [{
                        "type": "dom",
                        "payload": getattr(result, 'payload', 'Source-to-Sink Pattern'),
                        "evidence": getattr(result, 'snippet', '')[:200],
                        "source": getattr(result, 'source', 'unknown'),
                        "sink": getattr(result, 'sink', 'unknown'),
                        "confidence": getattr(result, 'confidence', 'Medium'),
                        "severity": "medium"
                    }]
                    
        except httpx.HTTPStatusError as e:
            logger.warning(f"DOM Scan HTTP Error: {e.response.status_code} for {url}")
        except Exception as e:
            logger.error(f"DOM Scan failed: {e}")
            
        return []

    async def _scan_stored(self, url: str, method: str, timeout: float, options: Dict) -> List[Dict]:
        """執行 Stored XSS 掃描"""
        view_url = options.get("view_url")
        if not view_url:
            return []
            
        task = FunctionTaskPayload(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            scan_id=f"scan_{uuid.uuid4().hex[:8]}",
            target=FunctionTaskTarget(
                url=url,
                method=method,
                parameter_location="query",
                parameter=options.get("param")
            ),
            test_config=FunctionTaskTestConfig(timeout=timeout)
        )
        
        detector = StoredXssDetector(task, timeout=timeout)
        payloads = self.payload_generator.generate_basic_payloads()[:3] # 限制 Payload 數量
        
        detections = await detector.execute(payloads, view_urls=[view_url])
        
        return [
            {
                "type": "stored",
                "payload": d.payload,
                "status": d.response_status,
                "evidence": d.response_text[:200],
                "severity": "critical"
            }
            for d in detections
        ]
