"""AIVA Dashboard API 客戶端

此模組封裝所有與 AIVA Core API 的通訊邏輯。

依據 AIVA Common 規範:
- 使用 Pydantic 模型進行數據驗證
- 統一錯誤處理
- 支援重試機制
"""

import json
from typing import Any, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from aiva_common.utils import get_logger

logger = get_logger(__name__)


class AIVAApiClient:
    """AIVA FastAPI 客戶端"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        """初始化 API 客戶端
        
        Args:
            base_url: API 基礎 URL
            timeout: 請求超時（秒）
            retry_attempts: 重試次數
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # 建立 Session 並配置重試策略
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    # ==================== 基礎方法 ====================
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """執行 HTTP 請求
        
        Args:
            method: HTTP 方法
            endpoint: API 端點
            **kwargs: requests 參數
        
        Returns:
            Response 對象
        
        Raises:
            requests.HTTPError: HTTP 錯誤
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API 請求失敗: {method} {url} - {e}")
            raise
    
    # ==================== 健康檢查 ====================
    
    def health_check(self) -> dict[str, Any]:
        """健康檢查
        
        Returns:
            健康狀態資訊
        """
        response = self._request("GET", "/health")
        return response.json()
    
    # ==================== 掃描管理 ====================
    
    def start_scan(
        self,
        target: str,
        scan_type: str = "comprehensive",
        max_depth: int = 3,
        timeout: int = 1800
    ) -> dict[str, Any]:
        """啟動掃描
        
        Args:
            target: 目標 URL
            scan_type: 掃描類型（quick/comprehensive/deep）
            max_depth: 掃描深度
            timeout: 超時時間（秒）
        
        Returns:
            掃描響應 {scan_id, status, message, target, estimated_time}
        """
        response = self._request(
            "POST",
            "/scan",
            json={
                "target": target,
                "scan_type": scan_type,
                "max_depth": max_depth,
                "timeout": timeout
            }
        )
        return response.json()
    
    def get_scan_status(self, scan_id: str) -> dict[str, Any]:
        """查詢掃描狀態
        
        Args:
            scan_id: 掃描 ID
        
        Returns:
            掃描狀態資訊
        """
        response = self._request("GET", f"/status/{scan_id}")
        return response.json()
    
    def list_scans(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> dict[str, Any]:
        """查詢掃描列表
        
        Args:
            status: 篩選狀態
            limit: 返回數量
            offset: 分頁偏移
        
        Returns:
            {total, scans, limit, offset}
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        response = self._request("GET", "/api/scans", params=params)
        return response.json()
    
    def get_scan_results(self, scan_id: str) -> dict[str, Any]:
        """取得完整掃描結果
        
        Args:
            scan_id: 掃描 ID
        
        Returns:
            完整掃描結果（含漏洞詳情）
        """
        response = self._request("GET", f"/api/scans/{scan_id}/results")
        return response.json()
    
    def stop_scan(self, scan_id: str) -> dict[str, str]:
        """停止掃描
        
        Args:
            scan_id: 掃描 ID
        
        Returns:
            操作結果訊息
        """
        response = self._request("POST", f"/api/scans/{scan_id}/stop")
        return response.json()
    
    def delete_scan(self, scan_id: str) -> dict[str, str]:
        """刪除掃描記錄
        
        Args:
            scan_id: 掃描 ID
        
        Returns:
            操作結果訊息
        """
        response = self._request("DELETE", f"/api/scans/{scan_id}")
        return response.json()
    
    # ==================== SSE 串流 ====================
    
    def stream_logs(self, scan_id: str) -> Iterator[dict]:
        """串流日誌（SSE）
        
        Args:
            scan_id: 掃描 ID
        
        Yields:
            日誌事件 {timestamp, level, message, scan_id}
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/logs/stream",
                params={"scan_id": scan_id},
                stream=True,
                timeout=None  # SSE 無超時
            )
            response.raise_for_status()
            
            # 解析 SSE 事件
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                # SSE 格式: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # 移除 "data: " 前綴
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️  無效的 JSON 數據: {data_str}")
                        continue
                
                # 檢查完成事件
                elif line.startswith("event: completed"):
                    logger.info(f"✅ 日誌串流完成: {scan_id}")
                    break
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 日誌串流失敗: {e}")
            raise
    
    def stream_status(self, scan_id: str) -> Iterator[dict]:
        """串流狀態更新（SSE）
        
        Args:
            scan_id: 掃描 ID
        
        Yields:
            狀態事件 {scan_id, status, progress, phase, ...}
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/status/stream",
                params={"scan_id": scan_id},
                stream=True,
                timeout=None  # SSE 無超時
            )
            response.raise_for_status()
            
            # 解析 SSE 事件
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                # SSE 格式: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️  無效的 JSON 數據: {data_str}")
                        continue
                
                # 檢查完成事件
                elif line.startswith("event: completed"):
                    logger.info(f"✅ 狀態串流完成: {scan_id}")
                    break
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 狀態串流失敗: {e}")
            raise
    
    # ==================== 工具方法 ====================
    
    def is_api_available(self) -> bool:
        """檢查 API 是否可用
        
        Returns:
            API 可用性
        """
        try:
            response = self.health_check()
            return response.get("status") == "healthy"
        except Exception:
            return False
