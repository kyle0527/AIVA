"""
WebSocket 連接管理器

用於管理 WebSocket 連接並向客戶端推送實時更新。
"""

import asyncio
import json
from typing import Set, Dict, Any, Optional
from datetime import datetime, UTC

try:
    from fastapi import WebSocket
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = Any  # type: ignore

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """WebSocket 連接管理器
    
    功能:
    1. 管理客戶端連接
    2. 廣播消息到所有客戶端
    3. 發送個人消息
    4. 處理連接/斷開
    5. 自動清理失效連接
    """
    
    def __init__(self):
        """初始化 WebSocket 管理器"""
        self.active_connections: Set[Any] = set()
        self.logger = logger
        self._lock = asyncio.Lock()
        
        if not FASTAPI_AVAILABLE:
            self.logger.warning("⚠️  FastAPI 未安裝，WebSocket 功能將不可用")
        
        self.logger.info("✅ WebSocket 管理器已初始化")
    
    async def connect(self, websocket: Any) -> None:
        """接受新的 WebSocket 連接
        
        Args:
            websocket: FastAPI WebSocket 對象
        """
        if not FASTAPI_AVAILABLE:
            self.logger.error("❌ FastAPI 未安裝，無法接受 WebSocket 連接")
            return
        
        await websocket.accept()
        
        async with self._lock:
            self.active_connections.add(websocket)
        
        self.logger.info(
            f"✅ WebSocket 連接建立，當前連接數: {len(self.active_connections)}"
        )
        
        # 發送歡迎消息
        await self.send_personal_message(
            {
                "type": "connected",
                "message": "已連接到 AIVA 實時更新服務",
                "timestamp": datetime.now(UTC).isoformat()
            },
            websocket
        )
    
    def disconnect(self, websocket: Any) -> None:
        """斷開 WebSocket 連接
        
        Args:
            websocket: FastAPI WebSocket 對象
        """
        self.active_connections.discard(websocket)
        self.logger.info(
            f"🔌 WebSocket 連接斷開，當前連接數: {len(self.active_connections)}"
        )
    
    async def send_personal_message(
        self, 
        message: Dict[str, Any], 
        websocket: Any
    ) -> bool:
        """發送個人消息
        
        Args:
            message: 消息內容（字典）
            websocket: 目標 WebSocket
            
        Returns:
            是否發送成功
        """
        try:
            # 確保消息有時間戳
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(UTC).isoformat()
            
            await websocket.send_json(message)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 發送個人消息失敗: {e}")
            self.disconnect(websocket)
            return False
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """廣播消息到所有連接
        
        Args:
            message: 消息內容（字典）
            
        Returns:
            成功發送的連接數
        """
        if not self.active_connections:
            return 0
        
        # 確保消息有時間戳
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(UTC).isoformat()
        
        disconnected = set()
        success_count = 0
        
        # 並行發送到所有連接
        tasks = []
        for connection in self.active_connections:
            tasks.append(self._send_to_connection(connection, message))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for connection, result in zip(self.active_connections, results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ 廣播消息失敗: {result}")
                disconnected.add(connection)
            elif result:
                success_count += 1
        
        # 清理斷開的連接
        async with self._lock:
            for conn in disconnected:
                self.disconnect(conn)
        
        if success_count > 0:
            self.logger.debug(
                f"📡 廣播消息到 {success_count}/{len(self.active_connections)} 個客戶端"
            )
        
        return success_count
    
    async def _send_to_connection(
        self, 
        connection: Any, 
        message: Dict[str, Any]
    ) -> bool:
        """發送消息到單個連接（內部方法）"""
        try:
            await connection.send_json(message)
            return True
        except Exception:
            return False
    
    # ===== 便捷方法 =====
    
    async def broadcast_progress(
        self,
        command_id: str,
        progress: float,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """廣播進度更新
        
        Args:
            command_id: 命令 ID
            progress: 進度值 (0.0 - 1.0)
            message: 進度消息
            metadata: 額外的元數據
            
        Returns:
            成功發送的連接數
        """
        return await self.broadcast({
            "type": "progress_update",
            "command_id": command_id,
            "progress": progress,
            "message": message,
            "metadata": metadata or {}
        })
    
    async def broadcast_status_change(
        self,
        command_id: str,
        new_status: str,
        old_status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """廣播狀態變更
        
        Args:
            command_id: 命令 ID
            new_status: 新狀態
            old_status: 舊狀態（可選）
            metadata: 額外的元數據
            
        Returns:
            成功發送的連接數
        """
        msg = {
            "type": "status_change",
            "command_id": command_id,
            "status": new_status,
            "metadata": metadata or {}
        }
        
        if old_status:
            msg["old_status"] = old_status
        
        return await self.broadcast(msg)
    
    async def broadcast_result(
        self,
        command_id: str,
        result_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """廣播執行結果
        
        Args:
            command_id: 命令 ID
            result_type: 結果類型
            data: 結果數據
            metadata: 額外的元數據
            
        Returns:
            成功發送的連接數
        """
        return await self.broadcast({
            "type": "result",
            "command_id": command_id,
            "result_type": result_type,
            "data": data,
            "metadata": metadata or {}
        })
    
    async def broadcast_error(
        self,
        command_id: str,
        error_message: str,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """廣播錯誤消息
        
        Args:
            command_id: 命令 ID
            error_message: 錯誤消息
            error_type: 錯誤類型（可選）
            metadata: 額外的元數據
            
        Returns:
            成功發送的連接數
        """
        msg = {
            "type": "error",
            "command_id": command_id,
            "error_message": error_message,
            "metadata": metadata or {}
        }
        
        if error_type:
            msg["error_type"] = error_type
        
        return await self.broadcast(msg)
    
    def get_connection_count(self) -> int:
        """獲取當前連接數"""
        return len(self.active_connections)
    
    def is_connected(self) -> bool:
        """是否有活躍連接"""
        return len(self.active_connections) > 0


# ===== 全局單例 =====

_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """獲取全局 WebSocket Manager 實例（單例模式）
    
    Returns:
        WebSocket Manager 實例
    """
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager


def reset_websocket_manager() -> None:
    """重置 WebSocket Manager（主要用於測試）"""
    global _websocket_manager
    _websocket_manager = None
