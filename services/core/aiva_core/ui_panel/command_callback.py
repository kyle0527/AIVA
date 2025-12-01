"""
UI 命令回調處理器

實現 CommandCallback 協議，用於將命令執行進度、狀態和結果推送到 UI。
"""

from typing import Dict, Any, Optional
from datetime import datetime, UTC

from services.aiva_common.schemas.commands import CommandCallback, CommandStatus
from services.aiva_common.utils import get_logger
from services.core.aiva_core.ui_panel.websocket_manager import get_websocket_manager

# 嘗試導入 Rich Console（可選）
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

logger = get_logger(__name__)


class UICommandCallback:
    """UI 命令回調處理器
    
    將命令執行的進度、狀態變更和中間結果推送到 UI。
    支持多種 UI 輸出方式:
    1. WebSocket 推送（用於 Web 儀表板）
    2. Rich Console 輸出（用於 CLI）
    3. 日誌記錄
    """
    
    def __init__(
        self, 
        use_websocket: bool = True,
        use_rich_console: bool = True,
        websocket_manager: Optional[Any] = None
    ):
        """初始化 UI 回調處理器
        
        Args:
            use_websocket: 是否使用 WebSocket 推送
            use_rich_console: 是否使用 Rich Console 輸出
            websocket_manager: WebSocket 管理器（可選，不提供則使用全局單例）
        """
        self.use_websocket = use_websocket
        self.use_rich_console = use_rich_console and RICH_AVAILABLE
        
        # WebSocket 管理器
        if use_websocket:
            self.websocket_manager = websocket_manager or get_websocket_manager()
        else:
            self.websocket_manager = None
        
        # 進度緩存（用於避免過於頻繁的更新）
        self.progress_cache: Dict[str, float] = {}
        self.last_update_time: Dict[str, float] = {}
        self.min_update_interval = 0.5  # 最小更新間隔（秒）
        
        self.logger = logger
        self.logger.debug("✅ UI 命令回調處理器已初始化")
    
    async def on_progress(
        self,
        command_id: str,
        progress: float,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """進度更新回調
        
        Args:
            command_id: 命令 ID
            progress: 進度值 (0.0 - 1.0)
            message: 進度消息
            metadata: 額外的元數據
        """
        # 檢查是否需要節流（避免過於頻繁的更新）
        current_time = datetime.now(UTC).timestamp()
        last_time = self.last_update_time.get(command_id, 0)
        
        if current_time - last_time < self.min_update_interval:
            # 距離上次更新時間太短，跳過（除非是 100% 完成）
            if progress < 1.0:
                return
        
        self.progress_cache[command_id] = progress
        self.last_update_time[command_id] = current_time
        
        # 構建更新消息
        update_message = {
            "type": "progress_update",
            "command_id": command_id,
            "progress": progress,
            "progress_percent": f"{progress * 100:.1f}%",
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        # 1. WebSocket 推送
        if self.use_websocket and self.websocket_manager:
            try:
                await self.websocket_manager.broadcast(update_message)
            except Exception as e:
                self.logger.error(f"WebSocket 推送失敗: {e}")
        
        # 2. Rich Console 輸出
        if self.use_rich_console and console:
            try:
                progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
                console.print(
                    f"[cyan]►[/cyan] [{command_id[:8]}] "
                    f"[yellow]{progress_bar}[/yellow] "
                    f"[green]{progress * 100:.1f}%[/green] - {message}"
                )
            except Exception as e:
                self.logger.error(f"Rich Console 輸出失敗: {e}")
        
        # 3. 日誌記錄
        self.logger.info(
            f"[進度更新] {command_id}: {progress:.1%} - {message}"
        )
    
    async def on_status_change(
        self,
        command_id: str,
        old_status: CommandStatus,
        new_status: CommandStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """狀態變更回調
        
        Args:
            command_id: 命令 ID
            old_status: 舊狀態
            new_status: 新狀態
            metadata: 額外的元數據
        """
        # 構建狀態變更消息
        status_message = {
            "type": "status_change",
            "command_id": command_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        # 1. WebSocket 推送
        if self.use_websocket and self.websocket_manager:
            try:
                await self.websocket_manager.broadcast(status_message)
            except Exception as e:
                self.logger.error(f"WebSocket 推送失敗: {e}")
        
        # 2. Rich Console 輸出
        if self.use_rich_console and console:
            try:
                status_color = self._get_status_color(new_status)
                status_icon = self._get_status_icon(new_status)
                
                console.print(
                    f"[{status_color}]{status_icon}[/{status_color}] "
                    f"[{command_id[:8]}] "
                    f"[dim]{old_status.value}[/dim] → "
                    f"[{status_color}]{new_status.value}[/{status_color}]"
                )
            except Exception as e:
                self.logger.error(f"Rich Console 輸出失敗: {e}")
        
        # 3. 日誌記錄
        self.logger.info(
            f"[狀態變更] {command_id}: {old_status.value} → {new_status.value}"
        )
    
    async def on_partial_result(
        self,
        command_id: str,
        result_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """中間結果回調
        
        Args:
            command_id: 命令 ID
            result_type: 結果類型
            data: 結果數據
            metadata: 額外的元數據
        """
        # 構建結果消息
        result_message = {
            "type": "partial_result",
            "command_id": command_id,
            "result_type": result_type,
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        # 1. WebSocket 推送
        if self.use_websocket and self.websocket_manager:
            try:
                await self.websocket_manager.broadcast(result_message)
            except Exception as e:
                self.logger.error(f"WebSocket 推送失敗: {e}")
        
        # 2. Rich Console 輸出
        if self.use_rich_console and console:
            try:
                self._display_partial_result_rich(command_id, result_type, data)
            except Exception as e:
                self.logger.error(f"Rich Console 輸出失敗: {e}")
        
        # 3. 日誌記錄
        self.logger.info(
            f"[中間結果] {command_id} [{result_type}]: {str(data)[:200]}"
        )
    
    # ===== 輔助方法 =====
    
    def _get_status_color(self, status: CommandStatus) -> str:
        """獲取狀態對應的顏色"""
        color_map = {
            CommandStatus.PENDING: "yellow",
            CommandStatus.RUNNING: "cyan",
            CommandStatus.COMPLETED: "green",
            CommandStatus.FAILED: "red",
            CommandStatus.TIMEOUT: "orange",
            CommandStatus.CANCELLED: "grey"
        }
        return color_map.get(status, "white")
    
    def _get_status_icon(self, status: CommandStatus) -> str:
        """獲取狀態對應的圖標"""
        icon_map = {
            CommandStatus.PENDING: "⏸️",
            CommandStatus.RUNNING: "▶️",
            CommandStatus.COMPLETED: "✅",
            CommandStatus.FAILED: "❌",
            CommandStatus.TIMEOUT: "⏱️",
            CommandStatus.CANCELLED: "🚫"
        }
        return icon_map.get(status, "●")
    
    def _display_partial_result_rich(
        self, 
        command_id: str, 
        result_type: str, 
        data: Any
    ) -> None:
        """在 Rich Console 顯示中間結果"""
        if not console:
            return
        
        # 根據結果類型顯示不同的格式
        if result_type == "vulnerability_found":
            vuln_type = data.get("type", "Unknown") if isinstance(data, dict) else "Unknown"
            severity = data.get("severity", "Unknown") if isinstance(data, dict) else "Unknown"
            
            console.print(
                f"[red]🚨 發現漏洞[/red]: "
                f"[yellow]{vuln_type}[/yellow] "
                f"([magenta]{severity}[/magenta])"
            )
        
        elif result_type == "url_discovered":
            url = data.get("url", str(data)) if isinstance(data, dict) else str(data)
            console.print(
                f"[cyan]🔗 發現 URL[/cyan]: [blue]{url}[/blue]"
            )
        
        elif result_type == "asset_found":
            asset_type = data.get("type", "Unknown") if isinstance(data, dict) else "Unknown"
            console.print(
                f"[green]📦 發現資產[/green]: [yellow]{asset_type}[/yellow]"
            )
        
        elif result_type == "error":
            error_msg = data.get("error", str(data)) if isinstance(data, dict) else str(data)
            console.print(
                f"[red]❌ 錯誤[/red]: {error_msg}"
            )
        
        else:
            # 通用格式
            console.print(
                f"[cyan]📋 {result_type}[/cyan]: {str(data)[:100]}"
            )
    
    def get_progress(self, command_id: str) -> Optional[float]:
        """獲取命令的當前進度
        
        Args:
            command_id: 命令 ID
            
        Returns:
            進度值 (0.0 - 1.0)，如果不存在則返回 None
        """
        return self.progress_cache.get(command_id)
    
    def clear_progress(self, command_id: str) -> None:
        """清除命令的進度緩存
        
        Args:
            command_id: 命令 ID
        """
        self.progress_cache.pop(command_id, None)
        self.last_update_time.pop(command_id, None)


# ===== 全局單例 =====

_ui_callback: Optional[UICommandCallback] = None


def get_ui_callback(
    use_websocket: bool = True,
    use_rich_console: bool = True
) -> UICommandCallback:
    """獲取全局 UI 回調處理器實例（單例模式）
    
    Args:
        use_websocket: 是否使用 WebSocket 推送
        use_rich_console: 是否使用 Rich Console 輸出
        
    Returns:
        UI 回調處理器實例
    """
    global _ui_callback
    if _ui_callback is None:
        _ui_callback = UICommandCallback(
            use_websocket=use_websocket,
            use_rich_console=use_rich_console
        )
    return _ui_callback


def reset_ui_callback() -> None:
    """重置 UI 回調處理器（主要用於測試）"""
    global _ui_callback
    _ui_callback = None
