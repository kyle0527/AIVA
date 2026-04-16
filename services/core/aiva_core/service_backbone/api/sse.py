"""SSE (Server-Sent Events) 端點處理器

此模組提供即時日誌和狀態串流功能，用於 Dashboard 即時監控。

依據 AIVA Common 規範:
- 使用 Pydantic 模型定義數據結構
- 使用 aiva_common.utils.logging 統一日誌
- 使用 aiva_common.schemas 數據合約
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

# 異步文件 I/O 支援
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# NOTE: Pylance 會報告 "無法解析匯入" 警告，這是預期的
# 在安裝 sse-starlette 套件前，SSE 功能會被禁用
# 安裝方式：pip install -r requirements-dashboard.txt
try:
    from sse_starlette.sse import EventSourceResponse
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    EventSourceResponse = None  # type: ignore

from aiva_common.utils import get_logger

# 初始化 API 路由
router = APIRouter(prefix="/api", tags=["SSE"])
logger = get_logger(__name__)

# 檢查 SSE 是否可用
if not SSE_AVAILABLE:
    logger.warning("⚠️  SSE功能未啟用：sse-starlette 套件未安裝。執行 'pip install -r requirements-dashboard.txt' 以啟用此功能。")

# ==================== SSE 事件數據模型 ====================

class LogEvent:
    """日誌事件"""
    def __init__(self, timestamp: str, level: str, message: str, scan_id: str):
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.scan_id = scan_id
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "scan_id": self.scan_id
        }


class StatusEvent:
    """狀態事件"""
    def __init__(
        self, 
        scan_id: str, 
        status: str, 
        progress: float,
        phase: str,
        current_task: str | None = None,
        findings_count: int = 0
    ):
        self.scan_id = scan_id
        self.status = status
        self.progress = progress
        self.phase = phase
        self.current_task = current_task
        self.findings_count = findings_count
    
    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "progress": self.progress,
            "phase": self.phase,
            "current_task": self.current_task,
            "findings_count": self.findings_count,
            "timestamp": datetime.now().isoformat()
        }


# ==================== 異步文件讀取輔助函數 ====================

async def _read_existing_logs(log_file: Path, scan_id: str) -> AsyncIterator[LogEvent]:
    """異步讀取現有日誌"""
    if AIOFILES_AVAILABLE:
        async with aiofiles.open(log_file, encoding='utf-8') as f:
            async for line in f:
                log_event = _parse_log_line(line, scan_id)
                if log_event:
                    yield log_event
    else:
        # 使用線程池執行同步 I/O
        def read_lines():
            with open(log_file, encoding='utf-8') as f:
                return f.readlines()
        
        lines = await asyncio.to_thread(read_lines)
        for line in lines:
            log_event = _parse_log_line(line, scan_id)
            if log_event:
                yield log_event


async def _read_new_logs_from_position(
    log_file: Path, 
    position: int, 
    scan_id: str
) -> tuple[list[LogEvent], int, bool]:
    """異步讀取指定位置之後的新日誌
    
    Returns:
        tuple: (日誌事件列表, 新位置, 是否完成)
    """
    events: list[LogEvent] = []
    scan_completed = False
    new_position = position
    
    if AIOFILES_AVAILABLE:
        async with aiofiles.open(log_file, encoding='utf-8') as f:
            await f.seek(position)
            content = await f.read()
            new_position = position + len(content.encode('utf-8'))
            
            for line in content.splitlines():
                log_event = _parse_log_line(line, scan_id)
                if log_event:
                    events.append(log_event)
                    if "completed" in line.lower() or "finished" in line.lower():
                        scan_completed = True
    else:
        def read_from_position():
            nonlocal new_position, scan_completed
            result = []
            with open(log_file, encoding='utf-8') as f:
                f.seek(position)
                for line in f:
                    log_event = _parse_log_line(line, scan_id)
                    if log_event:
                        result.append(log_event)
                        if "completed" in line.lower() or "finished" in line.lower():
                            scan_completed = True
                new_position = f.tell()
            return result
        
        events = await asyncio.to_thread(read_from_position)
    
    return events, new_position, scan_completed


# ==================== SSE 端點 ====================

@router.get("/logs/stream")
async def stream_logs(scan_id: str) -> Any:  # EventSourceResponse when available
    """串流掃描日誌（SSE）
    
    Args:
        scan_id: 掃描任務 ID
    
    Returns:
        EventSourceResponse: SSE 串流響應
    
    Event 格式:
        event: log
        data: {"timestamp": "...", "level": "INFO", "message": "...", "scan_id": "..."}
        
        event: completed
        data: {"scan_id": "..."}
    """
    logger.info(f"[SSE] Starting log stream for scan_id={scan_id}")
    
    async def log_generator() -> AsyncIterator[dict]:
        """日誌生成器"""
        # 確定日誌文件路徑（使用項目根目錄）
        project_root = Path(__file__).parents[6]  # app.py → api → service_backbone → aiva_core → core → services → AIVA-git
        log_file = project_root / "logs" / f"{scan_id}.log"
        
        # 如果日誌檔不存在，嘗試讀取全局日誌
        if not log_file.exists():
            log_file = project_root / "logs" / "aiva.log"
        
        logger.info(f"[SSE] Monitoring log file: {log_file}")
        
        # 步驟 1: 讀取現有日誌
        # 使用異步方式讀取日誌
        if log_file.exists():
            try:
                async for log_event in _read_existing_logs(log_file, scan_id):
                    yield {
                        "event": "log",
                        "data": json.dumps(log_event.to_dict())
                    }
            except Exception as e:
                logger.error(f"[SSE] Error reading existing logs: {e}")
        
        # 步驟 2: 持續監控新日誌
        last_position = log_file.stat().st_size if log_file.exists() else 0
        scan_completed = False
        
        while not scan_completed:
            await asyncio.sleep(0.5)  # 500ms 檢查間隔
            
            if not log_file.exists():
                continue
            
            try:
                current_size = log_file.stat().st_size
                
                # 如果檔案有新內容
                if current_size > last_position:
                    # 使用異步方式讀取新日誌
                    events, last_position, scan_completed = await _read_new_logs_from_position(
                        log_file, last_position, scan_id
                    )
                    for log_event in events:
                        yield {
                            "event": "log",
                            "data": json.dumps(log_event.to_dict())
                        }
                
                # 超時保護：30 分鐘後自動斷開
                # NOTE: 未來可從 SessionStateManager 檢查實際狀態來決定是否繼續串流
                
            except Exception as e:
                logger.error(f"[SSE] Error monitoring logs: {e}")
                break
        
        # 發送完成事件
        logger.info(f"[SSE] Scan {scan_id} completed, closing stream")
        yield {
            "event": "completed",
            "data": json.dumps({"scan_id": scan_id})
        }
    
    if not SSE_AVAILABLE or EventSourceResponse is None:
        raise HTTPException(status_code=501, detail="SSE 功能未啟用，請安裝 sse-starlette")
    return EventSourceResponse(log_generator())  # type: ignore[misc]


@router.get("/status/stream")
async def stream_status(scan_id: str) -> Any:  # EventSourceResponse when available
    """串流掃描狀態（SSE）
    
    Args:
        scan_id: 掃描任務 ID
    
    Returns:
        EventSourceResponse: SSE 串流響應
    
    Event 格式:
        event: status
        data: {
            "scan_id": "...",
            "status": "running",
            "progress": 0.45,
            "phase": "phase1",
            "current_task": "...",
            "findings_count": 12
        }
        
        event: completed
        data: {"scan_id": "..."}
    """
    logger.info(f"[SSE] Starting status stream for scan_id={scan_id}")
    
    async def status_generator() -> AsyncIterator[dict]:
        """狀態生成器"""
        # 需要導入 SessionStateManager（延遲導入避免循環依賴）
        from services.core.aiva_core.service_backbone.state.session_state_manager import (
            SessionStateManager,
        )
        
        state_manager = SessionStateManager()
        
        while True:
            try:
                # 獲取當前狀態
                status_data = state_manager.get_session_status(scan_id)
                
                # 構建狀態事件
                status_event = StatusEvent(
                    scan_id=scan_id,
                    status=status_data.get("status", "unknown"),
                    progress=float(status_data.get("progress", 0.0)),
                    phase=status_data.get("phase", "unknown"),
                    current_task=status_data.get("current_task"),
                    findings_count=int(status_data.get("findings_count", 0))
                )
                
                yield {
                    "event": "status",
                    "data": json.dumps(status_event.to_dict())
                }
                
                # 檢查是否已完成
                if status_data.get("status") in ["completed", "failed", "cancelled"]:
                    logger.info(f"[SSE] Scan {scan_id} reached terminal state: {status_data.get('status')}")
                    yield {
                        "event": "completed",
                        "data": json.dumps({
                            "scan_id": scan_id,
                            "final_status": status_data.get("status")
                        })
                    }
                    break
                
                # 2 秒更新間隔
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"[SSE] Error getting status for {scan_id}: {e}")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
                break
    
    if not SSE_AVAILABLE or EventSourceResponse is None:
        raise HTTPException(status_code=501, detail="SSE 功能未啟用，請安裝 sse-starlette")
    return EventSourceResponse(status_generator())  # type: ignore[misc]


# ==================== 工具函數 ====================

def _parse_log_line(line: str, scan_id: str) -> LogEvent | None:
    """解析日誌行
    
    Args:
        line: 日誌行
        scan_id: 掃描 ID（用於篩選）
    
    Returns:
        LogEvent 或 None
    """
    try:
        # 忽略空行
        if not line.strip():
            return None
        
        # 簡單解析（假設格式：timestamp - name - level - message）
        # 2026-02-13 12:34:56 - aiva.core - INFO - [Scan] scan_123 started
        parts = line.split(" - ", maxsplit=3)
        if len(parts) >= 4:
            timestamp = parts[0].strip()
            level = parts[2].strip()
            message = parts[3].strip()
            
            # 篩選相關日誌（包含 scan_id 或全局訊息）
            if scan_id in message or "[SSE]" in message or "[Scan]" in message:
                return LogEvent(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    scan_id=scan_id
                )
        
        return None
        
    except Exception as e:
        logger.debug(f"[SSE] Failed to parse log line: {e}")
        return None
