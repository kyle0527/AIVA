"""
Smart Detection Manager

智能檢測管理器，用於協調各種檢測功能。
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SmartDetectionManager:
    """智能檢測管理器
    
    協調和管理各種安全檢測功能的核心組件。
    """
    
    def __init__(self):
        """初始化智能檢測管理器"""
        self.detectors = {}
        self.active_scans = {}
        logger.info("SmartDetectionManager initialized")
    
    def register_detector(self, name: str, detector: Any) -> None:
        """註冊檢測器
        
        Args:
            name: 檢測器名稱
            detector: 檢測器實例
        """
        self.detectors[name] = detector
        logger.info(f"Registered detector: {name}")
    
    def start_detection(self, target: str, config: Dict[str, Any]) -> str:
        """開始檢測
        
        Args:
            target: 檢測目標
            config: 檢測配置
            
        Returns:
            檢測會話 ID
        """
        session_id = f"scan_{len(self.active_scans)}"
        self.active_scans[session_id] = {
            "target": target,
            "config": config,
            "status": "running"
        }
        logger.info(f"Started detection session: {session_id} for target: {target}")
        return session_id
    
    def get_detection_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """獲取檢測狀態
        
        Args:
            session_id: 檢測會話 ID
            
        Returns:
            檢測狀態信息
        """
        return self.active_scans.get(session_id)
    
    def stop_detection(self, session_id: str) -> bool:
        """停止檢測
        
        Args:
            session_id: 檢測會話 ID
            
        Returns:
            是否成功停止
        """
        if session_id in self.active_scans:
            self.active_scans[session_id]["status"] = "stopped"
            logger.info(f"Stopped detection session: {session_id}")
            return True
        return False
    
    def list_active_detections(self) -> List[str]:
        """列出活躍的檢測會話
        
        Returns:
            活躍會話 ID 列表
        """
        return [
            session_id for session_id, info in self.active_scans.items()
            if info["status"] == "running"
        ]
