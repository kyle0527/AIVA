"""
Mode Manager - 執行模式管理器

統一管理系統執行模式（sandbox/production），支援多種切換方式：
1. API 調用（程式化切換）
2. CLI 命令（終端切換）
3. 設定檔（持久化配置）
4. 環境變數（部署時配置）

設計原則：
- 遵循 aiva_common 規範：簡單預設值，研發階段開箱即用
- 優先級：API > CLI > 環境變數 > 設定檔 > 預設值
- 持久化：自動保存到設定檔
- 線程安全：支援並發訪問
"""

import os
import json
import threading
from pathlib import Path
from typing import Literal

from services.aiva_common.utils import get_logger
from .executor.execution_status_monitor import EnvironmentType

logger = get_logger(__name__)


class ModeManager:
    """執行模式管理器
    
    管理系統執行模式的持久化和切換邏輯。
    
    切換優先級（高到低）：
    1. 執行時 API 調用（臨時覆蓋）
    2. CLI 命令（用戶手動設定）
    3. 環境變數（部署配置）
    4. 設定檔（持久化配置）
    5. 預設值（sandbox，研發友好）
    
    使用範例：
        >>> manager = ModeManager()
        >>> manager.set_mode("production")  # API 切換
        >>> current = manager.get_mode()  # 獲取當前模式
    """
    
    def __init__(self, config_path: Path | None = None):
        """初始化模式管理器
        
        Args:
            config_path: 設定檔路徑（預設為 ./data/mode_config.json）
        """
        self._lock = threading.RLock()
        
        # 設定檔路徑（遵循 aiva_common 原則：自動推導）
        if config_path is None:
            config_path = Path("./data/aiva_core/mode_config.json")
        
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 當前模式（內存緩存）
        self._current_mode: EnvironmentType | None = None
        
        # 初始化時載入配置
        self._load_from_config()
        
        logger.info(f"✅ ModeManager initialized: current_mode={self.get_mode().value}")
    
    def get_mode(self) -> EnvironmentType:
        """獲取當前執行模式
        
        按優先級查找：環境變數 > 設定檔 > 預設值
        
        Returns:
            當前執行模式（sandbox 或 production）
        """
        with self._lock:
            # 1. 檢查環境變數（優先級最高）
            env_mode = os.getenv("AIVA_EXECUTION_MODE")
            if env_mode:
                try:
                    return EnvironmentType(env_mode.lower())
                except ValueError:
                    logger.warning(
                        f"Invalid AIVA_EXECUTION_MODE: {env_mode}, "
                        f"using config/default"
                    )
            
            # 2. 使用內存緩存（已從設定檔載入）
            if self._current_mode:
                return self._current_mode
            
            # 3. 預設值（研發階段友好）
            return EnvironmentType.SANDBOX
    
    def set_mode(
        self, 
        mode: Literal["sandbox", "production"] | EnvironmentType,
        persist: bool = True
    ) -> None:
        """設定執行模式
        
        Args:
            mode: 目標模式（"sandbox" 或 "production"）
            persist: 是否持久化到設定檔（預設 True）
        
        Raises:
            ValueError: 模式名稱無效
        """
        with self._lock:
            # 正規化輸入
            if isinstance(mode, str):
                try:
                    mode = EnvironmentType(mode.lower())
                except ValueError as e:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be 'sandbox' or 'production'"
                    ) from e
            
            # 更新內存緩存
            old_mode = self._current_mode
            self._current_mode = mode
            
            # 持久化到設定檔
            if persist:
                self._save_to_config()
            
            logger.info(f"🔄 Mode switched: {old_mode} → {mode.value}")
    
    def _load_from_config(self) -> None:
        """從設定檔載入模式配置"""
        if not self.config_path.exists():
            logger.debug(f"Config file not found: {self.config_path}, using default")
            return
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            mode_str = config.get("execution_mode")
            if mode_str:
                self._current_mode = EnvironmentType(mode_str)
                logger.debug(f"Loaded mode from config: {mode_str}")
        
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using default")
    
    def _save_to_config(self) -> None:
        """保存模式配置到設定檔"""
        try:
            config = {
                "execution_mode": self._current_mode.value if self._current_mode else "sandbox",
                "updated_at": str(Path(__file__).stat().st_mtime),
            }
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Config saved: {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def reset(self) -> None:
        """重置為預設模式（sandbox）"""
        with self._lock:
            self._current_mode = EnvironmentType.SANDBOX
            self._save_to_config()
            logger.info("🔄 Mode reset to default: sandbox")


# 全局單例（遵循 aiva_common 單例模式）
_mode_manager_instance: ModeManager | None = None
_instance_lock = threading.Lock()


def get_mode_manager(config_path: Path | None = None) -> ModeManager:
    """獲取 ModeManager 單例實例
    
    Args:
        config_path: 設定檔路徑（僅首次調用有效）
    
    Returns:
        ModeManager 實例
    """
    global _mode_manager_instance
    
    if _mode_manager_instance is None:
        with _instance_lock:
            if _mode_manager_instance is None:
                _mode_manager_instance = ModeManager(config_path)
    
    return _mode_manager_instance
