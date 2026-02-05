"""
Sensitivity Manager - 目標敏感度管理器

設計原則：
- 實際執行絕大部分是黑盒測試，不區分環境
- 策略選擇基於目標敏感度 (0.0-1.0)
- 所有執行都學習，經驗直接可遷移

敏感度說明：
- 0.0-0.3: 低敏感 - 可激進探索
- 0.3-0.7: 中敏感 - 平衡策略
- 0.7-1.0: 高敏感 - 保守謹慎
"""

import os
import json
import threading
from pathlib import Path

from aiva_common.utils import get_logger

logger = get_logger(__name__)


class SensitivityManager:
    """目標敏感度管理器
    
    控制執行策略的激進程度，基於目標特性而非環境
    """
    
    def __init__(self, config_path: Path | None = None):
        """初始化敏感度管理器"""
        self._lock = threading.RLock()
        
        # 設定檔路徑
        if config_path is None:
            config_path = Path("./data/aiva_core/sensitivity_config.json")
        
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._sensitivity: float = 0.5  # 預設中等敏感度
        
        # 初始化時載入配置
        self._load_from_config()
        
        logger.info(
            "✅ SensitivityManager initialized: sensitivity=%.1f", self._sensitivity
        )
    
    def get(self) -> float:
        """獲取當前目標敏感度
        
        Returns:
            目標敏感度 (0.0-1.0)
        """
        with self._lock:
            # 1. 檢查環境變數
            env_sens = os.getenv("AIVA_TARGET_SENSITIVITY")
            if env_sens:
                try:
                    return float(env_sens)
                except ValueError:
                    logger.warning("Invalid AIVA_TARGET_SENSITIVITY: %s", env_sens)
            
            # 2. 使用內存值
            return self._sensitivity
    
    def set(self, sensitivity: float, persist: bool = True) -> None:
        """設定目標敏感度
        
        Args:
            sensitivity: 目標敏感度 (0.0-1.0)
            persist: 是否持久化
        """
        with self._lock:
            if not 0.0 <= sensitivity <= 1.0:
                raise ValueError("sensitivity must be between 0.0 and 1.0")
            
            old_sens = self._sensitivity
            self._sensitivity = sensitivity
            
            if persist:
                self._save_to_config()
            
            logger.info("🔄 Sensitivity changed: %.1f → %.1f", old_sens, sensitivity)
    
    def get_level(self) -> str:
        """獲取敏感度等級描述
        
        Returns:
            'low' / 'medium' / 'high'
        """
        sens = self.get()
        if sens <= 0.3:
            return "low"
        elif sens <= 0.7:
            return "medium"
        else:
            return "high"
    
    def _load_from_config(self) -> None:
        """從設定檔載入配置"""
        if not self.config_path.exists():
            return
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            if "sensitivity" in config:
                self._sensitivity = float(config["sensitivity"])
        
        except Exception as e:
            logger.warning("Failed to load config: %s", e)
    
    def _save_to_config(self) -> None:
        """保存配置到設定檔"""
        try:
            config = {
                "sensitivity": self._sensitivity,
                "level": self.get_level(),
            }
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error("Failed to save config: %s", e)
    
    def reset(self) -> None:
        """重置為預設敏感度 (0.5)"""
        with self._lock:
            self._sensitivity = 0.5
            self._save_to_config()
            logger.info("🔄 Reset to default sensitivity: 0.5")


# 全局單例
_instance: SensitivityManager | None = None
_instance_lock = threading.Lock()


def get_sensitivity_manager(config_path: Path | None = None) -> SensitivityManager:
    """獲取 SensitivityManager 單例實例"""
    global _instance
    
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SensitivityManager(config_path)
    
    return _instance


# 別名（向後兼容 import）
ModeManager = SensitivityManager
get_mode_manager = get_sensitivity_manager

