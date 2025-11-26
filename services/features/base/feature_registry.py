"""
功能模組註冊表

管理所有功能模組的註冊和調用
"""

from typing import Any, Callable, Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    功能模組註冊表
    
    負責：
    1. 註冊功能模組
    2. 查找功能模組
    3. 管理功能模組的元數據
    
    使用方式：
        registry = FeatureRegistry()
        registry.register("sqli", SqliFeature)
        feature = registry.get("sqli")
    """
    
    def __init__(self):
        """初始化註冊表"""
        self._features: Dict[str, Type] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
    def register(
        self,
        name: str,
        feature_class: Type,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        註冊功能模組
        
        Args:
            name: 功能模組名稱（如 "sqli", "xss"）
            feature_class: 功能模組類
            metadata: 功能模組元數據
        """
        if name in self._features:
            logger.warning(f"功能模組 '{name}' 已存在，將被覆蓋")
        
        self._features[name] = feature_class
        self._metadata[name] = metadata or {}
        
        logger.info(f"註冊功能模組: {name}")
    
    def unregister(self, name: str) -> None:
        """
        取消註冊功能模組
        
        Args:
            name: 功能模組名稱
        """
        if name in self._features:
            del self._features[name]
            del self._metadata[name]
            logger.info(f"取消註冊功能模組: {name}")
        else:
            logger.warning(f"功能模組 '{name}' 不存在")
    
    def get(self, name: str) -> Optional[Type]:
        """
        獲取功能模組類
        
        Args:
            name: 功能模組名稱
            
        Returns:
            功能模組類，如果不存在則返回 None
        """
        return self._features.get(name)
    
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        獲取功能模組元數據
        
        Args:
            name: 功能模組名稱
            
        Returns:
            功能模組元數據，如果不存在則返回 None
        """
        return self._metadata.get(name)
    
    def list_features(self) -> list[str]:
        """
        列出所有已註冊的功能模組
        
        Returns:
            功能模組名稱列表
        """
        return list(self._features.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        檢查功能模組是否已註冊
        
        Args:
            name: 功能模組名稱
            
        Returns:
            True 如果已註冊，否則 False
        """
        return name in self._features


# 全局註冊表實例
_global_registry = FeatureRegistry()


def get_global_registry() -> FeatureRegistry:
    """
    獲取全局功能模組註冊表
    
    Returns:
        全局註冊表實例
    """
    return _global_registry
