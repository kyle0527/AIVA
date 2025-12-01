"""
功能模組註冊表

管理所有功能模組的註冊和調用
"""

from typing import Any, Callable, Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    功能模組註冊表（單例模式）
    
    負責：
    1. 註冊功能模組
    2. 查找功能模組
    3. 管理功能模組的元數據
    
    使用方式：
        FeatureRegistry.register("sqli", SqliFeature)
        feature = FeatureRegistry.get("sqli")
    """
    
    _instance: Optional['FeatureRegistry'] = None
    _features: Dict[str, Type] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        """確保單例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(
        cls,
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
        if name in cls._features:
            logger.warning(f"功能模組 '{name}' 已存在，將被覆蓋")
        
        cls._features[name] = feature_class
        cls._metadata[name] = metadata or {}
        
        logger.info(f"註冊功能模組: {name}")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        取消註冊功能模組
        
        Args:
            name: 功能模組名稱
        """
        if name in cls._features:
            del cls._features[name]
            del cls._metadata[name]
            logger.info(f"取消註冊功能模組: {name}")
        else:
            logger.warning(f"功能模組 '{name}' 不存在")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        獲取功能模組類
        
        Args:
            name: 功能模組名稱
            
        Returns:
            功能模組類，如果不存在則返回 None
        """
        return cls._features.get(name)
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        獲取功能模組元數據
        
        Args:
            name: 功能模組名稱
            
        Returns:
            功能模組元數據，如果不存在則返回 None
        """
        return cls._metadata.get(name)
    
    @classmethod
    def list_features(cls) -> Dict[str, Type]:
        """
        列出所有已註冊的功能模組
        
        Returns:
            功能模組字典 {名稱: 類}
        """
        return cls._features.copy()
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        檢查功能模組是否已註冊
        
        Args:
            name: 功能模組名稱
            
        Returns:
            True 如果已註冊，否則 False
        """
        return name in cls._features


# 全局註冊表實例（向後兼容）
_global_registry = FeatureRegistry()


def get_global_registry() -> FeatureRegistry:
    """
    獲取全局功能模組註冊表
    
    Returns:
        全局註冊表實例
    """
    return _global_registry
