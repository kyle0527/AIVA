# -*- coding: utf-8 -*-
from typing import Dict, Type
from .feature_base import FeatureBase

class FeatureRegistry:
    """
    簡單插件式註冊表：核心/PlanExecutor 可透過名稱取用功能模組。
    
    支援裝飾器註冊模式，讓新功能模組只需加上 @FeatureRegistry.register
    即可自動註冊到系統中。
    
    設計目的：
    - 模組化：每個功能獨立開發，統一管理
    - 動態載入：支援運行時註冊新功能
    - 簡單易用：透過名稱字串即可取得功能類別
    """
    _REG: Dict[str, Type[FeatureBase]] = {}

    @classmethod
    def register(cls, feature_cls: Type[FeatureBase]):
        """
        註冊功能模組到系統中
        
        同時支援：
        1. 類別名稱（如 IdorWorker -> "idorworker"）
        2. 簡化名稱（如 name 屬性 "idor"）
        """
        cls._REG[feature_cls.__name__.lower()] = feature_cls
        # 同時以簡名註冊（如 IdorWorker -> "idor"）
        key = getattr(feature_cls, "name", None)
        if key:
            cls._REG[key.lower()] = feature_cls
        return feature_cls

    @classmethod
    def get(cls, key: str) -> Type[FeatureBase]:
        """
        根據名稱取得功能模組類別
        
        Args:
            key: 模組名稱（不區分大小寫）
            
        Returns:
            功能模組類別
            
        Raises:
            KeyError: 當功能模組不存在時
        """
        k = key.lower()
        if k not in cls._REG:
            available = list(cls._REG.keys())
            raise KeyError(f"Feature `{key}` not found. Available: {available}")
        return cls._REG[k]
    
    @classmethod
    def list_features(cls) -> Dict[str, str]:
        """
        列出所有已註冊的功能模組
        
        Returns:
            {name: version} 的對應字典
        """
        result = {}
        for name, feature_cls in cls._REG.items():
            if hasattr(feature_cls, 'version'):
                result[name] = feature_cls.version
            else:
                result[name] = "unknown"
        return result