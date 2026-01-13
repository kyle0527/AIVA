"""
Information Leak Detection Module
檢測敏感信息洩露（API密鑰、JWT、密碼、信用卡等）
"""

from .sensitive_info_detector import SensitiveInfoDetector, SensitiveMatch

__all__ = ['SensitiveInfoDetector', 'SensitiveMatch']
__version__ = '1.0.0'
