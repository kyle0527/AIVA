# -*- coding: utf-8 -*-
"""
AIVA 功能模組基礎架構

提供統一的介面、註冊機制、HTTP 客戶端和結果格式，
支援高價值安全漏洞檢測模組的快速開發和整合。

本模組旨在實現可直接用於 Bug Bounty 和滲透測試的
實戰級功能，包含嚴格的安全控制和標準化輸出。
"""

from .feature_base import FeatureBase
from .feature_registry import FeatureRegistry
from .http_client import SafeHttp
from .result_schema import FeatureResult, Finding

__all__ = ["FeatureBase", "FeatureRegistry", "SafeHttp", "FeatureResult", "Finding"]