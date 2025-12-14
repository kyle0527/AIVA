# -*- coding: utf-8 -*-
"""
Admin Router - 空路由器 (待實作)
"""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

# 共享的 active_scans 存儲
_active_scans: Dict[str, Dict[str, Any]] = {}

def get_active_scans() -> Dict[str, Dict[str, Any]]:
    """獲取 active_scans 字典"""
    return _active_scans
