# -*- coding: utf-8 -*-
"""
AIVA API 路由模組初始化

導出所有路由模組，便於主應用程序集成。
"""

from . import auth
from . import security  
from . import admin

__all__ = ['auth', 'security', 'admin']