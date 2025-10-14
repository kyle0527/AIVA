"""
AuthZ Module - 授權與權限分析模組

提供權限矩陣分析、角色映射、權限視覺化等功能。
"""

from .authz_mapper import AuthZMapper
from .matrix_visualizer import MatrixVisualizer
from .permission_matrix import PermissionMatrix

__all__ = [
    "PermissionMatrix",
    "AuthZMapper",
    "MatrixVisualizer",
]

__version__ = "1.0.0"
