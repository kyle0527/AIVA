"""Manifest 系統 - 能力靜態註冊表

提供 AI 與 CLI 工具之間的標準化映射。

Architecture:
- manifest_loader.py: JSON 加載和驗證
- manifest_schema.py: Pydantic 模型定義 (遷移到 aiva_common)
- hard_mask.py: 標籤過濾器 (Phase 1)

Version: 1.0.0
"""

from .manifest_loader import ManifestLoader

__all__ = ["ManifestLoader"]
