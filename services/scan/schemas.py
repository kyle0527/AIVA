# scan/schemas.py
"""
⚠️ DEPRECATED - 已棄用

此文件包含的類已被以下替代：
- Target → services.aiva_common.schemas.Target (更完整的定義)
- ScanContext → 使用 services.scan.models 中的配置類

此文件保留僅用於向後兼容，新代碼請勿使用。
建議遷移到標準的 aiva_common.schemas 中的類。
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# 已棄用：請使用 services.aiva_common.schemas.Target
class Target(BaseModel):
    url: str
    meta: Optional[Dict[str, Any]] = None

# 已棄用：請使用 services.scan.models 中的配置類
class ScanContext(BaseModel):
    targets: List[Target] = []
    depth: int = Field(2)
    options: Optional[Dict[str, Any]] = None
