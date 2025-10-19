# scan/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class Target(BaseModel):
    url: str
    meta: Optional[Dict[str, Any]] = None

class ScanContext(BaseModel):
    targets: List[Target] = []
    depth: int = Field(2)
    options: Optional[Dict[str, Any]] = None
