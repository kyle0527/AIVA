from pydantic import BaseModel, Field
from typing import List

class IdorConfig(BaseModel):
    horizontal_enabled: bool = True
    vertical_enabled: bool = True
    max_id_variations: int = 5
    allow_active_network: bool = False
    safe_mode: bool = True
    privileged_urls: List[str] = Field(default_factory=list)
    request_timeout: float = 8.0
