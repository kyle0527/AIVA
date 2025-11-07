from pydantic import BaseModel

class SsrfConfig(BaseModel):
    enable_internal_scan: bool = True
    enable_cloud_metadata: bool = True
    enable_file_protocol: bool = False
    allow_active_network: bool = False
    request_timeout: float = 8.0
    max_redirects: int = 3
    safe_mode: bool = True
