from pydantic import BaseModel
from typing import List

class SsrfConfig(BaseModel):
    enable_internal_scan: bool = True
    enable_cloud_metadata: bool = True
    enable_file_protocol: bool = False
    allow_active_network: bool = False
    request_timeout: float = 8.0
    max_redirects: int = 3
    safe_mode: bool = True
    
    # 新增：雲端元數據檢測優先
    cloud_metadata_first: bool = True
    
    # 新增：雲端元數據端點列表
    cloud_metadata_endpoints: List[str] = [
        "169.254.169.254",
        "metadata.google.internal",
        "100.100.100.200",
        "metadata.tencentyun.com"
    ]
    
    # 新增：OAST 回調等待時間（秒）
    oast_wait_time: float = 5.0
