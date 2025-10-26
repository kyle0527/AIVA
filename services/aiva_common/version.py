"""
AIVA Common 版本信息
"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

# 構建信息
__build__ = "main"
__commit__ = "unknown"

# 兼容性信息
MIN_PYTHON_VERSION = (3, 11, 0)
SUPPORTED_PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]

# API 版本
API_VERSION = "v1"
SCHEMA_VERSION = "1.0"

def get_version() -> str:
    """獲取完整版本字符串"""
    version = __version__
    if __build__ != "main":
        version += f"+{__build__}"
    return version

def get_version_info() -> dict:
    """獲取版本信息字典"""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build": __build__,
        "commit": __commit__,
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "python_version": f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+",
        "supported_python": SUPPORTED_PYTHON_VERSIONS,
    }