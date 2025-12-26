"""
UI Panel 通用工具函数

提供端口查找、错误处理、格式化等通用功能
"""

import socket
from typing import Any

from services.aiva_common.error_handling import (
    AIVAError,
    ErrorSeverity,
    ErrorType,
    create_error_context,
)
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)
MODULE_NAME = "ui.utils"


def find_free_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """查找可用的端口号
    
    Args:
        start_port: 起始端口号
        max_attempts: 最大尝试次数
    
    Returns:
        可用的端口号
    
    Raises:
        AIVAError: 找不到可用的端口
    
    Example:
        >>> port = find_free_port(8080, 100)
        >>> print(f"Found free port: {port}")
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
                logger.info(f"Found free port: {port}")
                return port
        except OSError:
            continue
    
    msg = f"无法在 {start_port}-{start_port + max_attempts - 1} 范围内找到可用端口"
    raise AIVAError(
        msg,
        error_type=ErrorType.SYSTEM,
        severity=ErrorSeverity.HIGH,
        context=create_error_context(
            module=MODULE_NAME,
            function="find_free_port",
            start_port=start_port,
            max_attempts=max_attempts,
        ),
    )


def format_result_preview(result: Any, max_length: int = 100) -> str:
    """格式化结果预览
    
    Args:
        result: 结果数据（可以是字符串、字典、列表等）
        max_length: 最大长度
    
    Returns:
        格式化后的字符串
    
    Example:
        >>> preview = format_result_preview({"vuln": "XSS", "count": 5})
        >>> print(preview)
        '{"vuln": "XSS", "count": 5}'
    """
    if result is None:
        return "-"
    
    if isinstance(result, (dict, list)):
        result_str = str(result)
    else:
        result_str = str(result)
    
    if len(result_str) > max_length:
        return result_str[:max_length] + "..."
    return result_str


def get_mode_display(mode: str) -> str:
    """获取模式的显示名称
    
    Args:
        mode: 运行模式 (ui/ai/hybrid)
    
    Returns:
        显示名称
    
    Example:
        >>> display = get_mode_display("hybrid")
        >>> print(display)
        'UI + AI 混合模式'
    """
    mode_map = {
        "ui": "仅 UI 界面",
        "ai": "仅 AI 代理",
        "hybrid": "UI + AI 混合模式",
    }
    return mode_map.get(mode, "未知模式")


def validate_url(url: str) -> bool:
    """验证 URL 格式
    
    Args:
        url: 要验证的 URL
    
    Returns:
        是否为有效 URL
    
    Example:
        >>> is_valid = validate_url("https://example.com")
        >>> print(is_valid)
        True
    """
    if not url:
        return False
    
    # 基本验证：确保以 http:// 或 https:// 开头
    if not url.startswith(("http://", "https://")):
        return False
    
    # 确保至少有域名部分
    parts = url.split("://")
    if len(parts) != 2 or not parts[1]:
        return False
    
    return True


def safe_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """安全地从字典获取值
    
    Args:
        data: 字典数据
        key: 键名
        default: 默认值
    
    Returns:
        值或默认值
    
    Example:
        >>> value = safe_get({"name": "test"}, "name", "unknown")
        >>> print(value)
        'test'
    """
    try:
        return data.get(key, default)
    except (AttributeError, KeyError):
        return default


def format_timestamp(timestamp: Any) -> str:
    """格式化时间戳
    
    Args:
        timestamp: 时间戳（datetime 对象或字符串）
    
    Returns:
        格式化的时间字符串
    
    Example:
        >>> from datetime import datetime
        >>> ts = datetime(2025, 12, 16, 10, 30, 0)
        >>> formatted = format_timestamp(ts)
        >>> print(formatted)
        '2025-12-16 10:30:00'
    """
    if timestamp is None:
        return "-"
    
    from datetime import datetime
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            return str(timestamp)
    
    return str(timestamp)


def build_task_row(task: dict[str, Any]) -> str:
    """构建任务表格行
    
    Args:
        task: 任务数据字典
    
    Returns:
        HTML 表格行
    
    Example:
        >>> task = {"task_id": "123", "target": "example.com", "status": "running"}
        >>> row = build_task_row(task)
    """
    task_id = safe_get(task, "task_id", "-")
    target = safe_get(task, "target", "-")
    scan_type = safe_get(task, "scan_type", "-")
    status = safe_get(task, "status", "-")
    created_by = safe_get(task, "created_by", "-")
    ai_result_text = "AI" if task.get("ai_result") else "-"
    
    return (
        f"<tr>"
        f"<td>{task_id}</td>"
        f"<td>{target}</td>"
        f"<td>{scan_type}</td>"
        f"<td>{status}</td>"
        f"<td>{created_by}</td>"
        f"<td>{ai_result_text}</td>"
        f"</tr>"
    )


def build_detection_row(detection: dict[str, Any]) -> str:
    """构建检测结果表格行
    
    Args:
        detection: 检测结果数据字典
    
    Returns:
        HTML 表格行
    
    Example:
        >>> det = {"vuln_type": "XSS", "target": "example.com", "status": "found"}
        >>> row = build_detection_row(det)
    """
    vuln_type = safe_get(detection, "vuln_type", "-")
    target = safe_get(detection, "target", "-")
    status = safe_get(detection, "status", "-")
    method = safe_get(detection, "method", "-")
    result = safe_get(detection, "result", safe_get(detection, "findings", "-"))
    result_preview = format_result_preview(result, max_length=50)
    
    return (
        f"<tr>"
        f"<td>{vuln_type}</td>"
        f"<td>{target}</td>"
        f"<td>{status}</td>"
        f"<td>{method}</td>"
        f"<td>{result_preview}</td>"
        f"</tr>"
    )


def check_dependencies() -> dict[str, bool]:
    """检查UI模块的依赖项
    
    Returns:
        依赖项检查结果字典
    
    Example:
        >>> deps = check_dependencies()
        >>> print(deps)
        {'fastapi': True, 'uvicorn': True, 'rich': True, 'websockets': True}
    """
    dependencies = {}
    
    # 检查 FastAPI
    try:
        import fastapi  # noqa: F401
        dependencies["fastapi"] = True
    except ImportError:
        dependencies["fastapi"] = False
    
    # 检查 Uvicorn
    try:
        import uvicorn  # noqa: F401
        dependencies["uvicorn"] = True
    except ImportError:
        dependencies["uvicorn"] = False
    
    # 检查 Rich
    try:
        import rich  # noqa: F401
        dependencies["rich"] = True
    except ImportError:
        dependencies["rich"] = False
    
    # 检查 WebSockets
    try:
        import websockets  # noqa: F401
        dependencies["websockets"] = True
    except ImportError:
        dependencies["websockets"] = False
    
    return dependencies


def log_dependency_status() -> None:
    """记录依赖项状态到日志
    
    Example:
        >>> log_dependency_status()
        INFO: Checking UI dependencies...
        INFO: fastapi: ✓
        INFO: uvicorn: ✓
    """
    logger.info("Checking UI dependencies...")
    deps = check_dependencies()
    
    for dep_name, available in deps.items():
        status = "✓" if available else "✗"
        logger.info(f"{dep_name}: {status}")
    
    missing = [name for name, avail in deps.items() if not avail]
    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        logger.warning("Install with: pip install fastapi uvicorn rich websockets")
