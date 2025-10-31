"""
AIVA Common - 日誌工具模組

提供統一的日誌配置和管理功能，符合 AIVA 開發規範
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: str | None = None,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """
    設置並配置日誌記錄器

    Args:
        name: 日誌記錄器名稱
        level: 日誌級別
        format_string: 自定義格式字符串
        handler: 自定義處理器

    Returns:
        配置好的日誌記錄器
    """
    logger = logging.getLogger(name)

    # 避免重複配置
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 使用自定義處理器或默認處理器
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    # 設置格式
    if format_string is None:
        format_string = "%(asctime)s %(levelname)s %(name)s - %(message)s"

    formatter = logging.Formatter(fmt=format_string, datefmt="%Y-%m-%dT%H:%M:%S%z")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def configure_root_logger(
    level: int = logging.INFO, format_string: str | None = None
) -> None:
    """
    配置根日誌記錄器

    Args:
        level: 日誌級別
        format_string: 格式字符串
    """
    if format_string is None:
        format_string = "%(asctime)s %(levelname)s %(name)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
    )
