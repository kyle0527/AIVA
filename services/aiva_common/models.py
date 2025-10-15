"""
AIVA Common Models - 共享基礎模型

此文件已被棄用，所有類現在都定義在 schemas.py 中。
為了向後兼容，這裡重新導出 schemas.py 中的相應類。

請直接從 aiva_common.schemas 或 aiva_common 導入這些類。

包含內容：
1. 核心消息協議 (MessageHeader, AivaMessage)
2. 通用認證和限流 (Authentication, RateLimit)
3. 官方安全標準 (CVSS v3.1, SARIF v2.1.0, CVE/CWE/CAPEC)
"""

from __future__ import annotations

# 為向後兼容而重新導出 schemas.py 中的類
from .schemas import (
    AivaMessage,
    Authentication,
    CAPECReference,
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    MessageHeader,
    RateLimit,
    SARIFLocation,
    SARIFReport,
    SARIFResult,
    SARIFRule,
    SARIFRun,
    SARIFTool,
)

__all__ = [
    # 核心消息協議
    "MessageHeader",
    "AivaMessage",
    # 通用認證和控制
    "Authentication",
    "RateLimit",
    # CVSS v3.1
    "CVSSv3Metrics",
    # CVE/CWE/CAPEC
    "CVEReference",
    "CWEReference",
    "CAPECReference",
    # SARIF v2.1.0
    "SARIFLocation",
    "SARIFResult",
    "SARIFRule",
    "SARIFTool",
    "SARIFRun",
    "SARIFReport",
]

