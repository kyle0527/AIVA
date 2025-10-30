"""
敏感資料掃描器
檢測響應中的敏感資訊（API Keys, Secrets, PII 等）
"""


import re
from typing import Any

from services.aiva_common.enums import Severity
from services.aiva_common.schemas import SensitiveMatch
from services.aiva_common.utils import get_logger, new_id

logger = get_logger(__name__)


class SensitiveDataScanner:
    """
    掃描和識別敏感資料的工具類
    """

    # 敏感資料模式定義
    PATTERNS = {
        "aws_access_key": r"AKIA[0-9A-Z]{16}",
        "aws_secret_key": r"(?i)aws_secret[_-]?key[\"']\s*[:=]\s*[\"']([A-Za-z0-9/+=]{40})[\"']",
        "github_token": r"ghp_[0-9a-zA-Z]{36}",
        "slack_token": r"xox[baprs]-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24,32}",
        "google_api_key": r"AIza[0-9A-Za-z\\-_]{35}",
        "stripe_key": r"sk_live_[0-9a-zA-Z]{24,}",
        "jwt_token": r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        "private_key": r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "password_in_url": r"(?i)(password|passwd|pwd)=([^&\s]+)",
        "api_key_generic": r"(?i)(api[_-]?key|apikey)[\"']\s*[:=]\s*[\"']([a-zA-Z0-9_-]{20,})[\"']",
    }

    def __init__(self):
        """初始化敏感資料掃描器"""
        self.compiled_patterns = {
            name: re.compile(pattern) for name, pattern in self.PATTERNS.items()
        }
        self.matches: list[SensitiveMatch] = []
        logger.debug("SensitiveDataScanner initialized")

    def scan_content(
        self, content: str, source_url: str, content_type: str = "html"
    ) -> list[SensitiveMatch]:
        """
        掃描內容中的敏感資料

        Args:
            content: 要掃描的內容
            source_url: 內容來源 URL
            content_type: 內容類型（html, json, javascript 等）

        Returns:
            發現的敏感資料匹配列表
        """
        matches: list[SensitiveMatch] = []

        for pattern_name, pattern_regex in self.compiled_patterns.items():
            found_matches = pattern_regex.finditer(content)
            for match in found_matches:
                sensitive_match = SensitiveMatch(
                    match_id=new_id("sens"),
                    pattern_name=pattern_name,
                    matched_text=self._mask_sensitive_value(match.group(0)),
                    context=self._get_context(content, match.start(), match.end()),
                    confidence=0.9,
                    url=source_url,
                    severity=self._determine_severity(pattern_name),
                )
                matches.append(sensitive_match)
                self.matches.append(sensitive_match)

                logger.warning(
                    f"Sensitive data found: {pattern_name} at {source_url} "
                    f"(position {match.start()}-{match.end()})"
                )

        if matches:
            logger.info(f"Found {len(matches)} sensitive matches in {source_url}")
        return matches

    def scan_headers(
        self, headers: dict[str, str], source_url: str
    ) -> list[SensitiveMatch]:
        """
        掃描 HTTP 標頭中的敏感資料

        Args:
            headers: HTTP 標頭字典
            source_url: 來源 URL

        Returns:
            發現的敏感資料匹配列表
        """
        matches: list[SensitiveMatch] = []

        # 檢查可能暴露敏感資訊的標頭
        sensitive_headers = ["authorization", "x-api-key", "x-auth-token", "cookie"]

        for header_name, header_value in headers.items():
            if header_name.lower() in sensitive_headers:
                sensitive_match = SensitiveMatch(
                    match_id=new_id("sens"),
                    pattern_name=f"header_{header_name.lower()}",
                    matched_text=self._mask_sensitive_value(header_value),
                    context=f"HTTP Header: {header_name}",
                    confidence=0.8,
                    url=source_url,
                    severity=self._determine_severity(f"header_{header_name.lower()}"),
                )
                matches.append(sensitive_match)
                self.matches.append(sensitive_match)

                logger.warning(
                    f"Sensitive header found: {header_name} at {source_url}"
                )

        return matches

    def _mask_sensitive_value(self, value: str) -> str:
        """
        遮罩敏感值以避免在日誌中暴露

        Args:
            value: 原始敏感值

        Returns:
            遮罩後的值
        """
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    def _get_context(self, content: str, start: int, end: int, context_size: int = 50) -> str:
        """
        獲取匹配的上下文

        Args:
            content: 完整內容
            start: 匹配開始位置
            end: 匹配結束位置
            context_size: 上下文大小

        Returns:
            上下文字符串
        """
        context_start = max(0, start - context_size)
        context_end = min(len(content), end + context_size)
        context = content[context_start:context_end]

        # 遮罩敏感部分
        match_length = end - start
        match_in_context_start = start - context_start
        match_in_context_end = match_in_context_start + match_length

        masked_context = (
            context[:match_in_context_start]
            + "***REDACTED***"
            + context[match_in_context_end:]
        )

        return masked_context.replace("\n", " ").strip()

    def _determine_severity(self, pattern_type: str) -> Severity:
        """
        根據模式類型判斷嚴重程度

        Args:
            pattern_type: 模式類型

        Returns:
            嚴重程度
        """
        critical_patterns = ["aws_secret_key", "private_key", "stripe_key"]
        high_patterns = ["aws_access_key", "github_token", "google_api_key", "jwt_token", "header_authorization"]
        medium_patterns = ["api_key_generic", "password_in_url", "slack_token"]

        if pattern_type in critical_patterns:
            return Severity.CRITICAL
        elif pattern_type in high_patterns:
            return Severity.HIGH
        elif pattern_type in medium_patterns:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _get_severity(self, pattern_type: str) -> str:
        """
        根據模式類型判斷嚴重程度

        Args:
            pattern_type: 模式類型

        Returns:
            嚴重程度（critical, high, medium, low）
        """
        critical_patterns = ["aws_secret_key", "private_key", "stripe_key"]
        high_patterns = ["aws_access_key", "github_token", "google_api_key", "jwt_token"]
        medium_patterns = ["api_key_generic", "password_in_url", "slack_token"]

        if pattern_type in critical_patterns:
            return "critical"
        elif pattern_type in high_patterns:
            return "high"
        elif pattern_type in medium_patterns:
            return "medium"
        else:
            return "low"

    def get_all_matches(self) -> list[SensitiveMatch]:
        """
        獲取所有發現的敏感資料匹配

        Returns:
            所有匹配列表
        """
        return self.matches.copy()

    def get_statistics(self) -> dict[str, Any]:
        """
        獲取統計信息

        Returns:
            統計數據
        """
        stats: dict[str, Any] = {
            "total_matches": len(self.matches),
            "by_type": {},
            "by_severity": {},
        }

        for match in self.matches:
            # 按類型統計
            stats["by_type"][match.pattern_name] = (
                stats["by_type"].get(match.pattern_name, 0) + 1
            )
            # 按嚴重程度統計
            stats["by_severity"][match.severity] = (
                stats["by_severity"].get(match.severity, 0) + 1
            )

        return stats

    def clear_matches(self) -> None:
        """清空所有匹配記錄"""
        self.matches.clear()
        logger.debug("Cleared all sensitive data matches")
