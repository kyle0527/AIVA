

from dataclasses import dataclass, field
import re
from typing import Any

from services.aiva_common.enums import Location, SensitiveInfoType, Severity
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SensitiveMatch:
    """敏感信息匹配結果"""

    info_type: SensitiveInfoType
    value: str
    location: Location
    context: str
    line_number: int | None = None
    severity: Severity = Severity.MEDIUM
    description: str = ""
    recommendation: str = ""

    def __post_init__(self):
        """自動設置描述和建議"""
        if not self.description:
            self.description = self._get_default_description()
        if not self.recommendation:
            self.recommendation = self._get_default_recommendation()

    def _get_default_description(self) -> str:
        """獲取默認描述"""
        descriptions = {
            SensitiveInfoType.API_KEY: "API key exposed in response",
            SensitiveInfoType.ACCESS_TOKEN: "Access token exposed in response",
            SensitiveInfoType.PASSWORD: "Password exposed in response",
            SensitiveInfoType.EMAIL: "Email address exposed",
            SensitiveInfoType.CREDIT_CARD: "Credit card number exposed",
            SensitiveInfoType.DATABASE_CONNECTION: "Database connection string exposed",
            SensitiveInfoType.INTERNAL_IP: "Internal IP address exposed",
            SensitiveInfoType.AWS_KEY: "AWS credentials exposed",
            SensitiveInfoType.FILE_PATH: "Internal file path exposed",
            SensitiveInfoType.STACK_TRACE: "Stack trace exposed",
            SensitiveInfoType.DEBUG_INFO: "Debug information exposed",
        }
        return descriptions.get(self.info_type, f"{self.info_type.value} exposed")

    def _get_default_recommendation(self) -> str:
        """獲取默認建議"""
        if self.info_type in [
            SensitiveInfoType.API_KEY,
            SensitiveInfoType.ACCESS_TOKEN,
            SensitiveInfoType.SECRET_KEY,
            SensitiveInfoType.PASSWORD,
        ]:
            return (
                "Remove credentials from client-side code. "
                "Use secure server-side authentication."
            )
        elif self.info_type in [
            SensitiveInfoType.EMAIL,
            SensitiveInfoType.PHONE,
            SensitiveInfoType.CREDIT_CARD,
        ]:
            return "Mask or remove personal information from responses."
        elif self.info_type in [
            SensitiveInfoType.STACK_TRACE,
            SensitiveInfoType.DEBUG_INFO,
            SensitiveInfoType.ERROR_MESSAGE,
        ]:
            return "Disable debug mode in production. Use generic error messages."
        elif self.info_type == SensitiveInfoType.FILE_PATH:
            return "Remove internal file paths from responses."
        else:
            return "Review and remove sensitive information from public responses."


@dataclass
class DetectionResult:
    """檢測結果"""

    url: str
    matches: list[SensitiveMatch] = field(default_factory=list)
    total_checks: int = 0
    scan_errors: list[str] = field(default_factory=list)

    def get_stats(self) -> dict[str, Any]:
        """獲取統計信息"""
        return {
            "total_matches": len(self.matches),
            "critical_issues": len(
                [m for m in self.matches if m.severity == Severity.HIGH]
            ),
            "high_issues": len(
                [m for m in self.matches if m.severity == Severity.HIGH]
            ),
            "matches_by_type": self._count_by_type(),
            "matches_by_location": self._count_by_location(),
        }

    def _count_by_type(self) -> dict[str, int]:
        """按類型統計"""
        counts: dict[str, int] = {}
        for match in self.matches:
            key = match.info_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_by_location(self) -> dict[str, int]:
        """按位置統計"""
        counts: dict[str, int] = {}
        for match in self.matches:
            key = match.location.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class SensitiveInfoDetector:
    """
    檢測靜態內容中的敏感信息洩露。

    主要功能:
    - 檢測 API keys, tokens, passwords 等認證憑證
    - 檢測個人識別信息 (PII): email, phone, SSN, credit card
    - 檢測雲服務憑證 (AWS, GCP, Azure)
    - 檢測數據庫連接字符串
    - 檢測內部路徑和系統信息
    - 檢測 debug 信息和錯誤堆棧
    - 檢測 HTML 註釋中的敏感信息

    使用範例:
        detector = SensitiveInfoDetector()

        # 檢測 HTML 內容
        result = detector.detect_in_html(html_content, url="https://example.com")

        # 檢測響應頭
        result = detector.detect_in_headers(headers, url="https://example.com")

        # 檢測 JavaScript
        result = detector.detect_in_javascript(js_code, url="https://example.com/app.js")
    """

    def __init__(self, *, min_severity: Severity = Severity.INFORMATIONAL):
        """
        初始化檢測器

        Args:
            min_severity: 最低嚴重程度,低於此級別的匹配將被過濾
        """
        self.min_severity = min_severity
        self._patterns = self._build_patterns()

    def detect_in_html(self, html_content: str, url: str = "") -> DetectionResult:
        """
        檢測 HTML 內容中的敏感信息

        Args:
            html_content: HTML 內容
            url: 頁面 URL

        Returns:
            DetectionResult: 檢測結果
        """
        result = DetectionResult(url=url)

        if not html_content:
            return result

        # 檢測 HTML 註釋
        result.matches.extend(self._detect_html_comments(html_content))

        # 檢測 JavaScript 代碼塊
        result.matches.extend(self._detect_script_blocks(html_content))

        # 檢測 meta 標籤
        result.matches.extend(self._detect_meta_tags(html_content))

        # 檢測 HTML body 中的敏感信息
        result.matches.extend(self._detect_in_text(html_content, Location.HTML_BODY))

        # 過濾低嚴重程度的匹配
        result.matches = self._filter_by_severity(result.matches)

        logger.info(f"HTML detection complete: {len(result.matches)} matches found")
        return result

    def detect_in_headers(
        self, headers: dict[str, str], url: str = ""
    ) -> DetectionResult:
        """
        檢測 HTTP 響應頭中的敏感信息

        Args:
            headers: HTTP 響應頭
            url: 請求 URL

        Returns:
            DetectionResult: 檢測結果
        """
        result = DetectionResult(url=url)

        if not headers:
            return result

        for header_name, header_value in headers.items():
            # 檢測 Set-Cookie 中的敏感 token
            if header_name.lower() == "set-cookie" and re.search(
                r"(token|auth|session|jwt)=([a-zA-Z0-9._-]{20,})",
                header_value,
                re.IGNORECASE,
            ):
                result.matches.append(
                    SensitiveMatch(
                        info_type=SensitiveInfoType.AUTH_COOKIE,
                        value=header_value[:100],
                        location=Location.COOKIE,
                        context=f"{header_name}: {header_value[:200]}",
                        severity=Severity.MEDIUM,
                    )
                )

            # 檢測其他敏感信息
            combined = f"{header_name}: {header_value}"
            for match in self._detect_in_text(combined, Location.RESPONSE_HEADER):
                result.matches.append(match)

        result.matches = self._filter_by_severity(result.matches)
        logger.info(f"Headers detection complete: {len(result.matches)} matches found")
        return result

    def detect_in_javascript(self, js_code: str, url: str = "") -> DetectionResult:
        """
        檢測 JavaScript 代碼中的敏感信息

        Args:
            js_code: JavaScript 代碼
            url: 腳本 URL

        Returns:
            DetectionResult: 檢測結果
        """
        result = DetectionResult(url=url)

        if not js_code:
            return result

        result.matches = self._detect_in_text(js_code, Location.JAVASCRIPT)
        result.matches = self._filter_by_severity(result.matches)

        logger.info(
            f"JavaScript detection complete: {len(result.matches)} matches found"
        )
        return result

    def detect_in_response(
        self, response_body: str, headers: dict[str, str] | None = None, url: str = ""
    ) -> DetectionResult:
        """
        檢測完整的 HTTP 響應

        Args:
            response_body: 響應體
            headers: 響應頭（可選）
            url: 請求 URL

        Returns:
            DetectionResult: 檢測結果
        """
        result = DetectionResult(url=url)

        # 檢測響應體
        if response_body:
            # 嘗試作為 HTML 檢測
            if "<html" in response_body.lower() or "<body" in response_body.lower():
                html_result = self.detect_in_html(response_body, url)
                result.matches.extend(html_result.matches)
            else:
                # 作為純文本檢測
                body_matches = self._detect_in_text(
                    response_body, Location.RESPONSE_BODY
                )
                result.matches.extend(body_matches)

        # 檢測響應頭
        if headers:
            header_result = self.detect_in_headers(headers, url)
            result.matches.extend(header_result.matches)

        result.matches = self._filter_by_severity(result.matches)
        logger.info(f"Response detection complete: {len(result.matches)} matches found")
        return result

    def _detect_html_comments(self, html: str) -> list[SensitiveMatch]:
        """檢測 HTML 註釋中的敏感信息"""
        matches: list[SensitiveMatch] = []

        # 提取所有 HTML 註釋
        comment_pattern = r"<!--(.*?)-->"
        for comment_match in re.finditer(comment_pattern, html, re.DOTALL):
            comment_text = comment_match.group(1)

            # 檢查註釋是否包含敏感信息
            for match in self._detect_in_text(comment_text, Location.HTML_COMMENT):
                # 更新嚴重程度（註釋中的洩露更嚴重）
                if match.severity == Severity.LOW:
                    match.severity = Severity.MEDIUM
                elif match.severity == Severity.MEDIUM:
                    match.severity = Severity.HIGH

                matches.append(match)

        return matches

    def _detect_script_blocks(self, html: str) -> list[SensitiveMatch]:
        """檢測 <script> 標籤中的敏感信息"""
        matches: list[SensitiveMatch] = []

        # 提取所有 script 標籤
        script_pattern = r"<script[^>]*>(.*?)</script>"
        for script_match in re.finditer(
            script_pattern, html, re.DOTALL | re.IGNORECASE
        ):
            script_text = script_match.group(1)

            for match in self._detect_in_text(script_text, Location.JAVASCRIPT):
                matches.append(match)

        return matches

    def _detect_meta_tags(self, html: str) -> list[SensitiveMatch]:
        """檢測 meta 標籤中的敏感信息"""
        matches: list[SensitiveMatch] = []

        # 提取 meta 標籤
        meta_pattern = r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*>'
        for meta_match in re.finditer(meta_pattern, html, re.IGNORECASE):
            content = meta_match.group(1)

            for match in self._detect_in_text(content, Location.META_TAG):
                matches.append(match)

        return matches

    def _detect_in_text(self, text: str, location: Location) -> list[SensitiveMatch]:
        """在文本中檢測敏感信息"""
        matches: list[SensitiveMatch] = []

        lines = text.split("\n")

        for line_num, line in enumerate(lines, start=1):
            for info_type, pattern_info in self._patterns.items():
                pattern = pattern_info["pattern"]
                severity = pattern_info["severity"]

                for regex_match in re.finditer(pattern, line, re.IGNORECASE):
                    matched_value = regex_match.group(0)

                    # 提取上下文（前後 50 個字符）
                    start = max(0, regex_match.start() - 50)
                    end = min(len(line), regex_match.end() + 50)
                    context = line[start:end]

                    matches.append(
                        SensitiveMatch(
                            info_type=info_type,
                            value=matched_value,
                            location=location,
                            context=context,
                            line_number=line_num,
                            severity=severity,
                        )
                    )

        return matches

    def _filter_by_severity(
        self, matches: list[SensitiveMatch]
    ) -> list[SensitiveMatch]:
        """根據最低嚴重程度過濾匹配"""
        severity_order = {
            Severity.INFORMATIONAL: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
        }

        min_level = severity_order[self.min_severity]
        return [m for m in matches if severity_order[m.severity] >= min_level]

    def _build_patterns(self) -> dict[SensitiveInfoType, dict[str, Any]]:
        """構建檢測模式"""
        return {
            # API Keys and Tokens
            SensitiveInfoType.API_KEY: {
                "pattern": (
                    r"(?i)(api[_-]?key|apikey)\s*[=:]\s*"
                    r"['\"]([a-zA-Z0-9_\-]{20,})['\"]"
                ),
                "severity": Severity.HIGH,
            },
            SensitiveInfoType.ACCESS_TOKEN: {
                "pattern": (
                    r"(?i)(access[_-]?token|bearer[_-]?token)\s*[=:]\s*"
                    r"['\"]([a-zA-Z0-9._\-]{20,})['\"]"
                ),
                "severity": Severity.HIGH,
            },
            SensitiveInfoType.SECRET_KEY: {
                "pattern": (
                    r"(?i)(secret[_-]?key|app[_-]?secret)\s*[=:]\s*"
                    r"['\"]([a-zA-Z0-9_\-]{20,})['\"]"
                ),
                "severity": Severity.HIGH,
            },
            SensitiveInfoType.PASSWORD: {
                "pattern": r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]([^'\"]{4,})['\"]",
                "severity": Severity.HIGH,
            },
            SensitiveInfoType.JWT_TOKEN: {
                "pattern": (
                    r"eyJ[a-zA-Z0-9_-]*\."
                    r"eyJ[a-zA-Z0-9_-]*\."
                    r"[a-zA-Z0-9_-]*"
                ),
                "severity": Severity.HIGH,
            },
            # AWS Credentials
            SensitiveInfoType.AWS_KEY: {
                "pattern": (
                    r"(?i)(AKIA[0-9A-Z]{16}|"
                    r"aws[_-]?(access[_-]?key|secret))"
                ),
                "severity": Severity.HIGH,
            },
            # GitHub Token
            SensitiveInfoType.GITHUB_TOKEN: {
                "pattern": (r"(?i)(ghp_[a-zA-Z0-9]{36}|github[_-]?token)"),
                "severity": Severity.HIGH,
            },
            # Email
            SensitiveInfoType.EMAIL: {
                "pattern": (
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\."
                    r"[A-Z|a-z]{2,}\b"
                ),
                "severity": Severity.LOW,
            },
            # Phone
            SensitiveInfoType.PHONE: {
                "pattern": (
                    r"(?i)(phone|tel|mobile)\s*[=:]\s*"
                    r"['\"]?[\d\s\-\+\(\)]{10,}['\"]?"
                ),
                "severity": Severity.MEDIUM,
            },
            # Credit Card (簡化版)
            SensitiveInfoType.CREDIT_CARD: {
                "pattern": (
                    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"
                    r"5[1-5][0-9]{14}|3[47][0-9]{13})\b"
                ),
                "severity": Severity.HIGH,
            },
            # Database Connection
            SensitiveInfoType.DATABASE_CONNECTION: {
                "pattern": (
                    r"(?i)(mysql|postgresql|mongodb|redis)://"
                    r"[^\s<>\"']+"
                ),
                "severity": Severity.HIGH,
            },
            # Internal IP
            SensitiveInfoType.INTERNAL_IP: {
                "pattern": (
                    r"\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|"
                    r"192\.168\.)\d{1,3}\.\d{1,3}\b"
                ),
                "severity": Severity.MEDIUM,
            },
            # File Path
            SensitiveInfoType.FILE_PATH: {
                "pattern": (
                    r"(?i)([a-z]:\\|/var/|/home/|/etc/|/usr/)"
                    r"[\w/\\.-]+"
                ),
                "severity": Severity.LOW,
            },
            # Stack Trace
            SensitiveInfoType.STACK_TRACE: {
                "pattern": r"(?i)(traceback|stack trace|at\s+[\w.]+\([^\)]+:\d+:\d+\))",
                "severity": Severity.MEDIUM,
            },
            # Debug Info
            SensitiveInfoType.DEBUG_INFO: {
                "pattern": r"(?i)(debug|trace|verbose)\s*[=:]\s*(true|1|on|enabled)",
                "severity": Severity.MEDIUM,
            },
            # Private Key
            SensitiveInfoType.PRIVATE_KEY: {
                "pattern": r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
                "severity": Severity.HIGH,
            },
        }

    def format_report(self, result: DetectionResult) -> str:
        """格式化檢測報告"""
        lines = []
        lines.append("Sensitive Information Detection Report")
        lines.append(f"URL: {result.url}")
        lines.append("=" * 70)

        stats = result.get_stats()
        lines.append("\nStatistics:")
        lines.append(f"  Total Matches: {stats['total_matches']}")
        lines.append(f"  Critical Issues: {stats['critical_issues']}")
        lines.append(f"  High Issues: {stats['high_issues']}")

        if stats["matches_by_type"]:
            lines.append("\n  Matches by Type:")
            for info_type, count in stats["matches_by_type"].items():
                lines.append(f"    - {info_type}: {count}")

        if result.matches:
            lines.append("\n" + "Findings".center(70, "─"))

            # 按嚴重程度排序
            sorted_matches = sorted(
                result.matches,
                key=lambda m: (
                    {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}[
                        m.severity.value
                    ]
                ),
            )

            for match in sorted_matches:
                lines.append(
                    f"\n[{match.severity.value.upper()}] {match.info_type.value}"
                )
                lines.append(f"  Location: {match.location.value}")
                if match.line_number:
                    lines.append(f"  Line: {match.line_number}")
                lines.append(f"  Value: {match.value[:80]}")
                lines.append(f"  Context: {match.context[:100]}")
                lines.append(f"  Description: {match.description}")
                lines.append(f"  Recommendation: {match.recommendation}")

        return "\n".join(lines)

    def get_critical_issues(self, result: DetectionResult) -> list[SensitiveMatch]:
        """獲取關鍵問題"""
        return [m for m in result.matches if m.severity == Severity.HIGH]

    def get_high_risk_issues(self, result: DetectionResult) -> list[SensitiveMatch]:
        """獲取高風險問題"""
        return [
            m for m in result.matches if m.severity in [Severity.HIGH, Severity.HIGH]
        ]
