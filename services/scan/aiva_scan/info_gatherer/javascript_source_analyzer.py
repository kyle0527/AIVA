from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from services.aiva_common.enums import Severity
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class SinkType(Enum):
    """JavaScript sink 類型"""

    # DOM-based XSS sinks
    INNER_HTML = "innerHTML"
    OUTER_HTML = "outerHTML"
    DOCUMENT_WRITE = "document.write"
    EVAL = "eval"
    SET_TIMEOUT = "setTimeout"
    SET_INTERVAL = "setInterval"
    FUNCTION_CONSTRUCTOR = "Function"

    # DOM manipulation
    INSERT_ADJACENT_HTML = "insertAdjacentHTML"
    CREATE_ELEMENT = "createElement"
    SET_ATTRIBUTE = "setAttribute"

    # URL/Navigation sinks
    LOCATION_ASSIGN = "location.assign"
    LOCATION_REPLACE = "location.replace"
    LOCATION_HREF = "location.href"
    WINDOW_OPEN = "window.open"

    # Data storage
    LOCAL_STORAGE = "localStorage"
    SESSION_STORAGE = "sessionStorage"
    COOKIE = "document.cookie"

    # AJAX/Fetch
    FETCH = "fetch"
    XHR_OPEN = "XMLHttpRequest.open"
    XHR_SEND = "XMLHttpRequest.send"

    # WebSocket
    WEBSOCKET = "WebSocket"

    # PostMessage
    POST_MESSAGE = "postMessage"


class PatternType(Enum):
    """可疑模式類型"""

    # 輸入源
    URL_PARAMETER = "url_parameter"
    HASH_FRAGMENT = "hash_fragment"
    REFERRER = "referrer"
    POST_MESSAGE_DATA = "postmessage_data"

    # 敏感信息
    API_KEY = "api_key"
    TOKEN = "token"
    PASSWORD = "password"
    HARDCODED_SECRET = "hardcoded_secret"

    # 危險操作
    DYNAMIC_CODE = "dynamic_code"
    UNSAFE_REDIRECT = "unsafe_redirect"
    CORS_MISCONFIGURATION = "cors_misconfiguration"

    # 加密問題
    WEAK_CRYPTO = "weak_crypto"
    INSECURE_RANDOM = "insecure_random"


@dataclass
class SinkMatch:
    """Sink 匹配結果"""

    sink_type: SinkType
    line_number: int
    code_snippet: str
    context: str
    severity: Severity
    description: str
    tainted_source: str | None = None


@dataclass
class PatternMatch:
    """模式匹配結果"""

    pattern_type: PatternType
    line_number: int
    code_snippet: str
    matched_value: str
    severity: Severity
    description: str


@dataclass
class AnalysisResult:
    """分析結果"""

    url: str
    sinks: list[SinkMatch] = field(default_factory=list)
    patterns: list[PatternMatch] = field(default_factory=list)
    total_lines: int = 0
    analysis_errors: list[str] = field(default_factory=list)

    def get_stats(self) -> dict[str, Any]:
        """獲取統計信息"""
        return {
            "total_sinks": len(self.sinks),
            "total_patterns": len(self.patterns),
            "critical_issues": len(
                [s for s in self.sinks if s.severity == Severity.HIGH]
            )
            + len([p for p in self.patterns if p.severity == Severity.HIGH]),
            "high_issues": len([s for s in self.sinks if s.severity == Severity.HIGH])
            + len([p for p in self.patterns if p.severity == Severity.HIGH]),
            "total_lines": self.total_lines,
            "sinks_by_type": self._count_by_type(self.sinks),
            "patterns_by_type": self._count_by_type(self.patterns),
        }

    def _count_by_type(
        self, items: list[SinkMatch] | list[PatternMatch]
    ) -> dict[str, int]:
        """按類型統計"""
        counts: dict[str, int] = {}
        for item in items:
            if isinstance(item, SinkMatch):
                key = item.sink_type.value
            else:
                key = item.pattern_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class JavaScriptSourceAnalyzer:
    """
    分析 JavaScript 源碼，檢測安全漏洞、危險 sink 和可疑模式。

    主要功能:
    - 檢測 DOM-based XSS sink (innerHTML, eval, etc.)
    - 識別用戶輸入源 (URL 參數, location.hash, etc.)
    - 檢測硬編碼的敏感信息 (API keys, tokens, passwords)
    - 識別不安全的加密和隨機數生成
    - 檢測 CORS 配置問題
    - 分析數據流（source -> sink）

    使用範例:
        analyzer = JavaScriptSourceAnalyzer()
        result = analyzer.analyze(js_code, url="https://example.com/app.js")

        print(f"發現 {len(result.sinks)} 個 sink")
        for sink in result.sinks:
            print(f"  - {sink.sink_type.value} at line {sink.line_number}")
    """

    def __init__(self):
        """初始化分析器"""
        self._sink_patterns = self._build_sink_patterns()
        self._source_patterns = self._build_source_patterns()
        self._security_patterns = self._build_security_patterns()

    def analyze(self, source_code: str, url: str = "") -> AnalysisResult:
        """
        分析 JavaScript 源碼

        Args:
            source_code: JavaScript 源碼
            url: 源碼的 URL（用於報告）

        Returns:
            AnalysisResult: 分析結果
        """
        result = AnalysisResult(url=url)

        if not source_code or not source_code.strip():
            logger.debug("Empty source code provided")
            return result

        lines = source_code.split("\n")
        result.total_lines = len(lines)

        # 檢測 sinks
        result.sinks = self._detect_sinks(lines)

        # 檢測可疑模式
        result.patterns = self._detect_patterns(lines)

        # 分析數據流
        self._analyze_dataflow(lines, result)

        logger.info(
            f"Analysis complete: {len(result.sinks)} sinks, "
            f"{len(result.patterns)} patterns found"
        )

        return result

    def _detect_sinks(self, lines: list[str]) -> list[SinkMatch]:
        """檢測危險 sinks"""
        sinks: list[SinkMatch] = []

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", "/*")):
                continue

            for sink_type, pattern_info in self._sink_patterns.items():
                pattern = pattern_info["pattern"]
                severity = pattern_info["severity"]
                description = pattern_info["description"]

                if re.search(pattern, line, re.IGNORECASE):
                    # 提取上下文
                    context = self._extract_context(lines, line_num, window=2)

                    sinks.append(
                        SinkMatch(
                            sink_type=sink_type,
                            line_number=line_num,
                            code_snippet=stripped,
                            context=context,
                            severity=severity,
                            description=description,
                        )
                    )

        return sinks

    def _detect_patterns(self, lines: list[str]) -> list[PatternMatch]:
        """檢測可疑模式"""
        patterns: list[PatternMatch] = []

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # 檢測輸入源
            for source_type, pattern_info in self._source_patterns.items():
                pattern = pattern_info["pattern"]
                severity = pattern_info["severity"]
                description = pattern_info["description"]

                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    patterns.append(
                        PatternMatch(
                            pattern_type=source_type,
                            line_number=line_num,
                            code_snippet=stripped,
                            matched_value=match.group(0),
                            severity=severity,
                            description=description,
                        )
                    )

            # 檢測安全問題
            for security_type, pattern_info in self._security_patterns.items():
                pattern = pattern_info["pattern"]
                severity = pattern_info["severity"]
                description = pattern_info["description"]

                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    patterns.append(
                        PatternMatch(
                            pattern_type=security_type,
                            line_number=line_num,
                            code_snippet=stripped,
                            matched_value=match.group(0),
                            severity=severity,
                            description=description,
                        )
                    )

        return patterns

    def _analyze_dataflow(self, lines: list[str], result: AnalysisResult) -> None:
        """
        分析數據流（簡化版）

        檢測用戶輸入源是否流向危險 sink
        """
        # 簡單的污點分析：檢查是否有輸入源和 sink 在附近
        source_lines = {
            p.line_number
            for p in result.patterns
            if p.pattern_type
            in [
                PatternType.URL_PARAMETER,
                PatternType.HASH_FRAGMENT,
                PatternType.REFERRER,
                PatternType.POST_MESSAGE_DATA,
            ]
        }

        for sink in result.sinks:
            # 檢查 sink 附近（前後 10 行）是否有輸入源
            nearby_sources = [
                line for line in source_lines if abs(line - sink.line_number) <= 10
            ]

            if nearby_sources:
                sink.tainted_source = f"Potential taint from line(s): {nearby_sources}"
                # 提升嚴重程度
                if sink.severity == Severity.HIGH:
                    sink.severity = Severity.HIGH

    def _extract_context(self, lines: list[str], line_num: int, window: int = 2) -> str:
        """提取代碼上下文"""
        start = max(0, line_num - window - 1)
        end = min(len(lines), line_num + window)
        context_lines = lines[start:end]
        return "\n".join(
            f"{i + start + 1}: {line}" for i, line in enumerate(context_lines)
        )

    def _build_sink_patterns(self) -> dict[SinkType, dict[str, Any]]:
        """構建 sink 檢測模式"""
        return {
            # DOM XSS sinks
            SinkType.INNER_HTML: {
                "pattern": r"\.innerHTML\s*=",
                "severity": Severity.HIGH,
                "description": "innerHTML assignment can lead to XSS",
            },
            SinkType.OUTER_HTML: {
                "pattern": r"\.outerHTML\s*=",
                "severity": Severity.HIGH,
                "description": "outerHTML assignment can lead to XSS",
            },
            SinkType.DOCUMENT_WRITE: {
                "pattern": r"document\.write(ln)?\s*\(",
                "severity": Severity.HIGH,
                "description": "document.write can lead to XSS",
            },
            SinkType.EVAL: {
                "pattern": r"\beval\s*\(",
                "severity": Severity.HIGH,
                "description": "eval() executes arbitrary code",
            },
            SinkType.SET_TIMEOUT: {
                "pattern": r"setTimeout\s*\(\s*[\"']",
                "severity": Severity.HIGH,
                "description": "setTimeout with string argument acts like eval",
            },
            SinkType.SET_INTERVAL: {
                "pattern": r"setInterval\s*\(\s*[\"']",
                "severity": Severity.HIGH,
                "description": "setInterval with string argument acts like eval",
            },
            SinkType.FUNCTION_CONSTRUCTOR: {
                "pattern": r"new\s+Function\s*\(",
                "severity": Severity.HIGH,
                "description": "Function constructor can execute arbitrary code",
            },
            # DOM manipulation
            SinkType.INSERT_ADJACENT_HTML: {
                "pattern": r"\.insertAdjacentHTML\s*\(",
                "severity": Severity.HIGH,
                "description": "insertAdjacentHTML can lead to XSS",
            },
            SinkType.SET_ATTRIBUTE: {
                "pattern": r"\.setAttribute\s*\(\s*[\"']on",
                "severity": Severity.MEDIUM,
                "description": "Setting event handlers via setAttribute",
            },
            # URL/Navigation
            SinkType.LOCATION_HREF: {
                "pattern": r"location\.href\s*=",
                "severity": Severity.MEDIUM,
                "description": "location.href can cause open redirect",
            },
            SinkType.LOCATION_ASSIGN: {
                "pattern": r"location\.assign\s*\(",
                "severity": Severity.MEDIUM,
                "description": "location.assign can cause open redirect",
            },
            SinkType.WINDOW_OPEN: {
                "pattern": r"window\.open\s*\(",
                "severity": Severity.MEDIUM,
                "description": "window.open can be abused for phishing",
            },
            # Data storage
            SinkType.LOCAL_STORAGE: {
                "pattern": r"localStorage\.(setItem|set)\s*\(",
                "severity": Severity.LOW,
                "description": "localStorage can persist sensitive data",
            },
            SinkType.COOKIE: {
                "pattern": r"document\.cookie\s*=",
                "severity": Severity.MEDIUM,
                "description": "Cookie manipulation detected",
            },
            # AJAX
            SinkType.FETCH: {
                "pattern": r"\bfetch\s*\(",
                "severity": Severity.LOW,
                "description": "Fetch API call detected",
            },
            SinkType.XHR_OPEN: {
                "pattern": r"\.open\s*\(\s*[\"'](GET|POST|PUT|DELETE)",
                "severity": Severity.LOW,
                "description": "XMLHttpRequest detected",
            },
            # WebSocket
            SinkType.WEBSOCKET: {
                "pattern": r"new\s+WebSocket\s*\(",
                "severity": Severity.LOW,
                "description": "WebSocket connection detected",
            },
            # PostMessage
            SinkType.POST_MESSAGE: {
                "pattern": r"\.postMessage\s*\(",
                "severity": Severity.MEDIUM,
                "description": "postMessage can leak sensitive data",
            },
        }

    def _build_source_patterns(self) -> dict[PatternType, dict[str, Any]]:
        """構建輸入源檢測模式"""
        return {
            PatternType.URL_PARAMETER: {
                "pattern": (
                    r"(location\.search|URLSearchParams|"
                    r"window\.location\.search)"
                ),
                "severity": Severity.INFORMATIONAL,
                "description": "User input from URL parameters",
            },
            PatternType.HASH_FRAGMENT: {
                "pattern": (r"(location\.hash|window\.location\.hash)"),
                "severity": Severity.INFORMATIONAL,
                "description": "User input from URL hash fragment",
            },
            PatternType.REFERRER: {
                "pattern": r"document\.referrer",
                "severity": Severity.INFORMATIONAL,
                "description": "User input from referrer",
            },
            PatternType.POST_MESSAGE_DATA: {
                "pattern": r"addEventListener\s*\(\s*[\"']message[\"']",
                "severity": Severity.INFORMATIONAL,
                "description": "Receives postMessage data",
            },
        }

    def _build_security_patterns(self) -> dict[PatternType, dict[str, Any]]:
        """構建安全問題檢測模式"""
        return {
            # 敏感信息
            PatternType.API_KEY: {
                "pattern": (
                    r"(api[_-]?key|apikey)\s*[=:]\s*"
                    r"[\"'][a-zA-Z0-9]{20,}[\"']"
                ),
                "severity": Severity.HIGH,
                "description": "Hardcoded API key detected",
            },
            PatternType.TOKEN: {
                "pattern": (
                    r"(access[_-]?token|bearer[_-]?token)\s*[=:]\s*"
                    r"[\"'][a-zA-Z0-9._-]{20,}[\"']"
                ),
                "severity": Severity.HIGH,
                "description": "Hardcoded token detected",
            },
            PatternType.PASSWORD: {
                "pattern": r"(password|passwd|pwd)\s*[=:]\s*[\"'][^\"']+[\"']",
                "severity": Severity.HIGH,
                "description": "Hardcoded password detected",
            },
            # 弱加密
            PatternType.WEAK_CRYPTO: {
                "pattern": r"(MD5|SHA1|DES|RC4)\s*\(",
                "severity": Severity.HIGH,
                "description": "Weak cryptographic algorithm detected",
            },
            PatternType.INSECURE_RANDOM: {
                "pattern": r"Math\.random\s*\(\s*\)",
                "severity": Severity.MEDIUM,
                "description": "Math.random is not cryptographically secure",
            },
            # CORS
            PatternType.CORS_MISCONFIGURATION: {
                "pattern": r"Access-Control-Allow-Origin[\"']?\s*:\s*[\"']\*[\"']",
                "severity": Severity.HIGH,
                "description": "CORS wildcard allows any origin",
            },
        }

    def get_high_risk_issues(
        self, result: AnalysisResult
    ) -> list[SinkMatch | PatternMatch]:
        """獲取高風險問題"""
        high_risk: list[SinkMatch | PatternMatch] = []

        for sink in result.sinks:
            if sink.severity in [Severity.HIGH, Severity.HIGH]:
                high_risk.append(sink)

        for pattern in result.patterns:
            if pattern.severity in [Severity.HIGH, Severity.HIGH]:
                high_risk.append(pattern)

        return high_risk

    def format_report(self, result: AnalysisResult) -> str:
        """格式化報告"""
        lines = []
        lines.append("JavaScript Source Analysis Report")
        lines.append(f"URL: {result.url}")
        lines.append("=" * 60)

        stats = result.get_stats()
        lines.append("\nStatistics:")
        lines.append(f"  Total Lines: {stats['total_lines']}")
        lines.append(f"  Total Sinks: {stats['total_sinks']}")
        lines.append(f"  Total Patterns: {stats['total_patterns']}")
        lines.append(f"  Critical Issues: {stats['critical_issues']}")
        lines.append(f"  High Issues: {stats['high_issues']}")

        if result.sinks:
            lines.append("\n" + "Sinks Found".center(60, "─"))
            for sink in result.sinks:
                lines.append(
                    f"\n[{sink.severity.value.upper()}] {sink.sink_type.value}"
                )
                lines.append(f"  Line {sink.line_number}: {sink.code_snippet[:80]}")
                lines.append(f"  {sink.description}")
                if sink.tainted_source:
                    lines.append(f"  ⚠️  {sink.tainted_source}")

        if result.patterns:
            lines.append("\n" + "Patterns Found".center(60, "─"))
            for pattern in result.patterns:
                if pattern.severity in [Severity.HIGH, Severity.HIGH]:
                    lines.append(
                        f"\n[{pattern.severity.value.upper()}] "
                        f"{pattern.pattern_type.value}"
                    )
                    lines.append(
                        f"  Line {pattern.line_number}: {pattern.code_snippet[:80]}"
                    )
                    lines.append(f"  {pattern.description}")

        return "\n".join(lines)
