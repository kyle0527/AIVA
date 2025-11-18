"""
JavaScript 代碼分析器
靜態分析 JavaScript 代碼以識別潛在的安全問題
"""


import re
from typing import Any

from services.aiva_common.schemas import JavaScriptAnalysisResult
from services.aiva_common.utils import get_logger, new_id

logger = get_logger(__name__)


class JavaScriptAnalyzer:
    """
    分析 JavaScript 代碼的安全性
    """

    def __init__(self):
        """初始化 JavaScript 分析器"""
        self.analysis_results: list[JavaScriptAnalysisResult] = []
        logger.debug("JavaScriptAnalyzer initialized")

    async def analyze_javascript(
        self, js_content: str, file_url: str
    ) -> JavaScriptAnalysisResult:
        """
        分析 JavaScript 代碼

        Args:
            js_content: JavaScript 代碼內容
            file_url: JS 文件 URL

        Returns:
            分析結果
        """
        result = JavaScriptAnalysisResult(
            analysis_id=new_id("jsanalysis"),
            url=file_url,
            source_size_bytes=len(js_content.encode('utf-8')),
            dangerous_functions=[],
            external_resources=[],
            data_leaks=[],
        )

        # 檢測危險函數
        result.dangerous_functions = self._detect_dangerous_functions(js_content)

        # 檢測外部資源
        result.external_resources = self._extract_external_resources(js_content)

        # 檢測數據洩漏
        result.data_leaks = self._detect_data_leaks(js_content)

        # 計算安全分數
        result.security_score = self._calculate_security_score(result)

        self.analysis_results.append(result)
        logger.info(
            f"JS analysis completed for {file_url}: "
            f"score={result.security_score}, "
            f"dangerous_functions={len(result.dangerous_functions)}"
        )

        return result

    def _detect_dangerous_functions(self, js_content: str) -> list[str]:
        """
        檢測危險的 JavaScript 函數調用

        Args:
            js_content: JavaScript 代碼

        Returns:
            檢測到的危險函數列表
        """
        dangerous_patterns = {
            "eval": r"\beval\s*\(",
            "Function_constructor": r"\bnew\s+Function\s*\(",
            "setTimeout_string": r"\bsetTimeout\s*\(\s*['\"]",
            "setInterval_string": r"\bsetInterval\s*\(\s*['\"]",
            "innerHTML": r"\.innerHTML\s*=",
            "document.write": r"\bdocument\.write\s*\(",
            "document.writeln": r"\bdocument\.writeln\s*\(",
            "dangerouslySetInnerHTML": r"\bdangerouslySetInnerHTML\s*:",
        }

        detected: list[str] = []
        for func_name, pattern in dangerous_patterns.items():
            if re.search(pattern, js_content, re.IGNORECASE):
                detected.append(func_name)
                logger.warning(f"Dangerous function detected: {func_name}")

        return detected

    def _extract_external_resources(self, js_content: str) -> list[str]:
        """
        提取 JavaScript 中引用的外部資源

        Args:
            js_content: JavaScript 代碼

        Returns:
            外部資源 URL 列表
        """
        external_urls: list[str] = []

        # 查找 HTTP/HTTPS URL
        url_pattern = r'https?://[^\s\'"<>)]+[^\s\'"<>.,;:!?)]'
        matches = re.finditer(url_pattern, js_content, re.IGNORECASE)

        for match in matches:
            url = match.group(0)
            if url not in external_urls:
                external_urls.append(url)

        logger.debug(f"Found {len(external_urls)} external resources")
        return external_urls

    def _detect_data_leaks(self, js_content: str) -> list[dict[str, str]]:
        """
        檢測可能的數據洩漏

        Args:
            js_content: JavaScript 代碼

        Returns:
            數據洩漏信息列表
        """
        leaks: list[dict[str, str]] = []

        # 檢測 console.log 洩漏
        console_pattern = r'console\.(log|debug|info|warn|error)\s*\('
        console_matches = re.finditer(console_pattern, js_content, re.IGNORECASE)
        if any(console_matches):
            leaks.append({
                "type": "console_logging",
                "severity": "low",
                "description": "Console logging may expose sensitive data in production"
            })

        # 檢測 localStorage/sessionStorage 使用
        storage_pattern = r'(localStorage|sessionStorage)\.(setItem|getItem)'
        storage_matches = re.finditer(storage_pattern, js_content, re.IGNORECASE)
        if any(storage_matches):
            leaks.append({
                "type": "browser_storage",
                "severity": "medium",
                "description": "Browser storage may contain sensitive data"
            })

        # 檢測直接在 URL 中傳遞敏感資訊
        url_param_pattern = r'[?&](password|passwd|pwd|token|api_?key|secret)='
        url_param_matches = re.finditer(url_param_pattern, js_content, re.IGNORECASE)
        if any(url_param_matches):
            leaks.append({
                "type": "sensitive_url_params",
                "severity": "high",
                "description": "Sensitive data in URL parameters"
            })

        # 檢測硬編碼的認證資訊
        hardcoded_pattern = r'(password|apiKey|secretKey|token)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]'
        hardcoded_matches = re.finditer(hardcoded_pattern, js_content, re.IGNORECASE)
        if any(hardcoded_matches):
            leaks.append({
                "type": "hardcoded_credentials",
                "severity": "critical",
                "description": "Hardcoded credentials found in JavaScript"
            })

        logger.debug(f"Detected {len(leaks)} potential data leaks")
        return leaks

    def _calculate_security_score(self, result: JavaScriptAnalysisResult) -> int:
        """
        計算安全分數（0-100）

        Args:
            result: 分析結果

        Returns:
            安全分數
        """
        score = 100

        # 危險函數扣分
        score -= len(result.dangerous_functions) * 10

        # 數據洩漏扣分
        for leak in result.data_leaks:
            severity = leak.get("severity", "low")
            if severity == "critical":
                score -= 25
            elif severity == "high":
                score -= 15
            elif severity == "medium":
                score -= 10
            else:
                score -= 5

        # 外部資源過多扣分
        if len(result.external_resources) > 10:
            score -= 5

        # 確保分數在 0-100 之間
        return max(0, min(100, score))

    def get_all_results(self) -> list[JavaScriptAnalysisResult]:
        """
        獲取所有分析結果

        Returns:
            所有分析結果列表
        """
        return self.analysis_results.copy()

    def get_statistics(self) -> dict[str, Any]:
        """
        獲取統計信息

        Returns:
            統計數據
        """
        if not self.analysis_results:
            return {
                "total_files": 0,
                "average_score": 0,
                "total_dangerous_functions": 0,
                "total_data_leaks": 0,
            }

        total_score = sum(r.security_score for r in self.analysis_results)
        total_dangerous = sum(len(r.dangerous_functions) for r in self.analysis_results)
        total_leaks = sum(len(r.data_leaks) for r in self.analysis_results)

        return {
            "total_files": len(self.analysis_results),
            "average_score": total_score / len(self.analysis_results),
            "total_dangerous_functions": total_dangerous,
            "total_data_leaks": total_leaks,
            "min_score": min(r.security_score for r in self.analysis_results),
            "max_score": max(r.security_score for r in self.analysis_results),
        }

    def clear_results(self) -> None:
        """清空所有分析結果"""
        self.analysis_results.clear()
        logger.debug("Cleared all JavaScript analysis results")
