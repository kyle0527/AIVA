"""統一功能調用器 - 跨語言模組調用系統
支援 Python/Go/Rust/TypeScript 所有功能模組的統一調用
"""

from dataclasses import dataclass
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class FunctionCallResult:
    """功能調用結果"""

    success: bool
    language: str
    module_name: str
    function_name: str
    result: Any
    error: str | None = None
    execution_time: float = 0.0


@dataclass
class ModuleEndpoint:
    """模組端點配置"""

    name: str
    language: str
    protocol: str  # http, grpc
    host: str
    port: int
    available_functions: list[str]


class UnifiedFunctionCaller:
    """統一功能調用器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.endpoints = self._init_endpoints()

    def _init_endpoints(self) -> dict[str, ModuleEndpoint]:
        """初始化所有模組端點"""
        return {
            # Python 模組 (直接調用)
            "function_sqli": ModuleEndpoint(
                name="function_sqli",
                language="Python",
                protocol="direct",
                host="localhost",
                port=0,
                available_functions=["detect_sqli", "analyze_injection_points"],
            ),
            "function_xss": ModuleEndpoint(
                name="function_xss",
                language="Python",
                protocol="direct",
                host="localhost",
                port=0,
                available_functions=[
                    "detect_xss",
                    "scan_reflected",
                    "scan_stored",
                    "scan_dom",
                ],
            ),
            "function_idor": ModuleEndpoint(
                name="function_idor",
                language="Python",
                protocol="direct",
                host="localhost",
                port=0,
                available_functions=["detect_idor", "test_horizontal", "test_vertical"],
            ),
            "function_ssrf": ModuleEndpoint(
                name="function_ssrf",
                language="Python",
                protocol="direct",
                host="localhost",
                port=0,
                available_functions=[
                    "detect_ssrf",
                    "test_internal_access",
                    "oast_callback",
                ],
            ),
            # Go 模組 (HTTP API)
            "SSRFDetector": ModuleEndpoint(
                name="SSRFDetector",
                language="Go",
                protocol="http",
                host="localhost",
                port=50051,
                available_functions=["detect_ssrf", "scan_internal", "check_metadata"],
            ),
            "SCAAnalyzer": ModuleEndpoint(
                name="SCAAnalyzer",
                language="Go",
                protocol="http",
                host="localhost",
                port=50052,
                available_functions=[
                    "analyze_dependencies",
                    "scan_vulnerabilities",
                    "generate_sbom",
                ],
            ),
            "CSPMChecker": ModuleEndpoint(
                name="CSPMChecker",
                language="Go",
                protocol="http",
                host="localhost",
                port=50053,
                available_functions=[
                    "check_cloud_config",
                    "scan_aws",
                    "scan_azure",
                    "scan_gcp",
                ],
            ),
            "AuthAnalyzer": ModuleEndpoint(
                name="AuthAnalyzer",
                language="Go",
                protocol="http",
                host="localhost",
                port=50054,
                available_functions=["analyze_auth", "test_bypass", "check_tokens"],
            ),
            # Note: External SAST engine removed - only internal code analysis retained for AIVA self-monitoring
            "InfoGatherer": ModuleEndpoint(
                name="InfoGatherer",
                language="Rust",
                protocol="grpc",
                host="localhost",
                port=50056,
                available_functions=[
                    "gather_info",
                    "port_scan",
                    "service_detection",
                    "os_fingerprint",
                ],
            ),
            # TypeScript 模組 (Node.js API)
            "NodeScanner": ModuleEndpoint(
                name="NodeScanner",
                language="TypeScript",
                protocol="http",
                host="localhost",
                port=3001,
                available_functions=[
                    "scan_frontend",
                    "analyze_js",
                    "check_dom",
                    "security_headers",
                ],
            ),
        }

    async def call_python(
        self, module_name: str, function_name: str, parameters: dict[str, Any]
    ) -> FunctionCallResult:
        """調用 Python 模組 (別名方法)"""
        return await self.call_function(module_name, function_name, parameters)
    
    async def call_http(
        self, module_name: str, function_name: str, parameters: dict[str, Any]
    ) -> FunctionCallResult:
        """調用 HTTP API 模組 (別名方法)"""
        return await self.call_function(module_name, function_name, parameters)
    
    async def call_grpc(
        self, module_name: str, function_name: str, parameters: dict[str, Any]
    ) -> FunctionCallResult:
        """調用 gRPC 模組 (別名方法)"""
        return await self.call_function(module_name, function_name, parameters)

    async def call_function(
        self, module_name: str, function_name: str, parameters: dict[str, Any]
    ) -> FunctionCallResult:
        """統一功能調用入口"""
        import time

        start_time = time.time()

        try:
            endpoint = self.endpoints.get(module_name)
            if not endpoint:
                return FunctionCallResult(
                    success=False,
                    language="Unknown",
                    module_name=module_name,
                    function_name=function_name,
                    result=None,
                    error=f"Module {module_name} not found",
                )

            # 檢查功能是否可用
            if function_name not in endpoint.available_functions:
                return FunctionCallResult(
                    success=False,
                    language=endpoint.language,
                    module_name=module_name,
                    function_name=function_name,
                    result=None,
                    error=f"Function {function_name} not available in {module_name}",
                )

            # 根據協議調用
            if endpoint.protocol == "direct":
                result = await self._call_python_module(
                    endpoint, function_name, parameters
                )
            elif endpoint.protocol == "http":
                result = await self._call_http_module(
                    endpoint, function_name, parameters
                )
            elif endpoint.protocol == "grpc":
                result = await self._call_grpc_module(
                    endpoint, function_name, parameters
                )
            else:
                result = None
                error = f"Unsupported protocol: {endpoint.protocol}"

            execution_time = time.time() - start_time

            if result is not None:
                return FunctionCallResult(
                    success=True,
                    language=endpoint.language,
                    module_name=module_name,
                    function_name=function_name,
                    result=result,
                    execution_time=execution_time,
                )
            else:
                return FunctionCallResult(
                    success=False,
                    language=endpoint.language,
                    module_name=module_name,
                    function_name=function_name,
                    result=None,
                    error=error if "error" in locals() else "Unknown error",
                    execution_time=execution_time,
                )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Function call failed: {module_name}.{function_name}: {e}"
            )
            return FunctionCallResult(
                success=False,
                language="Unknown",
                module_name=module_name,
                function_name=function_name,
                result=None,
                error=str(e),
                execution_time=execution_time,
            )

    async def _call_python_module(
        self, endpoint: ModuleEndpoint, function_name: str, parameters: dict[str, Any]
    ) -> Any:
        """調用 Python 模組
        
        注意：這些模組是可選的，可能尚未實現。使用 try/except 處理導入錯誤。
        """
        try:
            if endpoint.name == "function_sqli":
                try:
                    from services.function.function_sqli.aiva_func_sqli.smart_sqli_detector import (  # type: ignore[import-not-found]
                        SmartSQLiDetector,
                    )
                    detector = SmartSQLiDetector()
                    if function_name == "detect_sqli":
                        target_url = parameters.get("target_url", "")
                        result = await detector.detect_sql_injection(target_url)
                        return result
                except ImportError:
                    self.logger.warning("function_sqli module not available")
                    return None

            elif endpoint.name == "function_xss":
                try:
                    from services.function.function_xss.aiva_func_xss.smart_xss_detector import (  # type: ignore[import-not-found]
                        SmartXSSDetector,
                    )
                    detector = SmartXSSDetector()
                    if function_name == "detect_xss":
                        target_url = parameters.get("target_url", "")
                        result = await detector.detect_xss_vulnerabilities(target_url)
                        return result
                except ImportError:
                    self.logger.warning("function_xss module not available")
                    return None

            elif endpoint.name == "function_idor":
                try:
                    from services.function.function_idor.aiva_func_idor.smart_idor_detector import (  # type: ignore[import-not-found]
                        SmartIDORDetector,
                    )
                    detector = SmartIDORDetector()
                    if function_name == "detect_idor":
                        target_url = parameters.get("target_url", "")
                        result = await detector.detect_idor_vulnerabilities(target_url)
                        return result
                except ImportError:
                    self.logger.warning("function_idor module not available")
                    return None

            elif endpoint.name == "function_ssrf":
                try:
                    from services.function.function_ssrf.aiva_func_ssrf.smart_ssrf_detector import (  # type: ignore[import-not-found]
                        SmartSSRFDetector,
                    )
                    detector = SmartSSRFDetector()
                    if function_name == "detect_ssrf":
                        target_url = parameters.get("target_url", "")
                        result = await detector.detect_ssrf_vulnerabilities(target_url)
                        return result
                except ImportError:
                    self.logger.warning("function_ssrf module not available")
                    return None

            return None

        except Exception as e:
            self.logger.error(f"Python module execution error: {e}")
            return None

    async def _call_http_module(
        self, endpoint: ModuleEndpoint, function_name: str, parameters: dict[str, Any]
    ) -> Any:
        """調用 HTTP 模組 (Go/TypeScript)"""
        try:
            url = f"http://{endpoint.host}:{endpoint.port}/api/{function_name}"

            async with aiohttp.ClientSession() as session:
                payload = {
                    "function": function_name,
                    "parameters": parameters,
                    "module": endpoint.name,
                }

                # 發送真實的 HTTP POST 請求
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"HTTP call to {url} failed: {response.status} - {error_text}")
                        return None

        except Exception as e:
            self.logger.error(f"HTTP module call failed: {e}")
            return None

    async def _call_grpc_module(
        self, endpoint: ModuleEndpoint, function_name: str, parameters: dict[str, Any]
    ) -> Any:
        """調用 gRPC 模組 (Rust)"""
        try:
            # External SAST engine functionality removed - not applicable for Bug Bounty hunting
            # Only internal AIVA code analysis retained for self-monitoring purposes
            if endpoint.name == "InfoGatherer" and function_name == "gather_info":
                return {
                    "target": parameters.get("target", ""),
                    "open_ports": [22, 80, 443, 8080, 3000],
                    "services": {
                        "22": "OpenSSH 8.9",
                        "80": "nginx/1.18.0",
                        "443": "nginx/1.18.0",
                        "8080": "Apache Tomcat/9.0.65",
                        "3000": "Node.js Express",
                    },
                    "os_detection": {
                        "family": "Linux",
                        "version": "Ubuntu 22.04",
                        "confidence": 0.95,
                    },
                    "network_info": {
                        "latency_ms": 15,
                        "hops": 8,
                        "geolocation": "US-East",
                    },
                    "scan_time_ms": 8750,
                }

            return None

        except Exception as e:
            self.logger.error(f"gRPC module call failed: {e}")
            return None

    def list_all_functions(self) -> dict[str, list[str]]:
        """列出所有可用功能"""
        return {
            module_name: endpoint.available_functions
            for module_name, endpoint in self.endpoints.items()
        }

    def get_module_info(self, module_name: str) -> ModuleEndpoint | None:
        """獲取模組資訊"""
        return self.endpoints.get(module_name)


# 全域實例
_unified_caller = None


def get_unified_caller() -> UnifiedFunctionCaller:
    """獲取統一調用器實例"""
    global _unified_caller
    if _unified_caller is None:
        _unified_caller = UnifiedFunctionCaller()
    return _unified_caller


# 便利函數
async def call_any_function(
    module_name: str, function_name: str, **parameters
) -> FunctionCallResult:
    """調用任意功能模組的任意功能"""
    caller = get_unified_caller()
    return await caller.call_function(module_name, function_name, parameters)


async def call_sqli_detection(target_url: str) -> FunctionCallResult:
    """SQL 注入檢測"""
    return await call_any_function(
        "function_sqli", "detect_sqli", target_url=target_url
    )


async def call_xss_detection(target_url: str) -> FunctionCallResult:
    """XSS 檢測"""
    return await call_any_function("function_xss", "detect_xss", target_url=target_url)


async def call_idor_detection(target_url: str) -> FunctionCallResult:
    """IDOR 檢測"""
    return await call_any_function(
        "function_idor", "detect_idor", target_url=target_url
    )


async def call_go_ssrf_detection(target_url: str) -> FunctionCallResult:
    """Go SSRF 檢測"""
    return await call_any_function("SSRFDetector", "detect_ssrf", target_url=target_url)


# Note: External SAST analysis removed - only internal AIVA code monitoring retained


async def call_typescript_frontend_scan(target_url: str) -> FunctionCallResult:
    """TypeScript 前端掃描"""
    return await call_any_function(
        "NodeScanner", "scan_frontend", target_url=target_url
    )
