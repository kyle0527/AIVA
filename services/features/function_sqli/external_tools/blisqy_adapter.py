"""
Blisqy SQLi Adapter - function_sqli 外部工具橋接層

將 Blisqy (HTTP Header 時序盲注工具) 橋接到 AIVA function_sqli 模組。
工具本體位置: tools/external_arsenal/Blisqy/

Blisqy 官方文檔: https://github.com/JohnTroony/Blisqy
測試目標：HTTP Header 注入（User-Agent, X-Forwarded-For, Referer 等）

使用方式:
    adapter = BlisqyAdapter()
    result = await adapter.run(
        target_url="https://example.com",
        inject_header="User-Agent",
        cookies={"session": "abc"}
    )
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from aiva_common.schemas.findings import ExternalToolFinding

logger = logging.getLogger(__name__)

# 工具本體路徑
_TOOL_DIR = Path(__file__).resolve().parents[5] / "tools" / "external_arsenal" / "Blisqy"
_FIND_SCRIPT = _TOOL_DIR / "FindBlindSpot.py"
_EXPLOIT_SCRIPT = _TOOL_DIR / "ExploitBlindSpot.py"

# 已知的 Blisqy 輸出關鍵字 (用於判斷是否找到注入點)
_VULN_KEYWORDS = [
    "blind spot found",
    "vulnerable",
    "injection point",
    "time-based",
    "blind sqli",
]


class BlisqyAdapter:
    """Blisqy HTTP Header 時序盲注適配器

    Blisqy 專注於 HTTP Header 注入，補強 function_sqli
    現有引擎不擅長處理的「Header SQLi」場景。

    目標 Header (預設):
        - User-Agent
        - X-Forwarded-For
        - Referer
        - X-Real-IP
    """

    TOOL_NAME = "blisqy"
    DEFAULT_TIMEOUT = 180  # 3 分鐘（時序注入需要時間）
    DEFAULT_INJECT_HEADERS = ["User-Agent", "X-Forwarded-For", "Referer"]

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._python_bin = sys.executable

    def is_available(self) -> bool:
        """檢查 Blisqy 是否就緒"""
        if not _FIND_SCRIPT.exists():
            logger.warning(f"Blisqy FindBlindSpot.py not found at {_FIND_SCRIPT}")
            return False
        return True

    async def run(
        self,
        target_url: str,
        inject_header: str = "User-Agent",
        cookies: dict[str, str] | None = None,
        time_delay: float = 5.0,
    ) -> ExternalToolFinding:
        """執行 Blisqy Header 盲注偵測

        Args:
            target_url: 目標 URL
            inject_header: 要注入的 HTTP Header 名稱
            cookies: Cookie 字典
            time_delay: 時序延遲閾值（秒）

        Returns:
            ExternalToolFinding: 標準化執行結果
        """
        if not self.is_available():
            return ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=f"blisqy_{int(time.time())}",
                target_url=target_url,
                return_code=-1,
                execution_time_seconds=0.0,
                is_success=False,
                raw_stderr=f"Blisqy not found at: {_TOOL_DIR}",
            )

        execution_id = f"blisqy_{int(time.time())}"
        cmd = self._build_command(target_url, inject_header, cookies, time_delay)

        logger.info(f"[Blisqy] Scanning Header={inject_header} → {target_url}")
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_TOOL_DIR),
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = time.monotonic() - start
                return ExternalToolFinding(
                    tool_name=self.TOOL_NAME,
                    execution_id=execution_id,
                    target_url=target_url,
                    return_code=-2,
                    execution_time_seconds=round(elapsed, 2),
                    is_success=False,
                    raw_stderr=f"Timeout after {self.timeout}s",
                )

            elapsed = time.monotonic() - start
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            rc = proc.returncode or 0

            vulns = self._parse_output(stdout, inject_header)
            finding = ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=execution_id,
                target_url=target_url,
                return_code=rc,
                execution_time_seconds=round(elapsed, 2),
                is_success=(rc == 0),
                raw_stdout=stdout,
                raw_stderr=stderr or None,
                identified_vulnerabilities=vulns,
                tool_specific_metrics={
                    "inject_header": inject_header,
                    "time_delay": time_delay,
                    "command": " ".join(str(c) for c in cmd),
                },
            )
            logger.info(
                f"[Blisqy] Done in {elapsed:.1f}s — "
                f"{'VULNERABLE' if vulns else 'clean'} Header={inject_header}"
            )
            return finding

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(f"[Blisqy] Execution error: {exc}")
            return ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=execution_id,
                target_url=target_url,
                return_code=-3,
                execution_time_seconds=round(elapsed, 2),
                is_success=False,
                raw_stderr=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(
        self,
        target_url: str,
        inject_header: str,
        cookies: dict[str, str] | None,
        time_delay: float,
    ) -> list[str]:
        cmd = [
            self._python_bin,
            str(_FIND_SCRIPT),
            "-u", target_url,
            "-p", inject_header,
            "-t", str(time_delay),
        ]
        if cookies:
            cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
            cmd += ["-c", cookie_str]
        return cmd

    def _parse_output(self, stdout: str, header: str) -> list[str]:
        """判斷輸出中是否有注入點"""
        vulns: list[str] = []
        lower = stdout.lower()
        for kw in _VULN_KEYWORDS:
            if kw in lower:
                vulns.append(f"Header SQLi ({header}): {kw}")
                break
        return vulns

    # ------------------------------------------------------------------
    # Multi-header sweep helper
    # ------------------------------------------------------------------

    async def sweep_headers(
        self,
        target_url: str,
        headers_to_test: list[str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> list[ExternalToolFinding]:
        """並行掃描多個 HTTP Header

        Args:
            target_url: 目標 URL
            headers_to_test: 要測試的 Header 列表（預設使用常見 3 個）
            cookies: Cookie 字典

        Returns:
            每個 Header 各一個 ExternalToolFinding
        """
        targets = headers_to_test or self.DEFAULT_INJECT_HEADERS
        tasks = [self.run(target_url, header, cookies) for header in targets]
        return list(await asyncio.gather(*tasks))
