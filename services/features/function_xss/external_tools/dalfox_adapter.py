"""
Dalfox XSS Adapter - function_xss 外部工具橋接層

將 Dalfox (Go 語言 XSS 掃描器) 橋接到 AIVA function_xss 模組。
工具本體位置: tools/external_arsenal/Dalfox/dalfox-windows-amd64.exe (Windows)
               tools/external_arsenal/Dalfox/dalfox           (Linux)

Dalfox 官方文檔: https://github.com/hahwul/dalfox
版本: v2.12.0

使用方式:
    adapter = DalfoxAdapter()
    result = await adapter.run(target_url="https://example.com", cookies={"session": "abc"})
"""

import asyncio
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

from aiva_common.schemas.findings import ExternalToolFinding

logger = logging.getLogger(__name__)

# 工具本體路徑 (相對於專案根目錄)
_TOOL_DIR = Path(__file__).resolve().parents[5] / "tools" / "external_arsenal" / "Dalfox"
_BINARY_NAME = "dalfox-windows-amd64.exe" if platform.system() == "Windows" else "dalfox"
_BINARY_PATH = _TOOL_DIR / _BINARY_NAME


class DalfoxAdapter:
    """Dalfox XSS 掃描器適配器

    將 Dalfox 的 CLI 輸出轉換為 AIVA 標準的 ExternalToolFinding。
    Dalfox 支援 --format json 輸出，解析簡單且結構化。
    """

    TOOL_NAME = "dalfox"
    DEFAULT_TIMEOUT = 300  # 5 分鐘

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._binary = _BINARY_PATH
        self._is_available: bool | None = None

    def is_available(self) -> bool:
        """檢查 Dalfox 是否已就緒"""
        if self._is_available is not None:
            return self._is_available
        self._is_available = self._binary.exists()
        if not self._is_available:
            logger.warning(
                f"Dalfox binary not found at {self._binary}. "
                "Please ensure tools/external_arsenal/Dalfox/ contains the binary."
            )
        return self._is_available

    async def run(
        self,
        target_url: str,
        cookies: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        blind_xss_callback: str | None = None,
        extra_flags: list[str] | None = None,
    ) -> ExternalToolFinding:
        """執行 Dalfox 掃描

        Args:
            target_url: 目標 URL（必填）
            cookies: Cookie 字典（如 {"session": "abc123"}）
            headers: 額外 HTTP Header
            blind_xss_callback: Blind XSS 回呼 URL（如 OAST 服務）
            extra_flags: 其他 Dalfox CLI 旗標

        Returns:
            ExternalToolFinding: 標準化執行結果
        """
        if not self.is_available():
            return ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=f"dalfox_{int(time.time())}",
                target_url=target_url,
                return_code=-1,
                execution_time_seconds=0.0,
                is_success=False,
                raw_stderr=f"Dalfox binary not found at: {self._binary}",
            )

        cmd = self._build_command(target_url, cookies, headers, blind_xss_callback, extra_flags)
        execution_id = f"dalfox_{int(time.time())}"

        logger.info(f"[Dalfox] Starting scan → {target_url}")
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = time.monotonic() - start
                logger.warning(f"[Dalfox] Timeout after {elapsed:.1f}s on {target_url}")
                return ExternalToolFinding(
                    tool_name=self.TOOL_NAME,
                    execution_id=execution_id,
                    target_url=target_url,
                    return_code=-2,
                    execution_time_seconds=elapsed,
                    is_success=False,
                    raw_stderr=f"Timeout after {self.timeout}s",
                )

            elapsed = time.monotonic() - start
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            rc = proc.returncode or 0

            parsed, vulns = self._parse_output(stdout)
            finding = ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=execution_id,
                target_url=target_url,
                return_code=rc,
                execution_time_seconds=round(elapsed, 2),
                is_success=(rc == 0),
                raw_stdout=stdout,
                raw_stderr=stderr if stderr else None,
                parsed_json_output=parsed,
                identified_vulnerabilities=vulns,
                tool_specific_metrics={"command": " ".join(str(c) for c in cmd)},
            )
            logger.info(
                f"[Dalfox] Done in {elapsed:.1f}s — "
                f"{len(vulns)} potential XSS(es) found on {target_url}"
            )
            return finding

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(f"[Dalfox] Execution error: {exc}")
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
        cookies: dict[str, str] | None,
        headers: dict[str, str] | None,
        blind_xss_callback: str | None,
        extra_flags: list[str] | None,
    ) -> list[str]:
        cmd: list[str] = [
            str(self._binary),
            "url", target_url,
            "--format", "json",  # 結構化輸出，方便解析
            "--silence",          # 抑制進度條
            "--no-color",
        ]
        if cookies:
            cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
            cmd += ["--cookie", cookie_str]
        if headers:
            for k, v in headers.items():
                cmd += ["--header", f"{k}: {v}"]
        if blind_xss_callback:
            cmd += ["--blind", blind_xss_callback]
        if extra_flags:
            cmd += extra_flags
        return cmd

    def _parse_output(self, stdout: str) -> tuple[dict[str, Any] | None, list[str]]:
        """解析 Dalfox JSON 輸出，提取漏洞清單"""
        # Dalfox --format json 每行一個 JSON 物件
        parsed_items: list[dict[str, Any]] = []
        vulns: list[str] = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                parsed_items.append(obj)
                # Dalfox JSON 結構: {"type": "G", "string": "XSS Found ...", "param": "q", ...}
                if obj.get("type") in ("G", "V", "POC"):
                    vuln_str = obj.get("string") or obj.get("data") or str(obj)
                    vulns.append(vuln_str)
            except json.JSONDecodeError:
                # 非 JSON 行（如進度輸出），直接略過
                pass

        aggregate = {"results": parsed_items, "total": len(parsed_items)} if parsed_items else None
        return aggregate, vulns
