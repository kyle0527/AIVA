"""
SecretFinder Adapter - function_web_scanner 外部工具橋接層

將 SecretFinder (JavaScript 敏感資訊挖掘工具) 橋接到 AIVA function_web_scanner 模組。
工具本體位置: tools/external_arsenal/SecretFinder/

SecretFinder 官方文檔: https://github.com/m4ll0k/SecretFinder
功能：從 JavaScript 檔案中提取 API Key, JWT, 密碼, AWS 金鑰等敏感資訊

輸出對接: aiva_common.schemas.findings.SensitiveMatch (已存在的 Schema)

使用方式:
    adapter = SecretFinderAdapter()
    result = await adapter.run(target_url="https://example.com/main.js")
    # 或掃描整個頁面（自動尋找 JS 檔案）
    result = await adapter.scan_page(target_url="https://example.com")
"""

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from aiva_common.schemas.findings import ExternalToolFinding, SensitiveMatch
from aiva_common.enums.common import Severity

logger = logging.getLogger(__name__)

# 工具本體路徑
_TOOL_DIR = Path(__file__).resolve().parents[5] / "tools" / "external_arsenal" / "SecretFinder"
_MAIN_SCRIPT = _TOOL_DIR / "SecretFinder.py"

# SecretFinder 輸出的 regex 模式（用於匹配敏感資訊類別）
_SEVERITY_MAP: dict[str, Severity] = {
    "google_api": Severity.HIGH,
    "firebase": Severity.HIGH,
    "google_captcha": Severity.MEDIUM,
    "google_oauth": Severity.HIGH,
    "amazon_aws_access_key_id": Severity.CRITICAL,
    "amazon_mws_auth_token": Severity.CRITICAL,
    "amazon_aws_url": Severity.HIGH,
    "facebook_access_token": Severity.HIGH,
    "authorization_basic": Severity.HIGH,
    "authorization_bearer": Severity.HIGH,
    "jwt": Severity.HIGH,
    "mailgun_api_key": Severity.HIGH,
    "twilio_api_key": Severity.HIGH,
    "github_access_token": Severity.HIGH,
    "private_ssh_key": Severity.CRITICAL,
    "slack_token": Severity.HIGH,
    "rsa_private_key": Severity.CRITICAL,
}


class SecretFinderAdapter:
    """SecretFinder 敏感資訊挖掘適配器

    掃描 JavaScript 文件中的 API Key, JWT, 私鑰等敏感資訊，
    並將結果映射到 AIVA 的 SensitiveMatch Schema。

    使用場景:
        - Phase 0 偵察：掃描目標網站所有 JS 文件
        - Phase 1 深度掃描：針對特定 JS URL 深入挖掘
    """

    TOOL_NAME = "secretfinder"
    DEFAULT_TIMEOUT = 120  # 2 分鐘

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._python_bin = sys.executable

    def is_available(self) -> bool:
        """檢查 SecretFinder 是否就緒"""
        if not _MAIN_SCRIPT.exists():
            logger.warning(f"SecretFinder.py not found at {_MAIN_SCRIPT}")
            return False
        return True

    async def run(
        self,
        target_url: str,
        output_json: bool = True,
        cookies: dict[str, str] | None = None,
    ) -> ExternalToolFinding:
        """掃描指定 URL（JavaScript 文件或包含 JS 的網頁）

        Args:
            target_url: JavaScript 文件 URL 或包含 JS 的頁面 URL
            output_json: 是否使用 JSON 輸出格式（推薦 True）
            cookies: Cookie 字典

        Returns:
            ExternalToolFinding: 標準化執行結果
        """
        if not self.is_available():
            return ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=f"sf_{int(time.time())}",
                target_url=target_url,
                return_code=-1,
                execution_time_seconds=0.0,
                is_success=False,
                raw_stderr=f"SecretFinder.py not found at: {_TOOL_DIR}",
            )

        execution_id = f"sf_{int(time.time())}"
        cmd = self._build_command(target_url, output_json, cookies)

        logger.info(f"[SecretFinder] Scanning → {target_url}")
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

            parsed, vulns, matches = self._parse_output(stdout, target_url)
            finding = ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=execution_id,
                target_url=target_url,
                return_code=rc,
                execution_time_seconds=round(elapsed, 2),
                is_success=(rc == 0),
                raw_stdout=stdout,
                raw_stderr=stderr or None,
                parsed_json_output=parsed,
                identified_vulnerabilities=vulns,
                tool_specific_metrics={
                    "sensitive_matches_count": len(matches),
                    "command": " ".join(str(c) for c in cmd),
                },
            )
            logger.info(
                f"[SecretFinder] Done in {elapsed:.1f}s — "
                f"{len(matches)} sensitive secrets found in {target_url}"
            )
            return finding

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(f"[SecretFinder] Execution error: {exc}")
            return ExternalToolFinding(
                tool_name=self.TOOL_NAME,
                execution_id=execution_id,
                target_url=target_url,
                return_code=-3,
                execution_time_seconds=round(elapsed, 2),
                is_success=False,
                raw_stderr=str(exc),
            )

    def to_sensitive_matches(self, finding: ExternalToolFinding) -> list[SensitiveMatch]:
        """將 ExternalToolFinding 轉換為 SensitiveMatch 列表

        對接 AIVA 標準的 SensitiveMatch Schema，供後續 FindingPayload 使用。

        Args:
            finding: SecretFinder 的執行結果

        Returns:
            符合 aiva_common 標準的 SensitiveMatch 列表
        """
        matches: list[SensitiveMatch] = []
        parsed = finding.parsed_json_output
        if not parsed:
            return matches

        for item in parsed.get("results", []):
            pattern_name = item.get("name", "unknown")
            matched_text = item.get("matched", "")
            context = item.get("string", matched_text)[:200]  # 限制長度

            severity = _SEVERITY_MAP.get(pattern_name.lower(), Severity.MEDIUM)
            match_id = f"sf_{finding.execution_id}_{len(matches)}"

            try:
                sm = SensitiveMatch(
                    match_id=match_id,
                    pattern_name=pattern_name,
                    matched_text=matched_text[:100],  # 截斷以避免過長
                    context=context,
                    confidence=0.85,
                    url=finding.target_url,
                    severity=severity,
                )
                matches.append(sm)
            except Exception as e:
                logger.warning(f"[SecretFinder] Failed to create SensitiveMatch: {e}")

        return matches

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(
        self,
        target_url: str,
        output_json: bool,
        cookies: dict[str, str] | None,
    ) -> list[str]:
        cmd = [
            self._python_bin,
            str(_MAIN_SCRIPT),
            "-i", target_url,
        ]
        if output_json:
            cmd += ["-o", "json"]
        if cookies:
            cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
            cmd += ["-c", cookie_str]
        return cmd

    def _parse_output(
        self, stdout: str, target_url: str
    ) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
        """解析 SecretFinder 輸出

        Returns:
            (parsed_dict, vuln_strings, raw_matches)
        """
        matches: list[dict[str, Any]] = []
        vulns: list[str] = []

        # 嘗試解析 JSON 輸出
        try:
            data = json.loads(stdout)
            if isinstance(data, list):
                matches = data
            elif isinstance(data, dict):
                matches = data.get("results", [data])
        except (json.JSONDecodeError, ValueError):
            # Fall back: 解析純文字輸出
            for line in stdout.splitlines():
                match = re.search(r"\[(.+?)\]\s+(.+)", line)
                if match:
                    matches.append({
                        "name": match.group(1).strip(),
                        "matched": match.group(2).strip(),
                    })

        for m in matches:
            name = m.get("name", "secret")
            vulns.append(f"Sensitive data: {name} @ {target_url}")

        parsed = {"results": matches, "source_url": target_url, "total": len(matches)}
        return parsed, vulns, matches
