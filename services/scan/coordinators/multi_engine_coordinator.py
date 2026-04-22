"""多引擎掃描協調器

整合 Python / Rust / Go / TypeScript 四個掃描引擎，提供統一的
coordinate(target, options) 入口。

設計原則：
- Python 引擎直接 import，其餘語言引擎透過 subprocess 呼叫 CLI
- 任何單一引擎失敗不中斷整體流程，結果合併後返回
- 若所有引擎皆不可用，仍返回結構化空結果而非拋出例外
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 可選 Python 引擎（靜默降級）
# ──────────────────────────────────────────────
try:
    from services.scan.python_engine import (
        DeserializationDetector,
        PassiveAnalyzer,
        XXEDetector,
    )
    _PYTHON_ENGINE_AVAILABLE = True
except ImportError as _e:
    logger.warning("⚠️  Python scan engine not available: %s", _e)
    _PYTHON_ENGINE_AVAILABLE = False
    DeserializationDetector = None  # type: ignore[assignment,misc]
    PassiveAnalyzer = None  # type: ignore[assignment,misc]
    XXEDetector = None  # type: ignore[assignment,misc]


# ──────────────────────────────────────────────
# 二進制路徑探測（編譯產物）
# ──────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]  # services/scan/coordinators → AIVA/

def _find_binary(candidates: list[str]) -> str | None:
    """在 PATH 和常用編譯輸出路徑中搜尋二進制檔案"""
    for name in candidates:
        # PATH 中搜尋
        found = shutil.which(name)
        if found:
            return found
        # 常見相對路徑
        for rel in [
            f"services/scan/rust_engine/target/release/{name}",
            f"services/scan/go_engine/cmd/{name}/{name}",
            f"services/scan/typescript_engine/dist/{name}.js",
        ]:
            abs_path = _REPO_ROOT / rel
            if abs_path.exists():
                return str(abs_path)
    return None


_RUST_BIN = _find_binary(["aiva_rust_scanner", "rust_scanner"])
_GO_BIN = _find_binary(["aiva_go_scanner", "go_scanner"])
_TS_BIN = _find_binary(["aiva_ts_scanner", "ts_scanner"])


class MultiEngineCoordinator:
    """多語言掃描引擎協調器

    提供統一入口 :meth:`coordinate`，並行運行各語言引擎，
    將結果彙整為標準格式後返回。
    """

    def __init__(self) -> None:
        self._python_available = _PYTHON_ENGINE_AVAILABLE
        self._rust_available = _RUST_BIN is not None
        self._go_available = _GO_BIN is not None
        self._ts_available = _TS_BIN is not None

        logger.info(
            "MultiEngineCoordinator initialized — "
            "python=%s rust=%s go=%s typescript=%s",
            self._python_available,
            self._rust_available,
            self._go_available,
            self._ts_available,
        )

    # ──────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────

    async def coordinate(
        self,
        target: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """並行運行所有可用引擎，合併結果

        Args:
            target: 目標 URL 或主機
            options: 可選的掃描參數（timeout、scan_type 等）

        Returns:
            合併後的掃描結果字典，結構：
            {
                "target": str,
                "findings": list[dict],
                "engines_used": list[str],
                "errors": list[str],
            }
        """
        options = options or {}
        timeout: int = int(options.get("timeout", 60))

        tasks: dict[str, asyncio.Task[list[dict[str, Any]]]] = {}

        if self._python_available:
            tasks["python"] = asyncio.create_task(
                self._run_python_engine(target, options)
            )
        if self._rust_available:
            tasks["rust"] = asyncio.create_task(
                self._run_subprocess_engine("rust", _RUST_BIN, target, options, timeout)
            )
        if self._go_available:
            tasks["go"] = asyncio.create_task(
                self._run_subprocess_engine("go", _GO_BIN, target, options, timeout)
            )
        if self._ts_available:
            tasks["typescript"] = asyncio.create_task(
                self._run_subprocess_engine("typescript", _TS_BIN, target, options, timeout)
            )

        if not tasks:
            logger.warning("⚠️  No scan engines available for target: %s", target)
            return {
                "target": target,
                "findings": [],
                "engines_used": [],
                "errors": ["No scan engines available"],
            }

        # 等待所有 task，不因單個失敗而中止
        done = await asyncio.gather(*tasks.values(), return_exceptions=True)

        all_findings: list[dict[str, Any]] = []
        engines_used: list[str] = []
        errors: list[str] = []

        for engine_name, result in zip(tasks.keys(), done):
            if isinstance(result, Exception):
                logger.error("Engine '%s' failed: %s", engine_name, result)
                errors.append(f"{engine_name}: {result}")
            else:
                all_findings.extend(result)
                engines_used.append(engine_name)

        return {
            "target": target,
            "findings": all_findings,
            "engines_used": engines_used,
            "errors": errors,
        }

    # ──────────────────────────────────────────
    # 引擎實作
    # ──────────────────────────────────────────

    async def _run_python_engine(
        self,
        target: str,
        options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """運行 Python 引擎（XXE / 反序列化 / 被動分析）"""
        findings: list[dict[str, Any]] = []
        timeout = int(options.get("timeout", 60))

        loop = asyncio.get_event_loop()

        # XXE 偵測
        try:
            xxe = XXEDetector()  # type: ignore[misc]
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, xxe.analyze_url, target),
                timeout=timeout,
            )
            for f in (raw or []):
                findings.append({
                    "engine": "python/xxe",
                    "type": getattr(f, "type", "xxe"),
                    "severity": str(getattr(f, "severity", "medium")),
                    "evidence": str(getattr(f, "evidence", "")),
                    "target": target,
                })
        except Exception as exc:
            logger.warning("XXEDetector on %s raised %s: %s", target, type(exc).__name__, exc)

        # 被動分析
        try:
            analyzer = PassiveAnalyzer()  # type: ignore[misc]
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, analyzer.analyze, target),
                timeout=timeout,
            )
            for f in (raw or []):
                findings.append({
                    "engine": "python/passive",
                    "type": getattr(f, "type", "passive"),
                    "severity": str(getattr(f, "severity", "info")),
                    "evidence": str(getattr(f, "evidence", "")),
                    "target": target,
                })
        except Exception as exc:
            logger.warning("PassiveAnalyzer on %s raised %s: %s", target, type(exc).__name__, exc)

        return findings

    async def _run_subprocess_engine(
        self,
        engine_name: str,
        binary: str | None,
        target: str,
        options: dict[str, Any],
        timeout: int,
    ) -> list[dict[str, Any]]:
        """透過 subprocess 調用已編譯的引擎 CLI

        期望引擎以 JSON 格式輸出結果到 stdout。
        """
        if not binary:
            return []

        cmd = [binary, "--target", target, "--output", "json"]
        if timeout:
            cmd += ["--timeout", str(timeout)]

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=5,  # 等待 proc 建立的超時
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            if proc.returncode != 0:
                logger.warning(
                    "Engine '%s' exited with code %d: %s",
                    engine_name,
                    proc.returncode,
                    stderr.decode(errors="replace")[:500],
                )
                return []

            raw_output = stdout.decode(errors="replace").strip()
            if not raw_output:
                return []

            data = json.loads(raw_output)
            findings_raw: list[Any] = data if isinstance(data, list) else data.get("findings", [])

            return [
                {
                    "engine": engine_name,
                    "type": f.get("type", "unknown"),
                    "severity": f.get("severity", "info"),
                    "evidence": f.get("evidence", ""),
                    "target": target,
                    **{k: v for k, v in f.items() if k not in ("type", "severity", "evidence")},
                }
                for f in findings_raw
                if isinstance(f, dict)
            ]

        except asyncio.TimeoutError:
            logger.warning("Engine '%s' timed out after %ds", engine_name, timeout)
            return []
        except json.JSONDecodeError as exc:
            logger.warning("Engine '%s' returned invalid JSON: %s", engine_name, exc)
            return []
        except Exception as exc:
            logger.error("Engine '%s' subprocess error: %s", engine_name, exc)
            return []
