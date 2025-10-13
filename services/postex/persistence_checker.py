"""
Persistence Checker - 持久化檢查器 (僅供授權測試使用)

⚠️ 嚴格警告: 此模組用於檢測持久化機制
- 僅在獲得明確書面授權的環境中使用
- 所有操作必須記錄並審計
- 禁止在生產環境或未授權系統使用

檢測場景:
- 啟動項檢測
- 計劃任務檢測
- 服務持久化檢測
- Registry 持久化檢測 (Windows)
- Cron 任務檢測 (Linux)
"""

from datetime import datetime
import hashlib
import platform
from typing import Any

import psutil
import structlog

logger = structlog.get_logger(__name__)


class PersistenceChecker:
    """
    持久化檢查器

    ⚠️ 僅用於安全測試和研究
    """

    def __init__(
        self,
        authorization_token: str | None = None,
        safe_mode: bool = True,
        deep_scan: bool = False,
    ):
        """
        初始化持久化檢查器

        Args:
            authorization_token: 授權令牌
            safe_mode: 安全模式(僅模擬,不執行)
            deep_scan: 深度掃描
        """
        self.authorization_token = authorization_token
        self.safe_mode = safe_mode
        self.deep_scan = deep_scan
        self.os_type = platform.system()
        self.test_results: list[dict[str, Any]] = []

        if not authorization_token:
            logger.warning(
                "persistence_checker_no_auth",
                message="No authorization token provided - running in safe mode only",
            )
            self.safe_mode = True

        logger.info(
            "persistence_checker_initialized",
            safe_mode=self.safe_mode,
            os_type=self.os_type,
            deep_scan=self.deep_scan,
        )

    def _log_action(self, action: str, details: dict[str, Any]) -> None:
        """記錄操作"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "safe_mode": self.safe_mode,
        }
        logger.info("persistence_check_action", **log_entry)

    def check_startup_items(self) -> dict[str, Any]:
        """
        檢查啟動項

        Returns:
            檢測結果
        """
        self._log_action("check_startup_items", {"os": self.os_type})

        result = {
            "test_name": "Startup Items Check",
            "timestamp": datetime.now().isoformat(),
            "os": self.os_type,
            "items": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            # 模擬啟動項
            if self.os_type == "Windows":
                result["items"].extend([
                    {
                        "location": "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                        "name": "SimulatedApp",
                        "value": "C:\\Program Files\\App\\app.exe",
                        "risk": "MEDIUM",
                        "simulated": True,
                    }
                ])
            else:  # Linux/macOS
                result["items"].extend([
                    {
                        "location": "/etc/systemd/system/",
                        "name": "simulated.service",
                        "risk": "MEDIUM",
                        "simulated": True,
                    }
                ])
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        logger.info("startup_items_checked", count=len(result["items"]))
        return result

    def check_scheduled_tasks(self) -> dict[str, Any]:
        """
        檢查計劃任務

        Returns:
            檢測結果
        """
        self._log_action("check_scheduled_tasks", {"os": self.os_type})

        result = {
            "test_name": "Scheduled Tasks Check",
            "timestamp": datetime.now().isoformat(),
            "os": self.os_type,
            "tasks": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            if self.os_type == "Windows":
                result["tasks"].append({
                    "name": "SimulatedTask",
                    "trigger": "Daily at 3:00 AM",
                    "action": "C:\\Scripts\\task.exe",
                    "risk": "HIGH",
                    "simulated": True,
                })
            else:
                result["tasks"].append({
                    "name": "simulated_cron",
                    "schedule": "0 3 * * *",
                    "command": "/usr/local/bin/task.sh",
                    "risk": "HIGH",
                    "simulated": True,
                })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def check_services(self) -> dict[str, Any]:
        """
        檢查系統服務

        Returns:
            檢測結果
        """
        self._log_action("check_services", {"os": self.os_type})

        result = {
            "test_name": "System Services Check",
            "timestamp": datetime.now().isoformat(),
            "os": self.os_type,
            "services": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            # 使用 psutil 獲取真實的進程信息(但標記為 simulated)
            try:
                process_count = len(list(psutil.process_iter()))
                result["services"].append({
                    "note": "Service enumeration would be performed",
                    "current_process_count": process_count,
                    "os": self.os_type,
                    "simulated": True,
                })
            except Exception as e:
                result["services"].append({
                    "error": str(e),
                    "simulated": True,
                })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def check_registry_persistence(self) -> dict[str, Any]:
        """
        檢查 Registry 持久化 (Windows)

        Returns:
            檢測結果
        """
        self._log_action("check_registry_persistence", {"os": self.os_type})

        result = {
            "test_name": "Registry Persistence Check",
            "timestamp": datetime.now().isoformat(),
            "os": self.os_type,
            "entries": [],
        }

        if self.os_type != "Windows":
            result["note"] = "Registry check only applies to Windows"
            self.test_results.append(result)
            return result

        if self.safe_mode:
            result["mode"] = "simulation"
            # 常見持久化 Registry 位置
            common_locations = [
                "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
                "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce",
                "HKLM\\System\\CurrentControlSet\\Services",
            ]
            result["entries"].extend([
                {
                    "location": loc,
                    "note": "Would be checked for suspicious entries",
                    "simulated": True,
                }
                for loc in common_locations
            ])
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def check_cron_jobs(self) -> dict[str, Any]:
        """
        檢查 Cron 任務 (Linux/macOS)

        Returns:
            檢測結果
        """
        self._log_action("check_cron_jobs", {"os": self.os_type})

        result = {
            "test_name": "Cron Jobs Check",
            "timestamp": datetime.now().isoformat(),
            "os": self.os_type,
            "jobs": [],
        }

        if self.os_type == "Windows":
            result["note"] = "Cron check only applies to Linux/macOS"
            self.test_results.append(result)
            return result

        if self.safe_mode:
            result["mode"] = "simulation"
            result["jobs"].append({
                "type": "user_crontab",
                "note": "User crontab would be checked",
                "simulated": True,
            })
            result["jobs"].append({
                "type": "system_cron",
                "locations": ["/etc/cron.d/", "/etc/cron.daily/", "/etc/cron.hourly/"],
                "note": "System cron directories would be scanned",
                "simulated": True,
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def run_full_assessment(self) -> dict[str, Any]:
        """
        執行完整的持久化檢查

        Returns:
            完整評估結果
        """
        logger.info("persistence_assessment_started", safe_mode=self.safe_mode)

        assessment = {
            "assessment_id": hashlib.sha256(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "safe_mode": self.safe_mode,
            "os": self.os_type,
            "deep_scan": self.deep_scan,
            "tests": [],
        }

        # 執行所有檢查
        assessment["tests"].append(self.check_startup_items())
        assessment["tests"].append(self.check_scheduled_tasks())
        assessment["tests"].append(self.check_services())

        # OS 特定檢查
        if self.os_type == "Windows":
            assessment["tests"].append(self.check_registry_persistence())
        else:
            assessment["tests"].append(self.check_cron_jobs())

        # 統計
        total_findings = sum(
            len(
                test.get("items", [])
                + test.get("tasks", [])
                + test.get("services", [])
                + test.get("entries", [])
                + test.get("jobs", [])
            )
            for test in assessment["tests"]
        )
        assessment["summary"] = {
            "total_tests": len(assessment["tests"]),
            "total_findings": total_findings,
            "mode": "simulation" if self.safe_mode else "actual",
            "os": self.os_type,
        }

        logger.info(
            "persistence_assessment_completed",
            total_tests=len(assessment["tests"]),
            total_findings=total_findings,
        )

        return assessment

    def get_results(self) -> list[dict[str, Any]]:
        """獲取所有測試結果"""
        return self.test_results


def main():
    """測試範例 - 僅在安全模式下運行"""
    print("⚠️  Persistence Checker - SAFE MODE DEMO")
    print("=" * 60)

    # 僅在安全模式下測試
    checker = PersistenceChecker(safe_mode=True)

    # 執行評估
    assessment = checker.run_full_assessment()

    print(f"\n📊 Assessment ID: {assessment['assessment_id']}")
    print(f"🔒 Safe Mode: {assessment['safe_mode']}")
    print(f"💻 OS: {assessment['os']}")
    print(f"\n📋 Tests Run: {assessment['summary']['total_tests']}")
    print(f"🔍 Findings: {assessment['summary']['total_findings']}")

    print("\n✅ Safe mode demo completed")
    print("⚠️  Remember: Never use this tool without explicit authorization!")


if __name__ == "__main__":
    main()
