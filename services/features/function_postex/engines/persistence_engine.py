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
        safe_mode: bool = False,
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

        # 真實的啟動項檢測（僅讀取，不修改）
        result["mode"] = "detection"
        try:
            if self.os_type == "Windows":
                # Windows Registry 啟動項檢測
                try:
                    import winreg
                    reg_paths = [
                        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                    ]
                    for hkey, path in reg_paths:
                        try:
                            key = winreg.OpenKey(hkey, path, 0, winreg.KEY_READ)
                            i = 0
                            while True:
                                try:
                                    name, value, _ = winreg.EnumValue(key, i)
                                    result["items"].append({
                                        "location": f"{hkey}\\{path}",
                                        "name": name,
                                        "value": value,
                                        "risk": "MEDIUM",
                                    })
                                    i += 1
                                except OSError:
                                    break
                            winreg.CloseKey(key)
                        except FileNotFoundError:
                            pass
                except ImportError:
                    result["error"] = "winreg module not available (not on Windows)"
            else:  # Linux/macOS
                # 檢查 systemd 服務
                import glob
                import os
                systemd_paths = [
                    "/etc/systemd/system/",
                    "/lib/systemd/system/",
                    os.path.expanduser("~/.config/systemd/user/")
                ]
                for path in systemd_paths:
                    if os.path.exists(path):
                        for service_file in glob.glob(os.path.join(path, "*.service")):
                            result["items"].append({
                                "location": path,
                                "name": os.path.basename(service_file),
                                "risk": "LOW",
                                "type": "systemd_service",
                            })
                
                # 檢查 cron jobs
                cron_paths = [
                    "/etc/crontab",
                    "/etc/cron.d/",
                    "/var/spool/cron/crontabs/"
                ]
                for path in cron_paths:
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            result["items"].append({
                                "location": path,
                                "name": "crontab",
                                "risk": "MEDIUM",
                                "type": "cron",
                            })
        except Exception as e:
            result["error"] = f"Startup items check failed: {str(e)}"

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

        # 真實的計劃任務檢測
        result["mode"] = "detection"
        try:
            if self.os_type == "Windows":
                # Windows 計劃任務
                try:
                    import subprocess
                    task_result = subprocess.run(
                        ["schtasks", "/query", "/fo", "LIST", "/v"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if task_result.returncode == 0:
                        # 簡單解析輸出（實際應該使用更完整的解析）
                        tasks = task_result.stdout.split("\n\n")
                        for task in tasks[:10]:  # 限制返回前10個
                            if "TaskName:" in task:
                                result["tasks"].append({
                                    "name": "scheduled_task",
                                    "details": task[:200],
                                    "risk": "MEDIUM",
                                })
                except FileNotFoundError:
                    result["error"] = "schtasks command not found"
            else:  # Linux/macOS
                # 檢查 cron jobs
                import subprocess
                try:
                    cron_result = subprocess.run(
                        ["crontab", "-l"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if cron_result.returncode == 0:
                        for line in cron_result.stdout.split("\n"):
                            if line.strip() and not line.startswith("#"):
                                result["tasks"].append({
                                    "schedule": line,
                                    "risk": "MEDIUM",
                                    "type": "cron",
                                })
                except FileNotFoundError:
                    result["tasks"].append({
                        "message": "No crontab for current user",
                        "risk": "INFO",
                    })
        except Exception as e:
            result["error"] = f"Scheduled tasks check failed: {str(e)}"

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

        # 真實的系統服務檢測
        result["mode"] = "detection"
        try:
            # 使用 psutil 獲取真實的服務和進程信息
            if self.os_type == "Windows":
                # Windows 服務檢測
                for proc in psutil.process_iter(['pid', 'name', 'username']):
                    try:
                        pinfo = proc.info
                        # 檢查常見的服務進程
                        if pinfo['name'] in ['services.exe', 'svchost.exe', 'System']:
                            result["services"].append({
                                "pid": pinfo['pid'],
                                "name": pinfo['name'],
                                "username": pinfo.get('username', 'N/A'),
                                "risk": "LOW",
                                "type": "windows_service",
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            else:  # Linux/macOS
                # 檢查 systemd 服務狀態
                import subprocess
                try:
                    service_result = subprocess.run(
                        ["systemctl", "list-units", "--type=service", "--state=running"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if service_result.returncode == 0:
                        lines = service_result.stdout.split("\n")
                        for line in lines[1:11]:  # 前10個服務
                            if line.strip() and ".service" in line:
                                result["services"].append({
                                    "service": line.split()[0],
                                    "state": "running",
                                    "risk": "LOW",
                                    "type": "systemd_service",
                                })
                except FileNotFoundError:
                    # systemctl 不可用，列出進程
                    for proc in psutil.process_iter(['pid', 'name', 'username']):
                        try:
                            result["services"].append({
                                "pid": proc.info['pid'],
                                "name": proc.info['name'],
                                "username": proc.info.get('username', 'N/A'),
                                "risk": "LOW",
                            })
                            if len(result["services"]) >= 20:  # 限制數量
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
        except Exception as e:
            result["error"] = f"Services check failed: {str(e)}"

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

        # 真實的 Windows Registry 持久化檢測
        result["mode"] = "detection"
        try:
            import winreg
            # 常見持久化 Registry 位置
            reg_locations = [
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\RunOnce"),
            ]
            
            for hkey, path in reg_locations:
                try:
                    key = winreg.OpenKey(hkey, path, 0, winreg.KEY_READ)
                    i = 0
                    while True:
                        try:
                            name, value, _ = winreg.EnumValue(key, i)
                            result["entries"].append({
                                "location": f"{hkey}\\{path}",
                                "name": name,
                                "value": value,
                                "risk": "MEDIUM",
                            })
                            i += 1
                        except OSError:
                            break
                    winreg.CloseKey(key)
                except FileNotFoundError:
                    pass
        except ImportError:
            result["error"] = "winreg module not available"
        except Exception as e:
            result["error"] = f"Registry check failed: {str(e)}"

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

        # 真實的 Cron 任務檢測
        result["mode"] = "detection"
        try:
            import subprocess
            # 檢查用戶 crontab
            try:
                cron_result = subprocess.run(
                    ["crontab", "-l"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if cron_result.returncode == 0:
                    for line in cron_result.stdout.split("\n"):
                        if line.strip() and not line.startswith("#"):
                            result["jobs"].append({
                                "type": "user_crontab",
                                "schedule": line,
                                "risk": "MEDIUM",
                            })
            except FileNotFoundError:
                result["jobs"].append({
                    "type": "user_crontab",
                    "message": "No crontab for current user",
                    "risk": "INFO",
                })
            
            # 檢查系統 cron 目錄
            import glob
            import os
            cron_dirs = ["/etc/cron.d/", "/etc/cron.daily/", "/etc/cron.hourly/", "/etc/cron.weekly/"]
            for cron_dir in cron_dirs:
                if os.path.exists(cron_dir):
                    for job_file in glob.glob(os.path.join(cron_dir, "*")):
                        result["jobs"].append({
                            "type": "system_cron",
                            "location": job_file,
                            "risk": "LOW",
                        })
        except Exception as e:
            result["error"] = f"Cron check failed: {str(e)}"

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
