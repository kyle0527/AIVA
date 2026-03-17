"""
function_postex.engines.persistence
持久性機制檢測引擎

檢測可用於建立持久性的系統配置和方法。
"""

import os
import platform
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from aiva_common.utils import get_logger
from aiva_common.enums.common import Severity, Confidence

logger = get_logger(__name__)


@dataclass
class PersistenceVector:
    """持久性向量"""
    id: str
    severity: Severity
    confidence: Confidence
    title: str
    description: str
    evidence: Dict[str, Any]
    recommendation: str
    technique: str  # cron, service, registry, startup, etc.
    persistence_commands: List[str]


class PersistenceEngine:
    """持久性機制檢測引擎"""
    
    def __init__(self, safe_mode: bool = False):
        """
        初始化持久性引擎
        
        Args:
            safe_mode: 安全模式（僅檢測，不執行）
        """
        self.safe_mode = safe_mode
        self.os_type = platform.system().lower()
        logger.info(f"持久性引擎初始化 (OS: {self.os_type}, SafeMode: {safe_mode})")
    
    def scan(self) -> List[PersistenceVector]:
        """
        掃描持久性機會
        
        Returns:
            List[PersistenceVector]: 發現的持久性向量
        """
        vectors = []
        
        if self.os_type == "linux":
            vectors.extend(self._check_linux_persistence())
        elif self.os_type == "windows":
            vectors.extend(self._check_windows_persistence())
        
        logger.info(f"持久性掃描完成，發現 {len(vectors)} 個向量")
        return vectors
    
    def _check_linux_persistence(self) -> List[PersistenceVector]:
        """Linux 持久性檢查"""
        vectors = []
        
        # 1. Cron Jobs
        vectors.extend(self._check_cron_persistence())
        
        # 2. Systemd Services
        vectors.extend(self._check_systemd_persistence())
        
        # 3. Shell RC Files
        vectors.extend(self._check_shell_rc_persistence())
        
        # 4. SSH Keys
        vectors.extend(self._check_ssh_persistence())
        
        # 5. LD_PRELOAD
        vectors.extend(self._check_ld_preload_persistence())
        
        return vectors
    
    def _check_cron_persistence(self) -> List[PersistenceVector]:
        """檢查 Cron 持久性機會"""
        vectors = []
        
        cron_paths = [
            "/etc/crontab",
            "/etc/cron.d",
            "/var/spool/cron/crontabs"
        ]
        
        for cron_path in cron_paths:
            if os.path.exists(cron_path) and os.access(cron_path, os.W_OK):
                vectors.append(PersistenceVector(
                    id=f"PERSIST-CRON-{Path(cron_path).name.upper()}",
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH,
                    title=f"Writable Cron Configuration: {cron_path}",
                    description="Cron configuration is writable, allowing persistence via scheduled tasks",
                    evidence={"path": cron_path, "writable": True},
                    recommendation="Restrict cron permissions to root only",
                    technique="cron",
                    persistence_commands=[
                        f"echo '* * * * * /tmp/backdoor.sh' >> {cron_path}",
                        "# Backdoor will run every minute"
                    ]
                ))
        
        return vectors
    
    def _check_systemd_persistence(self) -> List[PersistenceVector]:
        """檢查 Systemd 服務持久性"""
        vectors = []
        
        systemd_paths = [
            "/etc/systemd/system",
            "/usr/lib/systemd/system",
            Path.home() / ".config/systemd/user"
        ]
        
        for systemd_path in systemd_paths:
            if os.path.exists(systemd_path) and os.access(systemd_path, os.W_OK):
                vectors.append(PersistenceVector(
                    id=f"PERSIST-SYSTEMD-{Path(systemd_path).name.upper()}",
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH,
                    title=f"Writable Systemd Directory: {systemd_path}",
                    description="Systemd directory is writable, allowing malicious service creation",
                    evidence={"path": str(systemd_path), "writable": True},
                    recommendation="Restrict systemd directory permissions",
                    technique="systemd",
                    persistence_commands=[
                        f"cat > {systemd_path}/backdoor.service <<EOF",
                        "[Unit]",
                        "Description=Backdoor Service",
                        "[Service]",
                        "ExecStart=/tmp/backdoor.sh",
                        "[Install]",
                        "WantedBy=multi-user.target",
                        "EOF",
                        "systemctl enable backdoor.service"
                    ]
                ))
        
        return vectors
    
    def _check_shell_rc_persistence(self) -> List[PersistenceVector]:
        """檢查 Shell RC 文件持久性"""
        vectors = []
        
        rc_files = [
            Path.home() / ".bashrc",
            Path.home() / ".zshrc",
            Path.home() / ".profile",
            "/etc/profile",
            "/etc/bash.bashrc"
        ]
        
        for rc_file in rc_files:
            if os.path.exists(rc_file) and os.access(rc_file, os.W_OK):
                vectors.append(PersistenceVector(
                    id=f"PERSIST-RC-{rc_file.name.upper()}",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.HIGH,
                    title=f"Writable Shell RC File: {rc_file}",
                    description="Shell configuration file is writable, allowing command injection on login",
                    evidence={"path": str(rc_file), "writable": True},
                    recommendation="Verify RC file integrity and restrict permissions",
                    technique="shell_rc",
                    persistence_commands=[
                        f"echo '/tmp/backdoor.sh &' >> {rc_file}",
                        "# Backdoor runs on every shell login"
                    ]
                ))
        
        return vectors
    
    def _check_ssh_persistence(self) -> List[PersistenceVector]:
        """檢查 SSH 持久性"""
        vectors = []
        
        ssh_dir = Path.home() / ".ssh"
        authorized_keys = ssh_dir / "authorized_keys"
        
        if ssh_dir.exists() and os.access(ssh_dir, os.W_OK):
            vectors.append(PersistenceVector(
                id="PERSIST-SSH-DIR",
                severity=Severity.HIGH,
                confidence=Confidence.HIGH,
                title="Writable SSH Directory",
                description=".ssh directory is writable, allowing SSH key persistence",
                evidence={"path": str(ssh_dir), "writable": True},
                recommendation="Verify SSH keys and restrict .ssh permissions (chmod 700)",
                technique="ssh_keys",
                persistence_commands=[
                    "ssh-keygen -t rsa -f /tmp/key -N ''",
                    f"cat /tmp/key.pub >> {authorized_keys}",
                    "# Can now SSH with /tmp/key private key"
                ]
            ))
        
        return vectors
    
    def _check_ld_preload_persistence(self) -> List[PersistenceVector]:
        """檢查 LD_PRELOAD 持久性"""
        vectors = []
        
        ld_preload_paths = [
            "/etc/ld.so.preload",
            Path.home() / ".bashrc"  # Can set LD_PRELOAD in RC
        ]
        
        for ld_path in ld_preload_paths:
            if os.path.exists(ld_path) and os.access(ld_path, os.W_OK):
                vectors.append(PersistenceVector(
                    id=f"PERSIST-LDPRELOAD-{Path(ld_path).name.upper()}",
                    severity=Severity.CRITICAL,
                    confidence=Confidence.HIGH,
                    title=f"LD_PRELOAD Persistence Possible: {ld_path}",
                    description="Can inject malicious shared library via LD_PRELOAD",
                    evidence={"path": str(ld_path), "writable": True},
                    recommendation="Monitor LD_PRELOAD usage and restrict file permissions",
                    technique="ld_preload",
                    persistence_commands=[
                        "gcc -shared -fPIC -o /tmp/evil.so evil.c",
                        f"echo '/tmp/evil.so' > {ld_path}",
                        "# Malicious library loaded into all processes"
                    ]
                ))
        
        return vectors
    
    def _check_windows_persistence(self) -> List[PersistenceVector]:
        """Windows 持久性檢查"""
        vectors = []

        # 1. Registry Run Keys
        vectors.extend(self._check_registry_run_keys())

        # 2. Startup Folder
        vectors.extend(self._check_startup_folder())

        # 3. Scheduled Tasks
        vectors.extend(self._check_windows_scheduled_tasks_persistence())

        # 4. Windows Services
        vectors.extend(self._check_windows_services_persistence())

        return vectors

    def _check_registry_run_keys(self) -> List[PersistenceVector]:
        """檢查 Registry Run Keys 持久性機會"""
        import subprocess
        vectors = []

        run_keys = [
            (r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run", "Current User Run", Severity.HIGH),
            (r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce", "Current User RunOnce", Severity.HIGH),
            (r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run", "Local Machine Run", Severity.CRITICAL),
            (r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce", "Local Machine RunOnce", Severity.CRITICAL),
        ]

        for reg_path, description, severity in run_keys:
            try:
                # Check if we can write to this registry key
                result = subprocess.run(
                    ["reg", "query", reg_path],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # Try a harmless write test (query only, assess by hive)
                    is_hkcu = reg_path.startswith("HKCU")
                    if is_hkcu:
                        # HKCU is always writable by the current user
                        vectors.append(PersistenceVector(
                            id=f"PERSIST-WIN-REGRUN-{description.replace(' ', '_').upper()}",
                            severity=severity,
                            confidence=Confidence.HIGH,
                            title=f"Writable Registry Run Key: {description}",
                            description=f"Registry key {reg_path} is writable, allowing auto-start persistence",
                            evidence={"registry_path": reg_path, "writable": True},
                            recommendation="Monitor registry Run keys for unauthorized entries",
                            technique="registry_run",
                            persistence_commands=[
                                f'reg add "{reg_path}" /v Backdoor /t REG_SZ /d "C:\\Windows\\Temp\\payload.exe" /f',
                                "# Payload executes on every user login"
                            ]
                        ))
                    else:
                        # HKLM requires admin; check existing entries for suspicious items
                        entries = result.stdout.strip()
                        if entries:
                            vectors.append(PersistenceVector(
                                id=f"PERSIST-WIN-REGRUN-{description.replace(' ', '_').upper()}",
                                severity=Severity.MEDIUM,
                                confidence=Confidence.MEDIUM,
                                title=f"Registry Run Key Exists: {description}",
                                description=f"Registry key {reg_path} has entries; verify all are legitimate",
                                evidence={"registry_path": reg_path, "entries": entries[:500]},
                                recommendation="Audit all entries in registry Run keys",
                                technique="registry_run",
                                persistence_commands=[
                                    f'reg add "{reg_path}" /v Backdoor /t REG_SZ /d "C:\\Windows\\Temp\\payload.exe" /f'
                                ]
                            ))
            except FileNotFoundError:
                logger.debug("reg command not available")
            except Exception as e:
                logger.debug(f"Registry check failed for {reg_path}: {e}")

        return vectors

    def _check_startup_folder(self) -> List[PersistenceVector]:
        """檢查啟動資料夾持久性"""
        vectors = []

        startup_paths = []
        # User startup folder
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            startup_paths.append(
                (os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup"),
                 "User Startup Folder", Severity.HIGH)
            )
        # All Users startup folder
        programdata = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        startup_paths.append(
            (os.path.join(programdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup"),
             "All Users Startup Folder", Severity.CRITICAL)
        )

        for startup_path, description, severity in startup_paths:
            if os.path.exists(startup_path) and os.access(startup_path, os.W_OK):
                vectors.append(PersistenceVector(
                    id=f"PERSIST-WIN-STARTUP-{description.replace(' ', '_').upper()}",
                    severity=severity,
                    confidence=Confidence.HIGH,
                    title=f"Writable {description}",
                    description=f"Startup folder {startup_path} is writable, programs placed here execute on login",
                    evidence={"path": startup_path, "writable": True},
                    recommendation="Restrict write access to startup folders and monitor for changes",
                    technique="startup_folder",
                    persistence_commands=[
                        f'copy payload.exe "{startup_path}\\payload.exe"',
                        "# Payload runs on next user login"
                    ]
                ))

        return vectors

    def _check_windows_scheduled_tasks_persistence(self) -> List[PersistenceVector]:
        """檢查排程工作持久性機會"""
        import subprocess
        vectors = []

        try:
            # Check if we can create scheduled tasks
            result = subprocess.run(
                ["schtasks", "/query", "/fo", "CSV"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                # User can query tasks; check if creation is possible
                vectors.append(PersistenceVector(
                    id="PERSIST-WIN-SCHTASK",
                    severity=Severity.HIGH,
                    confidence=Confidence.MEDIUM,
                    title="Scheduled Task Creation Possible",
                    description="Current user can query scheduled tasks; task creation may be available for persistence",
                    evidence={"schtasks_accessible": True},
                    recommendation="Restrict schtasks access and monitor for unauthorized task creation",
                    technique="scheduled_task",
                    persistence_commands=[
                        'schtasks /create /tn "Backdoor" /tr "C:\\Windows\\Temp\\payload.exe" /sc onlogon /ru SYSTEM',
                        "# Task runs as SYSTEM on every logon"
                    ]
                ))
        except FileNotFoundError:
            logger.debug("schtasks not available")
        except Exception as e:
            logger.debug(f"Scheduled tasks check failed: {e}")

        return vectors

    def _check_windows_services_persistence(self) -> List[PersistenceVector]:
        """檢查 Windows 服務持久性機會"""
        import subprocess
        vectors = []

        try:
            # Check if sc.exe is accessible for service creation
            result = subprocess.run(
                ["sc", "query", "state=", "all"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                # Count services to provide context
                service_count = result.stdout.count("SERVICE_NAME:")
                vectors.append(PersistenceVector(
                    id="PERSIST-WIN-SERVICE",
                    severity=Severity.HIGH,
                    confidence=Confidence.MEDIUM,
                    title="Windows Service Management Accessible",
                    description=f"Can query {service_count} services; service creation may be available for persistence",
                    evidence={"service_count": service_count, "sc_accessible": True},
                    recommendation="Restrict service creation permissions and monitor for new services",
                    technique="windows_service",
                    persistence_commands=[
                        'sc create Backdoor binPath= "C:\\Windows\\Temp\\payload.exe" start= auto',
                        "sc start Backdoor",
                        "# Service runs with SYSTEM privileges on boot"
                    ]
                ))
        except FileNotFoundError:
            logger.debug("sc command not available")
        except Exception as e:
            logger.debug(f"Service check failed: {e}")

        return vectors
