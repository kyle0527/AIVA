"""
Privilege Escalator - 權限提升測試器 (僅供授權測試使用)

⚠️ 嚴格警告: 此模組用於測試權限提升漏洞
- 僅在獲得明確書面授權的環境中使用
- 所有操作必須記錄並審計
- 禁止在生產環境或未授權系統使用

測試場景:
- SUID/SGID 二進制文件濫用
- Sudo 配置錯誤
- 內核漏洞檢測
- 服務配置錯誤
- 計劃任務濫用
"""

from datetime import datetime
import hashlib
import platform
from typing import Any

from aiva_common.utils.logging import get_logger

logger = get_logger(__name__)


class AuthorizationError(Exception):
    """授權錯誤異常"""


class EnhancedPrivilegeAnalyzer:
    """增強權限分析器 - 實際檢測系統權限配置"""
    
    def __init__(self):
        self.analysis_results = []
        self.system_info = {}
    
    async def analyze_system_permissions(self) -> dict[str, Any]:
        """分析系統權限配置"""
        try:
            import os
            import platform
            import subprocess
            
            logger.info("開始分析系統權限配置")
            
            analysis_results = {
                'system_info': {
                    'platform': platform.system(),
                    'version': platform.version(),
                    'architecture': platform.architecture()[0]
                },
                'current_user': {
                    'username': os.getenv('USERNAME') or os.getenv('USER'),
                    'uid': os.getuid() if hasattr(os, 'getuid') else None,
                    'gid': os.getgid() if hasattr(os, 'getgid') else None
                },
                'vulnerabilities': [],
                'recommendations': []
            }
            
            # Windows系統檢測
            if platform.system() == 'Windows':
                try:
                    # 檢測當前用戶權限
                    result = subprocess.run(['whoami', '/priv'], capture_output=True, text=True, timeout=10)
                    if 'SeDebugPrivilege' in result.stdout:
                        analysis_results['vulnerabilities'].append({
                            'type': 'high_privilege_user',
                            'severity': 'high',
                            'description': '當前用戶具有調試權限'
                        })
                        
                except Exception as windows_error:
                    logger.debug(f"Windows檢測失敗: {windows_error}")
            
            # Linux/Unix系統檢測
            elif platform.system() in ['Linux', 'Darwin']:
                try:
                    # 檢測SUID文件
                    suid_result = subprocess.run(['find', '/', '-perm', '-4000', '-type', 'f', '2>/dev/null'], 
                                                capture_output=True, text=True, timeout=30)
                    suid_files = [f for f in suid_result.stdout.split('\n') if f.strip()]
                    if len(suid_files) > 20:
                        analysis_results['vulnerabilities'].append({
                            'type': 'excessive_suid_files',
                            'severity': 'medium',
                            'description': f'發現 {len(suid_files)} 個SUID文件',
                            'files': suid_files[:10]  # 只顯示前10個
                        })
                        
                except Exception as unix_error:
                    logger.debug(f"Unix檢測失敗: {unix_error}")
            
            # 生成建議
            if len(analysis_results['vulnerabilities']) == 0:
                analysis_results['recommendations'].append('系統權限配置目前看起來正常')
            else:
                analysis_results['recommendations'].append('建議檢查發現的權限問題並進行修復')
            
            analysis_results['analysis_time'] = datetime.now()
            analysis_results['status'] = 'completed'
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"系統權限分析失敗: {e}")
            return {'error': str(e), 'status': 'failed'}


class PrivilegeEscalator:
    """
    權限提升測試器

    ⚠️ 僅用於安全測試和研究
    """

    def __init__(
        self,
        authorization_token: str | None = None,
        log_all_actions: bool = True,
        safe_mode: bool = True,
    ):
        """
        初始化權限提升測試器

        Args:
            authorization_token: 授權令牌
            log_all_actions: 是否記錄所有操作
            safe_mode: 安全模式(僅模擬,不執行)
        """
        self.authorization_token = authorization_token
        self.log_all_actions = log_all_actions
        self.safe_mode = safe_mode
        self.test_results: list[dict[str, Any]] = []

        if not authorization_token:
            logger.warning(
                "privilege_escalator_no_auth",
                message="No authorization token provided - running in safe mode only",
            )
            self.safe_mode = True

        logger.info(
            "privilege_escalator_initialized",
            safe_mode=self.safe_mode,
            log_all_actions=self.log_all_actions,
        )

    def _check_authorization(self) -> bool:
        """檢查授權"""
        if not self.authorization_token:
            return False

        # 實際環境中應該驗證授權令牌
        # 這裡僅作示範
        return len(self.authorization_token) >= 32

    def _log_action(self, action: str, details: dict[str, Any]) -> None:
        """記錄操作"""
        if self.log_all_actions:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details,
                "safe_mode": self.safe_mode,
            }
            logger.info("privilege_escalation_action", **log_entry)

    def check_suid_binaries(self) -> dict[str, Any]:
        """
        檢查 SUID/SGID 二進制文件

        Returns:
            檢測結果
        """
        self._log_action("check_suid_binaries", {"os": platform.system()})

        result = {
            "test_name": "SUID/SGID Binary Check",
            "timestamp": datetime.now().isoformat(),
            "os": platform.system(),
            "findings": [],
        }

        # 真實的 SUID/SGID 檢測（僅讀取，不修改）
        result["mode"] = "detection"
        try:
            if platform.system() == "Linux":
                # 在 Linux 上檢查常見的危險 SUID 二進制文件
                dangerous_paths = [
                    "/usr/bin/passwd", "/usr/bin/sudo", "/usr/bin/su",
                    "/usr/bin/mount", "/usr/bin/umount", "/usr/bin/chsh",
                    "/usr/bin/newgrp", "/usr/bin/chfn", "/bin/ping"
                ]
                import os
                import stat
                for path in dangerous_paths:
                    if os.path.exists(path):
                        st = os.stat(path)
                        if st.st_mode & stat.S_ISUID or st.st_mode & stat.S_ISGID:
                            result["findings"].append({
                                "type": "suid_binary",
                                "path": path,
                                "permissions": oct(st.st_mode)[-4:],
                                "risk": "HIGH" if "sudo" in path or "su" in path else "MEDIUM",
                                "owner": st.st_uid,
                            })
            else:
                result["findings"].append({
                    "type": "not_applicable",
                    "message": "SUID check only applicable on Linux systems",
                    "risk": "INFO",
                })
        except Exception as e:
            result["error"] = f"SUID check failed: {str(e)}"

        self.test_results.append(result)
        logger.info("suid_check_completed", findings_count=len(result["findings"]))
        return result

    def check_sudo_misconfiguration(self) -> dict[str, Any]:
        """
        檢查 Sudo 配置錯誤

        Returns:
            檢測結果
        """
        self._log_action("check_sudo_misconfiguration", {})

        result = {
            "test_name": "Sudo Misconfiguration Check",
            "timestamp": datetime.now().isoformat(),
            "findings": [],
        }

        # 真實的 Sudo 配置檢測（僅讀取，不修改）
        result["mode"] = "detection"
        try:
            import subprocess
            if platform.system() == "Linux":
                # 嘗試讀取 sudo 配置
                try:
                    sudo_result = subprocess.run(
                        ["sudo", "-l"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if sudo_result.returncode == 0:
                        output = sudo_result.stdout
                        # 檢查危險配置
                        if "NOPASSWD" in output:
                            result["findings"].append({
                                "type": "sudo_nopasswd",
                                "message": "Found NOPASSWD entries in sudo config",
                                "risk": "HIGH",
                                "details": output[:500],
                            })
                        if "ALL" in output:
                            result["findings"].append({
                                "type": "sudo_all",
                                "message": "Found ALL permission in sudo config",
                                "risk": "CRITICAL",
                                "details": output[:500],
                            })
                    else:
                        result["findings"].append({
                            "type": "no_sudo_access",
                            "message": "Current user has no sudo privileges",
                            "risk": "INFO",
                        })
                except subprocess.TimeoutExpired:
                    result["error"] = "Sudo check timed out"
                except FileNotFoundError:
                    result["findings"].append({
                        "type": "sudo_not_installed",
                        "message": "Sudo command not found",
                        "risk": "INFO",
                    })
            else:
                result["findings"].append({
                    "type": "not_applicable",
                    "message": "Sudo check only applicable on Linux systems",
                    "risk": "INFO",
                })
        except Exception as e:
            result["error"] = f"Sudo check failed: {str(e)}"

        self.test_results.append(result)
        return result

    def check_kernel_exploits(self) -> dict[str, Any]:
        """
        檢查已知內核漏洞

        Returns:
            檢測結果
        """
        self._log_action("check_kernel_exploits", {})

        result = {
            "test_name": "Kernel Exploit Check",
            "timestamp": datetime.now().isoformat(),
            "kernel_version": platform.release(),
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["findings"].append({
                "type": "simulated",
                "message": "Kernel version check against known exploits would be performed",
                "kernel": platform.release(),
            })

        self.test_results.append(result)
        return result

    def check_writable_services(self) -> dict[str, Any]:
        """
        檢查可寫服務配置

        Returns:
            檢測結果
        """
        self._log_action("check_writable_services", {})

        result = {
            "test_name": "Writable Service Configuration Check",
            "timestamp": datetime.now().isoformat(),
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["findings"].append({
                "type": "simulated",
                "message": "Service configuration permissions would be checked",
            })

        self.test_results.append(result)
        return result

    def run_full_assessment(self) -> dict[str, Any]:
        """
        執行完整的權限提升評估

        Returns:
            完整評估結果
        """
        logger.info("privilege_escalation_assessment_started", safe_mode=self.safe_mode)

        assessment = {
            "assessment_id": hashlib.sha256(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "safe_mode": self.safe_mode,
            "os": platform.system(),
            "tests": [],
        }

        # 執行所有檢查
        assessment["tests"].append(self.check_suid_binaries())
        assessment["tests"].append(self.check_sudo_misconfiguration())
        assessment["tests"].append(self.check_kernel_exploits())
        assessment["tests"].append(self.check_writable_services())

        # 統計
        total_findings = sum(len(test.get("findings", [])) for test in assessment["tests"])
        assessment["summary"] = {
            "total_tests": len(assessment["tests"]),
            "total_findings": total_findings,
            "mode": "simulation" if self.safe_mode else "actual",
        }

        logger.info(
            "privilege_escalation_assessment_completed",
            total_tests=len(assessment["tests"]),
            total_findings=total_findings,
        )

        return assessment

    def get_results(self) -> list[dict[str, Any]]:
        """獲取所有測試結果"""
        return self.test_results

    def clear_results(self) -> None:
        """清除測試結果"""
        self.test_results.clear()
        logger.info("test_results_cleared")


def main():
    """測試範例 - 僅在安全模式下運行"""
    print("⚠️  Privilege Escalation Tester - SAFE MODE DEMO")
    print("=" * 60)

    # 僅在安全模式下測試
    escalator = PrivilegeEscalator(safe_mode=True)

    # 執行評估
    assessment = escalator.run_full_assessment()

    print(f"\n📊 Assessment ID: {assessment['assessment_id']}")
    print(f"🔒 Safe Mode: {assessment['safe_mode']}")
    print(f"💻 OS: {assessment['os']}")
    print(f"\n📋 Tests Run: {assessment['summary']['total_tests']}")
    print(f"🔍 Findings: {assessment['summary']['total_findings']}")

    print("\n✅ Safe mode demo completed")
    print("⚠️  Remember: Never use this tool without explicit authorization!")


if __name__ == "__main__":
    main()

