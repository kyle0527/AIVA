"""
Lateral Movement Tester - 橫向移動測試器 (僅供授權測試使用)

⚠️ 嚴格警告: 此模組用於測試橫向移動漏洞
- 僅在獲得明確書面授權的環境中使用
- 所有操作必須記錄並審計
- 禁止在生產環境或未授權系統使用

測試場景:
- 網絡掃描和探測
- 遠程服務枚舉
- 憑證重用檢測
- Pass-the-Hash 模擬
- RDP/SSH 訪問測試
"""

from datetime import datetime
import hashlib
import socket
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class LateralMovementTester:
    """
    橫向移動測試器

    ⚠️ 僅用於安全測試和研究
    """

    def __init__(
        self,
        authorization_token: str | None = None,
        target_network: str | None = None,
        safe_mode: bool = True,
    ):
        """
        初始化橫向移動測試器

        Args:
            authorization_token: 授權令牌
            target_network: 目標網絡 (CIDR)
            safe_mode: 安全模式(僅模擬,不執行)
        """
        self.authorization_token = authorization_token
        self.target_network = target_network
        self.safe_mode = safe_mode
        self.test_results: list[dict[str, Any]] = []

        if not authorization_token:
            logger.warning(
                "lateral_movement_no_auth",
                message="No authorization token provided - running in safe mode only",
            )
            self.safe_mode = True

        logger.info(
            "lateral_movement_initialized",
            safe_mode=self.safe_mode,
            target_network=self.target_network,
        )

    def _log_action(self, action: str, details: dict[str, Any]) -> None:
        """記錄操作"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "safe_mode": self.safe_mode,
        }
        logger.info("lateral_movement_action", **log_entry)

    def scan_network(self, network: str | None = None) -> dict[str, Any]:
        """
        掃描網絡中的活躍主機

        Args:
            network: 目標網絡 (CIDR)

        Returns:
            掃描結果
        """
        target = network or self.target_network or "127.0.0.1/32"
        self._log_action("scan_network", {"target": target})

        result = {
            "test_name": "Network Scan",
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "hosts": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["hosts"].append({
                "ip": "127.0.0.1",
                "status": "up",
                "hostname": socket.gethostname(),
                "note": "Simulated host in safe mode",
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        logger.info("network_scan_completed", hosts_found=len(result["hosts"]))
        return result

    def enumerate_services(self, target_host: str) -> dict[str, Any]:
        """
        枚舉目標主機服務

        Args:
            target_host: 目標主機 IP

        Returns:
            服務枚舉結果
        """
        self._log_action("enumerate_services", {"target": target_host})

        result = {
            "test_name": "Service Enumeration",
            "timestamp": datetime.now().isoformat(),
            "target": target_host,
            "services": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["services"].extend([
                {"port": 22, "service": "ssh", "state": "open", "simulated": True},
                {"port": 80, "service": "http", "state": "open", "simulated": True},
                {"port": 443, "service": "https", "state": "open", "simulated": True},
            ])
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def test_credential_reuse(
        self,
        username: str,
        target_hosts: list[str],
    ) -> dict[str, Any]:
        """
        測試憑證重用

        Args:
            username: 用戶名
            target_hosts: 目標主機列表

        Returns:
            測試結果
        """
        self._log_action(
            "test_credential_reuse",
            {"username": username, "target_count": len(target_hosts)},
        )

        result = {
            "test_name": "Credential Reuse Test",
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "targets": target_hosts,
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["findings"].append({
                "type": "simulated",
                "message": "Credential reuse testing would be performed here",
                "note": "Tests SSH, RDP, SMB credential reuse",
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def simulate_pass_the_hash(self) -> dict[str, Any]:
        """
        模擬 Pass-the-Hash 攻擊

        Returns:
            模擬結果
        """
        self._log_action("simulate_pass_the_hash", {})

        result = {
            "test_name": "Pass-the-Hash Simulation",
            "timestamp": datetime.now().isoformat(),
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["findings"].append({
                "type": "simulated",
                "message": "Pass-the-Hash vulnerability testing would be performed",
                "protocols": ["NTLM", "SMB", "WMI"],
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def test_remote_access(
        self,
        target_host: str,
        protocols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        測試遠程訪問協議

        Args:
            target_host: 目標主機
            protocols: 測試協議列表

        Returns:
            測試結果
        """
        protocols = protocols or ["ssh", "rdp", "vnc"]
        self._log_action(
            "test_remote_access",
            {"target": target_host, "protocols": protocols},
        )

        result = {
            "test_name": "Remote Access Test",
            "timestamp": datetime.now().isoformat(),
            "target": target_host,
            "protocols": protocols,
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            for protocol in protocols:
                result["findings"].append({
                    "protocol": protocol,
                    "status": "simulated",
                    "message": f"{protocol.upper()} access would be tested",
                })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def run_full_assessment(self) -> dict[str, Any]:
        """
        執行完整的橫向移動評估

        Returns:
            完整評估結果
        """
        logger.info("lateral_movement_assessment_started", safe_mode=self.safe_mode)

        assessment = {
            "assessment_id": hashlib.sha256(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "safe_mode": self.safe_mode,
            "target_network": self.target_network,
            "tests": [],
        }

        # 執行所有檢查
        assessment["tests"].append(self.scan_network())
        assessment["tests"].append(self.enumerate_services("127.0.0.1"))
        assessment["tests"].append(
            self.test_credential_reuse("testuser", ["127.0.0.1"])
        )
        assessment["tests"].append(self.simulate_pass_the_hash())
        assessment["tests"].append(self.test_remote_access("127.0.0.1"))

        # 統計
        total_findings = sum(
            len(test.get("findings", []) + test.get("hosts", []) + test.get("services", []))
            for test in assessment["tests"]
        )
        assessment["summary"] = {
            "total_tests": len(assessment["tests"]),
            "total_findings": total_findings,
            "mode": "simulation" if self.safe_mode else "actual",
        }

        logger.info(
            "lateral_movement_assessment_completed",
            total_tests=len(assessment["tests"]),
            total_findings=total_findings,
        )

        return assessment

    def get_results(self) -> list[dict[str, Any]]:
        """獲取所有測試結果"""
        return self.test_results


def main():
    """測試範例 - 僅在安全模式下運行"""
    print("⚠️  Lateral Movement Tester - SAFE MODE DEMO")
    print("=" * 60)

    # 僅在安全模式下測試
    tester = LateralMovementTester(safe_mode=True, target_network="127.0.0.1/32")

    # 執行評估
    assessment = tester.run_full_assessment()

    print(f"\n📊 Assessment ID: {assessment['assessment_id']}")
    print(f"🔒 Safe Mode: {assessment['safe_mode']}")
    print(f"🌐 Target Network: {assessment['target_network']}")
    print(f"\n📋 Tests Run: {assessment['summary']['total_tests']}")
    print(f"🔍 Findings: {assessment['summary']['total_findings']}")

    print("\n✅ Safe mode demo completed")
    print("⚠️  Remember: Never use this tool without explicit authorization!")


if __name__ == "__main__":
    main()
