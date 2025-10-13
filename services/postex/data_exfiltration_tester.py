"""
Data Exfiltration Tester - 數據外洩測試器 (僅供授權測試使用)

⚠️ 嚴格警告: 此模組用於測試數據外洩防護
- 僅在獲得明確書面授權的環境中使用
- 所有操作必須記錄並審計
- 禁止在生產環境或未授權系統使用

測試場景:
- DNS 隧道檢測
- HTTP/HTTPS 外洩模擬
- 文件傳輸協議測試
- 加密外洩通道檢測
- 數據洩露點識別
"""

import base64
from datetime import datetime
import hashlib
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class DataExfiltrationTester:
    """
    數據外洩測試器

    ⚠️ 僅用於安全測試和研究
    """

    def __init__(
        self,
        authorization_token: str | None = None,
        safe_mode: bool = True,
        log_all_actions: bool = True,
    ):
        """
        初始化數據外洩測試器

        Args:
            authorization_token: 授權令牌
            safe_mode: 安全模式(僅模擬,不執行)
            log_all_actions: 是否記錄所有操作
        """
        self.authorization_token = authorization_token
        self.safe_mode = safe_mode
        self.log_all_actions = log_all_actions
        self.test_results: list[dict[str, Any]] = []

        if not authorization_token:
            logger.warning(
                "data_exfiltration_no_auth",
                message="No authorization token provided - running in safe mode only",
            )
            self.safe_mode = True

        logger.info(
            "data_exfiltration_initialized",
            safe_mode=self.safe_mode,
        )

    def _log_action(self, action: str, details: dict[str, Any]) -> None:
        """記錄操作"""
        if self.log_all_actions:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details,
                "safe_mode": self.safe_mode,
            }
            logger.info("data_exfiltration_action", **log_entry)

    def test_dns_tunneling(self, domain: str) -> dict[str, Any]:
        """
        測試 DNS 隧道外洩

        Args:
            domain: 測試域名

        Returns:
            測試結果
        """
        self._log_action("test_dns_tunneling", {"domain": domain})

        result = {
            "test_name": "DNS Tunneling Test",
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            # 模擬 DNS 查詢
            sample_data = "sensitive_data_sample"
            encoded = base64.b32encode(sample_data.encode()).decode().lower()
            result["findings"].append({
                "type": "simulated",
                "method": "DNS Tunneling",
                "encoded_query": f"{encoded}.{domain}",
                "note": "This would send data via DNS queries",
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        logger.info("dns_tunneling_test_completed")
        return result

    def test_http_exfiltration(
        self,
        target_url: str,
        data_size: int = 1024,
    ) -> dict[str, Any]:
        """
        測試 HTTP/HTTPS 外洩

        Args:
            target_url: 目標 URL
            data_size: 測試數據大小(字節)

        Returns:
            測試結果
        """
        self._log_action(
            "test_http_exfiltration",
            {"target_url": target_url, "data_size": data_size},
        )

        result = {
            "test_name": "HTTP Exfiltration Test",
            "timestamp": datetime.now().isoformat(),
            "target_url": target_url,
            "data_size": data_size,
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["findings"].append({
                "type": "simulated",
                "method": "HTTP POST",
                "target": target_url,
                "data_size": data_size,
                "note": "Would send data via HTTP POST request",
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def test_file_transfer(
        self,
        protocol: str,
        target: str,
    ) -> dict[str, Any]:
        """
        測試文件傳輸協議外洩

        Args:
            protocol: 協議 (ftp, sftp, scp, smb)
            target: 目標地址

        Returns:
            測試結果
        """
        self._log_action(
            "test_file_transfer",
            {"protocol": protocol, "target": target},
        )

        result = {
            "test_name": "File Transfer Protocol Test",
            "timestamp": datetime.now().isoformat(),
            "protocol": protocol.upper(),
            "target": target,
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            result["findings"].append({
                "type": "simulated",
                "method": f"{protocol.upper()} Transfer",
                "target": target,
                "note": f"Would transfer file via {protocol.upper()}",
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def test_encrypted_channel(
        self,
        method: str = "base64",
    ) -> dict[str, Any]:
        """
        測試加密外洩通道

        Args:
            method: 加密方法

        Returns:
            測試結果
        """
        self._log_action("test_encrypted_channel", {"method": method})

        result = {
            "test_name": "Encrypted Channel Test",
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "findings": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            sample_data = "sensitive_info"

            if method == "base64":
                encoded = base64.b64encode(sample_data.encode()).decode()
            elif method == "hex":
                encoded = sample_data.encode().hex()
            else:
                encoded = "encrypted_data"

            result["findings"].append({
                "type": "simulated",
                "method": f"{method.upper()} Encoding",
                "encoded_sample": encoded[:50] + "...",
                "note": "Data would be encoded before exfiltration",
            })
        else:
            result["mode"] = "actual"
            result["error"] = "Actual execution requires authorization"

        self.test_results.append(result)
        return result

    def identify_exfiltration_points(self) -> dict[str, Any]:
        """
        識別可能的數據外洩點

        Returns:
            識別結果
        """
        self._log_action("identify_exfiltration_points", {})

        result = {
            "test_name": "Exfiltration Point Identification",
            "timestamp": datetime.now().isoformat(),
            "points": [],
        }

        if self.safe_mode:
            result["mode"] = "simulation"
            # 模擬識別常見外洩點
            common_points = [
                {
                    "type": "Network",
                    "description": "Outbound HTTP/HTTPS connections",
                    "risk": "HIGH",
                },
                {
                    "type": "Network",
                    "description": "DNS queries to external resolvers",
                    "risk": "MEDIUM",
                },
                {
                    "type": "Storage",
                    "description": "USB device connections",
                    "risk": "HIGH",
                },
                {
                    "type": "Cloud",
                    "description": "Cloud storage API access",
                    "risk": "MEDIUM",
                },
                {
                    "type": "Email",
                    "description": "Email attachments",
                    "risk": "MEDIUM",
                },
            ]
            result["points"].extend(common_points)

        self.test_results.append(result)
        logger.info("exfiltration_points_identified", count=len(result["points"]))
        return result

    def run_full_assessment(self) -> dict[str, Any]:
        """
        執行完整的數據外洩評估

        Returns:
            完整評估結果
        """
        logger.info("data_exfiltration_assessment_started", safe_mode=self.safe_mode)

        assessment = {
            "assessment_id": hashlib.sha256(
                f"{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "safe_mode": self.safe_mode,
            "tests": [],
        }

        # 執行所有檢查
        assessment["tests"].append(self.test_dns_tunneling("test.example.com"))
        assessment["tests"].append(
            self.test_http_exfiltration("https://test.example.com/upload")
        )
        assessment["tests"].append(self.test_file_transfer("sftp", "sftp.example.com"))
        assessment["tests"].append(self.test_encrypted_channel("base64"))
        assessment["tests"].append(self.identify_exfiltration_points())

        # 統計
        total_findings = sum(
            len(test.get("findings", []) + test.get("points", []))
            for test in assessment["tests"]
        )
        assessment["summary"] = {
            "total_tests": len(assessment["tests"]),
            "total_findings": total_findings,
            "mode": "simulation" if self.safe_mode else "actual",
        }

        logger.info(
            "data_exfiltration_assessment_completed",
            total_tests=len(assessment["tests"]),
            total_findings=total_findings,
        )

        return assessment

    def get_results(self) -> list[dict[str, Any]]:
        """獲取所有測試結果"""
        return self.test_results


def main():
    """測試範例 - 僅在安全模式下運行"""
    print("⚠️  Data Exfiltration Tester - SAFE MODE DEMO")
    print("=" * 60)

    # 僅在安全模式下測試
    tester = DataExfiltrationTester(safe_mode=True)

    # 執行評估
    assessment = tester.run_full_assessment()

    print(f"\n📊 Assessment ID: {assessment['assessment_id']}")
    print(f"🔒 Safe Mode: {assessment['safe_mode']}")
    print(f"\n📋 Tests Run: {assessment['summary']['total_tests']}")
    print(f"🔍 Findings: {assessment['summary']['total_findings']}")

    print("\n✅ Safe mode demo completed")
    print("⚠️  Remember: Never use this tool without explicit authorization!")


if __name__ == "__main__":
    main()
