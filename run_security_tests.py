#!/usr/bin/env python3
"""
統一安全測試運行器
Unified Security Testing Runner

支援:
- 權限提升與越權測試 (IDOR)
- 認證安全測試 (Auth)
- CORS 安全測試
- 依賴分析 (SCA)
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.function.function_idor.aiva_func_idor.privilege_escalation_tester import (
        PrivilegeEscalationTester,
        TestUser,
        EscalationType
    )
    from services.function.function_authn_go.internal.auth_cors_tester.auth_cors_tester import (
        AuthenticationTester,
        CORSTester
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 部分模組無法載入: {e}")
    MODULES_AVAILABLE = False


logger = logging.getLogger(__name__)


class UnifiedSecurityTester:
    """統一安全測試器"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.target_url = config.get("target_url", "")
        self.results: dict[str, Any] = {
            "idor": [],
            "auth": [],
            "cors": [],
            "sca": {}
        }

    async def run_all_tests(self):
        """運行所有測試"""
        logger.info("開始執行完整安全測試套件")

        # 1. IDOR 測試
        if self.config.get("enable_idor", True):
            await self.run_idor_tests()

        # 2. 認證測試
        if self.config.get("enable_auth", True):
            await self.run_auth_tests()

        # 3. CORS 測試
        if self.config.get("enable_cors", True):
            await self.run_cors_tests()

        # 4. SCA 測試
        if self.config.get("enable_sca", False):
            self.run_sca_tests()

        # 生成綜合報告
        self.generate_comprehensive_report()

        return self.results

    async def run_idor_tests(self):
        """執行 IDOR 測試"""
        logger.info("=" * 60)
        logger.info("開始 IDOR / 權限提升測試")
        logger.info("=" * 60)

        if not MODULES_AVAILABLE:
            logger.error("IDOR 模組未載入，跳過測試")
            return

        # 從配置讀取用戶信息
        users = self._load_test_users()
        if len(users) < 2:
            logger.warning("需要至少 2 個測試用戶進行 IDOR 測試")
            return

        attacker = users[0]
        victim = users[1] if len(users) > 1 else users[0]
        admin = users[2] if len(users) > 2 else users[0]

        async with PrivilegeEscalationTester(self.target_url) as tester:
            # 測試水平越權
            if self.config.get("test_horizontal_escalation", True):
                logger.info("測試水平越權...")
                for endpoint in self.config.get("horizontal_test_endpoints", []):
                    try:
                        finding = await tester.test_horizontal_escalation(
                            attacker=attacker,
                            victim=victim,
                            target_url=f"{self.target_url}{endpoint}",
                            method=self.config.get("test_method", "GET")
                        )
                        self.results["idor"].append(finding)
                        logger.info(
                            f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} - "
                            f"{endpoint} (CVSS: {finding.cvss_score})"
                        )
                    except Exception as e:
                        logger.error(f"  ❌ 測試失敗: {endpoint} - {e}")

            # 測試垂直越權
            if self.config.get("test_vertical_escalation", True):
                logger.info("測試垂直越權...")
                for endpoint in self.config.get("vertical_test_endpoints", []):
                    try:
                        finding = await tester.test_vertical_escalation(
                            low_priv_user=attacker,
                            high_priv_user=admin,
                            admin_url=f"{self.target_url}{endpoint}",
                            method=self.config.get("test_method", "GET")
                        )
                        self.results["idor"].append(finding)
                        logger.info(
                            f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} - "
                            f"{endpoint} (CVSS: {finding.cvss_score})"
                        )
                    except Exception as e:
                        logger.error(f"  ❌ 測試失敗: {endpoint} - {e}")

            # 測試資源枚舉
            if self.config.get("test_enumeration", True):
                logger.info("測試資源枚舉...")
                enum_config = self.config.get("enumeration_config", {})
                try:
                    finding = await tester.test_resource_enumeration(
                        user=attacker,
                        base_url=f"{self.target_url}{enum_config.get('endpoint', '/api/user')}",
                        id_param=enum_config.get("id_param", "id"),
                        id_range=tuple(enum_config.get("id_range", [1, 100])),
                        method=self.config.get("test_method", "GET")
                    )
                    self.results["idor"].append(finding)
                    accessible = finding.evidence.get("accessible_count", 0)
                    logger.info(
                        f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} - "
                        f"可訪問資源: {accessible} (CVSS: {finding.cvss_score})"
                    )
                except Exception as e:
                    logger.error(f"  ❌ 枚舉測試失敗: {e}")

            # 保存 IDOR 報告
            output_path = self.config.get("idor_report_path", "idor_test_report.json")
            tester.generate_report(output_path)
            logger.info(f"✅ IDOR 報告已生成: {output_path}")

    async def run_auth_tests(self):
        """執行認證測試"""
        logger.info("=" * 60)
        logger.info("開始認證安全測試")
        logger.info("=" * 60)

        if not MODULES_AVAILABLE:
            logger.error("認證模組未載入，跳過測試")
            return

        async with AuthenticationTester(self.target_url) as tester:
            # 弱密碼測試
            if self.config.get("test_weak_password", True):
                logger.info("測試弱密碼策略...")
                try:
                    finding = await tester.test_weak_password_policy(
                        register_url=f"{self.target_url}{self.config.get('register_endpoint', '/api/register')}",
                        weak_passwords=self.config.get("weak_passwords")
                    )
                    self.results["auth"].append(finding)
                    logger.info(
                        f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} "
                        f"(CVSS: {finding.cvss_score})"
                    )
                except Exception as e:
                    logger.error(f"  ❌ 測試失敗: {e}")

            # 暴力破解防護測試
            if self.config.get("test_brute_force", True):
                logger.info("測試暴力破解防護...")
                try:
                    finding = await tester.test_brute_force_protection(
                        login_url=f"{self.target_url}{self.config.get('login_endpoint', '/api/login')}",
                        username=self.config.get("test_username", "test_user"),
                        max_attempts=self.config.get("max_brute_force_attempts", 20)
                    )
                    self.results["auth"].append(finding)
                    logger.info(
                        f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} "
                        f"(CVSS: {finding.cvss_score})"
                    )
                except Exception as e:
                    logger.error(f"  ❌ 測試失敗: {e}")

            # JWT 安全測試
            if self.config.get("test_jwt", False) and self.config.get("jwt_token"):
                logger.info("測試 JWT 安全性...")
                try:
                    finding = await tester.test_jwt_security(
                        token=self.config["jwt_token"],
                        api_url=f"{self.target_url}{self.config.get('jwt_test_endpoint', '/api/protected')}"
                    )
                    self.results["auth"].append(finding)
                    logger.info(
                        f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} "
                        f"(CVSS: {finding.cvss_score})"
                    )
                except Exception as e:
                    logger.error(f"  ❌ 測試失敗: {e}")

            # Session Fixation 測試
            if self.config.get("test_session_fixation", True):
                logger.info("測試 Session Fixation...")
                try:
                    users = self._load_test_users()
                    if users:
                        finding = await tester.test_session_fixation(
                            login_url=f"{self.target_url}{self.config.get('login_endpoint', '/api/login')}",
                            username=users[0].username,
                            password=self.config.get("test_password", "test_password")
                        )
                        self.results["auth"].append(finding)
                        logger.info(
                            f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} "
                            f"(CVSS: {finding.cvss_score})"
                        )
                except Exception as e:
                    logger.error(f"  ❌ 測試失敗: {e}")

            # 保存認證報告
            output_path = self.config.get("auth_report_path", "auth_test_report.json")
            tester.generate_report(output_path)
            logger.info(f"✅ 認證報告已生成: {output_path}")

    async def run_cors_tests(self):
        """執行 CORS 測試"""
        logger.info("=" * 60)
        logger.info("開始 CORS 安全測試")
        logger.info("=" * 60)

        if not MODULES_AVAILABLE:
            logger.error("CORS 模組未載入，跳過測試")
            return

        async with CORSTester(self.target_url) as tester:
            test_endpoints = self.config.get(
                "cors_test_endpoints",
                ["/api/data", "/api/user"]
            )

            for endpoint in test_endpoints:
                url = f"{self.target_url}{endpoint}"

                # Null Origin 測試
                if self.config.get("test_null_origin", True):
                    logger.info(f"測試 Null Origin - {endpoint}...")
                    try:
                        finding = await tester.test_null_origin(url)
                        self.results["cors"].append(finding)
                        logger.info(
                            f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} "
                            f"(CVSS: {finding.cvss_score})"
                        )
                    except Exception as e:
                        logger.error(f"  ❌ 測試失敗: {e}")

                # Wildcard + Credentials 測試
                if self.config.get("test_wildcard_credentials", True):
                    logger.info(f"測試 Wildcard + Credentials - {endpoint}...")
                    try:
                        finding = await tester.test_wildcard_with_credentials(url)
                        self.results["cors"].append(finding)
                        logger.info(
                            f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} "
                            f"(CVSS: {finding.cvss_score})"
                        )
                    except Exception as e:
                        logger.error(f"  ❌ 測試失敗: {e}")

                # Reflected Origin 測試
                if self.config.get("test_reflected_origin", True):
                    logger.info(f"測試 Reflected Origin - {endpoint}...")
                    try:
                        finding = await tester.test_reflected_origin(
                            url,
                            test_origins=self.config.get("test_origins")
                        )
                        self.results["cors"].append(finding)
                        logger.info(
                            f"  {'❌ 發現漏洞' if finding.vulnerable else '✅ 安全'} "
                            f"(CVSS: {finding.cvss_score})"
                        )
                    except Exception as e:
                        logger.error(f"  ❌ 測試失敗: {e}")

            # 保存 CORS 報告
            output_path = self.config.get("cors_report_path", "cors_test_report.json")
            tester.generate_report(output_path)
            logger.info(f"✅ CORS 報告已生成: {output_path}")

    def run_sca_tests(self):
        """執行 SCA 測試"""
        logger.info("=" * 60)
        logger.info("開始 SCA 依賴分析測試")
        logger.info("=" * 60)
        logger.info("⚠️  SCA 測試需要 Go 環境，請直接運行 Go 程序")
        logger.info(f"    cd services/function/function_sca_go && go run cmd/worker/main.go")

    def _load_test_users(self) -> list[TestUser]:
        """載入測試用戶"""
        users_config = self.config.get("test_users", [])
        users = []

        for user_cfg in users_config:
            user = TestUser(
                user_id=user_cfg.get("user_id", ""),
                username=user_cfg.get("username", ""),
                role=user_cfg.get("role", "user"),
                token=user_cfg.get("token"),
                session=user_cfg.get("session"),
                cookies=user_cfg.get("cookies", {}),
                headers=user_cfg.get("headers", {})
            )
            users.append(user)

        return users

    def generate_comprehensive_report(self):
        """生成綜合報告"""
        logger.info("=" * 60)
        logger.info("生成綜合安全測試報告")
        logger.info("=" * 60)

        total_tests = (
            len(self.results["idor"]) +
            len(self.results["auth"]) +
            len(self.results["cors"])
        )

        vulnerable_tests = sum(
            1 for finding in (
                self.results["idor"] +
                self.results["auth"] +
                self.results["cors"]
            )
            if hasattr(finding, 'vulnerable') and finding.vulnerable
        )

        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "vulnerable_tests": vulnerable_tests,
                "safe_tests": total_tests - vulnerable_tests,
                "vulnerability_rate": f"{(vulnerable_tests / total_tests * 100):.1f}%"
                if total_tests > 0 else "0%"
            },
            "by_category": {
                "idor": {
                    "total": len(self.results["idor"]),
                    "vulnerable": sum(1 for f in self.results["idor"] if f.vulnerable)
                },
                "auth": {
                    "total": len(self.results["auth"]),
                    "vulnerable": sum(1 for f in self.results["auth"] if f.vulnerable)
                },
                "cors": {
                    "total": len(self.results["cors"]),
                    "vulnerable": sum(1 for f in self.results["cors"] if f.vulnerable)
                }
            },
            "critical_findings": self._get_critical_findings()
        }

        # 保存綜合報告
        output_path = self.config.get("comprehensive_report_path", "comprehensive_security_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 綜合報告已生成: {output_path}")
        logger.info(f"📊 總測試數: {total_tests}")
        logger.info(f"❌ 發現漏洞: {vulnerable_tests}")
        logger.info(f"✅ 安全測試: {total_tests - vulnerable_tests}")

    def _get_critical_findings(self) -> list[dict[str, Any]]:
        """獲取關鍵發現"""
        critical = []

        for finding in (self.results["idor"] + self.results["auth"] + self.results["cors"]):
            if hasattr(finding, 'vulnerable') and finding.vulnerable:
                if hasattr(finding, 'cvss_score') and finding.cvss_score >= 7.0:
                    critical.append({
                        "test_id": finding.test_id if hasattr(finding, 'test_id') else "unknown",
                        "severity": finding.severity if hasattr(finding, 'severity') else "UNKNOWN",
                        "cvss_score": finding.cvss_score if hasattr(finding, 'cvss_score') else 0.0,
                        "description": finding.description if hasattr(finding, 'description') else ""
                    })

        return sorted(critical, key=lambda x: x["cvss_score"], reverse=True)


def load_config(config_path: str) -> dict[str, Any]:
    """載入配置文件"""
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}，使用默認配置")
        return get_default_config()
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式錯誤: {e}")
        sys.exit(1)


def get_default_config() -> dict[str, Any]:
    """獲取默認配置"""
    return {
        "target_url": "http://localhost:3000",
        "enable_idor": True,
        "enable_auth": True,
        "enable_cors": True,
        "enable_sca": False,
        "test_horizontal_escalation": True,
        "test_vertical_escalation": True,
        "test_enumeration": True,
        "horizontal_test_endpoints": [
            "/api/user/profile",
            "/api/user/orders"
        ],
        "vertical_test_endpoints": [
            "/admin/dashboard",
            "/admin/users"
        ],
        "enumeration_config": {
            "endpoint": "/api/user/profile",
            "id_param": "user_id",
            "id_range": [1, 100]
        },
        "test_users": [
            {
                "user_id": "123",
                "username": "alice",
                "role": "user",
                "token": "token_alice"
            },
            {
                "user_id": "456",
                "username": "bob",
                "role": "user",
                "token": "token_bob"
            },
            {
                "user_id": "789",
                "username": "admin",
                "role": "admin",
                "token": "token_admin"
            }
        ]
    }


async def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="統一安全測試運行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默認配置
  python run_security_tests.py
  
  # 使用自定義配置
  python run_security_tests.py --config my_config.json
  
  # 只運行特定測試
  python run_security_tests.py --only-idor
  python run_security_tests.py --only-auth
  python run_security_tests.py --only-cors
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="security_test_config.json",
        help="配置文件路徑 (默認: security_test_config.json)"
    )

    parser.add_argument(
        "--only-idor",
        action="store_true",
        help="只運行 IDOR 測試"
    )

    parser.add_argument(
        "--only-auth",
        action="store_true",
        help="只運行認證測試"
    )

    parser.add_argument(
        "--only-cors",
        action="store_true",
        help="只運行 CORS 測試"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細輸出"
    )

    args = parser.parse_args()

    # 設置日誌
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 載入配置
    config = load_config(args.config)

    # 根據參數調整配置
    if args.only_idor:
        config["enable_idor"] = True
        config["enable_auth"] = False
        config["enable_cors"] = False
    elif args.only_auth:
        config["enable_idor"] = False
        config["enable_auth"] = True
        config["enable_cors"] = False
    elif args.only_cors:
        config["enable_idor"] = False
        config["enable_auth"] = False
        config["enable_cors"] = True

    # 運行測試
    tester = UnifiedSecurityTester(config)
    try:
        results = await tester.run_all_tests()
        logger.info("🎉 所有測試完成！")
        return 0
    except KeyboardInterrupt:
        logger.warning("⚠️  測試被用戶中斷")
        return 130
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
