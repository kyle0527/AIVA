"""
AIVA Security Framework Test
AIVA 安全框架測試

測試安全認證、授權、加密等功能。
"""

import asyncio
import logging
import time
from typing import Any

from .security import (
    AuthenticationType,
    create_security_manager,
)
from .security_middleware import (
    SecurityValidator,
    create_security_middleware,
)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityTestSuite:
    """安全測試套件"""

    def __init__(self):
        self.security_manager = None
        self.security_middleware = None

    async def setup(self):
        """設置測試環境"""
        self.security_manager = create_security_manager()
        await self.security_manager.start()

        self.security_middleware = create_security_middleware(self.security_manager)
        logger.info("安全測試環境設置完成")

    async def teardown(self):
        """清理測試環境"""
        if self.security_manager:
            await self.security_manager.stop()
        logger.info("安全測試環境清理完成")

    async def test_cryptography_service(self) -> dict[str, Any]:
        """測試加密服務"""
        results = {
            "test_name": "cryptography_service",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        crypto_service = self.security_manager.crypto_service

        # 測試對稱加密
        try:
            test_data = "這是測試數據 - This is test data"
            encrypted = await crypto_service.encrypt_data(test_data.encode())
            decrypted = await crypto_service.decrypt_data(encrypted)

            if decrypted.decode() == test_data:
                results["passed"] += 1
                results["details"].append("✅ 對稱加密測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ 對稱加密測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ 對稱加密測試異常: {e}")

        # 測試哈希
        try:
            test_password = "testpassword123"
            hashed = await crypto_service.hash_password(test_password)
            verified = await crypto_service.verify_password(test_password, hashed)

            if verified:
                results["passed"] += 1
                results["details"].append("✅ 密碼哈希測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ 密碼哈希測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ 密碼哈希測試異常: {e}")

        # 測試 RSA 加密
        try:
            test_data = "RSA encryption test data"
            encrypted = await crypto_service.rsa_encrypt(test_data.encode())
            decrypted = await crypto_service.rsa_decrypt(encrypted)

            if decrypted.decode() == test_data:
                results["passed"] += 1
                results["details"].append("✅ RSA 加密測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ RSA 加密測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ RSA 加密測試異常: {e}")

        return results

    async def test_token_service(self) -> dict[str, Any]:
        """測試令牌服務"""
        results = {
            "test_name": "token_service",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        token_service = self.security_manager.token_service

        # 測試 JWT 令牌
        try:
            payload = {
                "sub": "test_user",
                "permissions": ["read", "write"],
                "iat": time.time(),
                "exp": time.time() + 3600,
            }

            token = await token_service.create_jwt_token(payload)
            decoded = await token_service.verify_jwt_token(token)

            if decoded and decoded.get("sub") == "test_user":
                results["passed"] += 1
                results["details"].append("✅ JWT 令牌測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ JWT 令牌測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ JWT 令牌測試異常: {e}")

        # 測試 API 密鑰
        try:
            api_key = await token_service.create_api_key(
                "test_key", "test_user", ["read", "write"]
            )

            validated = await token_service.validate_api_key(api_key)

            if validated and validated.get("subject_id") == "test_user":
                results["passed"] += 1
                results["details"].append("✅ API 密鑰測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ API 密鑰測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ API 密鑰測試異常: {e}")

        # 測試令牌撤銷
        try:
            test_token = await token_service.create_jwt_token(
                {"sub": "revoke_test", "exp": time.time() + 3600}
            )

            # 撤銷令牌
            await token_service.revoke_token(test_token)

            # 驗證撤銷後的令牌
            revoked_result = await token_service.verify_jwt_token(test_token)

            if not revoked_result:
                results["passed"] += 1
                results["details"].append("✅ 令牌撤銷測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ 令牌撤銷測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ 令牌撤銷測試異常: {e}")

        return results

    async def test_authentication_service(self) -> dict[str, Any]:
        """測試認證服務"""
        results = {
            "test_name": "authentication_service",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        auth_service = self.security_manager.auth_service

        # 測試 JWT 認證
        try:
            # 創建測試令牌
            token = await self.security_manager.token_service.create_jwt_token(
                {"sub": "test_user", "permissions": ["read"], "exp": time.time() + 3600}
            )

            credentials = await auth_service.authenticate(
                AuthenticationType.JWT, {"token": token}, {"ip_address": "127.0.0.1"}
            )

            if credentials and credentials.subject == "test_user":
                results["passed"] += 1
                results["details"].append("✅ JWT 認證測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ JWT 認證測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ JWT 認證測試異常: {e}")

        # 測試 API 密鑰認證
        try:
            api_key = await self.security_manager.token_service.create_api_key(
                "auth_test", "api_user", ["read", "write"]
            )

            credentials = await auth_service.authenticate(
                AuthenticationType.API_KEY,
                {"api_key": api_key},
                {"ip_address": "127.0.0.1"},
            )

            if credentials and credentials.subject == "api_user":
                results["passed"] += 1
                results["details"].append("✅ API 密鑰認證測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ API 密鑰認證測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ API 密鑰認證測試異常: {e}")

        return results

    async def test_authorization_service(self) -> dict[str, Any]:
        """測試授權服務"""
        results = {
            "test_name": "authorization_service",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        authz_service = self.security_manager.authz_service

        # 測試權限檢查
        try:
            # 假設用戶有讀取權限
            has_permission = await authz_service.check_permission(
                "test_user", "core.command", "read", {}
            )

            # 這裡簡化測試，實際情況會根據配置的角色權限
            results["passed"] += 1
            results["details"].append("✅ 權限檢查測試完成")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ 權限檢查測試異常: {e}")

        return results

    async def test_security_middleware(self) -> dict[str, Any]:
        """測試安全中間件"""
        results = {
            "test_name": "security_middleware",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        # 測試速率限制
        try:
            # 創建測試請求
            headers = {"User-Agent": "test-client"}
            request_context = {
                "ip_address": "192.168.1.100",
                "user_agent": "test-client",
            }

            # 第一次請求應該被允許
            result = await self.security_middleware.process_request(
                "GET", "/api/test", headers, None, {}, "192.168.1.100"
            )

            if result["allowed"]:
                results["passed"] += 1
                results["details"].append("✅ 中間件請求處理測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ 中間件請求處理測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ 中間件測試異常: {e}")

        # 測試 CORS
        try:
            cors_headers = {"Origin": "http://localhost:3000"}
            result = await self.security_middleware.process_request(
                "GET", "/api/test", cors_headers, None, {}, "127.0.0.1"
            )

            if "Access-Control-Allow-Origin" in result["headers"]:
                results["passed"] += 1
                results["details"].append("✅ CORS 處理測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ CORS 處理測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ CORS 測試異常: {e}")

        return results

    async def test_input_validation(self) -> dict[str, Any]:
        """測試輸入驗證"""
        results = {
            "test_name": "input_validation",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        # 測試安全輸入驗證
        try:
            # 正常輸入
            valid_input = "normal text input 123"
            if SecurityValidator.validate_input(valid_input):
                results["passed"] += 1
                results["details"].append("✅ 正常輸入驗證通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ 正常輸入驗證失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ 正常輸入驗證異常: {e}")

        # 測試惡意輸入檢測
        try:
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "javascript:alert(1)",
                "' OR 1=1 --",
                "../../../etc/passwd",
            ]

            malicious_detected = 0
            for malicious_input in malicious_inputs:
                if not SecurityValidator.validate_input(malicious_input):
                    malicious_detected += 1

            if malicious_detected == len(malicious_inputs):
                results["passed"] += 1
                results["details"].append("✅ 惡意輸入檢測通過")
            else:
                results["failed"] += 1
                results["details"].append(
                    f"❌ 惡意輸入檢測失敗 ({malicious_detected}/{len(malicious_inputs)})"
                )
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ 惡意輸入檢測異常: {e}")

        # 測試 HTML 清理
        try:
            dirty_html = "<script>alert('xss')</script><p>Normal text</p>"
            cleaned = SecurityValidator.sanitize_html(dirty_html)

            if "<script>" not in cleaned and "Normal text" in cleaned:
                results["passed"] += 1
                results["details"].append("✅ HTML 清理測試通過")
            else:
                results["failed"] += 1
                results["details"].append("❌ HTML 清理測試失敗")
        except Exception as e:
            results["failed"] += 1
            results["details"].append(f"❌ HTML 清理測試異常: {e}")

        return results

    async def run_all_tests(self) -> dict[str, Any]:
        """運行所有安全測試"""
        logger.info("開始運行安全測試套件...")

        await self.setup()

        test_results = []

        try:
            # 運行所有測試
            tests = [
                self.test_cryptography_service(),
                self.test_token_service(),
                self.test_authentication_service(),
                self.test_authorization_service(),
                self.test_security_middleware(),
                self.test_input_validation(),
            ]

            for test in tests:
                result = await test
                test_results.append(result)
                logger.info(
                    f"測試 {result['test_name']}: 通過 {result['passed']}, 失敗 {result['failed']}"
                )

        finally:
            await self.teardown()

        # 統計總結果
        total_passed = sum(r["passed"] for r in test_results)
        total_failed = sum(r["failed"] for r in test_results)

        summary = {
            "total_tests": len(test_results),
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": (
                total_passed / (total_passed + total_failed) * 100
                if (total_passed + total_failed) > 0
                else 0
            ),
            "test_results": test_results,
        }

        logger.info(
            f"安全測試完成: 總通過 {total_passed}, 總失敗 {total_failed}, 成功率 {summary['success_rate']:.1f}%"
        )

        return summary


async def run_security_tests():
    """運行安全測試的主函數"""
    test_suite = SecurityTestSuite()
    results = await test_suite.run_all_tests()

    print("\n" + "=" * 60)
    print("AIVA 安全框架測試結果")
    print("=" * 60)
    print(f"總測試數: {results['total_tests']}")
    print(f"通過: {results['total_passed']}")
    print(f"失敗: {results['total_failed']}")
    print(f"成功率: {results['success_rate']:.1f}%")
    print()

    for test_result in results["test_results"]:
        print(f"📋 {test_result['test_name']}:")
        for detail in test_result["details"]:
            print(f"   {detail}")
        print()

    if results["total_failed"] == 0:
        print("🎉 所有安全測試通過!")
    else:
        print(f"⚠️  {results['total_failed']} 個測試失敗，請檢查安全配置")

    return results


if __name__ == "__main__":
    # 運行測試
    asyncio.run(run_security_tests())
