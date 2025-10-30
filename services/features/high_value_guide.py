# -*- coding: utf-8 -*-
"""
高價值功能模組使用指南

本文件提供了如何使用新建的高價值功能模組進行實戰級安全檢測的完整指南。
這些模組專門針對在 Bug Bounty 平台能獲得高額獎金的漏洞類型設計。

功能模組列表：
1. Mass Assignment / 權限提升 (mass_assignment)
2. JWT 混淆攻擊 (jwt_confusion) 
3. OAuth/OIDC 配置錯誤 (oauth_confusion)
4. GraphQL 權限檢測 (graphql_authz)
5. SSRF with OOB (ssrf_oob)

注意：使用前必須設置 ALLOWLIST_DOMAINS 環境變數！
"""

import os
from typing import Dict, Any, List

# 導入所有功能模組

from .feature_step_executor import FeatureStepExecutor

# 導入具體的功能模組以觸發註冊
from .mass_assignment import worker as mass_assignment_worker
from .jwt_confusion import worker as jwt_confusion_worker
from .oauth_confusion import worker as oauth_confusion_worker
from .graphql_authz import worker as graphql_authz_worker


class HighValueFeatureManager:
    """
    高價值功能模組管理器
    
    提供統一的介面來執行各種高價值安全檢測，
    並生成可直接用於 Bug Bounty 報告的結果。
    """
    
    def __init__(self):
        self.executor = FeatureStepExecutor()
        self.setup_allowlist_check()
    
    def setup_allowlist_check(self):
        """檢查並提示設置 ALLOWLIST_DOMAINS"""
        allowlist = os.getenv("ALLOWLIST_DOMAINS", "")
        if not allowlist:
            print("⚠️  警告：未設置 ALLOWLIST_DOMAINS 環境變數")
            print("   請執行：export ALLOWLIST_DOMAINS=example.com,api.example.com")
            print("   這是必要的安全措施，防止意外掃描未授權目標")
    
    def run_mass_assignment_test(self, target: str, update_endpoint: str, 
                                auth_headers: Dict[str, str],
                                baseline_body: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        執行 Mass Assignment 權限提升檢測
        
        適用場景：
        - 個人資料更新 API
        - 用戶設定 API  
        - 註冊 API
        
        Args:
            target: 目標網站，如 "https://app.example.com"
            update_endpoint: 更新端點，如 "/api/profile/update"
            auth_headers: 低權限用戶的認證頭
            baseline_body: 正常的更新請求內容
        """
        step = {
            "tool_type": "feature",
            "tool_name": "mass_assignment",
            "params": {
                "target": target,
                "path": update_endpoint,
                "method": "POST",
                "headers": auth_headers,
                "baseline_body": baseline_body or {},
                "check_endpoint": "/api/me",
                "check_key": "role"
            }
        }
        return self.executor.execute(step)
    
    def run_jwt_confusion_test(self, target: str, jwt_token: str,
                              validate_endpoint: str = "/api/me") -> Dict[str, Any]:
        """
        執行 JWT 混淆攻擊檢測
        
        適用場景：
        - API 使用 JWT 進行身分驗證
        - 可能存在 alg=none、kid 注入、RS→HS 混淆
        
        Args:
            target: 目標網站
            jwt_token: 有效的 JWT token（低權限）
            validate_endpoint: 驗證 JWT 的端點
        """
        step = {
            "tool_type": "feature", 
            "tool_name": "jwt_confusion",
            "params": {
                "target": target,
                "validate_endpoint": validate_endpoint,
                "victim_token": jwt_token,
                "attempts": {
                    "alg_none": True,
                    "kid_injection": True,
                    "symmetric_rs": True
                }
            }
        }
        return self.executor.execute(step)
    
    def run_oauth_confusion_test(self, idp_url: str, client_id: str,
                               legitimate_redirect: str,
                               attacker_redirect: str) -> Dict[str, Any]:
        """
        執行 OAuth/OIDC 配置錯誤檢測
        
        適用場景：
        - OAuth 2.0 / OpenID Connect 身分提供者
        - 可能存在 redirect_uri 繞過、PKCE 未強制執行
        
        Args:
            idp_url: 身分提供者 URL，如 "https://auth.example.com"
            client_id: OAuth 客戶端 ID
            legitimate_redirect: 合法的重定向 URI
            attacker_redirect: 攻擊者控制的重定向 URI
        """
        step = {
            "tool_type": "feature",
            "tool_name": "oauth_confusion",
            "params": {
                "target": idp_url,
                "auth_endpoint": "/oauth/authorize",
                "client_id": client_id,
                "legitimate_redirect": legitimate_redirect,
                "attacker_redirect": attacker_redirect,
                "scope": "openid profile email",
                "tests": {
                    "redirect_uri_bypass": True,
                    "pkce_downgrade": True,
                    "open_redirect_chain": True
                }
            }
        }
        return self.executor.execute(step)
    
    def run_graphql_authz_test(self, target: str, graphql_endpoint: str,
                              user_headers: Dict[str, str],
                              admin_headers: Dict[str, str] = None,
                              test_queries: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        執行 GraphQL 權限檢測
        
        適用場景：
        - GraphQL API 端點
        - 可能存在 introspection 暴露、欄位級權限缺失
        
        Args:
            target: 目標網站
            graphql_endpoint: GraphQL 端點，如 "/graphql"
            user_headers: 低權限用戶認證頭
            admin_headers: 可選，管理員認證頭（用於對比）
            test_queries: 測試查詢列表
        """
        if not test_queries:
            test_queries = [
                {
                    "name": "user_profile",
                    "query": "query { user(id: \"1001\") { id email role permissions } }",
                    "target_user_id": "1001"
                },
                {
                    "name": "sensitive_data",
                    "query": "query { me { id email phone address creditCard } }"
                }
            ]
        
        step = {
            "tool_type": "feature",
            "tool_name": "graphql_authz", 
            "params": {
                "target": target,
                "endpoint": graphql_endpoint,
                "headers_user": user_headers,
                "headers_admin": admin_headers,
                "test_queries": test_queries,
                "tests": {
                    "introspection": True,
                    "field_level_authz": True,
                    "object_level_authz": True
                }
            }
        }
        return self.executor.execute(step)
    
    def run_ssrf_oob_test(self, target: str, oob_http: str, oob_dns: str = None,
                         probe_endpoints: List[str] = None) -> Dict[str, Any]:
        """
        執行 SSRF OOB 檢測
        
        適用場景：
        - URL 抓取功能
        - PDF 生成服務
        - 圖片代理服務
        - Webhook 回調
        
        Args:
            target: 目標網站
            oob_http: OOB HTTP 回調 URL（如 Burp Collaborator）
            oob_dns: 可選，OOB DNS 域名
            probe_endpoints: 可選，指定要測試的端點
        """
        step = {
            "tool_type": "feature",
            "tool_name": "ssrf_oob",
            "params": {
                "target": target,
                "probe_endpoints": probe_endpoints or [],
                "url_params": ["url", "link", "fetch", "src", "href", "webhook"],
                "json_fields": ["url", "imageUrl", "webhookUrl", "callbackUrl"],
                "oob_http": oob_http,
                "oob_dns": oob_dns,
                "test_protocols": ["http", "https"],
                "payload_types": ["direct", "encoded"],
                "options": {
                    "delay_seconds": 10,
                    "auto_discover": True,
                    "test_internal": False  # 設為 True 可測試內網探測
                }
            }
        }
        return self.executor.execute(step)
    
    def run_comprehensive_scan(self, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        執行綜合掃描，運行所有適用的高價值檢測
        
        Args:
            target_config: 目標配置，包含：
              - target: 目標 URL
              - auth_headers: 認證頭
              - jwt_token: 可選，JWT token
              - oauth_config: 可選，OAuth 配置
              - oob_config: 可選，OOB 配置
        """
        results = {
            "target": target_config.get("target"),
            "scan_results": {},
            "summary": {
                "total_findings": 0,
                "critical_findings": 0,
                "high_findings": 0
            }
        }
        
        target = target_config["target"]
        auth_headers = target_config.get("auth_headers", {})
        
        # 1. Mass Assignment 檢測
        if auth_headers:
            print("🔍 執行 Mass Assignment 檢測...")
            result = self.run_mass_assignment_test(
                target, "/api/profile/update", auth_headers
            )
            results["scan_results"]["mass_assignment"] = result
            self._update_summary(results["summary"], result)
        
        # 2. JWT 混淆檢測
        jwt_token = target_config.get("jwt_token")
        if jwt_token:
            print("🔍 執行 JWT 混淆檢測...")
            result = self.run_jwt_confusion_test(target, jwt_token)
            results["scan_results"]["jwt_confusion"] = result
            self._update_summary(results["summary"], result)
        
        # 3. OAuth 配置檢測
        oauth_config = target_config.get("oauth_config")
        if oauth_config:
            print("🔍 執行 OAuth 配置檢測...")
            result = self.run_oauth_confusion_test(
                oauth_config["idp_url"],
                oauth_config["client_id"],
                oauth_config["legitimate_redirect"],
                oauth_config["attacker_redirect"]
            )
            results["scan_results"]["oauth_confusion"] = result
            self._update_summary(results["summary"], result)
        
        # 4. GraphQL 權限檢測
        if "/graphql" in target or target_config.get("has_graphql"):
            print("🔍 執行 GraphQL 權限檢測...")
            result = self.run_graphql_authz_test(
                target, "/graphql", auth_headers,
                target_config.get("admin_headers")
            )
            results["scan_results"]["graphql_authz"] = result
            self._update_summary(results["summary"], result)
        
        # 5. SSRF OOB 檢測
        oob_config = target_config.get("oob_config")
        if oob_config:
            print("🔍 執行 SSRF OOB 檢測...")
            result = self.run_ssrf_oob_test(
                target,
                oob_config["http_callback"],
                oob_config.get("dns_callback")
            )
            results["scan_results"]["ssrf_oob"] = result
            self._update_summary(results["summary"], result)
        
        print(f"\n✅ 掃描完成！")
        print(f"   總計發現: {results['summary']['total_findings']}")
        print(f"   Critical: {results['summary']['critical_findings']}")
        print(f"   High: {results['summary']['high_findings']}")
        
        return results
    
    def _update_summary(self, summary: Dict[str, int], result: Dict[str, Any]):
        """更新掃描摘要統計"""
        if result.get("ok") and "result" in result:
            findings = result["result"].get("findings", [])
            summary["total_findings"] += len(findings)
            
            for finding in findings:
                severity = finding.get("severity", "").lower()
                if severity == "critical":
                    summary["critical_findings"] += 1
                elif severity == "high":
                    summary["high_findings"] += 1

# 使用範例
def example_usage():
    """使用範例"""
    
    # 設置環境變數（必需！）
    os.environ["ALLOWLIST_DOMAINS"] = "example.com,api.example.com,auth.example.com"
    
    manager = HighValueFeatureManager()
    
    # 目標配置
    target_config = {
        "target": "https://app.example.com",
        "auth_headers": {
            "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
            "User-Agent": "AIVA Security Scanner/2.0"
        },
        "jwt_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "oauth_config": {
            "idp_url": "https://auth.example.com",
            "client_id": "app123",
            "legitimate_redirect": "https://app.example.com/callback",
            "attacker_redirect": "https://evil.example.net/callback"
        },
        "oob_config": {
            "http_callback": "https://your-collab-server.com/hit",
            "dns_callback": "your-token.dnslog.cn"
        },
        "has_graphql": True,
        "admin_headers": {  # 可選，如果有管理員帳號
            "Authorization": "Bearer admin_token_here"
        }
    }
    
    # 執行綜合掃描
    results = manager.run_comprehensive_scan(target_config)
    
    # 或者單獨執行某個檢測
    # mass_assign_result = manager.run_mass_assignment_test(
    #     "https://app.example.com",
    #     "/api/profile/update", 
    #     {"Authorization": "Bearer token"}
    # )
    
    return results

if __name__ == "__main__":
    # 執行範例（記得先設置 ALLOWLIST_DOMAINS）
    example_usage()