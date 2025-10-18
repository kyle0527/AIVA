# -*- coding: utf-8 -*-
"""
高價值功能管理器

提供統一的介面來執行高價值功能模組，並生成標準化的報告。
專門針對 Bug Bounty 和滲透測試場景設計。

這個管理器簡化了功能模組的使用，並提供了：
- 統一的參數驗證
- 自動化的錯誤處理
- 標準化的結果格式
- HackerOne 友善的報告輸出
"""

import os
import time
from typing import Dict, Any, List, Optional
from .base.feature_registry import FeatureRegistry
from .base.result_schema import FeatureResult, Finding
from .feature_step_executor import FeatureStepExecutor
from .example_config import get_config_for_feature, get_attack_route, get_report_template

class HighValueFeatureManager:
    """
    高價值功能管理器
    
    統一管理所有高價值功能模組的執行，提供簡化的 API 和
    自動化的報告生成功能。
    """
    
    def __init__(self, allowlist_domains: Optional[str] = None):
        """
        初始化管理器
        
        Args:
            allowlist_domains: 允許的目標域名，如 "example.com,api.example.com"
        """
        if allowlist_domains:
            os.environ["ALLOWLIST_DOMAINS"] = allowlist_domains
        
        self.executor = FeatureStepExecutor()
        self.results_history = []
        
    def run_mass_assignment_test(self, target: str, update_endpoint: str, 
                                auth_headers: Dict[str, str], **kwargs) -> FeatureResult:
        """
        執行 Mass Assignment 檢測
        
        Args:
            target: 目標基礎 URL
            update_endpoint: 更新 API 端點
            auth_headers: 認證頭
            **kwargs: 其他參數
            
        Returns:
            檢測結果
        """
        params = {
            "target": target,
            "path": update_endpoint,
            "headers": auth_headers,
            "method": kwargs.get("method", "POST"),
            "baseline_body": kwargs.get("baseline_body", {}),
            "check_endpoint": kwargs.get("check_endpoint", "/api/me"),
            "check_key": kwargs.get("check_key", "role")
        }
        
        result = self._execute_feature("mass_assignment", params)
        self.results_history.append(result)
        return result
    
    def run_jwt_confusion_test(self, target: str, victim_token: str,
                              validate_endpoint: str = "/api/me", **kwargs) -> FeatureResult:
        """
        執行 JWT 混淆檢測
        
        Args:
            target: 目標基礎 URL
            victim_token: 低權限 JWT token
            validate_endpoint: 驗證端點
            **kwargs: 其他參數
            
        Returns:
            檢測結果
        """
        params = {
            "target": target,
            "victim_token": victim_token,
            "validate_endpoint": validate_endpoint,
            "jwks_url": kwargs.get("jwks_url"),
            "attempts": kwargs.get("attempts", {
                "alg_none": True,
                "kid_injection": True,
                "symmetric_rs": True
            }),
            "headers": kwargs.get("headers", {})
        }
        
        result = self._execute_feature("jwt_confusion", params)
        self.results_history.append(result)
        return result
    
    def run_oauth_confusion_test(self, target: str, client_id: str,
                                legitimate_redirect: str, attacker_redirect: str,
                                **kwargs) -> FeatureResult:
        """
        執行 OAuth 混淆檢測
        
        Args:
            target: IdP 基礎 URL
            client_id: OAuth 客戶端 ID
            legitimate_redirect: 合法重定向 URI
            attacker_redirect: 攻擊者控制的 URI
            **kwargs: 其他參數
            
        Returns:
            檢測結果
        """
        params = {
            "target": target,
            "client_id": client_id,
            "legitimate_redirect": legitimate_redirect,
            "attacker_redirect": attacker_redirect,
            "auth_endpoint": kwargs.get("auth_endpoint", "/oauth/authorize"),
            "scope": kwargs.get("scope", "openid profile"),
            "tests": kwargs.get("tests", {
                "redirect_uri_bypass": True,
                "pkce_downgrade": True,
                "open_redirect_chain": True
            })
        }
        
        result = self._execute_feature("oauth_confusion", params)
        self.results_history.append(result)
        return result
    
    def run_graphql_authz_test(self, target: str, user_headers: Dict[str, str],
                              test_queries: List[Dict[str, Any]], **kwargs) -> FeatureResult:
        """
        執行 GraphQL 權限檢測
        
        Args:
            target: 目標基礎 URL
            user_headers: 低權限用戶認證頭
            test_queries: 測試查詢列表
            **kwargs: 其他參數
            
        Returns:
            檢測結果
        """
        params = {
            "target": target,
            "headers_user": user_headers,
            "test_queries": test_queries,
            "endpoint": kwargs.get("endpoint", "/graphql"),
            "headers_admin": kwargs.get("admin_headers"),
            "tests": kwargs.get("tests", {
                "introspection": True,
                "field_level_authz": True,
                "object_level_authz": True
            })
        }
        
        result = self._execute_feature("graphql_authz", params)
        self.results_history.append(result)
        return result
    
    def run_ssrf_oob_test(self, target: str, oob_callback: str,
                         test_endpoints: Optional[List[str]] = None, **kwargs) -> FeatureResult:
        """
        執行 SSRF OOB 檢測
        
        Args:
            target: 目標基礎 URL
            oob_callback: OOB 回調 URL 或域名
            test_endpoints: 測試端點列表
            **kwargs: 其他參數
            
        Returns:
            檢測結果
        """
        # 自動判斷 OOB 類型
        oob_http = None
        oob_dns = None
        
        if oob_callback.startswith("http"):
            oob_http = oob_callback
        else:
            oob_dns = oob_callback
        
        params = {
            "target": target,
            "oob_http": oob_http,
            "oob_dns": oob_dns,
            "probe_endpoints": test_endpoints or [
                "/api/fetch", "/api/screenshot", "/api/pdf/generate"
            ],
            "url_params": kwargs.get("url_params", ["url", "link", "src"]),
            "headers": kwargs.get("headers", {}),
            "options": kwargs.get("options", {"delay_seconds": 10})
        }
        
        result = self._execute_feature("ssrf_oob", params)
        self.results_history.append(result)
        return result
    
    def run_attack_route(self, route_name: str, target: str, **common_params) -> List[FeatureResult]:
        """
        執行預定義的攻擊路線
        
        Args:
            route_name: 路線名稱 (privilege_escalation_route, authentication_bypass_route, data_access_route)
            target: 目標 URL
            **common_params: 共用參數（認證頭等）
            
        Returns:
            所有步驟的結果列表
        """
        route_steps = get_attack_route(route_name)
        results = []
        
        for step in route_steps:
            # 合併通用參數
            params = step["params"].copy()
            params["target"] = target
            params.update(common_params)
            
            print(f"執行步驟: {step['description']}")
            result = self._execute_feature(step["tool_name"], params)
            results.append(result)
            
            # 如果找到 Critical 漏洞，可以選擇性停止
            if result.has_critical_findings():
                print(f"🚨 發現 Critical 漏洞於 {step['tool_name']}，建議優先處理")
        
        self.results_history.extend(results)
        return results
    
    def generate_hackerone_report(self, result: FeatureResult) -> Dict[str, Any]:
        """
        生成 HackerOne 友善的報告格式
        
        Args:
            result: 功能執行結果
            
        Returns:
            HackerOne 報告字典
        """
        if not result.findings:
            return {"error": "沒有發現漏洞"}
        
        # 取得報告模板
        template = get_report_template(result.feature)
        
        # 使用最高嚴重度的發現作為主要報告
        critical_findings = result.get_findings_by_severity("critical")
        high_findings = result.get_findings_by_severity("high")
        
        main_finding = None
        if critical_findings:
            main_finding = critical_findings[0]
        elif high_findings:
            main_finding = high_findings[0]
        else:
            main_finding = result.findings[0]
        
        # 生成 HackerOne 格式
        h1_report = main_finding.to_hackerone_format()
        
        # 添加模板資訊
        if template:
            h1_report.update({
                "asset": template.get("asset", ""),
                "weakness": template.get("weakness", ""),
                "suggested_title": template.get("title", h1_report["title"])
            })
        
        # 添加統計資訊
        h1_report["summary_stats"] = result.get_summary()
        
        return h1_report
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        取得會話摘要
        
        Returns:
            會話統計和摘要
        """
        if not self.results_history:
            return {"message": "尚未執行任何檢測"}
        
        total_findings = sum(len(r.findings) for r in self.results_history)
        critical_count = sum(len(r.get_findings_by_severity("critical")) for r in self.results_history)
        high_count = sum(len(r.get_findings_by_severity("high")) for r in self.results_history)
        
        features_used = list(set(r.feature for r in self.results_history))
        
        summary = {
            "executions": len(self.results_history),
            "features_tested": features_used,
            "total_findings": total_findings,
            "critical_findings": critical_count,
            "high_findings": high_count,
            "success_rate": len([r for r in self.results_history if r.ok]) / len(self.results_history),
            "high_value_targets": critical_count + high_count
        }
        
        # 推薦下一步
        if critical_count > 0:
            summary["recommendation"] = "發現 Critical 漏洞，建議立即報告並停止進一步測試"
        elif high_count > 0:
            summary["recommendation"] = "發現 High 嚴重度漏洞，建議完善 PoC 並準備報告"
        else:
            summary["recommendation"] = "繼續探索其他功能點或嘗試不同的攻擊路線"
        
        return summary
    
    def _execute_feature(self, feature_name: str, params: Dict[str, Any]) -> FeatureResult:
        """
        內部方法：執行功能模組
        
        Args:
            feature_name: 功能名稱
            params: 參數
            
        Returns:
            執行結果
        """
        step = {
            "tool_type": "feature",
            "tool_name": feature_name,
            "params": params,
            "step_id": f"{feature_name}_{int(time.time())}"
        }
        
        try:
            execution_result = self.executor.execute(step)
            
            if execution_result["ok"]:
                return FeatureResult(
                    ok=True,
                    feature=feature_name,
                    command_record=execution_result["result"]["command_record"],
                    findings=[
                        Finding(**f) for f in execution_result["result"]["findings"]
                    ],
                    meta=execution_result["result"]["meta"]
                )
            else:
                return FeatureResult(
                    ok=False,
                    feature=feature_name,
                    command_record={"command": f"{feature_name}.error", "description": "執行失敗"},
                    findings=[],
                    meta={"error": execution_result.get("error", "未知錯誤")}
                )
                
        except Exception as e:
            return FeatureResult(
                ok=False,
                feature=feature_name,
                command_record={"command": f"{feature_name}.exception", "description": f"執行異常: {e}"},
                findings=[],
                meta={"exception": str(e)}
            )