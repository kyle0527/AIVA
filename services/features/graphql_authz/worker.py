# -*- coding: utf-8 -*-
from typing import Dict, Any, List
import json
from urllib.parse import urljoin
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

# GraphQL Introspection 查詢
INTROSPECTION_QUERY = {
    "query": """
    {
        __schema {
            types {
                name
                fields {
                    name
                    type {
                        name
                        ofType {
                            name
                        }
                    }
                }
            }
        }
    }
    """
}

# 敏感欄位關鍵字
SENSITIVE_FIELDS = [
    "email", "password", "token", "secret", "key", "role", "admin", 
    "permission", "privilege", "credit", "balance", "ssn", "phone",
    "address", "salary", "internal", "private", "confidential"
]

@FeatureRegistry.register
class GraphQLAuthzWorker(FeatureBase):
    """
    GraphQL 權限檢測模組
    
    專門檢測 GraphQL API 中的權限控制漏洞，這類漏洞在現代 Web 應用中
    越來越常見，且容易導致敏感資料洩漏。
    
    檢測的漏洞類型：
    1. Introspection 端點暴露 - 洩漏完整的 schema 資訊
    2. 欄位級授權缺失 - 低權限用戶可存取敏感欄位  
    3. 物件級授權缺失 - 可存取其他用戶的資料
    4. 查詢深度/複雜度控制缺失 - DoS 攻擊風險
    
    這些漏洞特別容易在 GraphQL 實作中被忽略，且影響程度通常較高。
    """
    
    name = "graphql_authz"
    version = "2.0.0"
    tags = ["graphql", "authorization", "introspection", "field-level-authz"]

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 GraphQL 權限檢測
        
        Args:
            params: 檢測參數
              - target (str): 基礎 URL
              - endpoint (str): GraphQL 端點，如 /graphql
              - headers_user (dict): 低權限用戶的認證頭
              - headers_admin (dict): 可選，管理員認證頭（用於對比）
              - test_queries (list): 測試查詢列表，每個包含：
                - name: 查詢名稱
                - query: GraphQL 查詢字串
                - variables: 可選，查詢變數
                - target_user_id: 目標用戶 ID（測試 IDOR）
              - tests (dict): 測試類型開關
                - introspection: 是否測試 introspection
                - field_level_authz: 是否測試欄位級權限
                - object_level_authz: 是否測試物件級權限
              
        Returns:
            FeatureResult: 檢測結果
        """
        http = SafeHttp()
        base = params.get("target", "")
        endpoint = params.get("endpoint", "/graphql")
        graphql_url = urljoin(base, endpoint)
        headers_user = params.get("headers_user", {})
        headers_admin = params.get("headers_admin")
        test_queries = params.get("test_queries", [])
        tests = params.get("tests", {
            "introspection": True,
            "field_level_authz": True, 
            "object_level_authz": True
        })
        
        findings: List[Finding] = []
        trace = []
        
        cmd = self.build_command_record(
            "graphql.authz",
            "GraphQL 權限與授權檢測",
            {
                "endpoint": endpoint,
                "test_count": len(test_queries),
                "has_admin_compare": bool(headers_admin),
                "tests": list(tests.keys())
            }
        )
        
        # 1) Introspection 檢測
        if tests.get("introspection"):
            try:
                response = http.request(
                    "POST", graphql_url,
                    headers={**headers_user, "Content-Type": "application/json"},
                    data=json.dumps(INTROSPECTION_QUERY)
                )
                
                trace.append({
                    "step": "introspection_test",
                    "status": response.status_code,
                    "response_length": len(response.text),
                    "has_schema": "__schema" in response.text
                })
                
                if (response.status_code == 200 and 
                    "__schema" in response.text and 
                    "types" in response.text):
                    
                    # 解析 schema 查找敏感類型
                    try:
                        data = response.json()
                        schema = data.get("data", {}).get("__schema", {})
                        types = schema.get("types", [])
                        
                        sensitive_types = []
                        for type_info in types:
                            type_name = type_info.get("name", "")
                            fields = type_info.get("fields", [])
                            
                            if fields:  # 只檢查有欄位的類型
                                for field in fields:
                                    field_name = field.get("name", "").lower()
                                    if any(sensitive in field_name for sensitive in SENSITIVE_FIELDS):
                                        sensitive_types.append({
                                            "type": type_name,
                                            "sensitive_field": field.get("name")
                                        })
                        
                        severity = "medium" if sensitive_types else "low"
                        findings.append(Finding(
                            vuln_type="GraphQL Introspection Enabled",
                            severity=severity,
                            title=f"GraphQL Introspection 暴露 {len(types)} 個類型",
                            evidence={
                                "endpoint": graphql_url,
                                "types_count": len(types),
                                "sensitive_types": sensitive_types[:10],  # 限制輸出
                                "response_sample": response.text[:1000]
                            },
                            reproduction=[
                                {
                                    "step": 1,
                                    "description": "發送 introspection 查詢",
                                    "request": {
                                        "method": "POST",
                                        "url": graphql_url,
                                        "headers": {"Content-Type": "application/json"},
                                        "json": INTROSPECTION_QUERY
                                    },
                                    "expect": "生產環境應禁用 introspection"
                                }
                            ],
                            impact="完整的 schema 資訊洩漏，包含敏感欄位和類型結構，便於攻擊者構造針對性查詢",
                            recommendation="1. 在生產環境禁用 introspection\n2. 實施基於角色的 schema 存取控制\n3. 避免在 schema 中暴露敏感欄位名稱"
                        ))
                        
                    except (ValueError, KeyError) as e:
                        trace.append({
                            "step": "introspection_parse_error",
                            "error": str(e)
                        })
                        
            except Exception as e:
                trace.append({
                    "step": "introspection_error",
                    "error": str(e)
                })
        
        # 2) 欄位級和物件級權限檢測
        if test_queries and (tests.get("field_level_authz") or tests.get("object_level_authz")):
            for query_info in test_queries:
                query_name = query_info.get("name", "unknown")
                query = query_info.get("query", "")
                variables = query_info.get("variables", {})
                target_user_id = query_info.get("target_user_id")
                
                if not query:
                    continue
                
                try:
                    # 用低權限用戶執行查詢
                    request_body = {"query": query}
                    if variables:
                        request_body["variables"] = variables
                    
                    user_response = http.request(
                        "POST", graphql_url,
                        headers={**headers_user, "Content-Type": "application/json"},
                        data=json.dumps(request_body)
                    )
                    
                    trace.append({
                        "step": f"query_test_{query_name}",
                        "user_status": user_response.status_code,
                        "user_response_length": len(user_response.text),
                        "has_errors": "errors" in user_response.text
                    })
                    
                    # 如果有管理員認證頭，做對比測試
                    if headers_admin and tests.get("field_level_authz"):
                        admin_response = http.request(
                            "POST", graphql_url,
                            headers={**headers_admin, "Content-Type": "application/json"},
                            data=json.dumps(request_body)
                        )
                        
                        trace.append({
                            "step": f"admin_query_test_{query_name}",
                            "admin_status": admin_response.status_code,
                            "admin_response_length": len(admin_response.text)
                        })
                        
                        # 比較用戶和管理員的回應
                        if (user_response.status_code == 200 and 
                            admin_response.status_code == 200 and
                            "errors" not in user_response.text):
                            
                            try:
                                user_data = user_response.json()
                                admin_data = admin_response.json()
                                
                                # 簡單比較：如果內容幾乎相同，可能存在權限問題
                                user_str = json.dumps(user_data, sort_keys=True)
                                admin_str = json.dumps(admin_data, sort_keys=True)
                                
                                if len(user_str) > 50 and abs(len(user_str) - len(admin_str)) < max(50, len(admin_str) * 0.1):
                                    findings.append(Finding(
                                        vuln_type="GraphQL Field-Level Authorization Bypass",
                                        severity="high",
                                        title=f"欄位級權限缺失：{query_name}",
                                        evidence={
                                            "query": query,
                                            "user_response_length": len(user_str),
                                            "admin_response_length": len(admin_str),
                                            "user_sample": user_response.text[:500],
                                            "admin_sample": admin_response.text[:500]
                                        },
                                        reproduction=[
                                            {
                                                "step": 1,
                                                "description": f"以低權限用戶執行查詢：{query_name}",
                                                "request": {
                                                    "method": "POST",
                                                    "url": graphql_url,
                                                    "json": request_body
                                                },
                                                "expect": "敏感欄位應被過濾或拒絕存取"
                                            }
                                        ],
                                        impact="低權限用戶可存取與管理員相同的敏感欄位和資料",
                                        recommendation="1. 在 resolver 層實施欄位級權限檢查\n2. 根據用戶角色動態過濾敏感欄位\n3. 使用 GraphQL 權限框架如 graphql-shield"
                                    ))
                                    
                            except (ValueError, KeyError):
                                pass
                    
                    # 物件級權限檢測（如果指定了目標用戶ID）
                    elif (tests.get("object_level_authz") and target_user_id and 
                          user_response.status_code == 200 and "errors" not in user_response.text):
                        
                        try:
                            user_data = user_response.json()
                            
                            # 檢查是否返回了其他用戶的資料
                            response_str = json.dumps(user_data).lower()
                            if any(field in response_str for field in ["email", "id", "userid", "username"]):
                                findings.append(Finding(
                                    vuln_type="GraphQL Object-Level Authorization Bypass (IDOR)",
                                    severity="high", 
                                    title=f"物件級權限缺失：{query_name}",
                                    evidence={
                                        "query": query,
                                        "target_user_id": target_user_id,
                                        "response_sample": user_response.text[:500]
                                    },
                                    reproduction=[
                                        {
                                            "step": 1,
                                            "description": f"嘗試存取其他用戶資料：{target_user_id}",
                                            "request": {
                                                "method": "POST",
                                                "url": graphql_url,
                                                "json": request_body
                                            },
                                            "expect": "應拒絕存取其他用戶的資料"
                                        }
                                    ],
                                    impact="用戶可存取其他用戶的私人資料，違反資料隔離原則",
                                    recommendation="1. 在 resolver 中實施物件級權限檢查\n2. 驗證請求用戶對資源的所有權\n3. 使用上下文感知的權限控制"
                                ))
                        except (ValueError, KeyError):
                            pass
                    
                    # 檢查是否有敏感欄位洩漏（無管理員對比時的啟發式檢測）
                    elif (user_response.status_code == 200 and "errors" not in user_response.text):
                        response_lower = user_response.text.lower()
                        found_sensitive = [field for field in SENSITIVE_FIELDS if field in response_lower]
                        
                        if found_sensitive and len(user_response.text) > 100:
                            findings.append(Finding(
                                vuln_type="GraphQL Sensitive Field Exposure",
                                severity="medium",
                                title=f"敏感欄位暴露：{query_name}",
                                evidence={
                                    "query": query,
                                    "sensitive_fields_found": found_sensitive,
                                    "response_sample": user_response.text[:500]
                                },
                                reproduction=[
                                    {
                                        "step": 1,
                                        "description": f"執行查詢：{query_name}",
                                        "request": {
                                            "method": "POST",
                                            "url": graphql_url,
                                            "json": request_body
                                        },
                                        "expect": "敏感欄位應被適當保護"
                                    }
                                ],
                                impact="查詢結果中包含潛在敏感資訊，可能導致資料洩漏",
                                recommendation="1. 預設隱藏敏感欄位\n2. 實施基於角色的欄位可見性\n3. 對敏感資料進行脫敏處理"
                            ))
                
                except Exception as e:
                    trace.append({
                        "step": f"query_error_{query_name}",
                        "error": str(e)
                    })
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "trace": trace,
                "queries_tested": len(test_queries),
                "has_admin_comparison": bool(headers_admin)
            }
        )