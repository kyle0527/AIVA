# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional, Tuple
import json
from urllib.parse import urljoin
from datetime import datetime
import concurrent.futures
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

# v2.5 新增：高價值欄位權重矩陣
FIELD_VALUE_MATRIX = {
    "password": 10, "secret": 10, "token": 10, "api_key": 10,
    "ssn": 9, "credit_card": 9, "cvv": 9,
    "email": 7, "phone": 7, "address": 7,
    "role": 8, "permission": 8, "admin": 8,
    "salary": 6, "balance": 6, "credit": 6,
    "internal": 5, "private": 5, "confidential": 5
}

# v2.5 新增：批次查詢模板
BATCH_QUERY_TEMPLATES = [
    {"name": "parallel_users", "count": 5},
    {"name": "nested_depth", "depth": 10},
    {"name": "alias_explosion", "aliases": 20}
]

@FeatureRegistry.register
class GraphQLAuthzWorker(FeatureBase):
    """
    GraphQL 權限檢測模組 v2.5
    
    專門檢測 GraphQL API 中的權限控制漏洞，這類漏洞在現代 Web 應用中
    越來越常見，且容易導致敏感資料洩漏。
    
    v2.5 新增功能：
    - 深度分析優化：基於欄位權重的智能分析
    - 批次查詢測試：並發測試多個查詢組合
    - 字段級權限矩陣：構建完整的訪問控制矩陣
    - 錯誤訊息增強：從錯誤消息中提取敏感信息
    
    檢測的漏洞類型：
    1. Introspection 端點暴露 - 洩漏完整的 schema 資訊
    2. 欄位級授權缺失 - 低權限用戶可存取敏感欄位  
    3. 物件級授權缺失 - 可存取其他用戶的資料
    4. 查詢深度/複雜度控制缺失 - DoS 攻擊風險
    5. 批次查詢濫用 - 資源耗盡攻擊
    6. 錯誤消息洩露 - 敏感信息洩露
    
    這些漏洞特別容易在 GraphQL 實作中被忽略，且影響程度通常較高。
    """
    
    name = "graphql_authz"
    version = "2.5.0"
    tags = ["graphql", "authorization", "introspection", "field-level-authz"]
    
    def _analyze_field_value_weights(
        self,
        schema_types: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        v2.5 新增：欄位價值權重分析
        根據欄位名稱計算敏感度權重
        
        Returns:
            按權重排序的欄位列表
        """
        weighted_fields = []
        
        for type_info in schema_types:
            type_name = type_info.get("name", "")
            fields = type_info.get("fields", [])
            
            if fields:
                for field in fields:
                    field_name = field.get("name", "")
                    field_lower = field_name.lower()
                    
                    # 計算權重
                    weight = 0
                    for sensitive_key, value in FIELD_VALUE_MATRIX.items():
                        if sensitive_key in field_lower:
                            weight = max(weight, value)
                    
                    if weight > 0:
                        weighted_fields.append({
                            "type": type_name,
                            "field": field_name,
                            "weight": weight,
                            "field_type": field.get("type", {}).get("name", "Unknown")
                        })
        
        # 按權重降序排序
        weighted_fields.sort(key=lambda x: x["weight"], reverse=True)
        return weighted_fields
    
    def _batch_query_test(
        self,
        http: SafeHttp,
        graphql_url: str,
        headers: Dict[str, Any],
        base_query: str,
        batch_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        v2.5 新增：批次查詢測試
        測試 GraphQL 伺服器的批次查詢處理能力
        
        Returns:
            批次查詢測試結果
        """
        batch_name = batch_config.get("name", "unknown")
        
        try:
            if batch_name == "parallel_users":
                # 並行查詢多個用戶
                count = batch_config.get("count", 5)
                queries = [
                    {"query": base_query.replace("$id", str(i))}
                    for i in range(1, count + 1)
                ]
                
            elif batch_name == "nested_depth":
                # 深度嵌套查詢
                depth = batch_config.get("depth", 10)
                nested_query = base_query
                for _ in range(depth):
                    nested_query = f"{{ nested {nested_query} }}"
                queries = [{"query": nested_query}]
                
            elif batch_name == "alias_explosion":
                # 別名爆炸攻擊
                aliases = batch_config.get("aliases", 20)
                alias_query = "{ " + " ".join([
                    f"alias{i}: {base_query}"
                    for i in range(aliases)
                ]) + " }"
                queries = [{"query": alias_query}]
            else:
                return {"error": f"Unknown batch type: {batch_name}"}
            
            start_time = datetime.utcnow()
            response = http.request(
                "POST", graphql_url,
                headers={**headers, "Content-Type": "application/json"},
                data=json.dumps(queries if len(queries) > 1 else queries[0])
            )
            end_time = datetime.utcnow()
            
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                "batch_type": batch_name,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "response_size": len(response.text),
                "accepted": response.status_code == 200,
                "timestamp": end_time.isoformat()
            }
        except Exception as e:
            return {
                "batch_type": batch_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _build_field_permission_matrix(
        self,
        user_accessible: List[str],
        admin_accessible: List[str]
    ) -> Dict[str, Any]:
        """
        v2.5 新增：構建欄位級權限矩陣
        對比不同權限用戶的欄位訪問能力
        
        Returns:
            權限矩陣字典
        """
        user_set = set(user_accessible)
        admin_set = set(admin_accessible)
        
        return {
            "user_only": list(user_set - admin_set),
            "admin_only": list(admin_set - user_set),
            "shared": list(user_set & admin_set),
            "total_user_fields": len(user_set),
            "total_admin_fields": len(admin_set),
            "overlap_percentage": len(user_set & admin_set) / len(admin_set) * 100 if admin_set else 0
        }
    
    def _extract_field_names(
        self,
        data: Any,
        prefix: str = ""
    ) -> List[str]:
        """
        v2.5 輔助方法：遞歸提取 JSON 中所有欄位名稱
        
        Returns:
            欄位名稱列表
        """
        fields = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                field_name = f"{prefix}.{key}" if prefix else key
                fields.append(field_name)
                # 遞歸提取嵌套欄位
                if isinstance(value, (dict, list)):
                    fields.extend(self._extract_field_names(value, field_name))
        elif isinstance(data, list) and data:
            # 只檢查列表第一個元素
            fields.extend(self._extract_field_names(data[0], prefix))
        
        return fields
    
    def _extract_error_messages(
        self,
        response_text: str
    ) -> List[Dict[str, Any]]:
        """
        v2.5 新增：從錯誤消息中提取敏感信息
        分析 GraphQL 錯誤響應中的信息洩露
        
        Returns:
            提取的敏感信息列表
        """
        sensitive_info = []
        
        try:
            data = json.loads(response_text)
            errors = data.get("errors", [])
            
            for error in errors:
                message = error.get("message", "")
                message_lower = message.lower()
                
                # 檢查是否包含敏感信息
                leaks = []
                if "path" in message_lower or "file" in message_lower:
                    leaks.append("file_path")
                if "table" in message_lower or "column" in message_lower:
                    leaks.append("database_schema")
                if "error" in message_lower and ("stack" in message_lower or "trace" in message_lower):
                    leaks.append("stack_trace")
                if any(keyword in message_lower for keyword in ["user", "email", "id"]):
                    leaks.append("user_data")
                
                if leaks:
                    sensitive_info.append({
                        "error_message": message[:200],
                        "leak_types": leaks,
                        "extensions": error.get("extensions", {})
                    })
        except:
            pass
        
        return sensitive_info

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 GraphQL 權限檢測 v2.5
        
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
                - batch_queries: v2.5 批次查詢測試
                - error_analysis: v2.5 錯誤消息分析
              - batch_base_query (str): v2.5 批次查詢基礎模板
              
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
            "object_level_authz": True,
            "batch_queries": True,
            "error_analysis": True
        })
        batch_base_query = params.get("batch_base_query", "{ user(id: $id) { id name } }")
        
        findings: List[Finding] = []
        trace = []
        start_time = datetime.utcnow()
        
        # v2.5 新增：記錄時間戳
        timestamps = {
            "start": start_time.isoformat()
        }
        
        cmd = self.build_command_record(
            "graphql.authz.v2.5",
            "GraphQL 權限與授權檢測 v2.5",
            {
                "endpoint": endpoint,
                "test_count": len(test_queries),
                "has_admin_compare": bool(headers_admin),
                "tests": list(tests.keys()),
                "v2.5_features": ["field_value_weights", "batch_queries", "permission_matrix", "error_analysis"]
            }
        )
        
        # v2.5 新增：初始化統計數據
        v2_5_stats = {
            "weighted_fields_analyzed": 0,
            "batch_tests_performed": 0,
            "permission_matrices_built": 0,
            "error_messages_analyzed": 0
        }
        
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
                        
                        # v2.5 新增：欄位價值權重分析
                        weighted_fields = self._analyze_field_value_weights(types)
                        v2_5_stats["weighted_fields_analyzed"] = len(weighted_fields)
                        timestamps["field_analysis_complete"] = datetime.utcnow().isoformat()
                        
                        trace.append({
                            "step": "v2.5_field_weight_analysis",
                            "total_weighted_fields": len(weighted_fields),
                            "top_5_fields": weighted_fields[:5]
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
                                "response_sample": response.text[:1000],
                                "v2_5_weighted_fields": weighted_fields[:10],  # v2.5: 前10個高權重欄位
                                "v2_5_highest_risk_field": weighted_fields[0] if weighted_fields else None
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
                                
                                # v2.5 新增：構建欄位級權限矩陣
                                user_fields = self._extract_field_names(user_data)
                                admin_fields = self._extract_field_names(admin_data)
                                permission_matrix = self._build_field_permission_matrix(
                                    user_fields, admin_fields
                                )
                                v2_5_stats["permission_matrices_built"] += 1
                                
                                # v2.5 新增：錯誤消息分析
                                error_leaks = []
                                if tests.get("error_analysis"):
                                    error_leaks = self._extract_error_messages(user_response.text)
                                    v2_5_stats["error_messages_analyzed"] += 1
                                
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
                                            "admin_sample": admin_response.text[:500],
                                            "v2_5_permission_matrix": permission_matrix,  # v2.5
                                            "v2_5_error_leaks": error_leaks  # v2.5
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
        
        # v2.5 新增：批次查詢測試
        if tests.get("batch_queries") and batch_base_query:
            timestamps["batch_test_start"] = datetime.utcnow().isoformat()
            
            batch_results = []
            for batch_config in BATCH_QUERY_TEMPLATES:
                result = self._batch_query_test(
                    http, graphql_url, headers_user,
                    batch_base_query, batch_config
                )
                batch_results.append(result)
                v2_5_stats["batch_tests_performed"] += 1
                
                # 如果批次查詢被接受，可能存在資源耗盡風險
                if result.get("accepted") and result.get("duration_ms", 0) > 1000:
                    findings.append(Finding(
                        vuln_type="GraphQL Batch Query Abuse",
                        severity="medium",
                        title=f"批次查詢濫用風險：{result['batch_type']}",
                        evidence={
                            "batch_type": result["batch_type"],
                            "duration_ms": result["duration_ms"],
                            "response_size": result["response_size"],
                            "test_config": batch_config
                        },
                        reproduction=[
                            {
                                "step": 1,
                                "description": f"發送 {result['batch_type']} 批次查詢",
                                "request": {
                                    "method": "POST",
                                    "url": graphql_url,
                                    "note": f"使用 {batch_config} 配置"
                                },
                                "expect": "應限制批次查詢的深度和複雜度"
                            }
                        ],
                        impact="未限制的批次查詢可能導致資源耗盡和 DoS 攻擊",
                        recommendation="1. 實施查詢複雜度限制\n2. 限制查詢深度和廣度\n3. 實施速率限制和資源配額"
                    ))
            
            timestamps["batch_test_complete"] = datetime.utcnow().isoformat()
            trace.append({
                "step": "v2.5_batch_query_tests",
                "total_tests": len(batch_results),
                "results": batch_results
            })
        
        # v2.5 新增：計算總執行時間
        end_time = datetime.utcnow()
        timestamps["end"] = end_time.isoformat()
        total_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "trace": trace,
                "queries_tested": len(test_queries),
                "has_admin_comparison": bool(headers_admin),
                "v2_5_stats": v2_5_stats,  # v2.5
                "timestamps": timestamps,  # v2.5
                "total_duration_ms": total_duration_ms,  # v2.5
                "version": "2.5.0"  # v2.5
            }
        )