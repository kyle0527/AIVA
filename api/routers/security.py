# -*- coding: utf-8 -*-
"""
AIVA API 路由模組

將 API 端點分模組組織，提供更好的代碼結構和維護性。
包含高價值功能模組、傳統安全檢測、系統管理等端點。
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import sys
import os

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.features.high_value_manager import HighValueFeatureManager
from services.features.base.result_schema import FeatureResult

# 創建路由器
router = APIRouter()

# 全域變數
high_value_manager = None
active_scans: Dict[str, Dict[str, Any]] = {}

def get_high_value_manager():
    """獲取高價值模組管理器"""
    global high_value_manager
    if high_value_manager is None:
        try:
            high_value_manager = HighValueFeatureManager()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize high value manager: {str(e)}"
            )
    return high_value_manager

# === 高價值功能模組端點 ===

@router.post("/mass-assignment")
async def mass_assignment_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = None
):
    """
    Mass Assignment 權限提升檢測
    
    檢測目標應用是否存在 Mass Assignment 漏洞，
    可能導致權限提升或數據洩露。
    
    預期市場價值: $2.1K-$8.2K
    """
    scan_id = f"mass_assignment_{int(time.time())}"
    
    # 驗證必需參數
    required_fields = ["target", "update_endpoint", "auth_headers"]
    for field in required_fields:
        if field not in request:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    # 記錄掃描狀態
    active_scans[scan_id] = {
        "type": "mass_assignment",
        "status": "started",
        "user": current_user.get("username") if current_user else "anonymous",
        "start_time": datetime.utcnow(),
        "request": request
    }
    
    # 後台執行掃描
    background_tasks.add_task(execute_mass_assignment_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "Mass Assignment scan initiated",
        "timestamp": datetime.utcnow(),
        "estimated_duration": "30-120 seconds",
        "potential_value": "$2.1K-$8.2K"
    }

@router.post("/jwt-confusion")
async def jwt_confusion_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = None
):
    """
    JWT 混淆攻擊檢測
    
    檢測 JWT 實現中的算法混淆漏洞，
    可能導致身份驗證繞過。
    
    預期市場價值: $1.8K-$7.5K
    """
    scan_id = f"jwt_confusion_{int(time.time())}"
    
    required_fields = ["target", "victim_token"]
    for field in required_fields:
        if field not in request:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    active_scans[scan_id] = {
        "type": "jwt_confusion",
        "status": "started",
        "user": current_user.get("username") if current_user else "anonymous",
        "start_time": datetime.utcnow(),
        "request": request
    }
    
    background_tasks.add_task(execute_jwt_confusion_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started", 
        "message": "JWT Confusion scan initiated",
        "timestamp": datetime.utcnow(),
        "estimated_duration": "15-60 seconds",
        "potential_value": "$1.8K-$7.5K"
    }

@router.post("/oauth-confusion")
async def oauth_confusion_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = None
):
    """
    OAuth 配置錯誤檢測
    
    檢測 OAuth/OIDC 實現中的配置錯誤，
    可能導致帳戶接管。
    
    預期市場價值: $2.5K-$10.2K
    """
    scan_id = f"oauth_confusion_{int(time.time())}"
    
    required_fields = ["target", "client_id", "legitimate_redirect", "attacker_redirect"]
    for field in required_fields:
        if field not in request:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    active_scans[scan_id] = {
        "type": "oauth_confusion",
        "status": "started",
        "user": current_user.get("username") if current_user else "anonymous",
        "start_time": datetime.utcnow(),
        "request": request
    }
    
    background_tasks.add_task(execute_oauth_confusion_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "OAuth Confusion scan initiated", 
        "timestamp": datetime.utcnow(),
        "estimated_duration": "20-90 seconds",
        "potential_value": "$2.5K-$10.2K"
    }

@router.post("/graphql-authz")
async def graphql_authz_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = None
):
    """
    GraphQL 權限檢測
    
    檢測 GraphQL 端點的權限控制缺陷，
    可能導致未授權數據存取。
    
    預期市場價值: $1.9K-$7.8K
    """
    scan_id = f"graphql_authz_{int(time.time())}"
    
    required_fields = ["target", "user_headers", "test_queries"]
    for field in required_fields:
        if field not in request:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    active_scans[scan_id] = {
        "type": "graphql_authz",
        "status": "started",
        "user": current_user.get("username") if current_user else "anonymous",
        "start_time": datetime.utcnow(),
        "request": request
    }
    
    background_tasks.add_task(execute_graphql_authz_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "GraphQL Authorization scan initiated",
        "timestamp": datetime.utcnow(),
        "estimated_duration": "30-120 seconds",
        "potential_value": "$1.9K-$7.8K"
    }

@router.post("/ssrf-oob")
async def ssrf_oob_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = None
):
    """
    SSRF OOB 檢測
    
    檢測服務器端請求偽造漏洞，
    使用 Out-of-Band 技術進行檢測。
    
    預期市場價值: $2.2K-$8.7K
    """
    scan_id = f"ssrf_oob_{int(time.time())}"
    
    required_fields = ["target", "oob_callback"]
    for field in required_fields:
        if field not in request:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    active_scans[scan_id] = {
        "type": "ssrf_oob",
        "status": "started",
        "user": current_user.get("username") if current_user else "anonymous",
        "start_time": datetime.utcnow(),
        "request": request
    }
    
    background_tasks.add_task(execute_ssrf_oob_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "SSRF OOB scan initiated",
        "timestamp": datetime.utcnow(),
        "estimated_duration": "45-180 seconds",
        "potential_value": "$2.2K-$8.7K"
    }

# === 背景任務執行函數 ===

async def execute_mass_assignment_scan(scan_id: str, request: dict):
    """執行 Mass Assignment 掃描"""
    try:
        manager = get_high_value_manager()
        
        result = manager.run_mass_assignment_test(
            target=request["target"],
            update_endpoint=request["update_endpoint"],
            auth_headers=request["auth_headers"],
            **{k: v for k, v in request.items() if k not in ["target", "update_endpoint", "auth_headers"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.utcnow()
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.utcnow()

async def execute_jwt_confusion_scan(scan_id: str, request: dict):
    """執行 JWT Confusion 掃描"""
    try:
        manager = get_high_value_manager()
        
        result = manager.run_jwt_confusion_test(
            target=request["target"],
            victim_token=request["victim_token"],
            **{k: v for k, v in request.items() if k not in ["target", "victim_token"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.utcnow()
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.utcnow()

async def execute_oauth_confusion_scan(scan_id: str, request: dict):
    """執行 OAuth Confusion 掃描"""
    try:
        manager = get_high_value_manager()
        
        result = manager.run_oauth_confusion_test(
            target=request["target"],
            client_id=request["client_id"],
            legitimate_redirect=request["legitimate_redirect"],
            attacker_redirect=request["attacker_redirect"],
            **{k: v for k, v in request.items() if k not in ["target", "client_id", "legitimate_redirect", "attacker_redirect"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.utcnow()
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.utcnow()

async def execute_graphql_authz_scan(scan_id: str, request: dict):
    """執行 GraphQL AuthZ 掃描"""
    try:
        manager = get_high_value_manager()
        
        result = manager.run_graphql_authz_test(
            target=request["target"],
            user_headers=request["user_headers"],
            test_queries=request["test_queries"],
            **{k: v for k, v in request.items() if k not in ["target", "user_headers", "test_queries"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.utcnow()
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.utcnow()

async def execute_ssrf_oob_scan(scan_id: str, request: dict):
    """執行 SSRF OOB 掃描"""
    try:
        manager = get_high_value_manager()
        
        result = manager.run_ssrf_oob_test(
            target=request["target"],
            oob_callback=request["oob_callback"],
            **{k: v for k, v in request.items() if k not in ["target", "oob_callback"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.utcnow()
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.utcnow()

# === 掃描管理端點 ===

@router.get("/scans/{scan_id}")
async def get_scan_status(scan_id: str, current_user: dict = None):
    """獲取掃描狀態"""
    if scan_id not in active_scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    scan_info = active_scans[scan_id]
    
    # 檢查用戶權限（如果有認證）
    if current_user and current_user.get("role") != "admin" and scan_info.get("user") != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "scan_id": scan_id,
        "type": scan_info["type"],
        "status": scan_info["status"],
        "user": scan_info["user"],
        "start_time": scan_info["start_time"],
        "result": scan_info.get("result"),
        "error": scan_info.get("error"),
        "timestamp": datetime.utcnow()
    }

@router.get("/scans")
async def list_scans(current_user: dict = None):
    """列出掃描記錄"""
    scans = []
    for scan_id, scan_info in active_scans.items():
        # 如果有認證，管理員可以看到所有掃描，用戶只能看到自己的
        if not current_user or current_user.get("role") == "admin" or scan_info.get("user") == current_user.get("username"):
            scans.append({
                "scan_id": scan_id,
                "type": scan_info["type"],
                "status": scan_info["status"],
                "user": scan_info["user"],
                "start_time": scan_info["start_time"]
            })
    
    return {
        "scans": scans,
        "total": len(scans),
        "timestamp": datetime.utcnow()
    }

# === 統計端點 ===

@router.get("/stats")
async def get_security_stats():
    """獲取安全檢測統計"""
    # 統計掃描類型
    scan_types = {}
    scan_statuses = {}
    vulnerability_values = {
        "mass_assignment": {"min": 2100, "max": 8200},
        "jwt_confusion": {"min": 1800, "max": 7500},
        "oauth_confusion": {"min": 2500, "max": 10200},
        "graphql_authz": {"min": 1900, "max": 7800},
        "ssrf_oob": {"min": 2200, "max": 8700}
    }
    
    for scan_info in active_scans.values():
        scan_type = scan_info["type"]
        scan_status = scan_info["status"]
        
        scan_types[scan_type] = scan_types.get(scan_type, 0) + 1
        scan_statuses[scan_status] = scan_statuses.get(scan_status, 0) + 1
    
    return {
        "total_scans": len(active_scans),
        "scan_types": scan_types,
        "scan_statuses": scan_statuses,
        "vulnerability_values": vulnerability_values,
        "potential_total_value": sum(v["max"] for v in vulnerability_values.values()),
        "timestamp": datetime.utcnow()
    }