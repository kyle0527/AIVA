# -*- coding: utf-8 -*-
"""
AIVA REST API 主應用

提供完整的 REST API 介面，支援高價值功能模組、傳統安全檢測、
系統管理和用戶認證。專門為商業化部署設計。
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any, List, Optional
import asyncio
import time
import json
from datetime import datetime, timedelta, UTC
import jwt
import hashlib
import os
import sys

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from config.api_keys import get_api_key, has_api_key
from services.features.high_value_manager import HighValueFeatureManager
from services.features.base.result_schema import FeatureResult
from services.aiva_common.schemas import APIResponse

# 初始化配置
config = get_config()
security = HTTPBearer()

# 創建 FastAPI 應用
app = FastAPI(
    title="AIVA Security Platform API",
    description="Advanced Intelligent Vulnerability Assessment - Commercial API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", "cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域狀態
active_scans: Dict[str, Dict[str, Any]] = {}
high_value_manager = None

# JWT 密鑰
JWT_SECRET = get_api_key("jwt_secret", "aiva-default-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# === 認證系統 ===

def create_access_token(data: dict) -> str:
    """創建 JWT 訪問令牌"""
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """驗證 JWT 令牌"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

def get_current_user(payload: dict = Depends(verify_token)) -> dict:
    """獲取當前用戶"""
    return {
        "user_id": payload.get("sub"),
        "username": payload.get("username", "unknown"),
        "role": payload.get("role", "user")
    }

# === 基礎端點 ===

@app.get("/")
async def root():
    """API 根端點"""
    return {
        "message": "AIVA Security Platform API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now(UTC).isoformat(),
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    global high_value_manager
    
    # 初始化高價值模組管理器
    if high_value_manager is None:
        try:
            allowlist_domains = config.get("features", "allowlist_domains", [])
            if allowlist_domains:
                os.environ["ALLOWLIST_DOMAINS"] = ",".join(allowlist_domains)
            high_value_manager = HighValueFeatureManager()
        except Exception as e:
            response = APIResponse(
                success=False,
                message=f"Failed to initialize high value manager: {str(e)}",
                data={
                    "status": "unhealthy",
                    "services": {}
                }
            )
            return JSONResponse(
                status_code=503,
                content=response.model_dump()
            )
    
    response = APIResponse(
        success=True,
        message="All services operational",
        data={
            "status": "healthy",
            "services": {
                "high_value_manager": "operational",
                "api_gateway": "operational",
                "authentication": "operational"
            },
            "active_scans": len(active_scans)
        }
    )
    return response.model_dump()

# === 認證端點 ===

@app.post("/auth/login")
async def login(credentials: dict):
    """用戶登入"""
    username = credentials.get("username")
    password = credentials.get("password")
    
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password required"
        )
    
    # 簡化認證（生產環境需要實際的用戶資料庫）
    if username == "admin" and password == "aiva-admin-2025":
        token_data = {
            "sub": "admin",
            "username": username,
            "role": "admin"
        }
        access_token = create_access_token(token_data)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRATION_HOURS * 3600,
            "user": {
                "username": username,
                "role": "admin"
            }
        }
    elif username == "user" and password == "aiva-user-2025":
        token_data = {
            "sub": "user",
            "username": username,
            "role": "user"
        }
        access_token = create_access_token(token_data)
        return {
            "access_token": access_token,
            "token_type": "bearer", 
            "expires_in": JWT_EXPIRATION_HOURS * 3600,
            "user": {
                "username": username,
                "role": "user"
            }
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """獲取當前用戶信息"""
    return {
        "user": current_user,
        "timestamp": datetime.now(UTC).isoformat()
    }

# === 高價值功能模組 API ===

@app.post("/api/v1/security/mass-assignment")
async def mass_assignment_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Mass Assignment 權限提升檢測"""
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
        "user": current_user["username"],
        "start_time": datetime.now(UTC),
        "request": request
    }
    
    # 後台執行掃描
    background_tasks.add_task(execute_mass_assignment_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "Mass Assignment scan initiated",
        "timestamp": datetime.now(UTC),
        "estimated_duration": "30-120 seconds"
    }

@app.post("/api/v1/security/jwt-confusion")
async def jwt_confusion_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """JWT 混淆攻擊檢測"""
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
        "user": current_user["username"],
        "start_time": datetime.now(UTC),
        "request": request
    }
    
    background_tasks.add_task(execute_jwt_confusion_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started", 
        "message": "JWT Confusion scan initiated",
        "timestamp": datetime.now(UTC),
        "estimated_duration": "15-60 seconds"
    }

@app.post("/api/v1/security/oauth-confusion")
async def oauth_confusion_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """OAuth 配置錯誤檢測"""
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
        "user": current_user["username"],
        "start_time": datetime.now(UTC),
        "request": request
    }
    
    background_tasks.add_task(execute_oauth_confusion_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "OAuth Confusion scan initiated", 
        "timestamp": datetime.now(UTC),
        "estimated_duration": "20-90 seconds"
    }

@app.post("/api/v1/security/graphql-authz")
async def graphql_authz_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """GraphQL 權限檢測"""
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
        "user": current_user["username"],
        "start_time": datetime.now(UTC),
        "request": request
    }
    
    background_tasks.add_task(execute_graphql_authz_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "GraphQL Authorization scan initiated",
        "timestamp": datetime.now(UTC),
        "estimated_duration": "30-120 seconds"
    }

@app.post("/api/v1/security/ssrf-oob")
async def ssrf_oob_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """SSRF OOB 檢測"""
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
        "user": current_user["username"],
        "start_time": datetime.now(UTC),
        "request": request
    }
    
    background_tasks.add_task(execute_ssrf_oob_scan, scan_id, request)
    
    return {
        "scan_id": scan_id,
        "status": "started",
        "message": "SSRF OOB scan initiated",
        "timestamp": datetime.now(UTC),
        "estimated_duration": "45-180 seconds"
    }

# === 掃描狀態和結果端點 ===

@app.get("/api/v1/scans/{scan_id}")
async def get_scan_status(scan_id: str, current_user: dict = Depends(get_current_user)):
    """獲取掃描狀態"""
    if scan_id not in active_scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    scan_info = active_scans[scan_id]
    
    # 檢查用戶權限
    if current_user["role"] != "admin" and scan_info.get("user") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "scan_id": scan_id,
        "type": scan_info["type"],
        "status": scan_info["status"],
        "user": scan_info["user"],
        "start_time": scan_info["start_time"],
        "result": scan_info.get("result"),
        "error": scan_info.get("error"),
        "timestamp": datetime.now(UTC)
    }

@app.get("/api/v1/scans")
async def list_scans(current_user: dict = Depends(get_current_user)):
    """列出掃描記錄"""
    scans = []
    for scan_id, scan_info in active_scans.items():
        # 管理員可以看到所有掃描，用戶只能看到自己的
        if current_user["role"] == "admin" or scan_info.get("user") == current_user["username"]:
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
        "timestamp": datetime.now(UTC)
    }

# === 背景任務執行函數 ===

def execute_mass_assignment_scan(scan_id: str, request: dict):
    """執行 Mass Assignment 掃描"""
    try:
        global high_value_manager
        if high_value_manager is None:
            high_value_manager = HighValueFeatureManager()
        
        result = high_value_manager.run_mass_assignment_test(
            target=request["target"],
            update_endpoint=request["update_endpoint"],
            auth_headers=request["auth_headers"],
            **{k: v for k, v in request.items() if k not in ["target", "update_endpoint", "auth_headers"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.now(UTC)
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.now(UTC)

def execute_jwt_confusion_scan(scan_id: str, request: dict):
    """執行 JWT Confusion 掃描"""
    try:
        global high_value_manager
        if high_value_manager is None:
            high_value_manager = HighValueFeatureManager()
        
        result = high_value_manager.run_jwt_confusion_test(
            target=request["target"],
            victim_token=request["victim_token"],
            **{k: v for k, v in request.items() if k not in ["target", "victim_token"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.now(UTC)
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.now(UTC)

def execute_oauth_confusion_scan(scan_id: str, request: dict):
    """執行 OAuth Confusion 掃描"""
    try:
        global high_value_manager
        if high_value_manager is None:
            high_value_manager = HighValueFeatureManager()
        
        result = high_value_manager.run_oauth_confusion_test(
            target=request["target"],
            client_id=request["client_id"],
            legitimate_redirect=request["legitimate_redirect"],
            attacker_redirect=request["attacker_redirect"],
            **{k: v for k, v in request.items() if k not in ["target", "client_id", "legitimate_redirect", "attacker_redirect"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.now(UTC)
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.now(UTC)

def execute_graphql_authz_scan(scan_id: str, request: dict):
    """執行 GraphQL AuthZ 掃描"""
    try:
        global high_value_manager
        if high_value_manager is None:
            high_value_manager = HighValueFeatureManager()
        
        result = high_value_manager.run_graphql_authz_test(
            target=request["target"],
            user_headers=request["user_headers"],
            test_queries=request["test_queries"],
            **{k: v for k, v in request.items() if k not in ["target", "user_headers", "test_queries"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.now(UTC)
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.now(UTC)

def execute_ssrf_oob_scan(scan_id: str, request: dict):
    """執行 SSRF OOB 掃描"""
    try:
        global high_value_manager
        if high_value_manager is None:
            high_value_manager = HighValueFeatureManager()
        
        result = high_value_manager.run_ssrf_oob_test(
            target=request["target"],
            oob_callback=request["oob_callback"],
            **{k: v for k, v in request.items() if k not in ["target", "oob_callback"]}
        )
        
        active_scans[scan_id]["status"] = "completed"
        active_scans[scan_id]["result"] = result.to_dict() if hasattr(result, 'to_dict') else result
        active_scans[scan_id]["end_time"] = datetime.now(UTC)
        
    except Exception as e:
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)
        active_scans[scan_id]["end_time"] = datetime.now(UTC)

# === 系統管理端點 ===

@app.get("/api/v1/admin/stats")
async def get_system_stats(current_user: dict = Depends(get_current_user)):
    """獲取系統統計信息（僅管理員）"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # 統計掃描類型
    scan_types = {}
    scan_statuses = {}
    
    for scan_info in active_scans.values():
        scan_type = scan_info["type"]
        scan_status = scan_info["status"]
        
        scan_types[scan_type] = scan_types.get(scan_type, 0) + 1
        scan_statuses[scan_status] = scan_statuses.get(scan_status, 0) + 1
    
    return {
        "total_scans": len(active_scans),
        "scan_types": scan_types,
        "scan_statuses": scan_statuses,
        "uptime": "system_uptime", # 可以添加實際的 uptime 計算
        "timestamp": datetime.now(UTC)
    }

@app.delete("/api/v1/admin/scans/{scan_id}")
async def delete_scan(scan_id: str, current_user: dict = Depends(get_current_user)):
    """刪除掃描記錄（僅管理員）"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if scan_id not in active_scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    del active_scans[scan_id]
    return {"message": f"Scan {scan_id} deleted", "timestamp": datetime.now(UTC)}

# 整合路由模組
from api.routers import auth, security, admin
# from api.routers import capabilities  # 暫時禁用 - 啟動時有問題

# 添加路由
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(security.router, prefix="/api/v1/security", tags=["High-Value Security"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["System Administration"])
# app.include_router(capabilities.router, prefix="/api/v1/capabilities", tags=["Capability Execution"])  # 暫時禁用

# 更新主要掃描端點以使用共享存儲
from api.routers.admin import get_active_scans
active_scans = get_active_scans()

# === 啟動配置 ===

if __name__ == "__main__":
    # 從配置文件讀取設置
    host = config.get("api", "host", "0.0.0.0")
    port = config.get("api", "port", 8000)
    debug = config.get("api", "debug", False)
    
    print(f"🚀 Starting AIVA Security Platform API on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print("🔒 Authentication required for all endpoints except /health")
    print("👤 Default credentials - admin:aiva-admin-2025, user:aiva-user-2025")
    
    uvicorn.run(
        app,  # 直接傳遞 app 對象而不是字符串
        host=host,
        port=port,
        reload=debug,
        access_log=True
    )
