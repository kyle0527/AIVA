# -*- coding: utf-8 -*-
"""
AIVA API 認證路由

提供 JWT 基礎的認證系統，支援用戶登入、令牌驗證、
用戶管理等功能。支援多角色權限控制。
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
import jwt
from datetime import UTC, datetime, timedelta
import hashlib
import os
import sys

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.api_keys import get_api_key

# 使用標準合約
from services.aiva_common.schemas import (
    APIResponse,
    Authentication,
    MessageHeader,
    ExecutionError
)

# 創建路由器
router = APIRouter()
security = HTTPBearer()

# JWT 配置
JWT_SECRET = get_api_key("jwt_secret", "aiva-default-secret-change-in-production-2025")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# 請求模型 - 使用標準Authentication合約概念
class LoginRequest(BaseModel):
    """登入請求 - 基於標準Authentication合約"""
    username: str = Field(description="用戶名", min_length=1, max_length=50)
    password: str = Field(description="密碼", min_length=1, max_length=100)

class TokenResponse(BaseModel):
    """令牌響應 - 整合MessageHeader概念"""
    access_token: str = Field(description="訪問令牌")
    token_type: str = Field(default="bearer", description="令牌類型")
    expires_in: int = Field(description="過期時間(秒)")
    user: dict = Field(description="用戶信息")
    issued_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="簽發時間")

class UserInfo(BaseModel):
    """用戶信息 - 標準化用戶資料合約"""
    user_id: str = Field(description="用戶ID")
    username: str = Field(description="用戶名")
    role: str = Field(description="角色")
    permissions: list[str] = Field(default_factory=list, description="權限列表")
    last_login: Optional[datetime] = Field(None, description="最後登入時間")

# 默認用戶（生產環境應該使用數據庫）
DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("aiva-admin-2025".encode()).hexdigest(),
        "role": "admin",
        "permissions": ["read", "write", "admin", "delete"]
    },
    "user": {
        "password_hash": hashlib.sha256("aiva-user-2025".encode()).hexdigest(),
        "role": "user", 
        "permissions": ["read", "write"]
    },
    "viewer": {
        "password_hash": hashlib.sha256("aiva-viewer-2025".encode()).hexdigest(),
        "role": "viewer",
        "permissions": ["read"]
    }
}

def create_access_token(data: dict) -> str:
    """創建 JWT 訪問令牌"""
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire, "iat": datetime.now(UTC)})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """驗證密碼"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """認證用戶"""
    user = DEFAULT_USERS.get(username)
    if not user:
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    return {
        "username": username,
        "role": user["role"],
        "permissions": user["permissions"]
    }

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """驗證 JWT 令牌"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("username")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(payload: dict = Depends(verify_token)) -> dict:
    """獲取當前用戶"""
    return {
        "user_id": payload.get("sub"),
        "username": payload.get("username", "unknown"),
        "role": payload.get("role", "user"),
        "permissions": payload.get("permissions", [])
    }

def require_permission(required_permission: str):
    """需要特定權限的依賴函數"""
    def permission_checker(current_user: dict = Depends(get_current_user)):
        if required_permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{required_permission}' required"
            )
        return current_user
    return permission_checker

def require_admin(current_user: dict = Depends(get_current_user)):
    """需要管理員權限"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# === 認證端點 ===

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    用戶登入
    
    支援的默認帳戶:
    - admin/aiva-admin-2025: 管理員權限
    - user/aiva-user-2025: 一般用戶權限  
    - viewer/aiva-viewer-2025: 唯讀權限
    """
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = {
        "sub": user["username"],
        "username": user["username"],
        "role": user["role"],
        "permissions": user["permissions"]
    }
    
    access_token = create_access_token(token_data)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user={
            "username": user["username"],
            "role": user["role"],
            "permissions": user["permissions"]
        },
        issued_at=datetime.now(UTC)
    )

@router.get("/me", response_model=APIResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """獲取當前用戶信息"""
    user_info = UserInfo(
        user_id=current_user["user_id"] or current_user["username"],
        username=current_user["username"],
        role=current_user["role"],
        permissions=current_user.get("permissions", []),
        last_login=datetime.now(UTC)
    )
    
    return APIResponse(
        success=True,
        message="用戶信息獲取成功",
        data=user_info.model_dump()
    )

@router.post("/refresh", response_model=APIResponse)
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """刷新令牌"""
    token_data = {
        "sub": current_user["username"],
        "username": current_user["username"],
        "role": current_user["role"],
        "permissions": current_user["permissions"]
    }
    
    access_token = create_access_token(token_data)
    
    token_response = TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=current_user,
        issued_at=datetime.now(UTC)
    )
    
    return APIResponse(
        success=True,
        message="令牌刷新成功",
        data=token_response.model_dump()
    )

@router.post("/logout", response_model=APIResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """用戶登出（客戶端需要刪除令牌）"""
    return APIResponse(
        success=True,
        message="登出成功",
        data={
            "username": current_user["username"],
            "logout_time": datetime.now(UTC)
        }
    )

# === 權限檢查端點 ===

@router.get("/permissions", response_model=APIResponse)
async def get_user_permissions(current_user: dict = Depends(get_current_user)):
    """獲取用戶權限"""
    return APIResponse(
        success=True,
        message="權限查詢成功",
        data={
            "username": current_user["username"],
            "role": current_user["role"],
            "permissions": current_user["permissions"]
        }
    )

@router.get("/check-permission/{permission}", response_model=APIResponse)
async def check_permission(permission: str, current_user: dict = Depends(get_current_user)):
    """檢查特定權限"""
    has_permission = permission in current_user.get("permissions", [])
    
    return APIResponse(
        success=True,
        message=f"權限檢查完成: {'擁有' if has_permission else '缺少'} {permission} 權限",
        data={
            "username": current_user["username"],
            "permission": permission,
            "has_permission": has_permission
        }
    )

# === 用戶管理端點（僅管理員） ===

@router.get("/users", response_model=APIResponse)
async def list_users(admin_user: dict = Depends(require_admin)):
    """列出所有用戶（僅管理員）"""
    users = []
    for username, user_data in DEFAULT_USERS.items():
        users.append({
            "username": username,
            "role": user_data["role"],
            "permissions": user_data["permissions"]
        })
    
    return APIResponse(
        success=True,
        message=f"用戶列表查詢成功，共 {len(users)} 個用戶",
        data={
            "users": users,
            "total": len(users)
        }
    )

@router.get("/users/{username}", response_model=APIResponse)
async def get_user(username: str, admin_user: dict = Depends(require_admin)):
    """獲取特定用戶信息（僅管理員）"""
    user = DEFAULT_USERS.get(username)
    if not user:
        return APIResponse(
            success=False,
            message=f"用戶 {username} 不存在",
            data=None
        )
    
    return APIResponse(
        success=True,
        message=f"用戶 {username} 信息查詢成功",
        data={
            "username": username,
            "role": user["role"],
            "permissions": user["permissions"]
        }
    )

# === 認證狀態端點 ===

@router.get("/status", response_model=APIResponse)
async def auth_status():
    """認證系統狀態"""
    return APIResponse(
        success=True,
        message="認證系統運行正常",
        data={
            "service": "AIVA Authentication",
            "status": "operational",
            "jwt_algorithm": JWT_ALGORITHM,
            "token_expiration_hours": JWT_EXPIRATION_HOURS,
            "available_roles": ["admin", "user", "viewer"],
            "available_permissions": ["read", "write", "admin", "delete"]
        }
    )

# 導出認證依賴函數供其他模組使用
__all__ = [
    'router',
    'get_current_user', 
    'require_permission',
    'require_admin',
    'verify_token'
]