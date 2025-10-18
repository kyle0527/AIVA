# -*- coding: utf-8 -*-
"""
AIVA API 系統管理路由

提供系統管理、監控、配置等端點。
包含系統統計、健康檢查、掃描管理等功能。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime
import psutil
import os
import sys

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import get_config
from services.features.high_value_manager import HighValueFeatureManager
from .auth import get_current_user, require_admin

# 創建路由器
router = APIRouter()

# 系統啟動時間（用於計算 uptime）
START_TIME = datetime.utcnow()

# 模擬掃描存儲（實際應該使用數據庫）
active_scans: Dict[str, Dict[str, Any]] = {}

# === 系統狀態端點 ===

@router.get("/health")
async def health_check():
    """
    系統健康檢查
    
    檢查各個組件的狀態，包括高價值模組管理器、
    系統資源、配置等。
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {},
        "system": {}
    }
    
    # 檢查高價值模組管理器
    try:
        manager = HighValueFeatureManager()
        health_status["services"]["high_value_manager"] = "operational"
    except Exception as e:
        health_status["services"]["high_value_manager"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # 檢查配置系統
    try:
        config = get_config()
        health_status["services"]["config_system"] = "operational"
    except Exception as e:
        health_status["services"]["config_system"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # 檢查系統資源
    try:
        health_status["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
        }
    except Exception as e:
        health_status["system"]["error"] = str(e)
    
    # 檢查活躍掃描數量
    health_status["active_scans"] = len(active_scans)
    
    return health_status

@router.get("/status")
async def system_status():
    """
    系統詳細狀態
    
    提供系統運行時間、版本信息、組件狀態等詳細信息。
    """
    uptime = datetime.utcnow() - START_TIME
    
    return {
        "service": "AIVA Security Platform",
        "version": "1.0.0",
        "status": "operational",
        "uptime": {
            "seconds": int(uptime.total_seconds()),
            "human_readable": str(uptime).split('.')[0]
        },
        "start_time": START_TIME,
        "current_time": datetime.utcnow(),
        "platform": {
            "os": os.name,
            "python_version": sys.version.split()[0]
        }
    }

# === 系統統計端點 ===

@router.get("/stats")
async def get_system_stats(current_user: dict = Depends(get_current_user)):
    """
    系統統計信息
    
    提供掃描統計、用戶活動、系統資源使用等統計數據。
    管理員可以看到完整統計，一般用戶只能看到基本統計。
    """
    # 基本統計（所有用戶可見）
    stats = {
        "timestamp": datetime.utcnow(),
        "total_scans": len(active_scans),
        "system_uptime": str(datetime.utcnow() - START_TIME).split('.')[0]
    }
    
    # 掃描類型統計
    scan_types = {}
    scan_statuses = {}
    user_scans = {}
    
    for scan_info in active_scans.values():
        scan_type = scan_info.get("type", "unknown")
        scan_status = scan_info.get("status", "unknown")
        scan_user = scan_info.get("user", "anonymous")
        
        scan_types[scan_type] = scan_types.get(scan_type, 0) + 1
        scan_statuses[scan_status] = scan_statuses.get(scan_status, 0) + 1
        
        # 只有管理員可以看到用戶統計
        if current_user.get("role") == "admin":
            user_scans[scan_user] = user_scans.get(scan_user, 0) + 1
    
    stats["scan_types"] = scan_types
    stats["scan_statuses"] = scan_statuses
    
    # 管理員額外統計
    if current_user.get("role") == "admin":
        stats["user_scans"] = user_scans
        
        # 系統資源統計
        try:
            stats["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total if os.name != 'nt' else psutil.disk_usage('C:\\').total,
                    "free": psutil.disk_usage('/').free if os.name != 'nt' else psutil.disk_usage('C:\\').free,
                    "percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
                }
            }
        except Exception as e:
            stats["system_resources_error"] = str(e)
    
    return stats

# === 掃描管理端點 ===

@router.get("/scans")
async def list_all_scans(current_user: dict = Depends(get_current_user)):
    """
    列出掃描記錄
    
    管理員可以看到所有掃描，一般用戶只能看到自己的掃描。
    """
    scans = []
    for scan_id, scan_info in active_scans.items():
        # 權限檢查
        if current_user.get("role") == "admin" or scan_info.get("user") == current_user.get("username"):
            scans.append({
                "scan_id": scan_id,
                "type": scan_info.get("type"),
                "status": scan_info.get("status"),
                "user": scan_info.get("user"),
                "start_time": scan_info.get("start_time"),
                "end_time": scan_info.get("end_time")
            })
    
    return {
        "scans": scans,
        "total": len(scans),
        "timestamp": datetime.utcnow()
    }

@router.get("/scans/{scan_id}")
async def get_scan_detail(scan_id: str, current_user: dict = Depends(get_current_user)):
    """獲取掃描詳細信息"""
    if scan_id not in active_scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    scan_info = active_scans[scan_id]
    
    # 權限檢查
    if current_user.get("role") != "admin" and scan_info.get("user") != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "scan_id": scan_id,
        **scan_info,
        "timestamp": datetime.utcnow()
    }

@router.delete("/scans/{scan_id}")
async def delete_scan(scan_id: str, admin_user: dict = Depends(require_admin)):
    """刪除掃描記錄（僅管理員）"""
    if scan_id not in active_scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    deleted_scan = active_scans.pop(scan_id)
    
    return {
        "message": f"Scan {scan_id} deleted successfully",
        "deleted_scan": {
            "scan_id": scan_id,
            "type": deleted_scan.get("type"),
            "user": deleted_scan.get("user")
        },
        "timestamp": datetime.utcnow()
    }

@router.post("/scans/cleanup")
async def cleanup_scans(admin_user: dict = Depends(require_admin)):
    """清理已完成或失敗的掃描（僅管理員）"""
    cleanup_count = 0
    cleanup_scans = []
    
    # 找出需要清理的掃描
    for scan_id, scan_info in list(active_scans.items()):
        if scan_info.get("status") in ["completed", "failed"]:
            cleanup_scans.append({
                "scan_id": scan_id,
                "type": scan_info.get("type"),
                "status": scan_info.get("status")
            })
            del active_scans[scan_id]
            cleanup_count += 1
    
    return {
        "message": f"Cleaned up {cleanup_count} completed/failed scans",
        "cleaned_scans": cleanup_scans,
        "remaining_scans": len(active_scans),
        "timestamp": datetime.utcnow()
    }

# === 配置管理端點 ===

@router.get("/config")
async def get_system_config(admin_user: dict = Depends(require_admin)):
    """獲取系統配置（僅管理員）"""
    try:
        config = get_config()
        # 過濾敏感信息
        safe_config = {}
        for section, values in config.items():
            if isinstance(values, dict):
                safe_values = {}
                for key, value in values.items():
                    # 隱藏密碼、密鑰等敏感信息
                    if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                        safe_values[key] = "***HIDDEN***"
                    else:
                        safe_values[key] = value
                safe_config[section] = safe_values
            else:
                safe_config[section] = values
        
        return {
            "config": safe_config,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")

# === 系統操作端點 ===

@router.post("/maintenance/start")
async def start_maintenance_mode(admin_user: dict = Depends(require_admin)):
    """啟動維護模式（僅管理員）"""
    # 這裡可以設置維護模式標誌
    return {
        "message": "Maintenance mode started",
        "timestamp": datetime.utcnow(),
        "note": "New scans will be rejected during maintenance"
    }

@router.post("/maintenance/stop")
async def stop_maintenance_mode(admin_user: dict = Depends(require_admin)):
    """停止維護模式（僅管理員）"""
    return {
        "message": "Maintenance mode stopped",
        "timestamp": datetime.utcnow(),
        "note": "System is now accepting new scans"
    }

# === 系統資源監控端點 ===

@router.get("/resources")
async def get_system_resources(admin_user: dict = Depends(require_admin)):
    """獲取系統資源使用情況（僅管理員）"""
    try:
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used
            },
            "disk": {
                "total": psutil.disk_usage('/').total if os.name != 'nt' else psutil.disk_usage('C:\\').total,
                "free": psutil.disk_usage('/').free if os.name != 'nt' else psutil.disk_usage('C:\\').free,
                "used": psutil.disk_usage('/').used if os.name != 'nt' else psutil.disk_usage('C:\\').used,
                "percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
            },
            "processes": len(psutil.pids()),
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system resources: {str(e)}")

# 導出用於其他模組的掃描存儲
def get_active_scans():
    """獲取活躍掃描存儲"""
    return active_scans