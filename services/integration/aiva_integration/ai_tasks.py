#!/usr/bin/env python3
"""
AI Tasks API Routes

提供 AI 任務調度的 REST API 接口
符合計劃中的 services/api/routes/ai_tasks.py 規範
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from .unified_data_manager_v2 import UnifiedDataManagerV2

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["AI Tasks"])


# 請求模型
class AITaskRequest(BaseModel):
    """AI 任務請求"""
    description: str = Field(..., description="任務描述")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="任務參數")
    task_id: Optional[str] = Field(None, description="可選的任務 ID")


class ScanRequest(BaseModel):
    """掃描請求"""
    target: str = Field(..., description="掃描目標")
    scan_type: str = Field(default="passive", description="掃描類型 (passive/active)")


class AnalysisRequest(BaseModel):
    """分析請求"""
    target: str = Field(..., description="分析目標")
    depth: str = Field(default="basic", description="分析深度 (basic/deep)")


class AttackPlanRequest(BaseModel):
    """攻擊計劃請求"""
    target: str = Field(..., description="攻擊目標")
    attack_type: str = Field(default="auto", description="攻擊類型")


# 全局 Data Manager 實例
_data_manager: Optional[UnifiedDataManagerV2] = None


async def get_data_manager() -> UnifiedDataManagerV2:
    """獲取 Data Manager 實例"""
    global _data_manager
    
    if _data_manager is None:
        _data_manager = UnifiedDataManagerV2()
        await _data_manager.initialize_ai()
    
    return _data_manager


@router.post("/tasks/execute", response_model=Dict[str, Any])
async def execute_ai_task(request: AITaskRequest):
    """執行 AI 任務
    
    通用的 AI 任務執行接口，支持所有類型的任務。
    
    Args:
        request: AI 任務請求
    
    Returns:
        任務執行結果
    
    Example:
        ```
        POST /api/v1/ai/tasks/execute
        {
            "description": "Analyze code vulnerabilities",
            "parameters": {
                "target": "code.py",
                "depth": "basic"
            }
        }
        ```
    """
    try:
        manager = await get_data_manager()
        
        result = await manager.execute_ai_task(
            description=request.description,
            parameters=request.parameters,
            task_id=request.task_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"AI task execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan", response_model=Dict[str, Any])
async def execute_scan(request: ScanRequest):
    """執行掃描任務
    
    快捷掃描接口，支持被動和主動掃描。
    
    Args:
        request: 掃描請求
    
    Returns:
        掃描結果
    
    Example:
        ```
        POST /api/v1/ai/scan
        {
            "target": "example.com",
            "scan_type": "passive"
        }
        ```
    """
    try:
        manager = await get_data_manager()
        
        result = await manager.execute_scan(
            target=request.target,
            scan_type=request.scan_type
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Scan execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_code(request: AnalysisRequest):
    """分析代碼漏洞
    
    快捷代碼分析接口。
    
    Args:
        request: 分析請求
    
    Returns:
        分析結果
    
    Example:
        ```
        POST /api/v1/ai/analyze
        {
            "target": "src/app.py",
            "depth": "deep"
        }
        ```
    """
    try:
        manager = await get_data_manager()
        
        result = await manager.analyze_code_vulnerabilities(
            target=request.target,
            depth=request.depth
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Code analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attack/plan", response_model=Dict[str, Any])
async def plan_attack(request: AttackPlanRequest):
    """規劃攻擊策略
    
    使用 AI 規劃攻擊路徑。
    
    Args:
        request: 攻擊計劃請求
    
    Returns:
        攻擊計劃
    
    Example:
        ```
        POST /api/v1/ai/attack/plan
        {
            "target": "example.com",
            "attack_type": "xss"
        }
        ```
    """
    try:
        manager = await get_data_manager()
        
        result = await manager.plan_attack_strategy(
            target=request.target,
            attack_type=request.attack_type
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Attack planning failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/status", response_model=Dict[str, Any])
async def get_task_status(task_id: str):
    """獲取任務狀態
    
    查詢指定任務的執行狀態。
    
    Args:
        task_id: 任務 ID
    
    Returns:
        任務狀態信息
    
    Example:
        ```
        GET /api/v1/ai/tasks/task_12345/status
        ```
    """
    try:
        manager = await get_data_manager()
        
        status = await manager.get_task_status(task_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}", response_model=Dict[str, Any])
async def cancel_task(task_id: str):
    """取消任務
    
    取消正在執行的任務。
    
    Args:
        task_id: 任務 ID
    
    Returns:
        取消結果
    
    Example:
        ```
        DELETE /api/v1/ai/tasks/task_12345
        ```
    """
    try:
        manager = await get_data_manager()
        
        success = await manager.cancel_task(task_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to cancel task {task_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plugins", response_model=Dict[str, Any])
async def list_plugins():
    """列出可用的 AI 插件
    
    獲取所有已註冊的 AI 插件列表。
    
    Returns:
        插件列表
    
    Example:
        ```
        GET /api/v1/ai/plugins
        ```
    """
    try:
        manager = await get_data_manager()
        
        plugins = await manager.list_available_ai_plugins()
        
        return {
            "total": len(plugins),
            "plugins": plugins
        }
        
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plugins/{plugin_id}", response_model=Dict[str, Any])
async def get_plugin_info(plugin_id: str):
    """獲取插件詳細信息
    
    查詢指定插件的元數據和狀態。
    
    Args:
        plugin_id: 插件 ID
    
    Returns:
        插件信息
    
    Example:
        ```
        GET /api/v1/ai/plugins/bio_neuron
        ```
    """
    try:
        manager = await get_data_manager()
        
        info = await manager.get_ai_plugin_info(plugin_id)
        
        if info is None:
            raise HTTPException(status_code=404, detail=f"Plugin {plugin_id} not found")
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """AI 系統健康檢查
    
    檢查 AI Commander 和所有插件的健康狀態。
    
    Returns:
        健康狀態報告
    
    Example:
        ```
        GET /api/v1/ai/health
        ```
    """
    try:
        manager = await get_data_manager()
        
        health = await manager.health_check()
        
        return health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
