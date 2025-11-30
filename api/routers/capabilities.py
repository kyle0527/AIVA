# -*- coding: utf-8 -*-
"""
AIVA API - Capability Execution Router

提供能力查詢和執行的 REST API 端點
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import asyncio
import time
from datetime import datetime
from urllib.parse import urlparse
import sys
import os

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.core.aiva_core.core_capabilities.capability_registry import CapabilityRegistry
from services.core.aiva_core.service_backbone.api.unified_function_caller import UnifiedFunctionCaller
from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
from services.core.aiva_core.cognitive_core.rag.unified_vector_store import UnifiedVectorStore
from services.aiva_common.config.settings import get_settings

logger = logging.getLogger(__name__)

# 創建路由器
router = APIRouter()

# 全域實例
_capability_registry = None
_function_caller = None
_internal_loop_connector = None


# === Pydantic Models ===

class CapabilityQueryRequest(BaseModel):
    """能力查詢請求"""
    query: str = Field(..., description="自然語言查詢描述，例如: 'SQL 注入檢測'")
    limit: int = Field(default=10, ge=1, le=50, description="返回結果數量限制")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="過濾條件 (module, language 等)")


class CapabilityExecuteRequest(BaseModel):
    """能力執行請求"""
    capability_id: Optional[int] = Field(default=None, description="能力 ID（從資料庫）")
    capability_name: Optional[str] = Field(default=None, description="能力名稱")
    module: str = Field(..., description="模組名稱")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="執行參數")
    timeout: int = Field(default=30, ge=1, le=300, description="超時時間（秒）")


class CapabilityInfo(BaseModel):
    """能力資訊"""
    id: Optional[int] = None
    name: str
    module: str
    language: str
    description: Optional[str] = None
    invocation_metadata: Optional[Dict[str, Any]] = None
    parameters: Optional[List[str]] = None


class CapabilityQueryResponse(BaseModel):
    """能力查詢響應"""
    query: str
    total_found: int
    capabilities: List[CapabilityInfo]
    timestamp: datetime


class CapabilityExecuteResponse(BaseModel):
    """能力執行響應"""
    success: bool
    capability_name: str
    module: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: datetime


# === 依賴注入 ===

async def get_capability_registry() -> CapabilityRegistry:
    """獲取能力註冊表"""
    global _capability_registry
    if _capability_registry is None:
        _capability_registry = CapabilityRegistry()
        # 載入能力
        load_result = await _capability_registry.load_from_exploration()
        logger.info(f"Loaded {load_result['capabilities_loaded']} capabilities")
    return _capability_registry


def get_function_caller() -> UnifiedFunctionCaller:
    """獲取統一功能調用器"""
    global _function_caller
    if _function_caller is None:
        _function_caller = UnifiedFunctionCaller()
    return _function_caller


def get_internal_loop_connector() -> InternalLoopConnector:
    """獲取內循環連接器"""
    global _internal_loop_connector
    if _internal_loop_connector is None:
        vector_store = UnifiedVectorStore()
        kb = KnowledgeBase(vector_store=vector_store)
        _internal_loop_connector = InternalLoopConnector(rag_knowledge_base=kb)
    return _internal_loop_connector


# === API 端點 ===

@router.post("/query", response_model=CapabilityQueryResponse)
async def query_capabilities(
    request: CapabilityQueryRequest
):
    """
    查詢可用能力
    
    根據自然語言描述查詢匹配的能力，支援過濾條件。
    返回能力清單包含 invocation 元數據，可直接用於執行。
    
    Args:
        request: 查詢請求（query, limit, filters）
    
    Returns:
        CapabilityQueryResponse: 匹配的能力清單
    
    Example:
        ```
        POST /api/v1/capabilities/query
        {
            "query": "SQL injection detection",
            "limit": 5,
            "filters": {"language": "Python"}
        }
        ```
    """
    try:
        logger.info(f"📥 Capability query: {request.query}")
        
        # 獲取內循環連接器（已包含 PostgreSQL 查詢功能）
        # connector = get_internal_loop_connector()  # 預留供未來使用
        
        # 從資料庫查詢能力
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="aiva_user",
            password="aiva_password",
            database="aiva_db"
        )
        
        # 構建查詢 SQL（使用全文搜索）
        filters_sql = ""
        if request.filters:
            if "language" in request.filters:
                filters_sql += f" AND language = '{request.filters['language']}'"
            if "module" in request.filters:
                filters_sql += f" AND module LIKE '%{request.filters['module']}%'"
        
        query_sql = f"""
            SELECT id, name, module, language, description, metadata_json, invocation_metadata
            FROM capability_records
            WHERE (name ILIKE '%{request.query}%' 
                   OR description ILIKE '%{request.query}%'
                   OR module ILIKE '%{request.query}%')
                {filters_sql}
            LIMIT {request.limit}
        """
        
        rows = await conn.fetch(query_sql)
        await conn.close()
        
        # 轉換為 CapabilityInfo
        capabilities = []
        for row in rows:
            cap = CapabilityInfo(
                id=row['id'],
                name=row['name'],
                module=row['module'],
                language=row['language'],
                description=row['description'],
                invocation_metadata=eval(row['invocation_metadata']) if row['invocation_metadata'] else None,
                parameters=[]  # 可從 metadata_json 解析
            )
            capabilities.append(cap)
        
        logger.info(f"✅ Found {len(capabilities)} capabilities")
        
        return CapabilityQueryResponse(
            query=request.query,
            total_found=len(capabilities),
            capabilities=capabilities,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/execute", response_model=CapabilityExecuteResponse)
async def execute_capability(
    request: CapabilityExecuteRequest
):
    """
    執行指定能力
    
    根據能力 ID 或名稱執行對應的功能模組，支援多語言（Python/Go/Rust/TypeScript）。
    使用 UnifiedFunctionCaller 進行跨語言調用，並返回執行結果。
    
    Args:
        request: 執行請求（capability_id/name, module, parameters, timeout）
    
    Returns:
        CapabilityExecuteResponse: 執行結果（success, result, error, execution_time）
    
    Example:
        ```
        POST /api/v1/capabilities/execute
        {
            "capability_name": "detect_sqli",
            "module": "function_sqli",
            "parameters": {
                "target_url": "http://localhost:8080/WebGoat/SqlInjection/attack1",
                "method": "POST",
                "payloads": ["' OR '1'='1", "admin'--"]
            },
            "timeout": 30
        }
        ```
    """
    try:
        start_time = time.time()
        
        # 1. 查詢能力資訊（獲取 invocation_metadata）
        if request.capability_id:
            logger.info(f"📥 Execute capability ID: {request.capability_id}")
            query_field = f"id = {request.capability_id}"
        elif request.capability_name:
            logger.info(f"📥 Execute capability: {request.capability_name}")
            query_field = f"name = '{request.capability_name}'"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide capability_id or capability_name"
            )
        
        # 從資料庫獲取 invocation_metadata
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="aiva_user",
            password="aiva_password",
            database="aiva_db"
        )
        
        query_sql = f"""
            SELECT name, module, language, invocation_metadata
            FROM capability_records
            WHERE {query_field}
        """
        
        row = await conn.fetchrow(query_sql)
        await conn.close()
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Capability not found"
            )
        
        capability_name = row['name']
        module = row['module']
        invocation_meta = eval(row['invocation_metadata']) if row['invocation_metadata'] else None
        
        if not invocation_meta:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Capability has no invocation metadata"
            )
        
        logger.info(f"📋 Invocation: {invocation_meta['protocol']} @ {invocation_meta['endpoint']}")
        
        # 2. 使用 UnifiedFunctionCaller 執行
        function_caller = get_function_caller()
        
        # 根據 protocol 調用不同方法
        protocol = invocation_meta['protocol']
        
        if protocol == 'unified_caller':
            # Python 直接調用
            result = await function_caller.call_python(
                module_name=invocation_meta['module_arg'],
                function_name=invocation_meta['function_arg'],
                **request.parameters
            )
        elif protocol == 'http':
            # HTTP API 調用
            result = await function_caller.call_http(
                module_name=module,
                function_name=capability_name,
                **request.parameters
            )
        elif protocol == 'grpc':
            # gRPC 調用
            result = await function_caller.call_grpc(
                module_name=module,
                function_name=capability_name,
                **request.parameters
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported protocol: {protocol}"
            )
        
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Execution completed in {execution_time:.2f}s")
        
        return CapabilityExecuteResponse(
            success=result.success,
            capability_name=capability_name,
            module=module,
            result=result.result if result.success else None,
            error=result.error if not result.success else None,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ Execution failed: {e}")
        
        return CapabilityExecuteResponse(
            success=False,
            capability_name=request.capability_name or "unknown",
            module=request.module,
            result=None,
            error=str(e),
            execution_time=execution_time,
            timestamp=datetime.now()
        )


@router.get("/list")
async def list_capabilities(
    language: Optional[str] = None,
    module: Optional[str] = None,
    limit: int = 50
):
    """
    列出所有可用能力
    
    返回能力清單，支援按語言和模組過濾。
    
    Args:
        language: 過濾語言（Python/Go/Rust/TypeScript）
        module: 過濾模組名稱
        limit: 返回數量限制（預設 50）
    
    Returns:
        Dict: {"total": int, "capabilities": List[CapabilityInfo]}
    """
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="aiva_user",
            password="aiva_password",
            database="aiva_db"
        )
        
        filters_sql = ""
        if language:
            filters_sql += f" WHERE language = '{language}'"
        if module:
            if filters_sql:
                filters_sql += f" AND module LIKE '%{module}%'"
            else:
                filters_sql += f" WHERE module LIKE '%{module}%'"
        
        query_sql = f"""
            SELECT id, name, module, language, description, invocation_metadata
            FROM capability_records
            {filters_sql}
            LIMIT {limit}
        """
        
        rows = await conn.fetch(query_sql)
        await conn.close()
        
        capabilities = []
        for row in rows:
            cap = CapabilityInfo(
                id=row['id'],
                name=row['name'],
                module=row['module'],
                language=row['language'],
                description=row['description'],
                invocation_metadata=eval(row['invocation_metadata']) if row['invocation_metadata'] else None
            )
            capabilities.append(cap)
        
        return {
            "total": len(capabilities),
            "capabilities": [cap.dict() for cap in capabilities],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ List failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List failed: {str(e)}"
        )


@router.get("/stats")
async def get_capability_stats():
    """
    獲取能力統計資訊
    
    返回系統能力的統計數據（總數、語言分佈、模組分佈等）。
    
    Returns:
        Dict: 統計資訊
    """
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="aiva_user",
            password="aiva_password",
            database="aiva_db"
        )
        
        # 總數統計
        total = await conn.fetchval("SELECT COUNT(*) FROM capability_records")
        
        # 語言分佈
        language_dist = await conn.fetch("""
            SELECT language, COUNT(*) as count
            FROM capability_records
            GROUP BY language
            ORDER BY count DESC
        """)
        
        # 模組分佈（Top 10）
        module_dist = await conn.fetch("""
            SELECT module, COUNT(*) as count
            FROM capability_records
            GROUP BY module
            ORDER BY count DESC
            LIMIT 10
        """)
        
        # invocation 覆蓋率
        invocation_coverage = await conn.fetchval("""
            SELECT COUNT(*) FROM capability_records
            WHERE invocation_metadata IS NOT NULL AND invocation_metadata != ''
        """)
        
        await conn.close()
        
        return {
            "total_capabilities": total,
            "invocation_coverage": f"{invocation_coverage}/{total} ({invocation_coverage/total*100:.1f}%)",
            "language_distribution": {row['language']: row['count'] for row in language_dist},
            "top_modules": {row['module']: row['count'] for row in module_dist},
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats failed: {str(e)}"
        )
