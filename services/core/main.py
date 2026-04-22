"""
AIVA 系统对外入口（安全防护層）
====================================

版本: 2.0.0
作用: 整个程序的对外 HTTP 入口，提供第一道安全防线

架构設計（統一入口點 + 雙層安全）:
- main.py (本脚本 Port 9000): 第一道安全防线，检测恶意请求后转发
- app.py (AI核心 Port 8000): 程序與 AI 溝通的接口（AI 可插拔，实现双路分离）

✅ v2.0 更新 (2026-02-03):
  - main.py 啟動時自動檢查 app.py 是否就緒
  - 提供統一的啟動腳本建議
  - 整合 ai_menu.py CLI 介面說明
  - 連接到 scan 模組的 MultiEngineCoordinator

安全职責（為什麼分兩個脚本）:
  ✓ 檢測惡意請求（木馬、注入攻擊、異常參數）
  ✓ 速率限制和訪問控制
  ✓ 請求白名單/黑名單
  ✓ 保護 AI 系統不受污染

數據流（三層架構 + 雙路分離）:
  外部 HTTP 請求 / CLI 選單
      ↓
  main.py (Port 9000 - 第一道防線)
      ↓ 通過檢測後轉發
      ↓
  app.py (Port 8000 - AI 核心服務)
      ↓
  雙路分離處理:
    ├─ 路徑1: 整合模組存儲（實時記錄）
    ├─ 路徑2: AI 決策引擎 → scan 模組 → 執行
    └─ 路徑3: Internal/External Loop（學習系統）

三種使用方式:
  1. HTTP API: 外部系統 → main.py (9000) → app.py (8000)
  2. CLI 選單: python ai_menu.py → app.py (8000) 直接調用
  3. 直接調用: 測試環境 → app.py (8000) 直接訪問
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field

# 使用 aiva_common 統一的掃描請求/響應模型
from aiva_common.schemas import ScanRequest, ScanResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 应用
app = FastAPI(
    title="AIVA 系统入口",
    version="2.0.0",
    description="AIVA 智能渗透测试系统的唯一对外入口（安全網關層）"
)

# AI 系统地址（内部）
AI_CORE_URL = "http://localhost:8000"  # AI 系统运行在 8000 端口


# ============================================================================
# 启动前置检查
# ============================================================================

async def check_app_py_availability() -> bool:
    """檢查 app.py (Port 8000) 是否就緒
    
    Returns:
        bool: app.py 是否可用
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AI_CORE_URL}/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


@app.on_event("startup")
async def startup_checks():
    """啟動前置檢查 - 確保 app.py 已啟動"""
    logger.info("🚀 [main.py] AIVA External Gateway starting up...")
    logger.info(f"🔗 [main.py] Checking app.py availability at {AI_CORE_URL}...")
    
    # 檢查 app.py 是否就緒
    app_py_ready = await check_app_py_availability()
    
    if not app_py_ready:
        logger.warning("⚠️  [main.py] app.py (Port 8000) is NOT running!")
        logger.warning("⚠️  [main.py] main.py will start but requests will fail")
        logger.warning("")
        logger.warning("📋 [建議] 請先啟動 app.py:")
        logger.warning("   1. 打開新的終端")
        logger.warning("   2. cd services/core")
        logger.warning("   3. python aiva_core/service_backbone/api/app.py")
        logger.warning("")
        logger.warning("💡 [提示] 或者使用 CLI 選單模式:")
        logger.warning("   python services/core/aiva_core/core_capabilities/dialog/ai_menu.py")
        logger.warning("")
    else:
        logger.info("✅ [main.py] app.py (Port 8000) is ready!")
        logger.info("✅ [main.py] AIVA External Gateway is ready to accept requests")


# ============================================================================
# HTTP 端点
# ============================================================================

@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    Returns:
        系统健康状态
    """
    try:
        # 检查 AI 系统是否可用
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AI_CORE_URL}/health", timeout=5.0)
            ai_healthy = response.status_code == 200
    except Exception:
        ai_healthy = False
    
    return {
        "status": "healthy" if ai_healthy else "degraded",
        "service": "AIVA System Gateway",
        "version": "1.0.0",
        "ai_core_status": "online" if ai_healthy else "offline"
    }


@app.get("/status/{scan_id}")
async def get_scan_status(scan_id: str):
    """
    查询扫描状态
    
    Args:
        scan_id: 扫描任务 ID
        
    Returns:
        扫描任务状态
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{AI_CORE_URL}/status/{scan_id}",
                timeout=10.0
            )
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI 系统响应超时")
    except httpx.RequestError as e:
        logger.error(f"转发状态查询失败: {e}")
        raise HTTPException(status_code=503, detail="AI 系统不可用")


@app.post("/scan", response_model=ScanResponse)
async def create_scan(request: ScanRequest):
    """
    创建扫描任务（整个程序对外入口）
    
    接收外部扫描请求，转发给 app.py（程序与 AI 的沟通接口）
    
    数据流: 外部 → main.py (本函数) → app.py → AI 内部
    
    Args:
        request: 扫描请求参数
        
    Returns:
        扫描任务信息
    """
    logger.info(f"[main.py] 收到外部扫描请求: target={request.target}, type={request.scan_type}")
    
    try:
        # 转发请求给 AI 系统
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AI_CORE_URL}/scan",
                json=request.dict(),
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"扫描任务已创建: scan_id={result.get('scan_id')}")
                return result
            else:
                logger.error(f"AI 系统返回错误: {response.status_code}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json().get("detail", "AI 系统处理失败")
                )
                
    except httpx.TimeoutException:
        logger.error("AI 系统响应超时")
        raise HTTPException(status_code=504, detail="AI 系统响应超时")
    except httpx.RequestError as e:
        logger.error(f"转发请求失败: {e}")
        raise HTTPException(status_code=503, detail="AI 系统不可用")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "系统内部错误"}
    )


# ============================================================================
# 启动配置
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,  # 对外端口 9000（AI 在 8000）
        reload=True,
        log_level="info"
    )
