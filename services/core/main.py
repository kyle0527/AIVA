"""
AIVA 系统对外入口（安全防护层）
====================================

版本: 1.0.0
作用: 整个程序的对外 HTTP 入口，提供第一道安全防线

架构设计（可插拔式 AI + 安全防护）:
- main.py (本脚本): 第一道安全防线，检测恶意请求后转发
- app.py: 程序与 AI 沟通的接口（AI 可插拔，实现双路分离）
- AI 内部: aiva_core/* 中的其他文件

安全职责（为什么分两个脚本）:
  ✓ 检测恶意请求（木马、注入攻击、异常参数）
  ✓ 速率限制和访问控制
  ✓ 请求白名单/黑名单
  ✓ 保护 AI 系统不受污染

数据流（三层架构 + 双路分离）:
  外部 HTTP 请求
      ↓
  main.py (第一道防线 - 安全检测)
      ↓ 通过检测后转发
      ↓
  app.py (程序与 AI 的沟通接口 - 可插拔)
      ↓
  双路分离处理:
    ├─ 路径1: 整合模块存储（实时记录）
    └─ 路径2: AI 内部文件 → 任务规划与下令 → 执行模块

学习系统（异步独立）:
  - 任务结束后从整合模块读取完整数据
  - 本次记录与历史记录比对和评估
  - 学习目标：如何调整参数才能让响应变好
  - 不在任务执行期间介入（安全设计）
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
from typing import Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 应用
app = FastAPI(
    title="AIVA 系统入口",
    version="1.0.0",
    description="AIVA 智能渗透测试系统的唯一对外入口"
)

# AI 系统地址（内部）
AI_CORE_URL = "http://localhost:8000"  # AI 系统运行在 8000 端口


# ============================================================================
# 请求/响应模型
# ============================================================================

class ScanRequest(BaseModel):
    """扫描请求模型"""
    target: str = Field(..., description="目标 URL")
    scan_type: str = Field(default="full", description="扫描类型")
    max_depth: int = Field(default=3, description="最大爬取深度")
    timeout: int = Field(default=300, description="超时时间（秒）")


class ScanResponse(BaseModel):
    """扫描响应模型"""
    scan_id: str
    status: str
    message: str
    target: str
    estimated_time: Optional[int] = None


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
