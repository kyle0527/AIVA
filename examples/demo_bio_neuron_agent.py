"""
AIVA BioNeuronRAGAgent API 服務
將原本的示範程式改寫為實際的 RESTful API 服務
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

from services.core.aiva_core.ai_engine import BioNeuronRAGAgent
from services.aiva_common.schemas import APIResponse

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全域代理實例
agent: Optional[BioNeuronRAGAgent] = None


async def initialize_storage():
    """初始化 AIVA 數據存儲"""
    try:
        from services.core.aiva_core.storage import StorageManager
        
        # 配置
        data_root = project_root / "data"  # 使用相對路徑
        db_type = "hybrid"  # 推薦：hybrid (SQLite + JSONL)

        logger.info(f"Initializing AIVA storage: {data_root}")
        logger.info(f"Database type: {db_type}")

        # 創建存儲管理器
        storage = StorageManager(
            data_root=data_root, db_type=db_type, auto_create_dirs=True
        )

        logger.info("✅ Storage initialized successfully!")
        
        # 獲取統計資訊
        stats = await storage.get_statistics()
        logger.info(f"Storage statistics: {stats.get('backend')}, Total size: {stats.get('total_size', 0) / (1024*1024):.2f} MB")

        return storage
        
    except Exception as e:
        logger.warning(f"Storage initialization failed: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命周期管理"""
    global agent
    
    # 自動化儲存初始化
    logger.info("🚀 正在自動初始化 AIVA 存儲...")
    try:
        await initialize_storage()
        logger.info("✅ 存儲初始化完成")
    except Exception as e:
        logger.error(f"❌ 存儲初始化失敗: {e}")
        # 可以選擇是否要中斷啟動，這裡我們繼續執行
        logger.warning("⚠️ 存儲初始化失敗，但將繼續啟動服務...")
    
    # 啟動時初始化代理
    logger.info("🚀 正在初始化 BioNeuronRAGAgent...")
    try:
        # 使用相對路徑
        agent = BioNeuronRAGAgent(codebase_path=str(project_root))
        logger.info("✅ BioNeuronRAGAgent 初始化完成")
    except Exception as e:
        logger.error(f"❌ BioNeuronRAGAgent 初始化失敗: {e}")
        raise e
    
    yield
    
    # 關閉時清理
    logger.info("🔄 正在關閉 AIVA 服務...")


# 建立 FastAPI 應用
app = FastAPI(
    title="AIVA BioNeuronRAGAgent API",
    description="AIVA 核心 AI 代理服務 - 提供程式碼分析、漏洞掃描、系統命令執行等功能",
    version="1.0.0",
    lifespan=lifespan
)


# 請求資料模型 - 使用標準合約
class AIAgentRequest(BaseModel):
    """AI代理請求 - 統一代理呼叫格式"""
    query: str = Field(..., description="要執行的查詢或指令")
    path: Optional[str] = Field(None, description="檔案路徑 (用於程式碼讀取/寫入/分析)")
    target_url: Optional[str] = Field(None, description="目標 URL (用於掃描)")
    scan_type: Optional[str] = Field(None, description="掃描類型 (如: full, quick)")
    command: Optional[str] = Field(None, description="系統命令 (用於命令執行)")
    content: Optional[str] = Field(None, description="檔案內容 (用於檔案寫入)")

# 使用標準 APIResponse 替代本地響應模型
# InvokeResponse, StatsResponse, HistoryResponse 已標準化為 APIResponse


# API 端點
@app.get("/")
async def root():
    """根端點 - 服務狀態檢查"""
    response = APIResponse(
        success=True,
        message="AIVA 核心 AI 代理服務已啟動並運行中",
        data={
            "service": "AIVA BioNeuronRAGAgent API",
            "status": "running"
        }
    )
    return response.model_dump()


@app.post("/invoke")
async def invoke_agent(request: AIAgentRequest):
    """呼叫 AI 代理執行任務 - 使用標準APIResponse"""
    if agent is None:
        raise HTTPException(status_code=503, detail="AI 代理尚未初始化")
    
    try:
        logger.info(f"🔍 處理請求: {request.query[:50]}...")
        
        # 準備參數 (排除 None 值)
        params = request.model_dump(exclude_unset=True)
        
        # 呼叫代理
        result = agent.invoke(**params)
        
        logger.info(f"✅ 請求處理完成，使用工具: {result.get('tool_used', 'unknown')}")
        
        # 使用標準 APIResponse
        response = APIResponse(
            success=True,
            message="AI代理執行成功",
            data=result
        )
        return response.model_dump()
        
    except Exception as e:
        logger.error(f"❌ 處理請求時發生錯誤: {e}")
        response = APIResponse(
            success=False,
            message="AI代理執行失敗",
            errors=[str(e)]
        )
        return response.model_dump()


@app.get("/stats")
async def get_knowledge_stats():
    """取得知識庫統計資訊 - 使用標準APIResponse"""
    if agent is None:
        raise HTTPException(status_code=503, detail="AI 代理尚未初始化")
    
    try:
        stats = agent.get_knowledge_stats()
        response = APIResponse(
            success=True,
            message="知識庫統計資訊檢索成功",
            data=stats
        )
        return response.model_dump()
    except Exception as e:
        logger.error(f"❌ 取得統計資訊時發生錯誤: {e}")
        response = APIResponse(
            success=False,
            message="取得統計資訊失敗",
            errors=[str(e)]
        )
        return response.model_dump()


@app.get("/history")
async def get_execution_history():
    """取得執行歷史 - 使用標準APIResponse"""
    if agent is None:
        raise HTTPException(status_code=503, detail="AI 代理尚未初始化")
    
    try:
        history = agent.get_history()
        response = APIResponse(
            success=True,
            message="執行歷史檢索成功",
            data={"history": history}
        )
        return response.model_dump()
    except Exception as e:
        logger.error(f"❌ 取得執行歷史時發生錯誤: {e}")
        response = APIResponse(
            success=False,
            message="取得執行歷史失敗",
            errors=[str(e)]
        )
        return response.model_dump()


# 健康檢查端點
@app.get("/health")
async def health_check():
    """健康檢查端點"""
    agent_status = "ready" if agent is not None else "not_initialized"
    
    return {
        "status": "healthy",
        "agent_status": agent_status,
        "timestamp": "2025-10-23",
        "service": "AIVA BioNeuronRAGAgent API"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 啟動 AIVA BioNeuronRAGAgent API 服務...")
    print("📖 API 文件可在以下位置查看:")
    print("   - Swagger UI: http://127.0.0.1:8000/docs")
    print("   - ReDoc: http://127.0.0.1:8000/redoc")
    print("🔗 服務端點: http://127.0.0.1:8000")
    print("-" * 50)
    
    # 啟動服務
    uvicorn.run(
        "demo_bio_neuron_agent:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # 關閉自動重載避免干擾
        log_level="info"
    )
