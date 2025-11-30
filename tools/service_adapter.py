"""Service Adapter - 將現有功能模組轉換為 HTTP 微服務
用途：這個腳本會載入您現有的 features 模組，並啟動一個 FastAPI 伺服器供 Core 調用。
"""
import importlib
import logging
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ServiceAdapter")


class ExecutionRequest(BaseModel):
    command: str
    params: Dict[str, Any]


def create_service_app(module_path: str, service_name: str):
    """動態載入模組並建立 API"""
    app = FastAPI(title=service_name)
    worker_instance = None

    try:
        # 嘗試動態導入 worker 模組
        logger.info(f"Loading module: {module_path}")
        module = importlib.import_module(module_path)

        # 嘗試尋找並實例化 Worker 類別
        # 會自動尋找名稱結尾為 Worker 或 Service 的類別
        for attr_name in dir(module):
            if attr_name.endswith("Worker") or attr_name.endswith("Service"):
                worker_class = getattr(module, attr_name)
                try:
                    # 嘗試無參數初始化
                    worker_instance = worker_class()
                    logger.info(f"✅ Successfully loaded worker: {attr_name}")
                    break
                except Exception as init_error:
                    logger.warning(f"Failed to init {attr_name}: {init_error}")
                    continue

        if not worker_instance:
            logger.error(f"❌ Could not auto-load worker class from {module_path}")

    except Exception as e:
        logger.error(f"❌ Failed to load module {module_path}: {e}")

    @app.post("/api/execute")
    async def execute(request: ExecutionRequest):
        logger.info(f"[{service_name}] Executing: {request.command} | Params: {request.params}")

        if not worker_instance:
            raise HTTPException(status_code=500, detail="Worker not initialized")

        try:
            # 優先調用 process_task (標準接口)
            if hasattr(worker_instance, "process_task"):
                result = await worker_instance.process_task(request.params)
            # 其次調用 scan (常見接口)
            elif hasattr(worker_instance, "scan"):
                url = request.params.get("url")
                result = await worker_instance.scan(url)
            # 或者嘗試調用 command 指定的方法
            elif hasattr(worker_instance, request.command):
                method = getattr(worker_instance, request.command)
                result = await method(request.params)
            else:
                raise NotImplementedError(
                    f"Worker has no method 'process_task', 'scan' or '{request.command}'"
                )

            return result

        except Exception as e:
            logger.error(f"Execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": service_name, "worker_loaded": worker_instance is not None}

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        required=True,
        help="Python path to worker module (e.g., services.function.function_sqli.worker)",
    )
    parser.add_argument("--name", required=True, help="Service Name")
    parser.add_argument("--port", type=int, required=True, help="Port to run on")

    args = parser.parse_args()

    # 確保專案根目錄在 path 中
    sys.path.insert(0, ".")

    app = create_service_app(args.module, args.name)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
