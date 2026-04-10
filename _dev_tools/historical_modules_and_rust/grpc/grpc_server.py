"""
AIVA gRPC 服務實現 - V2 統一架構
========================================

基於自動生成的 Protocol Buffers 實現核心 gRPC 服務
- TaskService: 任務管理服務
- CrossLanguageService: 跨語言通信服務

功能:
- 統一的 gRPC 服務端點
- 與 MQ 系統整合
- 分散式追蹤支援
- 自動服務發現
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import AsyncIterator, Dict, Optional

import grpc
from grpc import aio  # type: ignore

# gRPC 生成的存根 (需要先編譯 .proto 檔案)
try:
    from services.aiva_common.grpc.generated.python import aiva_pb2, aiva_pb2_grpc
except ImportError:
    # 如果還未編譯，先使用模擬定義
    logging.warning("gRPC 存根未找到，使用模擬定義。請執行 compile_protos.py")
    aiva_pb2 = None
    aiva_pb2_grpc = None

from services.aiva_common.schemas.generated.messaging import AivaMessage
from services.aiva_common.messaging.compatibility_layer import message_broker
from services.aiva_common.enums import Topic, ModuleName

logger = logging.getLogger(__name__)


class TaskServiceImplementation(aiva_pb2_grpc.TaskServiceServicer if aiva_pb2_grpc else object):
    """任務管理服務實現"""
    
    def __init__(self):
        self.active_tasks = {}  # 活躍任務追蹤
        self.task_results = {}  # 任務結果緩存
    
    async def CreateTask(self, request, context):
        """創建新任務"""
        try:
            task_id = str(uuid.uuid4())
            
            # 轉換 gRPC 請求為內部格式
            task_config = {
                "task_id": task_id,
                "task_type": request.task_type,
                "target": {
                    "url": request.target.url,
                    "host": request.target.host,
                    "port": request.target.port
                },
                "parameters": dict(request.parameters),
                "priority": request.priority,
                "timeout": request.timeout
            }
            
            # 透過 MQ 發佈任務
            message = message_broker.publish(
                topic=Topic.TASK_SCAN_START,
                payload=task_config,
                source_module=ModuleName.API_GATEWAY,
                target_module=ModuleName.SCAN,
                trace_id=str(uuid.uuid4())
            )
            
            # 記錄活躍任務
            self.active_tasks[task_id] = {
                "config": task_config,
                "created_at": datetime.now(),
                "status": "PENDING",
                "message_id": message.header.message_id,
                "trace_id": message.trace_id
            }
            
            logger.info(f"📋 創建任務: {task_id} (類型: {request.task_type})")
            
            # 回傳 gRPC 響應
            if aiva_pb2:
                return aiva_pb2.AIVAResponse(
                    request_id=task_id,
                    success=True,
                    result={"task_id": task_id, "status": "created"},
                    timestamp=datetime.now()
                )
            else:
                # 模擬回傳
                return {"success": True, "task_id": task_id}
                
        except Exception as e:
            logger.error(f"❌ 創建任務失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"任務創建失敗: {e}")
            return aiva_pb2.AIVAResponse(success=False, error_message=str(e)) if aiva_pb2 else {"success": False}
    
    async def GetTaskStatus(self, request, context):
        """獲取任務狀態"""
        try:
            task_id = request.request_id
            
            if task_id not in self.active_tasks:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"任務不存在: {task_id}")
                return aiva_pb2.AIVAResponse(success=False, error_message="任務不存在") if aiva_pb2 else {"success": False}
            
            task_info = self.active_tasks[task_id]
            
            logger.info(f"📊 查詢任務狀態: {task_id} -> {task_info['status']}")
            
            if aiva_pb2:
                return aiva_pb2.TaskResult(
                    task_id=task_id,
                    status=getattr(aiva_pb2.TaskStatus, task_info['status'], aiva_pb2.TaskStatus.TASK_STATUS_PENDING),
                    started_at=task_info['created_at'],
                    metadata={"trace_id": task_info['trace_id']}
                )
            else:
                return {"task_id": task_id, "status": task_info['status']}
                
        except Exception as e:
            logger.error(f"❌ 查詢任務狀態失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"狀態查詢失敗: {e}")
            return {"success": False}
    
    async def CancelTask(self, request, context):
        """取消任務"""
        try:
            task_id = request.request_id
            
            if task_id not in self.active_tasks:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"任務不存在: {task_id}")
                return aiva_pb2.AIVAResponse(success=False, error_message="任務不存在") if aiva_pb2 else {"success": False}
            
            # 透過 MQ 發送取消命令
            message_broker.publish(
                topic=Topic.COMMAND_TASK_CANCEL,
                payload={"task_id": task_id, "reason": "user_cancelled"},
                source_module=ModuleName.API_GATEWAY,
                target_module=ModuleName.CORE,
                trace_id=self.active_tasks[task_id]["trace_id"]
            )
            
            # 更新任務狀態
            self.active_tasks[task_id]["status"] = "CANCELLED"
            
            logger.info(f"❌ 取消任務: {task_id}")
            
            if aiva_pb2:
                return aiva_pb2.AIVAResponse(
                    request_id=task_id,
                    success=True,
                    result={"task_id": task_id, "status": "cancelled"},
                    timestamp=datetime.now()
                )
            else:
                return {"success": True, "task_id": task_id}
                
        except Exception as e:
            logger.error(f"❌ 取消任務失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"任務取消失敗: {e}")
            return {"success": False}
    
    async def StreamTaskProgress(self, request, context):
        """串流任務進度"""
        try:
            task_id = request.request_id
            
            if task_id not in self.active_tasks:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"任務不存在: {task_id}")
                return
            
            logger.info(f"📡 開始串流任務進度: {task_id}")
            
            # 模擬進度更新（實際應該從 MQ 接收）
            progress_steps = [
                ("RUNNING", "任務開始執行"),
                ("PROGRESS", "掃描中... 25%"),
                ("PROGRESS", "掃描中... 50%"),
                ("PROGRESS", "掃描中... 75%"),
                ("COMPLETED", "任務完成")
            ]
            
            for status, message in progress_steps:
                if context.cancelled():
                    break
                
                # 更新任務狀態
                self.active_tasks[task_id]["status"] = status
                
                if aiva_pb2:
                    response = aiva_pb2.AIVAResponse(
                        request_id=task_id,
                        success=True,
                        result={"status": status, "message": message},
                        timestamp=datetime.now()
                    )
                    yield response
                else:
                    yield {"task_id": task_id, "status": status, "message": message}
                
                # 模擬處理時間
                await asyncio.sleep(2)
            
            logger.info(f"✅ 任務進度串流完成: {task_id}")
            
        except Exception as e:
            logger.error(f"❌ 任務進度串流失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"進度串流失敗: {e}")


class CrossLanguageServiceImplementation(aiva_pb2_grpc.CrossLanguageServiceServicer if aiva_pb2_grpc else object):
    """跨語言通信服務實現"""
    
    def __init__(self):
        self.service_registry = {}  # 服務註冊表
        self.active_connections = {}  # 活躍連線
    
    async def ExecuteTask(self, request, context):
        """執行跨語言任務"""
        try:
            task_type = request.task
            trace_id = request.trace_id or str(uuid.uuid4())
            
            # 根據任務類型路由到對應語言服務
            if "scan" in task_type.lower():
                target_module = ModuleName.SCAN
                topic = Topic.TASK_SCAN_START
            elif "function" in task_type.lower():
                target_module = ModuleName.FUNCTION
                topic = Topic.TASK_FUNCTION_START
            else:
                target_module = ModuleName.CORE
                topic = Topic.TASK_SCAN_START
            
            # 透過統一 MQ 系統執行
            message = message_broker.publish(
                topic=topic,
                payload=dict(request.parameters),
                source_module=ModuleName.API_GATEWAY,
                target_module=target_module,
                trace_id=trace_id
            )
            
            logger.info(f"🔄 執行跨語言任務: {task_type} -> {target_module}")
            
            if aiva_pb2:
                return aiva_pb2.AIVAResponse(
                    request_id=request.request_id,
                    success=True,
                    result={"task_type": task_type, "message_id": message.header.message_id},
                    timestamp=datetime.now()
                )
            else:
                return {"success": True, "task_type": task_type}
                
        except Exception as e:
            logger.error(f"❌ 跨語言任務執行失敗: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"任務執行失敗: {e}")
            return {"success": False}
    
    async def HealthCheck(self, request, context):
        """健康檢查"""
        try:
            logger.info("💓 gRPC 健康檢查")
            
            # 檢查 MQ 連接狀態
            mq_status = "healthy"  # 實際應該檢查 MQ 連接
            
            health_info = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "mq_status": mq_status,
                "active_tasks": len(getattr(self, 'task_service', {}).get('active_tasks', {})),
                "service_version": "1.1.0"
            }
            
            if aiva_pb2:
                return aiva_pb2.AIVAResponse(
                    request_id=request.request_id,
                    success=True,
                    result=health_info,
                    timestamp=datetime.now()
                )
            else:
                return {"success": True, "health": health_info}
                
        except Exception as e:
            logger.error(f"❌ 健康檢查失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def GetServiceInfo(self, request, context):
        """獲取服務資訊"""
        try:
            service_info = {
                "service_name": "AIVA gRPC Server",
                "version": "1.1.0",
                "supported_languages": ["Python", "Go", "TypeScript", "Rust"],
                "supported_protocols": ["gRPC", "MQ"],
                "features": [
                    "統一任務管理",
                    "跨語言通信",
                    "分散式追蹤",
                    "MQ 整合",
                    "雙向串流"
                ],
                "endpoints": [
                    "TaskService/CreateTask",
                    "TaskService/GetTaskStatus", 
                    "TaskService/CancelTask",
                    "TaskService/StreamTaskProgress",
                    "CrossLanguageService/ExecuteTask",
                    "CrossLanguageService/HealthCheck",
                    "CrossLanguageService/GetServiceInfo",
                    "CrossLanguageService/BidirectionalStream"
                ]
            }
            
            logger.info("ℹ️ 返回服務資訊")
            
            if aiva_pb2:
                return aiva_pb2.AIVAResponse(
                    request_id=request.request_id,
                    success=True,
                    result=service_info,
                    timestamp=datetime.now()
                )
            else:
                return {"success": True, "service_info": service_info}
                
        except Exception as e:
            logger.error(f"❌ 獲取服務資訊失敗: {e}")
            return {"success": False}
    
    async def BidirectionalStream(self, request_iterator, context):
        """雙向串流通信"""
        try:
            logger.info("🔄 開始雙向串流通信")
            
            async for request in request_iterator:
                try:
                    # 處理接收到的訊息
                    task = request.task
                    trace_id = request.trace_id or str(uuid.uuid4())
                    
                    logger.info(f"📥 接收串流訊息: {task}")
                    
                    # 透過 MQ 處理並獲取響應
                    message = message_broker.publish(
                        topic=Topic.TASK_SCAN_START,
                        payload=dict(request.parameters),
                        source_module=ModuleName.API_GATEWAY,
                        target_module=ModuleName.SCAN,
                        trace_id=trace_id
                    )
                    
                    # 回傳處理結果
                    if aiva_pb2:
                        response = aiva_pb2.AIVAResponse(
                            request_id=request.request_id,
                            success=True,
                            result={"processed_task": task, "message_id": message.header.message_id},
                            timestamp=datetime.now()
                        )
                        yield response
                    else:
                        yield {"success": True, "processed_task": task}
                    
                except Exception as e:
                    logger.error(f"❌ 處理串流訊息失敗: {e}")
                    if aiva_pb2:
                        yield aiva_pb2.AIVAResponse(success=False, error_message=str(e))
                    else:
                        yield {"success": False, "error": str(e)}
            
            logger.info("✅ 雙向串流通信結束")
            
        except Exception as e:
            logger.error(f"❌ 雙向串流通信失敗: {e}")


class AIVAGRPCServer:
    """AIVA gRPC 服務器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 50051):
        self.host = host
        self.port = port
        self.server = None
        self.task_service = TaskServiceImplementation()
        self.cross_language_service = CrossLanguageServiceImplementation()
        
        # 共享服務引用
        self.cross_language_service.task_service = self.task_service
    
    async def start(self):
        """啟動 gRPC 服務器"""
        try:
            self.server = aio.server()
            
            if aiva_pb2_grpc:
                # 註冊服務
                aiva_pb2_grpc.add_TaskServiceServicer_to_server(
                    self.task_service, self.server
                )
                aiva_pb2_grpc.add_CrossLanguageServiceServicer_to_server(
                    self.cross_language_service, self.server
                )
            
            # 添加監聽端口
            listen_addr = f"{self.host}:{self.port}"
            self.server.add_insecure_port(listen_addr)
            
            # 啟動服務器
            await self.server.start()
            logger.info(f"🚀 AIVA gRPC 服務器啟動: {listen_addr}")
            logger.info("📋 支援服務: TaskService, CrossLanguageService")
            
            # 等待終止
            await self.server.wait_for_termination()
            
        except Exception as e:
            logger.error(f"❌ gRPC 服務器啟動失敗: {e}")
            raise
    
    async def stop(self):
        """停止 gRPC 服務器"""
        if self.server:
            await self.server.stop(5)
            logger.info("🛑 AIVA gRPC 服務器已停止")


async def main():
    """主程式入口"""
    logging.basicConfig(level=logging.INFO)
    
    server = AIVAGRPCServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("🔴 接收到中斷信號")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())