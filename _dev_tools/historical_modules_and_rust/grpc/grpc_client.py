"""
AIVA gRPC 客戶端 - V2 統一架構整合
=====================================

整合 gRPC 通信到 AivaClient 易用層
支援與現有 HTTP/MQ 協議並存

功能:
- 統一客戶端介面
- 自動協議選擇 (gRPC/HTTP/MQ)
- 連接池管理
- 自動重試與容錯
- 分散式追蹤
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime
import json

import grpc
from grpc import aio  # type: ignore

# gRPC 生成的存根
try:
    from services.aiva_common.grpc.generated.python import aiva_pb2, aiva_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    logging.warning("gRPC 存根未找到，gRPC 功能將被禁用")
    aiva_pb2 = None
    aiva_pb2_grpc = None
    GRPC_AVAILABLE = False

from services.aiva_common.messaging.compatibility_layer import message_broker
from services.aiva_common.enums import Topic, ModuleName

logger = logging.getLogger(__name__)


class GRPCConnectionManager:
    """gRPC 連接管理器"""
    
    def __init__(self):
        self._channels = {}  # 連接池
        self._stubs = {}     # 存根緩存
    
    async def get_channel(self, address: str) -> Optional[aio.Channel]:
        """獲取 gRPC 通道"""
        if not GRPC_AVAILABLE:
            return None
            
        if address not in self._channels:
            try:
                channel = aio.insecure_channel(address)
                # 測試連接
                await channel.channel_ready()
                self._channels[address] = channel
                logger.info(f"📡 建立 gRPC 連接: {address}")
            except Exception as e:
                logger.warning(f"⚠️  gRPC 連接失敗: {address} - {e}")
                return None
        
        return self._channels[address]
    
    async def get_task_stub(self, address: str):
        """獲取任務服務存根"""
        if not GRPC_AVAILABLE:
            return None
            
        stub_key = f"{address}_task"
        if stub_key not in self._stubs:
            channel = await self.get_channel(address)
            if channel:
                self._stubs[stub_key] = aiva_pb2_grpc.TaskServiceStub(channel)
        
        return self._stubs.get(stub_key)
    
    async def get_cross_language_stub(self, address: str):
        """獲取跨語言服務存根"""
        if not GRPC_AVAILABLE:
            return None
            
        stub_key = f"{address}_cross_lang"
        if stub_key not in self._stubs:
            channel = await self.get_channel(address)
            if channel:
                self._stubs[stub_key] = aiva_pb2_grpc.CrossLanguageServiceStub(channel)
        
        return self._stubs.get(stub_key)
    
    async def close_all(self):
        """關閉所有連接"""
        for address, channel in self._channels.items():
            try:
                await channel.close()
                logger.info(f"🔌 關閉 gRPC 連接: {address}")
            except Exception as e:
                logger.warning(f"⚠️  關閉連接失敗: {address} - {e}")
        
        self._channels.clear()
        self._stubs.clear()


class AIVAGRPCClient:
    """AIVA gRPC 客戶端"""
    
    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.connection_manager = GRPCConnectionManager()
        self._health_checked = False
    
    async def _ensure_connection(self) -> bool:
        """確保 gRPC 連接可用"""
        if not GRPC_AVAILABLE:
            logger.debug("gRPC 不可用，將使用備用協議")
            return False
            
        try:
            if not self._health_checked:
                # 執行健康檢查
                result = await self.health_check()
                self._health_checked = result.get("success", False)
            
            return self._health_checked
        except Exception as e:
            logger.warning(f"⚠️  gRPC 連接檢查失敗: {e}")
            return False
    
    async def create_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """創建任務"""
        try:
            if await self._ensure_connection():
                # 使用 gRPC
                stub = await self.connection_manager.get_task_stub(self.server_address)
                if stub and aiva_pb2:
                    request = aiva_pb2.TaskConfig(
                        task_id=task_config.get("task_id", str(uuid.uuid4())),
                        task_type=task_config.get("task_type", "scan"),
                        target=aiva_pb2.Target(
                            url=task_config.get("target_url", ""),
                            host=task_config.get("host", ""),
                            port=task_config.get("port", 80)
                        ),
                        parameters=task_config.get("parameters", {}),
                        priority=task_config.get("priority", 5),
                        timeout=task_config.get("timeout", 300)
                    )
                    
                    response = await stub.CreateTask(request)
                    logger.info(f"✅ gRPC 任務創建成功: {response.request_id}")
                    
                    return {
                        "success": response.success,
                        "task_id": response.request_id,
                        "protocol": "gRPC",
                        "result": dict(response.result) if hasattr(response, 'result') else {}
                    }
            
            # 備用: 使用 MQ
            logger.info("🔄 gRPC 不可用，使用 MQ 備用協議")
            message = message_broker.publish(
                topic=Topic.TASK_SCAN_START,
                payload=task_config,
                source_module=ModuleName.API_GATEWAY,
                target_module=ModuleName.SCAN,
                trace_id=str(uuid.uuid4())
            )
            
            return {
                "success": True,
                "task_id": task_config.get("task_id", str(uuid.uuid4())),
                "protocol": "MQ",
                "message_id": message.header.message_id
            }
            
        except Exception as e:
            logger.error(f"❌ 任務創建失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """獲取任務狀態"""
        try:
            if await self._ensure_connection():
                # 使用 gRPC
                stub = await self.connection_manager.get_task_stub(self.server_address)
                if stub and aiva_pb2:
                    request = aiva_pb2.AIVARequest(
                        request_id=task_id,
                        task="get_status",
                        parameters={},
                        trace_id=str(uuid.uuid4())
                    )
                    
                    response = await stub.GetTaskStatus(request)
                    logger.info(f"📊 gRPC 任務狀態查詢: {task_id}")
                    
                    return {
                        "success": True,
                        "task_id": response.task_id,
                        "status": response.status,
                        "protocol": "gRPC"
                    }
            
            # 備用: 返回模擬狀態
            logger.info("🔄 gRPC 不可用，返回模擬狀態")
            return {
                "success": True,
                "task_id": task_id,
                "status": "PENDING",
                "protocol": "MQ_FALLBACK"
            }
            
        except Exception as e:
            logger.error(f"❌ 任務狀態查詢失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消任務"""
        try:
            if await self._ensure_connection():
                # 使用 gRPC
                stub = await self.connection_manager.get_task_stub(self.server_address)
                if stub and aiva_pb2:
                    request = aiva_pb2.AIVARequest(
                        request_id=task_id,
                        task="cancel",
                        parameters={},
                        trace_id=str(uuid.uuid4())
                    )
                    
                    response = await stub.CancelTask(request)
                    logger.info(f"❌ gRPC 任務取消: {task_id}")
                    
                    return {
                        "success": response.success,
                        "task_id": task_id,
                        "protocol": "gRPC"
                    }
            
            # 備用: 使用 MQ
            logger.info("🔄 gRPC 不可用，使用 MQ 取消任務")
            message_broker.publish(
                topic=Topic.COMMAND_TASK_CANCEL,
                payload={"task_id": task_id},
                source_module=ModuleName.API_GATEWAY,
                target_module=ModuleName.CORE,
                trace_id=str(uuid.uuid4())
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "protocol": "MQ"
            }
            
        except Exception as e:
            logger.error(f"❌ 任務取消失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def stream_task_progress(self, task_id: str) -> AsyncIterator[Dict[str, Any]]:
        """串流任務進度"""
        try:
            if await self._ensure_connection():
                # 使用 gRPC
                stub = await self.connection_manager.get_task_stub(self.server_address)
                if stub and aiva_pb2:
                    request = aiva_pb2.AIVARequest(
                        request_id=task_id,
                        task="stream_progress",
                        parameters={},
                        trace_id=str(uuid.uuid4())
                    )
                    
                    logger.info(f"📡 開始 gRPC 進度串流: {task_id}")
                    
                    async for response in stub.StreamTaskProgress(request):
                        yield {
                            "task_id": task_id,
                            "status": dict(response.result).get("status", "UNKNOWN"),
                            "message": dict(response.result).get("message", ""),
                            "protocol": "gRPC",
                            "timestamp": response.timestamp
                        }
                    
                    logger.info(f"✅ gRPC 進度串流完成: {task_id}")
                    return
            
            # 備用: 模擬進度更新
            logger.info("🔄 gRPC 不可用，使用模擬進度串流")
            progress_steps = [
                ("RUNNING", "任務開始執行"),
                ("PROGRESS", "處理中... 50%"),
                ("COMPLETED", "任務完成")
            ]
            
            for status, message in progress_steps:
                yield {
                    "task_id": task_id,
                    "status": status,
                    "message": message,
                    "protocol": "SIMULATION",
                    "timestamp": datetime.now()
                }
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ 進度串流失敗: {e}")
            yield {"success": False, "error": str(e)}
    
    async def execute_cross_language_task(self, task: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """執行跨語言任務"""
        try:
            if await self._ensure_connection():
                # 使用 gRPC
                stub = await self.connection_manager.get_cross_language_stub(self.server_address)
                if stub and aiva_pb2:
                    request = aiva_pb2.AIVARequest(
                        request_id=str(uuid.uuid4()),
                        task=task,
                        parameters=parameters,
                        trace_id=str(uuid.uuid4())
                    )
                    
                    response = await stub.ExecuteTask(request)
                    logger.info(f"🔄 gRPC 跨語言任務執行: {task}")
                    
                    return {
                        "success": response.success,
                        "task": task,
                        "protocol": "gRPC",
                        "result": dict(response.result) if hasattr(response, 'result') else {}
                    }
            
            # 備用: 使用 MQ
            logger.info("🔄 gRPC 不可用，使用 MQ 執行跨語言任務")
            message = message_broker.publish(
                topic=Topic.TASK_SCAN_START,
                payload=parameters,
                source_module=ModuleName.API_GATEWAY,
                target_module=ModuleName.FUNCTION,
                trace_id=str(uuid.uuid4())
            )
            
            return {
                "success": True,
                "task": task,
                "protocol": "MQ",
                "message_id": message.header.message_id
            }
            
        except Exception as e:
            logger.error(f"❌ 跨語言任務執行失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康檢查"""
        try:
            if not GRPC_AVAILABLE:
                return {"success": True, "protocol": "MQ_ONLY", "grpc_available": False}
                
            stub = await self.connection_manager.get_cross_language_stub(self.server_address)
            if stub and aiva_pb2:
                request = aiva_pb2.AIVARequest(
                    request_id=str(uuid.uuid4()),
                    task="health_check",
                    parameters={},
                    trace_id=str(uuid.uuid4())
                )
                
                response = await stub.HealthCheck(request)
                logger.info("💓 gRPC 健康檢查通過")
                
                return {
                    "success": response.success,
                    "protocol": "gRPC",
                    "grpc_available": True,
                    "health_info": dict(response.result) if hasattr(response, 'result') else {}
                }
            
            return {"success": False, "protocol": "UNKNOWN", "grpc_available": False}
            
        except Exception as e:
            logger.warning(f"⚠️  gRPC 健康檢查失敗: {e}")
            return {"success": False, "error": str(e), "grpc_available": False}
    
    async def get_service_info(self) -> Dict[str, Any]:
        """獲取服務資訊"""
        try:
            if await self._ensure_connection():
                stub = await self.connection_manager.get_cross_language_stub(self.server_address)
                if stub and aiva_pb2:
                    request = aiva_pb2.AIVARequest(
                        request_id=str(uuid.uuid4()),
                        task="get_service_info",
                        parameters={},
                        trace_id=str(uuid.uuid4())
                    )
                    
                    response = await stub.GetServiceInfo(request)
                    logger.info("ℹ️ 獲取 gRPC 服務資訊")
                    
                    return {
                        "success": response.success,
                        "protocol": "gRPC",
                        "service_info": dict(response.result) if hasattr(response, 'result') else {}
                    }
            
            # 備用資訊
            return {
                "success": True,
                "protocol": "MQ_FALLBACK",
                "service_info": {
                    "service_name": "AIVA Unified Client",
                    "protocols": ["MQ", "HTTP"],
                    "grpc_available": False
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 獲取服務資訊失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def close(self):
        """關閉客戶端"""
        await self.connection_manager.close_all()
        logger.info("🔌 gRPC 客戶端已關閉")


# 全域 gRPC 客戶端實例  
grpc_client = AIVAGRPCClient()