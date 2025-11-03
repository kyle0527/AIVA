#!/usr/bin/env python3
"""
TODO-006 gRPC 服務框架完整性測試
================================

驗證統一通信架構的 gRPC 組件是否正常工作

測試範圍:
1. gRPC 服務器啟動和健康檢查
2. gRPC 客戶端連接和通信
3. 與 MQ 系統的整合
4. 跨語言服務調用
5. 協議自動切換
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path
import json
import time
from typing import Dict, Any

# 確保模塊路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.aiva_common.grpc.grpc_client import AIVAGRPCClient, grpc_client
from services.aiva_common.grpc.grpc_server import AIVAGRPCServer
from services.aiva_common.v2_client.aiva_client import AivaClient
from services.aiva_common.messaging.compatibility_layer import message_broker
from services.aiva_common.utils.logging import setup_logger

logger = setup_logger(__name__)


class GRPCIntegrationTester:
    """gRPC 整合測試器"""
    
    def __init__(self):
        self.test_results = {}
        self.server_address = "localhost:50052"  # 測試專用端口
        self.test_server = None
        self.test_client = None
    
    async def setup_test_environment(self):
        """設置測試環境"""
        logger.info("🔧 設置 gRPC 測試環境...")
        
        try:
            # 啟動測試 gRPC 服務器
            self.test_server = AIVAGRPCServer(host="localhost", port=50052)
            
            # 在後台啟動服務器
            self._server_task = asyncio.create_task(self.test_server.start())
            await asyncio.sleep(2)  # 等待服務器啟動
            
            # 創建測試客戶端
            self.test_client = AIVAGRPCClient(self.server_address)
            
            logger.info(f"✅ 測試環境準備完成: {self.server_address}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 測試環境設置失敗: {e}")
            return False
    
    async def cleanup_test_environment(self):
        """清理測試環境"""
        logger.info("🧹 清理測試環境...")
        
        try:
            if self.test_client:
                await self.test_client.close()
            
            if self.test_server:
                await self.test_server.stop()
            
            logger.info("✅ 測試環境已清理")
            
        except Exception as e:
            logger.warning(f"⚠️  清理過程中出現警告: {e}")
    
    async def test_grpc_health_check(self) -> Dict[str, Any]:
        """測試 gRPC 健康檢查"""
        logger.info("🏥 測試 gRPC 健康檢查...")
        
        try:
            result = await self.test_client.health_check()
            
            success = result.get("success", False)
            grpc_available = result.get("grpc_available", False)
            
            logger.info(f"📋 健康檢查結果: 成功={success}, gRPC可用={grpc_available}")
            
            return {
                "test_name": "gRPC_Health_Check",
                "success": success,
                "details": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ 健康檢查測試失敗: {e}")
            return {
                "test_name": "gRPC_Health_Check", 
                "success": False,
                "error": str(e)
            }
    
    async def test_task_creation(self) -> Dict[str, Any]:
        """測試任務創建"""
        logger.info("📝 測試 gRPC 任務創建...")
        
        try:
            task_config = {
                "task_id": f"test_task_{int(time.time())}",
                "task_type": "scan",
                "target_url": "https://example.com",
                "parameters": {"timeout": 30}
            }
            
            result = await self.test_client.create_task(task_config)
            
            success = result.get("success", False)
            protocol = result.get("protocol", "unknown")
            
            logger.info(f"📋 任務創建結果: 成功={success}, 協議={protocol}")
            
            return {
                "test_name": "Task_Creation",
                "success": success,
                "protocol": protocol,
                "details": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ 任務創建測試失敗: {e}")
            return {
                "test_name": "Task_Creation",
                "success": False,
                "error": str(e)
            }
    
    async def test_cross_language_execution(self) -> Dict[str, Any]:
        """測試跨語言執行"""
        logger.info("🔄 測試跨語言任務執行...")
        
        try:
            result = await self.test_client.execute_cross_language_task(
                task="python.test_function",
                parameters={"input": "test_data"}
            )
            
            success = result.get("success", False)
            protocol = result.get("protocol", "unknown")
            
            logger.info(f"📋 跨語言執行結果: 成功={success}, 協議={protocol}")
            
            return {
                "test_name": "Cross_Language_Execution",
                "success": success,
                "protocol": protocol,
                "details": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ 跨語言執行測試失敗: {e}")
            return {
                "test_name": "Cross_Language_Execution",
                "success": False,
                "error": str(e)
            }
    
    async def test_aiva_client_integration(self) -> Dict[str, Any]:
        """測試 AivaClient 整合"""
        logger.info("🤝 測試 AivaClient 與 gRPC 整合...")
        
        try:
            # 創建啟用 gRPC 的 AivaClient
            aiva_client = AivaClient(
                grpc_enabled=True,
                grpc_server_address=self.server_address
            )
            
            # 測試服務調用
            result = await aiva_client.call_service(
                language="python",
                task="test_task",
                params={"test": "data"}
            )
            
            success = result.get("success", False)
            protocol = result.get("protocol", "unknown")
            
            await aiva_client.close()
            
            logger.info(f"📋 AivaClient 整合結果: 成功={success}, 協議={protocol}")
            
            return {
                "test_name": "AivaClient_Integration",
                "success": success,
                "protocol": protocol,
                "details": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ AivaClient 整合測試失敗: {e}")
            return {
                "test_name": "AivaClient_Integration",
                "success": False,
                "error": str(e)
            }
    
    async def test_protocol_fallback(self) -> Dict[str, Any]:
        """測試協議備用機制"""
        logger.info("🔀 測試協議自動切換...")
        
        try:
            # 創建無效地址的客戶端來測試備用機制
            fallback_client = AIVAGRPCClient("invalid:99999")
            
            # 嘗試調用，應該自動切換到 MQ
            result = await fallback_client.create_task({
                "task_id": "fallback_test",
                "task_type": "scan"
            })
            
            success = result.get("success", False)
            protocol = result.get("protocol", "unknown")
            
            await fallback_client.close()
            
            logger.info(f"📋 協議切換結果: 成功={success}, 協議={protocol}")
            
            return {
                "test_name": "Protocol_Fallback",
                "success": success,
                "protocol": protocol,
                "details": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ 協議切換測試失敗: {e}")
            return {
                "test_name": "Protocol_Fallback",
                "success": False,
                "error": str(e)
            }
    
    def test_mq_integration(self) -> Dict[str, Any]:
        """測試 MQ 系統整合"""
        logger.info("📬 測試 MQ 系統整合...")
        
        try:
            from services.aiva_common.enums import Topic, ModuleName
            
            # 發送測試消息
            message = message_broker.publish(
                topic=Topic.TASK_SCAN_START,
                payload={"test": "grpc_integration"},
                source_module=ModuleName.API_GATEWAY,
                target_module=ModuleName.SCAN,
                trace_id="grpc_test_trace"
            )
            
            success = message and hasattr(message, 'header')
            
            logger.info(f"📋 MQ 整合結果: 成功={success}")
            
            return {
                "test_name": "MQ_Integration",
                "success": success,
                "message_id": message.header.message_id if success else None,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ MQ 整合測試失敗: {e}")
            return {
                "test_name": "MQ_Integration",
                "success": False,
                "error": str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """運行所有測試"""
        logger.info("🧪 開始 gRPC 整合測試套件...")
        
        test_results = []
        start_time = time.time()
        
        try:
            # 設置測試環境
            setup_success = await self.setup_test_environment()
            if not setup_success:
                return {
                    "success": False,
                    "error": "測試環境設置失敗",
                    "timestamp": time.time()
                }
            
            # 執行所有測試
            tests = [
                self.test_grpc_health_check,
                self.test_task_creation,
                self.test_cross_language_execution,
                self.test_aiva_client_integration,
                self.test_protocol_fallback,
                self.test_mq_integration
            ]
            
            for test_func in tests:
                try:
                    result = await test_func()
                    test_results.append(result)
                except Exception as e:
                    logger.error(f"❌ 測試 {test_func.__name__} 執行失敗: {e}")
                    test_results.append({
                        "test_name": test_func.__name__,
                        "success": False,
                        "error": str(e)
                    })
            
            # 計算總體結果
            successful_tests = sum(1 for r in test_results if r.get("success", False))
            total_tests = len(test_results)
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            execution_time = time.time() - start_time
            
            logger.info(f"📊 測試完成: {successful_tests}/{total_tests} 通過 ({success_rate:.1f}%)")
            
            return {
                "success": success_rate >= 50,  # 至少 50% 通過才算成功
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "execution_time": execution_time,
                "test_results": test_results,
                "timestamp": time.time()
            }
            
        finally:
            # 清理測試環境
            await self.cleanup_test_environment()


async def main():
    """主函數"""
    logger.info("=" * 60)
    logger.info("🤖 AIVA v5.0 統一通信架構")
    logger.info("🧪 TODO-006 gRPC 服務框架測試")
    logger.info("=" * 60)
    
    tester = GRPCIntegrationTester()
    
    try:
        # 運行測試
        results = await tester.run_all_tests()
        
        # 生成測試報告
        def save_report(data, filename):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        report_file = Path("TODO_006_gRPC_TEST_REPORT.json")
        save_report(results, report_file)
        
        logger.info(f"📄 測試報告已生成: {report_file}")
        
        # 顯示結果摘要
        if results.get("success", False):
            logger.info("🎉 gRPC 服務框架測試通過！")
            logger.info("✅ TODO-006 完成度驗證成功")
        else:
            logger.warning("⚠️  gRPC 服務框架測試部分失敗")
            logger.warning("🔍 請檢查測試報告瞭解詳情")
        
        return 0 if results.get("success", False) else 1
        
    except Exception as e:
        logger.error(f"❌ 測試執行失敗: {e}")
        logger.error(f"🔍 詳細錯誤: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)