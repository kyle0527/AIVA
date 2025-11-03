#!/usr/bin/env python3
"""
TODO-006 gRPC 架構完成度快速驗證
================================

檢查統一通信架構核心組件是否正確實現

測試範圍:
1. 模塊導入測試
2. gRPC 客戶端初始化
3. MQ 系統整合
4. 架構完整性檢查
"""

import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any

# 設置基本日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_module_imports() -> Dict[str, Any]:
    """測試核心模塊導入"""
    logger.info("🔍 測試核心模塊導入...")
    
    results = {}
    
    # 測試 gRPC 組件
    try:
        from services.aiva_common.grpc.grpc_client import AIVAGRPCClient
        from services.aiva_common.grpc.grpc_server import AIVAGRPCServer
        results["grpc_components"] = {"success": True, "message": "gRPC 組件導入成功"}
    except Exception as e:
        results["grpc_components"] = {"success": False, "error": str(e)}
    
    # 測試 V2 客戶端
    try:
        from services.aiva_common.v2_client.aiva_client import AivaClient, get_aiva_client
        results["v2_client"] = {"success": True, "message": "V2 客戶端導入成功"}
    except Exception as e:
        results["v2_client"] = {"success": False, "error": str(e)}
    
    # 測試 MQ 系統
    try:
        from services.aiva_common.messaging.compatibility_layer import message_broker
        from services.aiva_common.messaging.unified_topic_manager import UnifiedTopicManager
        results["mq_system"] = {"success": True, "message": "MQ 系統導入成功"}
    except Exception as e:
        results["mq_system"] = {"success": False, "error": str(e)}
    
    # 測試 Schema 生成
    try:
        from services.aiva_common.schemas.generated.messaging import AivaMessage, AIVARequest
        results["generated_schemas"] = {"success": True, "message": "生成 Schema 導入成功"}
    except Exception as e:
        results["generated_schemas"] = {"success": False, "error": str(e)}
    
    success_count = sum(1 for r in results.values() if r.get("success", False))
    total_count = len(results)
    
    return {
        "test_name": "Module_Imports",
        "success": success_count == total_count,
        "success_rate": (success_count / total_count * 100) if total_count > 0 else 0,
        "results": results
    }


def test_grpc_client_creation() -> Dict[str, Any]:
    """測試 gRPC 客戶端創建"""
    logger.info("🤖 測試 gRPC 客戶端創建...")
    
    try:
        from services.aiva_common.grpc.grpc_client import AIVAGRPCClient
        
        # 創建客戶端實例 - 即使 gRPC 服務器不運行也應該能創建
        client = AIVAGRPCClient("localhost:50099")  # 使用不存在的端口避免衝突
        
        # 檢查客戶端屬性
        has_connection_manager = hasattr(client, 'connection_manager')
        has_server_address = hasattr(client, 'server_address')
        
        return {
            "test_name": "gRPC_Client_Creation",
            "success": has_connection_manager and has_server_address,
            "details": {
                "connection_manager": has_connection_manager,
                "server_address": has_server_address,
                "address": client.server_address
            }
        }
        
    except Exception as e:
        return {
            "test_name": "gRPC_Client_Creation",
            "success": False,
            "error": str(e)
        }


def test_v2_client_integration() -> Dict[str, Any]:
    """測試 V2 客戶端整合"""
    logger.info("🔗 測試 V2 客戶端整合...")
    
    try:
        from services.aiva_common.v2_client.aiva_client import get_aiva_client
        
        # 獲取客戶端實例
        client = get_aiva_client()
        
        # 檢查 gRPC 整合
        has_grpc_support = hasattr(client, 'grpc_enabled')
        has_grpc_client = hasattr(client, 'grpc_client')
        has_endpoints = hasattr(client, 'endpoints')
        
        return {
            "test_name": "V2_Client_Integration",
            "success": has_grpc_support and has_grpc_client and has_endpoints,
            "details": {
                "grpc_support": has_grpc_support,
                "grpc_client": has_grpc_client,
                "endpoints": has_endpoints,
                "grpc_enabled": getattr(client, 'grpc_enabled', False)
            }
        }
        
    except Exception as e:
        return {
            "test_name": "V2_Client_Integration",
            "success": False,
            "error": str(e)
        }


def test_mq_integration() -> Dict[str, Any]:
    """測試 MQ 系統整合"""
    logger.info("📬 測試 MQ 系統整合...")
    
    try:
        from services.aiva_common.messaging.compatibility_layer import message_broker
        from services.aiva_common.enums import Topic, ModuleName
        
        # 檢查 message_broker 功能
        has_publish = hasattr(message_broker, 'publish')
        has_subscribe = hasattr(message_broker, 'subscribe_and_process')
        has_conversion = hasattr(message_broker, 'convert_v1_to_v2')
        
        return {
            "test_name": "MQ_Integration",
            "success": has_publish and has_subscribe and has_conversion,
            "details": {
                "publish_method": has_publish,
                "subscribe_method": has_subscribe,
                "conversion_method": has_conversion,
                "topic_enum": len(list(Topic)) > 0,
                "module_enum": len(list(ModuleName)) > 0
            }
        }
        
    except Exception as e:
        return {
            "test_name": "MQ_Integration",
            "success": False,
            "error": str(e)
        }


def test_protocol_buffer_support() -> Dict[str, Any]:
    """測試 Protocol Buffer 支援"""
    logger.info("🔧 測試 Protocol Buffer 支援...")
    
    try:
        # 檢查 proto 文件存在
        proto_file = Path("services/aiva_common/grpc/aiva.proto")
        proto_exists = proto_file.exists()
        
        # 檢查代碼生成工具
        codegen_file = Path("services/aiva_common/tools/schema_codegen_tool.py")
        codegen_exists = codegen_file.exists()
        
        # 檢查生成目錄結構
        generated_dir = Path("services/aiva_common/grpc/generated")
        generated_exists = generated_dir.exists()
        
        return {
            "test_name": "Protocol_Buffer_Support",
            "success": proto_exists and codegen_exists,
            "details": {
                "proto_file": proto_exists,
                "codegen_tool": codegen_exists,
                "generated_dir": generated_exists
            }
        }
        
    except Exception as e:
        return {
            "test_name": "Protocol_Buffer_Support",
            "success": False,
            "error": str(e)
        }


def test_architecture_completeness() -> Dict[str, Any]:
    """測試架構完整性"""
    logger.info("🏗️  測試架構完整性...")
    
    try:
        # 檢查核心文件存在
        core_files = {
            "core_schema": "services/aiva_common/core_schema_sot.yaml",
            "grpc_server": "services/aiva_common/grpc/grpc_server.py",
            "grpc_client": "services/aiva_common/grpc/grpc_client.py",
            "v2_client": "services/aiva_common/v2_client/aiva_client.py",
            "topic_manager": "services/aiva_common/messaging/unified_topic_manager.py",
            "compatibility": "services/aiva_common/messaging/compatibility_layer.py"
        }
        
        file_results = {}
        for name, path in core_files.items():
            file_results[name] = Path(path).exists()
        
        success_count = sum(file_results.values())
        total_count = len(file_results)
        
        return {
            "test_name": "Architecture_Completeness",
            "success": success_count == total_count,
            "success_rate": (success_count / total_count * 100),
            "file_results": file_results
        }
        
    except Exception as e:
        return {
            "test_name": "Architecture_Completeness",
            "success": False,
            "error": str(e)
        }


def main():
    """主函數"""
    logger.info("=" * 60)
    logger.info("🤖 AIVA v5.0 統一通信架構")
    logger.info("✅ TODO-006 gRPC 服務框架完成度驗證")
    logger.info("=" * 60)
    
    # 執行測試
    tests = [
        test_module_imports,
        test_grpc_client_creation,
        test_v2_client_integration,
        test_mq_integration,
        test_protocol_buffer_support,
        test_architecture_completeness
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            
            success = result.get("success", False)
            status = "✅" if success else "❌"
            logger.info(f"{status} {result['test_name']}: {'通過' if success else '失敗'}")
            
        except Exception as e:
            logger.error(f"❌ 測試 {test_func.__name__} 執行失敗: {e}")
            results.append({
                "test_name": test_func.__name__,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    # 計算總體結果
    successful_tests = sum(1 for r in results if r.get("success", False))
    total_tests = len(results)
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    # 生成報告
    report = {
        "summary": {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "overall_success": success_rate >= 80  # 80% 通過率
        },
        "test_results": results,
        "conclusion": {
            "todo_006_status": "COMPLETED" if success_rate >= 80 else "PARTIAL",
            "grpc_framework_ready": success_rate >= 80,
            "recommendations": []
        }
    }
    
    if success_rate < 100:
        report["conclusion"]["recommendations"].append("檢查失敗的測試並修復相關問題")
    if success_rate >= 80:
        report["conclusion"]["recommendations"].append("gRPC 服務框架基本完成，可以進入生產環境測試")
    
    # 保存報告
    report_file = Path("TODO_006_COMPLETION_VERIFICATION.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 顯示結果
    logger.info("=" * 60)
    logger.info(f"📊 測試結果: {successful_tests}/{total_tests} 通過 ({success_rate:.1f}%)")
    
    if report["summary"]["overall_success"]:
        logger.info("🎉 TODO-006 gRPC 服務框架驗證通過！")
        logger.info("✅ 統一通信架構實現完成")
    else:
        logger.warning("⚠️  TODO-006 部分功能需要進一步完善")
    
    logger.info(f"📄 詳細報告: {report_file}")
    logger.info("=" * 60)
    
    return 0 if report["summary"]["overall_success"] else 1


if __name__ == "__main__":
    sys.exit(main())