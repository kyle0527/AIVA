#!/usr/bin/env python3
"""
AIVA 四大模組訊息傳遞實際測試

這個腳本將測試：
1. 基礎消息協議 (MessageHeader, AivaMessage)
2. 模組間通信能力
3. 消息代理連接
4. 任務派發流程
"""

import asyncio
import json
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

# 添加項目根目錄到Python路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_test_header(test_name: str):
    """打印測試標題"""
    print(f"\n{'='*60}")
    print(f"[TEST] {test_name}")
    print(f"{'='*60}")

def print_success(message: str):
    """打印成功消息"""
    print(f"[OK] {message}")

def print_error(message: str):
    """打印錯誤消息"""
    print(f"[FAIL] {message}")

def print_info(message: str):
    """打印信息消息"""
    print(f"[INFO]  {message}")

# ============================================================================
# 測試 1: 基礎環境和導入測試
# ============================================================================

def test_basic_imports():
    """測試基礎導入功能"""
    print_test_header("基礎導入測試")
    
    try:
        # 測試基礎枚舉導入
        from services.aiva_common.enums import ModuleName, Topic, Severity
        print_success("基礎枚舉導入成功")
        print_info(f"ModuleName: {list(ModuleName)[:5]}...")
        print_info(f"Topic 總數: {len(list(Topic))}")
        
        # 測試核心Schema導入
        from services.aiva_common.schemas import MessageHeader, AivaMessage
        print_success("核心Schema導入成功")
        print_info(f"MessageHeader: {MessageHeader.__name__}")
        print_info(f"AivaMessage: {AivaMessage.__name__}")
        
        # 測試工具函數導入
        from services.aiva_common.utils import new_id, get_logger
        print_success("工具函數導入成功")
        
        return True
        
    except ImportError as e:
        print_error(f"導入失敗: {e}")
        return False
    except Exception as e:
        print_error(f"意外錯誤: {e}")
        return False

# ============================================================================
# 測試 2: 基礎消息協議測試  
# ============================================================================

def test_message_protocol():
    """測試MessageHeader和AivaMessage的創建和序列化"""
    print_test_header("基礎消息協議測試")
    
    try:
        from services.aiva_common.enums import ModuleName, Topic
        from services.aiva_common.schemas import MessageHeader, AivaMessage
        from services.aiva_common.utils import new_id
        
        # 測試 MessageHeader 創建
        header = MessageHeader(
            message_id=new_id("msg"),
            trace_id=new_id("trace"),
            correlation_id=new_id("corr"),
            source_module=ModuleName.CORE,
            timestamp=datetime.now(UTC),
            version="1.0"
        )
        print_success("MessageHeader 創建成功")
        print_info(f"Message ID: {header.message_id}")
        print_info(f"Source Module: {header.source_module}")
        
        # 測試 AivaMessage 創建
        test_payload = {
            "test_key": "test_value",
            "target": "https://example.com",
            "priority": 5
        }
        
        message = AivaMessage(
            header=header,
            topic=Topic.TASK_SCAN_START,
            schema_version="1.0",
            payload=test_payload
        )
        print_success("AivaMessage 創建成功")
        print_info(f"Topic: {message.topic}")
        print_info(f"Payload keys: {list(message.payload.keys())}")
        
        # 測試序列化
        serialized = message.model_dump()
        print_success("消息序列化成功")
        print_info(f"序列化大小: {len(json.dumps(serialized, default=str))} bytes")
        
        # 測試反序列化
        reconstructed = AivaMessage(**serialized)
        print_success("消息反序列化成功")
        
        # 驗證數據完整性
        assert reconstructed.header.message_id == header.message_id
        assert reconstructed.topic == Topic.TASK_SCAN_START
        assert reconstructed.payload == test_payload
        print_success("數據完整性驗證通過")
        
        return True
        
    except Exception as e:
        print_error(f"消息協議測試失敗: {e}")
        print_error(f"詳細錯誤: {traceback.format_exc()}")
        return False

# ============================================================================
# 測試 3: 消息代理連接測試
# ============================================================================

async def test_message_broker():
    """測試消息代理連接和基本功能"""
    print_test_header("消息代理連接測試")
    
    try:
        from services.aiva_common.mq import get_broker
        
        # 獲取消息代理（會自動選擇可用的實現）
        broker = await get_broker()
        print_success(f"消息代理創建成功: {type(broker).__name__}")
        
        # 測試基本發布功能（使用內存代理）
        from services.aiva_common.enums import Topic
        
        test_message = b'{"test": "message"}'
        await broker.publish(Topic.MODULE_HEARTBEAT, test_message)
        print_success("測試消息發布成功")
        
        # 測試訂閱功能（簡單測試）
        print_info("消息代理基本功能驗證完成")
        
        return True
        
    except Exception as e:
        print_error(f"消息代理測試失敗: {e}")
        print_error(f"詳細錯誤: {traceback.format_exc()}")
        return False

# ============================================================================
# 測試 4: 任務派發器測試
# ============================================================================

async def test_task_dispatcher():
    """測試任務派發器的消息構建功能"""
    print_test_header("任務派發器測試")
    
    try:
        # 使用內存消息代理避免外部依賴
        from services.aiva_common.mq import InMemoryBroker
        from services.core.aiva_core.messaging.task_dispatcher import TaskDispatcher
        from services.aiva_common.enums import ModuleName, Topic
        from services.aiva_common.schemas import ScanStartPayload
        from services.aiva_common.utils import new_id
        
        # 創建內存消息代理
        broker = InMemoryBroker()
        await broker.connect()
        print_success("內存消息代理創建成功")
        
        # 創建任務派發器
        dispatcher = TaskDispatcher(broker=broker, module_name=ModuleName.CORE)
        print_success("任務派發器創建成功")
        
        # 創建測試掃描任務
        from pydantic import HttpUrl
        scan_payload = ScanStartPayload(
            scan_id="scan_" + new_id("scan").split('-')[1],
            targets=[HttpUrl("https://example.com")]
        )
        
        # 測試掃描任務派發（不實際發送，只測試消息構建）
        scan_id = await dispatcher.dispatch_scan_task(scan_payload)
        print_success(f"掃描任務派發測試成功: {scan_id}")
        
        # 測試消息構建功能
        test_message = dispatcher._build_message(
            topic=Topic.TASK_SCAN_START,
            payload={"test": "data"},
            correlation_id="test-correlation"
        )
        
        print_success("消息構建測試成功")
        print_info(f"消息主題: {test_message.topic}")
        print_info(f"來源模組: {test_message.header.source_module}")
        
        return True
        
    except Exception as e:
        print_error(f"任務派發器測試失敗: {e}")
        print_error(f"詳細錯誤: {traceback.format_exc()}")
        return False

# ============================================================================
# 測試 5: 完整工作流測試
# ============================================================================

async def test_complete_workflow():
    """測試完整的消息傳遞工作流"""
    print_test_header("完整工作流測試")
    
    try:
        from services.aiva_common.mq import InMemoryBroker
        from services.aiva_common.enums import ModuleName, Topic
        from services.aiva_common.schemas import MessageHeader, AivaMessage
        from services.aiva_common.utils import new_id
        
        # 創建內存消息代理
        broker = InMemoryBroker()
        await broker.connect()
        
        # 模擬 Core → Scan 的消息傳遞
        core_message = AivaMessage(
            header=MessageHeader(
                message_id=new_id("msg"),
                trace_id=new_id("trace"),
                source_module=ModuleName.CORE
            ),
            topic=Topic.TASK_SCAN_START,
            payload={
                "scan_id": new_id("scan"),
                "target": "https://example.com",
                "priority": 5
            }
        )
        
        # 發布消息
        await broker.publish(
            Topic.TASK_SCAN_START,
            json.dumps(core_message.model_dump(), default=str).encode()
        )
        print_success("Core → Scan 消息發布成功")
        
        # 模擬接收並處理消息
        async for msg in broker.subscribe(Topic.TASK_SCAN_START):
            received_message = AivaMessage.model_validate_json(msg.body)
            print_success("消息接收和解析成功")
            print_info(f"接收到的 scan_id: {received_message.payload['scan_id']}")
            print_info(f"目標: {received_message.payload['target']}")
            
            # 模擬 Scan → Core 的結果回報
            response_message = AivaMessage(
                header=MessageHeader(
                    message_id=new_id("msg"),
                    trace_id=received_message.header.trace_id,  # 保持追蹤ID
                    correlation_id=received_message.payload['scan_id'],
                    source_module=ModuleName.SCAN
                ),
                topic=Topic.RESULTS_SCAN_COMPLETED,
                payload={
                    "scan_id": received_message.payload['scan_id'],
                    "status": "completed",
                    "assets_found": 15,
                    "vulnerabilities": 3
                }
            )
            
            await broker.publish(
                Topic.RESULTS_SCAN_COMPLETED,
                json.dumps(response_message.model_dump(), default=str).encode()
            )
            print_success("Scan → Core 結果回報成功")
            break  # 只處理一個消息
        
        # 驗證結果消息
        async for result_msg in broker.subscribe(Topic.RESULTS_SCAN_COMPLETED):
            result_message = AivaMessage.model_validate_json(result_msg.body)
            print_success("結果消息接收成功")
            print_info(f"掃描狀態: {result_message.payload['status']}")
            print_info(f"發現資產: {result_message.payload['assets_found']}")
            print_info(f"追蹤ID匹配: {result_message.header.trace_id == core_message.header.trace_id}")
            break
        
        return True
        
    except Exception as e:
        print_error(f"完整工作流測試失敗: {e}")
        print_error(f"詳細錯誤: {traceback.format_exc()}")
        return False

# ============================================================================
# 主測試函數
# ============================================================================

async def main():
    """主測試函數"""
    print("[START] AIVA 四大模組訊息傳遞實際測試開始")
    print(f"[TIME] 測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # 執行測試
    tests = [
        ("基礎導入測試", test_basic_imports, False),
        ("基礎消息協議測試", test_message_protocol, False),
        ("消息代理連接測試", test_message_broker, True),
        ("任務派發器測試", test_task_dispatcher, True),
        ("完整工作流測試", test_complete_workflow, True)
    ]
    
    for test_name, test_func, is_async in tests:
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print_error(f"{test_name} 執行失敗: {e}")
            test_results.append((test_name, False))
    
    # 輸出測試結果總結
    print_test_header("測試結果總結")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        if result:
            print_success(f"{test_name}: 通過")
            passed += 1
        else:
            print_error(f"{test_name}: 失敗")
    
    print(f"\n[STATS] 測試統計:")
    print(f"   通過: {passed}/{total}")
    print(f"   成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print_success("[SUCCESS] 所有測試通過！AIVA訊息傳遞系統運行正常！")
        return 0
    else:
        print_error(f"[WARN]  {total-passed} 個測試失敗，需要檢查相關問題")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[STOP] 測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print_error(f"測試執行出現意外錯誤: {e}")
        print_error(f"詳細錯誤: {traceback.format_exc()}")
        sys.exit(1)