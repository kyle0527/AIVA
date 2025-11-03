#!/usr/bin/env python3
"""
AIVA 統一 MQ 系統測試器
=====================

測試 TODO-005 實施的統一 Envelope 與 Topic 系統
- V2 增強版 AivaMessage
- 統一 Topic 管理
- 雙軌運行兼容性
- 自動格式轉換
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any

from services.aiva_common.schemas.generated.messaging import AivaMessage
from services.aiva_common.schemas.generated.base_types import MessageHeader
from services.aiva_common.enums import Topic, ModuleName
from services.aiva_common.messaging.unified_topic_manager import topic_manager
from services.aiva_common.messaging.compatibility_layer import compatibility_layer, message_broker


def test_enhanced_aiva_message():
    """測試增強版 AivaMessage"""
    print("\n🧪 測試 V2 增強版 AivaMessage")
    print("=" * 50)
    
    try:
        # 使用 Topic 管理器創建訊息
        message = topic_manager.create_enhanced_message(
            topic=Topic.TASK_SCAN_START,
            payload={
                "target_url": "https://example.com",
                "scan_type": "comprehensive",
                "timeout": 300
            },
            source_module=ModuleName.SCAN,
            target_module=ModuleName.CORE,
            trace_id=str(uuid.uuid4())
        )
        
        print("✅ 訊息創建成功")
        print(f"  📧 ID: {message.header.message_id[:8]}...")
        print(f"  🏷️  Topic: {message.topic}")
        print(f"  📡 來源: {message.source_module}")
        print(f"  🎯 目標: {message.target_module}")
        print(f"  🔍 Trace ID: {message.trace_id[:8]}...")
        print(f"  🚀 路由策略: {message.routing_strategy}")
        print(f"  ⚡ 優先級: {message.priority}")
        print(f"  📦 載荷: {len(message.payload)} 個欄位")
        
        # 序列化測試
        json_str = message.model_dump_json()
        restored = AivaMessage.model_validate_json(json_str)
        assert restored.header.message_id == message.header.message_id
        print(f"  ✅ 序列化/反序列化成功 ({len(json_str)} bytes)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 測試失敗: {e}")
        return False


def test_topic_management():
    """測試 Topic 管理功能"""
    print("\n🏷️  測試統一 Topic 管理")
    print("=" * 50)
    
    try:
        # 測試 Topic 元資料
        metadata = topic_manager.get_topic_metadata(Topic.TASK_SCAN_START)
        print(f"✅ Topic 元資料獲取成功")
        print(f"  📂 類別: {metadata.category}")
        print(f"  🎯 模組: {metadata.module}")
        print(f"  ⚡ 動作: {metadata.action}")
        print(f"  📝 描述: {metadata.description}")
        print(f"  🚀 路由策略: {metadata.default_routing}")
        print(f"  🔢 優先級: {metadata.priority}")
        
        # 測試舊版映射
        legacy_topic = "scan.start"
        normalized = topic_manager.normalize_topic(legacy_topic)
        print(f"✅ 舊版映射成功: {legacy_topic} -> {normalized}")
        
        # 測試 Topic 分類
        task_topics = topic_manager.get_topics_by_category("tasks")
        scan_topics = topic_manager.get_topics_by_module("scan")
        print(f"✅ Topic 分類功能")
        print(f"  📋 任務類: {len(task_topics)} 個")
        print(f"  🔍 掃描類: {len(scan_topics)} 個")
        
        # 生成遷移報告
        report = topic_manager.get_migration_report()
        print(f"✅ 遷移報告生成")
        print(f"  📊 總 Topic 數: {report['total_topics']}")
        print(f"  🔗 舊版映射: {report['legacy_mappings']}")
        print(f"  📂 類別數: {len(report['categories'])}")
        print(f"  🎯 模組數: {len(report['modules'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 測試失敗: {e}")
        return False


def test_compatibility_layer():
    """測試兼容性層功能"""
    print("\n🔄 測試 MQ 兼容性層")
    print("=" * 50)
    
    try:
        # 測試 V1 -> V2 轉換
        v1_message = {
            "routing_key": "scan.start",
            "exchange": "aiva_exchange",
            "payload": {
                "target": "https://test.com",
                "type": "full_scan"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        v2_message = compatibility_layer.convert_v1_to_v2(
            v1_message, 
            source_module=ModuleName.SCAN
        )
        
        print(f"✅ V1->V2 轉換成功")
        print(f"  🔄 {v1_message['routing_key']} -> {v2_message.topic}")
        print(f"  📦 載荷保留: {v2_message.payload}")
        print(f"  🏷️  轉換標記: {v2_message.metadata.get('converted_from')}")
        
        # 測試 V2 -> V1 轉換
        v1_fallback = compatibility_layer.convert_v2_to_v1(v2_message)
        print(f"✅ V2->V1 回退成功")
        print(f"  🔙 Routing Key: {v1_fallback['routing_key']}")
        print(f"  📦 載荷保留: {v1_fallback['payload']}")
        
        # 測試自動格式檢測
        v1_format = compatibility_layer.detect_message_format(v1_message)
        v2_format = compatibility_layer.detect_message_format(v2_message.model_dump())
        print(f"✅ 格式檢測成功")
        print(f"  🔍 V1 檢測: {v1_format}")
        print(f"  🔍 V2 檢測: {v2_format}")
        
        # 測試統計功能
        stats = compatibility_layer.get_migration_stats()
        print(f"✅ 統計功能正常")
        print(f"  📊 V1 訊息: {stats['v1_messages_received']}")
        print(f"  📊 V2 訊息: {stats['v2_messages_received']}")
        print(f"  📊 轉換次數: {stats['v1_to_v2_conversions']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 測試失敗: {e}")
        return False


def test_unified_message_broker():
    """測試統一訊息代理器"""
    print("\n📡 測試統一訊息代理器")
    print("=" * 50)
    
    try:
        # 測試統一發佈介面
        published_message = message_broker.publish(
            topic=Topic.FINDING_DETECTED,
            payload={
                "vulnerability_type": "XSS",
                "severity": "high",
                "target_url": "https://vulnerable.com/search?q=<script>",
                "evidence": ["Reflected XSS in search parameter"]
            },
            source_module=ModuleName.FUNCTION,
            target_module=ModuleName.CORE
        )
        
        print(f"✅ 統一發佈成功")
        print(f"  📧 訊息ID: {published_message.header.message_id[:8]}...")
        print(f"  🏷️  Topic: {published_message.topic}")
        print(f"  🔍 漏洞類型: {published_message.payload['vulnerability_type']}")
        print(f"  ⚡ 嚴重性: {published_message.payload['severity']}")
        
        # 測試統一訂閱處理（模擬接收 V1 格式）
        v1_incoming = {
            "routing_key": "finding.new",
            "payload": {
                "type": "SQLI",
                "url": "https://test.com/login",
                "parameter": "username"
            }
        }
        
        processed_message = message_broker.subscribe_and_process(
            v1_incoming,
            source_module=ModuleName.FUNCTION
        )
        
        print(f"✅ 統一訂閱處理成功")
        print(f"  🔄 格式轉換: V1 -> V2")
        print(f"  🏷️  標準化 Topic: {processed_message.topic}")
        print(f"  📦 載荷: {processed_message.payload}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 測試失敗: {e}")
        return False


def test_end_to_end_flow():
    """測試端到端流程"""
    print("\n🌊 測試端到端流程")
    print("=" * 50)
    
    try:
        # 啟用模組 V2 支援
        compatibility_layer.enable_v2_for_module(ModuleName.SCAN)
        compatibility_layer.enable_v2_for_module(ModuleName.CORE)
        
        print(f"✅ 啟用 V2 支援")
        enabled_modules = compatibility_layer.get_v2_enabled_modules()
        print(f"  🎯 啟用模組: {', '.join(enabled_modules)}")
        
        # 模擬完整工作流程
        workflow_steps = [
            {
                "topic": Topic.TASK_SCAN_START,
                "source": ModuleName.CORE,
                "target": ModuleName.SCAN,
                "payload": {"target_url": "https://demo.com", "scan_type": "quick"}
            },
            {
                "topic": Topic.RESULTS_SCAN_COMPLETED,
                "source": ModuleName.SCAN,
                "target": ModuleName.CORE,
                "payload": {"scan_id": "scan_001", "urls_found": 25, "status": "completed"}
            },
            {
                "topic": Topic.FINDING_DETECTED,
                "source": ModuleName.FUNCTION,
                "target": None,  # 廣播
                "payload": {
                    "finding_id": "vuln_001",
                    "type": "XSS",
                    "severity": "medium",
                    "confidence": 0.85
                }
            }
        ]
        
        trace_id = str(uuid.uuid4())
        
        for i, step in enumerate(workflow_steps, 1):
            message = message_broker.publish(
                topic=step["topic"],
                payload=step["payload"],
                source_module=step["source"],
                target_module=step["target"],
                trace_id=trace_id
            )
            
            print(f"  {i}. 📤 {step['topic']} ({step['source']} -> {step['target'] or 'ALL'})")
            print(f"     🔍 Trace: {message.trace_id[:8]}...")
        
        # 統計報告
        final_stats = compatibility_layer.get_migration_stats()
        print(f"\n📊 最終統計")
        print(f"  📈 V2 採用率: {final_stats['v2_adoption_rate']:.1%}")
        print(f"  🎯 V2 模組數: {final_stats['v2_enabled_modules']}")
        print(f"  🔄 總轉換次數: {final_stats['v1_to_v2_conversions']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 測試失敗: {e}")
        return False


def main():
    """主測試流程"""
    print("🚀 AIVA 統一 MQ 系統測試開始")
    print("=" * 70)
    
    test_results = []
    
    # 執行測試
    tests = [
        ("V2 增強版 AivaMessage", test_enhanced_aiva_message),
        ("統一 Topic 管理", test_topic_management), 
        ("MQ 兼容性層", test_compatibility_layer),
        ("統一訊息代理器", test_unified_message_broker),
        ("端到端流程", test_end_to_end_flow)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 測試異常: {e}")
            test_results.append((test_name, False))
    
    # 測試結果摘要
    print("\n🎯 測試結果摘要")
    print("=" * 70)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {status} {test_name}")
    
    print(f"\n📊 總計: {passed}/{total} 通過 ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 所有測試通過！TODO-005 實施成功")
        return 0
    else:
        print("⚠️  部分測試失敗，需要修復")
        return 1


if __name__ == "__main__":
    exit(main())