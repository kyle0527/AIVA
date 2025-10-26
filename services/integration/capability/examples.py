#!/usr/bin/env python3
"""
AIVA 能力註冊中心使用示例
展示如何使用能力註冊系統的各種功能

此示例包括:
- 註冊新的能力
- 發現現有能力
- 測試能力連接性
- 產生跨語言綁定
- 匯出文件和報告
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# 加入 AIVA 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aiva_common.utils.logging import get_logger
from aiva_common.enums import ProgrammingLanguage

from .registry import CapabilityRegistry
from .toolkit import CapabilityToolkit
from .models import (
    CapabilityRecord, 
    CapabilityType, 
    CapabilityStatus,
    InputParameter,
    OutputParameter,
    create_capability_id
)

# 設定結構化日誌
logger = get_logger(__name__)


async def example_register_capabilities():
    """示例：註冊新的能力"""
    
    print("🔧 示例：註冊新的能力")
    print("=" * 50)
    
    registry = CapabilityRegistry()
    
    # 創建示例能力 1: Python SQL 注入掃描器
    sqli_capability = CapabilityRecord(
        id=create_capability_id("security", "sqli", "boolean_detection"),
        name="SQL 注入布爾盲注檢測",
        description="檢測 Web 應用中的 SQL 注入布爾盲注漏洞，使用自動化負載技術",
        module="function_sqli",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="services.features.function_sqli.worker:run_boolean_sqli",
        capability_type=CapabilityType.SCANNER,
        inputs=[
            InputParameter(
                name="url",
                type="str",
                required=True,
                description="目標 URL",
                validation_rules={"format": "url"}
            ),
            InputParameter(
                name="timeout",
                type="int",
                required=False,
                description="超時時間(秒)",
                default=30,
                validation_rules={"min": 1, "max": 300}
            ),
            InputParameter(
                name="user_agent",
                type="str",
                required=False,
                description="自訂 User-Agent",
                default="AIVA-Scanner/1.0"
            )
        ],
        outputs=[
            OutputParameter(
                name="vulnerabilities",
                type="List[Dict]",
                description="發現的漏洞列表",
                sample_value=[
                    {
                        "type": "sqli_boolean",
                        "severity": "high",
                        "parameter": "id",
                        "payload": "1' AND (SELECT COUNT(*) FROM information_schema.tables)>0--"
                    }
                ]
            ),
            OutputParameter(
                name="scan_report",
                type="Dict",
                description="掃描報告摘要",
                sample_value={
                    "total_tests": 25,
                    "vulnerabilities_found": 1,
                    "scan_duration": 45.2
                }
            )
        ],
        tags=["security", "sqli", "web", "injection", "database"],
        category="vulnerability_scanner",
        prerequisites=["network.connectivity", "http.client"],
        dependencies=["security.http.client"],
        timeout_seconds=300,
        priority=80
    )
    
    # 創建示例能力 2: Go 端口掃描器
    port_scanner_capability = CapabilityRecord(
        id=create_capability_id("network", "scanner", "port_scan"),
        name="高性能端口掃描器",
        description="使用 Go 實現的高性能 TCP 端口掃描器，支援多線程和自訂超時",
        module="port_scanner_go",
        language=ProgrammingLanguage.GO,
        entrypoint="http://localhost:8081/scan",
        capability_type=CapabilityType.SCANNER,
        inputs=[
            InputParameter(
                name="target",
                type="str",
                required=True,
                description="目標主機或 IP 地址"
            ),
            InputParameter(
                name="ports",
                type="List[int]",
                required=True,
                description="要掃描的端口列表",
                validation_rules={"min_items": 1, "max_items": 1000}
            ),
            InputParameter(
                name="threads",
                type="int",
                required=False,
                description="並發線程數",
                default=50,
                validation_rules={"min": 1, "max": 500}
            )
        ],
        outputs=[
            OutputParameter(
                name="open_ports",
                type="List[int]",
                description="開放的端口列表",
                sample_value=[22, 80, 443, 8080]
            ),
            OutputParameter(
                name="scan_stats",
                type="Dict",
                description="掃描統計信息",
                sample_value={
                    "total_ports": 100,
                    "open_ports": 4,
                    "scan_time": 2.3
                }
            )
        ],
        tags=["network", "port", "scan", "tcp", "security"],
        category="network_scanner",
        timeout_seconds=120,
        priority=70
    )
    
    # 創建示例能力 3: Rust 檔案雜湊計算器
    hash_calculator_capability = CapabilityRecord(
        id=create_capability_id("crypto", "hash", "file_hash"),
        name="高速檔案雜湊計算器",
        description="使用 Rust 實現的高性能檔案雜湊計算器，支援 MD5、SHA1、SHA256 等演算法",
        module="file_hasher_rust",
        language=ProgrammingLanguage.RUST,
        entrypoint="target/release/file_hasher",
        capability_type=CapabilityType.UTILITY,
        inputs=[
            InputParameter(
                name="file_path",
                type="str",
                required=True,
                description="檔案路徑",
                validation_rules={"format": "path"}
            ),
            InputParameter(
                name="algorithms",
                type="List[str]",
                required=False,
                description="雜湊演算法列表",
                default=["sha256"],
                validation_rules={
                    "allowed_values": ["md5", "sha1", "sha256", "sha512"]
                }
            )
        ],
        outputs=[
            OutputParameter(
                name="hashes",
                type="Dict[str, str]",
                description="檔案雜湊值",
                sample_value={
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                }
            ),
            OutputParameter(
                name="file_info",
                type="Dict",
                description="檔案資訊",
                sample_value={
                    "size": 1024,
                    "modified": "2024-01-15T10:30:00Z"
                }
            )
        ],
        tags=["crypto", "hash", "file", "security", "integrity"],
        category="crypto_utility",
        timeout_seconds=60,
        priority=60
    )
    
    # 註冊能力
    capabilities = [sqli_capability, port_scanner_capability, hash_calculator_capability]
    
    for capability in capabilities:
        success = await registry.register_capability(capability)
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{status} 註冊能力: {capability.name} ({capability.id})")
    
    print(f"\n📊 註冊統計:")
    stats = await registry.get_capability_stats()
    print(f"   總能力數: {stats['total_capabilities']}")
    print(f"   語言分布: {stats['by_language']}")
    
    return capabilities


async def example_discover_capabilities():
    """示例：發現現有能力"""
    
    print("\n🔍 示例：發現現有能力")
    print("=" * 50)
    
    registry = CapabilityRegistry()
    
    # 執行能力發現
    discovery_stats = await registry.discover_capabilities()
    
    print(f"發現結果:")
    print(f"   總計: {discovery_stats['discovered_count']} 個能力")
    
    for lang, count in discovery_stats.get('languages', {}).items():
        print(f"   {lang}: {count} 個")
    
    # 列出所有能力
    all_capabilities = await registry.list_capabilities()
    
    if all_capabilities:
        print(f"\n📋 已註冊的能力 ({len(all_capabilities)} 個):")
        for cap in all_capabilities[:5]:  # 只顯示前5個
            print(f"   • {cap.name} ({cap.id}) - {cap.language.value}")
    
    return discovery_stats


async def example_test_capabilities():
    """示例：測試能力連接性"""
    
    print("\n🧪 示例：測試能力連接性")
    print("=" * 50)
    
    registry = CapabilityRegistry()
    toolkit = CapabilityToolkit()
    
    # 獲取所有能力
    capabilities = await registry.list_capabilities()
    
    if not capabilities:
        print("   沒有可測試的能力")
        return
    
    # 測試前3個能力
    test_results = []
    
    for capability in capabilities[:3]:
        print(f"🔍 測試: {capability.name}")
        
        try:
            evidence = await toolkit.test_capability_connectivity(capability)
            test_results.append(evidence)
            
            status = "✅ 成功" if evidence.success else "❌ 失敗"
            print(f"   {status} 延遲: {evidence.latency_ms}ms")
            
            if evidence.error_message:
                print(f"   錯誤: {evidence.error_message}")
        
        except Exception as e:
            print(f"   ❌ 測試異常: {str(e)}")
    
    # 統計測試結果
    successful_tests = sum(1 for result in test_results if result.success)
    print(f"\n📊 測試統計:")
    print(f"   成功: {successful_tests}/{len(test_results)}")
    
    return test_results


async def example_generate_bindings():
    """示例：產生跨語言綁定"""
    
    print("\n🔧 示例：產生跨語言綁定")
    print("=" * 50)
    
    registry = CapabilityRegistry()
    toolkit = CapabilityToolkit()
    
    # 獲取第一個能力
    capabilities = await registry.list_capabilities()
    
    if not capabilities:
        print("   沒有可用的能力")
        return
    
    capability = capabilities[0]
    print(f"為能力 '{capability.name}' 產生綁定...")
    
    # 產生跨語言綁定
    bindings = await toolkit.generate_cross_language_bindings(capability)
    
    if bindings:
        print(f"✅ 成功產生 {len(bindings)} 種語言的綁定:")
        for lang in bindings.keys():
            print(f"   • {lang}")
        
        # 顯示 Python 綁定示例（如果存在）
        if "python" in bindings:
            print(f"\n📄 Python 綁定示例（前10行）:")
            lines = bindings["python"].split('\n')[:10]
            for i, line in enumerate(lines, 1):
                print(f"   {i:2}: {line}")
            if len(bindings["python"].split('\n')) > 10:
                print("   ... (更多內容)")
    else:
        print("❌ 無法產生綁定")
    
    return bindings


async def example_generate_documentation():
    """示例：產生能力文件"""
    
    print("\n📄 示例：產生能力文件")
    print("=" * 50)
    
    registry = CapabilityRegistry()
    toolkit = CapabilityToolkit()
    
    # 獲取所有能力
    capabilities = await registry.list_capabilities()
    
    if not capabilities:
        print("   沒有可用的能力")
        return
    
    # 為第一個能力產生文件
    capability = capabilities[0]
    print(f"為能力 '{capability.name}' 產生文件...")
    
    doc = await toolkit.generate_capability_documentation(capability)
    
    # 儲存文件
    doc_path = Path(f"capability_{capability.id.replace('.', '_')}_doc.md")
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    print(f"✅ 文件已儲存至: {doc_path}")
    
    # 產生系統摘要報告
    print(f"\n📊 產生系統摘要報告...")
    summary = await toolkit.export_capabilities_summary(capabilities)
    
    summary_path = Path("capability_system_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ 摘要報告已儲存至: {summary_path}")
    
    return doc, summary


async def example_validate_schemas():
    """示例：驗證能力模式"""
    
    print("\n✅ 示例：驗證能力模式")
    print("=" * 50)
    
    registry = CapabilityRegistry()
    toolkit = CapabilityToolkit()
    
    # 獲取所有能力
    capabilities = await registry.list_capabilities()
    
    if not capabilities:
        print("   沒有可驗證的能力")
        return
    
    validation_results = []
    
    for capability in capabilities[:3]:  # 驗證前3個能力
        print(f"🔍 驗證: {capability.name}")
        
        try:
            is_valid, errors = await toolkit.validate_capability_schema(capability)
            validation_results.append((capability.id, is_valid, errors))
            
            if is_valid:
                print(f"   ✅ 模式有效")
            else:
                print(f"   ❌ 模式無效:")
                for error in errors:
                    print(f"      • {error}")
        
        except Exception as e:
            print(f"   ❌ 驗證異常: {str(e)}")
            validation_results.append((capability.id, False, [str(e)]))
    
    # 統計驗證結果
    valid_count = sum(1 for _, is_valid, _ in validation_results if is_valid)
    print(f"\n📊 驗證統計:")
    print(f"   有效: {valid_count}/{len(validation_results)}")
    
    return validation_results


async def main():
    """主示例程式"""
    
    print("🚀 AIVA 能力註冊中心使用示例")
    print("=" * 70)
    print(f"開始時間: {datetime.now().isoformat()}")
    
    try:
        # 1. 註冊能力
        capabilities = await example_register_capabilities()
        
        # 2. 發現能力
        discovery_stats = await example_discover_capabilities()
        
        # 3. 測試能力
        test_results = await example_test_capabilities()
        
        # 4. 驗證模式
        validation_results = await example_validate_schemas()
        
        # 5. 產生綁定
        bindings = await example_generate_bindings()
        
        # 6. 產生文件
        doc, summary = await example_generate_documentation()
        
        # 總結
        print("\n🎉 示例執行完成")
        print("=" * 70)
        print(f"• 註冊了 {len(capabilities)} 個能力")
        print(f"• 發現了 {discovery_stats.get('discovered_count', 0)} 個能力")
        print(f"• 測試了 {len(test_results)} 個能力")
        print(f"• 驗證了 {len(validation_results)} 個能力")
        print(f"• 產生了 {len(bindings) if bindings else 0} 種語言綁定")
        print(f"• 產生了能力文件和系統摘要")
        
        print(f"\n📁 產生的檔案:")
        for file_path in Path('.').glob('capability_*.md'):
            print(f"   • {file_path}")
        for file_path in Path('.').glob('capability_*.json'):
            print(f"   • {file_path}")
        
    except Exception as e:
        logger.error("示例執行失敗", error=str(e), exc_info=True)
        print(f"❌ 示例執行失敗: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())