#!/usr/bin/env python3
"""
兩階段掃描測試腳本
直接調用 Core 模組的 TwoPhaseScanOrchestrator
"""
import asyncio
import sys
import os

# 添加路徑
sys.path.insert(0, os.path.dirname(__file__))

from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import TwoPhaseScanOrchestrator
from services.aiva_common.mq import RabbitBroker
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


async def test_two_phase_scan():
    """測試兩階段掃描"""
    
    # 測試目標（從 Docker 截圖）
    targets = [
        "http://localhost:8080",  # WebGoat
        "http://localhost:3000",  # Juice Shop
        "http://localhost:3001",  # Juice Shop 2
        "http://localhost:3003",  # Juice Shop 3
    ]
    
    print("=" * 80)
    print("🎯 兩階段掃描測試")
    print("=" * 80)
    print(f"\n📋 目標列表:")
    for i, target in enumerate(targets, 1):
        print(f"  [{i}] {target}")
    
    # 初始化
    broker = None
    try:
        print("\n🔗 連接 MQ...")
        rabbitmq_url = "amqp://guest:guest@localhost:5672/"
        broker = RabbitBroker(rabbitmq_url)
        await broker.connect()
        print("✅ MQ 連接成功")
        
        # 創建兩階段掃描器
        scanner = TwoPhaseScanOrchestrator(broker)
        
        print("\n🚀 開始兩階段掃描...")
        print("  Phase0: 快速偵察 (5-10 分鐘)")
        print("  Phase1: 深度掃描 (10-30 分鐘)")
        
        # 執行掃描
        result = await scanner.execute_two_phase_scan(
            targets=targets,
            trace_id="test-trace-001",
        )
        
        print("\n" + "=" * 80)
        print("✅ 掃描完成")
        print("=" * 80)
        
        # 顯示結果
        print("\n📊 掃描結果:")
        print(f"  掃描 ID: {result.scan_id}")
        print(f"  狀態: {result.status}")
        print(f"  總耗時: {result.execution_time:.2f} 秒")
        
        if result.summary:
            print(f"\n  掃描摘要:")
            print(f"    - URLs: {result.summary.urls_found}")
            print(f"    - Forms: {result.summary.forms_found}")
            print(f"    - APIs: {result.summary.apis_found}")
            print(f"    - 總資產: {len(result.assets)}")
        
        if result.fingerprints:
            print(f"\n  指紋識別:")
            fp = result.fingerprints
            if fp.web_server:
                print(f"    - Web 伺服器: {list(fp.web_server.keys())}")
            if fp.frameworks:
                print(f"    - 框架: {list(fp.frameworks.keys())}")
        
        if result.engine_results:
            print(f"\n  引擎結果:")
            for engine, result_info in result.engine_results.items():
                print(f"    - {engine}: {result_info.get('status', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.exception(f"掃描失敗: {e}")
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        if broker:
            await broker.close()
            print("\n🔌 已關閉 MQ 連接")


async def main():
    """主函數"""
    try:
        result = await test_two_phase_scan()
        return 0 if result and result.status == "success" else 1
    except KeyboardInterrupt:
        print("\n\n🛑 用戶中斷")
        return 130


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
