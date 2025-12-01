"""
指令系統優化 - 測試腳本

測試新增的回調機制和搜索功能
"""

import asyncio
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.aiva_common.schemas.commands import (
    AICommand,
    CommandType,
    CommandStatus,
    CommandPriority
)
from services.aiva_common.command_center import get_command_center
from services.core.aiva_core.ui_panel.command_callback import UICommandCallback
from services.integration.search_command_handler import SearchCommandHandler


async def test_callback_mechanism():
    """測試回調機制"""
    print("\n" + "="*60)
    print("測試 1: 回調機制")
    print("="*60 + "\n")
    
    # 創建 UI 回調處理器
    ui_callback = UICommandCallback(
        use_websocket=False,  # 測試時不使用 WebSocket
        use_rich_console=True  # 使用 Rich Console 輸出
    )
    
    # 創建測試命令
    command = AICommand(
        command_id="test_callback_001",
        command_type=CommandType.SCAN_PHASE0,
        target_module="test",
        payload={"test": True},
        enable_callbacks=True
    )
    
    # 設置回調
    command.set_callback(ui_callback)
    
    # 模擬進度更新
    print("模擬進度更新:")
    await ui_callback.on_progress(
        command.command_id,
        0.0,
        "開始測試..."
    )
    
    await asyncio.sleep(0.5)
    
    await ui_callback.on_progress(
        command.command_id,
        0.5,
        "處理中..."
    )
    
    await asyncio.sleep(0.5)
    
    await ui_callback.on_progress(
        command.command_id,
        1.0,
        "完成！"
    )
    
    # 模擬狀態變更
    print("\n模擬狀態變更:")
    await ui_callback.on_status_change(
        command.command_id,
        CommandStatus.PENDING,
        CommandStatus.RUNNING
    )
    
    await asyncio.sleep(0.5)
    
    await ui_callback.on_status_change(
        command.command_id,
        CommandStatus.RUNNING,
        CommandStatus.COMPLETED
    )
    
    # 模擬中間結果
    print("\n模擬中間結果:")
    await ui_callback.on_partial_result(
        command.command_id,
        "vulnerability_found",
        {
            "type": "XSS",
            "severity": "High",
            "url": "http://example.com/test"
        }
    )
    
    await asyncio.sleep(0.5)
    
    await ui_callback.on_partial_result(
        command.command_id,
        "url_discovered",
        {
            "url": "http://example.com/admin"
        }
    )
    
    print("\n✅ 回調機制測試完成\n")


async def test_search_duckduckgo():
    """測試 DuckDuckGo 搜索"""
    print("\n" + "="*60)
    print("測試 2: DuckDuckGo 搜索")
    print("="*60 + "\n")
    
    # 創建搜索處理器
    search_handler = SearchCommandHandler()
    
    # 創建 UI 回調
    ui_callback = UICommandCallback(
        use_websocket=False,
        use_rich_console=True
    )
    search_handler.set_callback(ui_callback)
    
    # 創建命令中心並註冊搜索處理器
    command_center = get_command_center()
    command_center.register_module("search", search_handler)
    
    # 創建搜索命令
    command = AICommand(
        command_id="search_ddg_001",
        command_type=CommandType.SEARCH_DUCKDUCKGO,
        target_module="search",
        payload={
            "query": "SQL injection vulnerability tutorial",
            "max_results": 5
        },
        enable_callbacks=True,
        priority=CommandPriority.HIGH
    )
    
    # 設置回調
    command.set_callback(ui_callback)
    
    # 執行搜索
    print("執行 DuckDuckGo 搜索...")
    result = await command_center.execute(command)
    
    print(f"\n搜索結果:")
    print(f"  狀態: {result.status.value}")
    print(f"  成功: {result.success}")
    print(f"  執行時間: {result.execution_time:.2f}秒")
    
    if result.success:
        search_result = result.result
        print(f"\n  查詢: {search_result.get('query')}")
        print(f"  來源: {search_result.get('source')}")
        print(f"  摘要: {search_result.get('abstract_text', 'N/A')[:200]}")
        
        related = search_result.get('related_topics', [])
        print(f"\n  相關主題 ({len(related)}):")
        for i, topic in enumerate(related[:3], 1):
            print(f"    {i}. {topic.get('text', 'N/A')[:100]}")
            print(f"       URL: {topic.get('url', 'N/A')}")
    else:
        print(f"  錯誤: {result.error}")
    
    print("\n✅ DuckDuckGo 搜索測試完成\n")


async def test_search_github():
    """測試 GitHub 搜索"""
    print("\n" + "="*60)
    print("測試 3: GitHub 搜索")
    print("="*60 + "\n")
    
    command_center = get_command_center()
    
    # 創建 UI 回調
    ui_callback = UICommandCallback(
        use_websocket=False,
        use_rich_console=True
    )
    
    # 創建搜索命令
    command = AICommand(
        command_id="search_github_001",
        command_type=CommandType.SEARCH_GITHUB,
        target_module="search",
        payload={
            "query": "XSS vulnerability scanner",
            "search_type": "repositories",
            "max_results": 5
        },
        enable_callbacks=True
    )
    
    command.set_callback(ui_callback)
    
    # 執行搜索
    print("執行 GitHub 搜索...")
    result = await command_center.execute(command)
    
    print(f"\n搜索結果:")
    print(f"  狀態: {result.status.value}")
    print(f"  成功: {result.success}")
    print(f"  執行時間: {result.execution_time:.2f}秒")
    
    if result.success:
        search_result = result.result
        print(f"\n  查詢: {search_result.get('query')}")
        print(f"  來源: {search_result.get('source')}")
        print(f"  總結果數: {search_result.get('total_count', 0)}")
        
        items = search_result.get('items', [])
        print(f"\n  倉庫列表 ({len(items)}):")
        for i, item in enumerate(items[:5], 1):
            print(f"    {i}. {item.get('name')} ({item.get('stars', 0)} ⭐)")
            print(f"       {item.get('description', 'N/A')[:100]}")
            print(f"       URL: {item.get('url')}")
            print()
    else:
        print(f"  錯誤: {result.error}")
    
    print("✅ GitHub 搜索測試完成\n")


async def test_search_cve():
    """測試 CVE 搜索"""
    print("\n" + "="*60)
    print("測試 4: CVE/NVD 搜索")
    print("="*60 + "\n")
    
    command_center = get_command_center()
    
    # 創建 UI 回調
    ui_callback = UICommandCallback(
        use_websocket=False,
        use_rich_console=True
    )
    
    # 創建搜索命令
    command = AICommand(
        command_id="search_cve_001",
        command_type=CommandType.SEARCH_CVE_DETAILS,
        target_module="search",
        payload={
            "keyword": "SQL injection",
            "max_results": 5
        },
        enable_callbacks=True
    )
    
    command.set_callback(ui_callback)
    
    # 執行搜索
    print("執行 CVE/NVD 搜索...")
    result = await command_center.execute(command)
    
    print(f"\n搜索結果:")
    print(f"  狀態: {result.status.value}")
    print(f"  成功: {result.success}")
    print(f"  執行時間: {result.execution_time:.2f}秒")
    
    if result.success:
        search_result = result.result
        print(f"\n  查詢: {search_result.get('query')}")
        print(f"  來源: {search_result.get('source')}")
        print(f"  總結果數: {search_result.get('total_results', 0)}")
        
        vulns = search_result.get('vulnerabilities', [])
        print(f"\n  漏洞列表 ({len(vulns)}):")
        for i, vuln in enumerate(vulns[:5], 1):
            print(f"    {i}. {vuln.get('id')}")
            print(f"       嚴重性: {vuln.get('severity')} (分數: {vuln.get('score')})")
            print(f"       描述: {vuln.get('description', 'N/A')[:150]}...")
            print()
    else:
        print(f"  錯誤: {result.error}")
    
    print("✅ CVE/NVD 搜索測試完成\n")


async def test_command_with_callback():
    """測試帶回調的完整命令執行流程"""
    print("\n" + "="*60)
    print("測試 5: 帶回調的完整命令執行流程")
    print("="*60 + "\n")
    
    command_center = get_command_center()
    
    # 創建 UI 回調
    ui_callback = UICommandCallback(
        use_websocket=False,
        use_rich_console=True
    )
    
    # 創建多個搜索命令並並行執行
    commands = [
        AICommand(
            command_id=f"search_parallel_{i}",
            command_type=CommandType.SEARCH_DUCKDUCKGO,
            target_module="search",
            payload={
                "query": query,
                "max_results": 3
            },
            enable_callbacks=True
        )
        for i, query in enumerate([
            "XSS attack prevention",
            "SQL injection detection",
            "CSRF protection"
        ], 1)
    ]
    
    # 為每個命令設置回調
    for cmd in commands:
        cmd.set_callback(ui_callback)
    
    # 並行執行
    print("並行執行 3 個搜索命令...\n")
    tasks = [command_center.execute(cmd) for cmd in commands]
    results = await asyncio.gather(*tasks)
    
    # 統計結果
    print("\n" + "="*60)
    print("執行結果統計:")
    print("="*60)
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_time = sum(r.execution_time for r in results)
    
    print(f"\n  總命令數: {len(results)}")
    print(f"  成功: {successful}")
    print(f"  失敗: {failed}")
    print(f"  總執行時間: {total_time:.2f}秒")
    print(f"  平均執行時間: {total_time/len(results):.2f}秒")
    
    print("\n✅ 並行命令執行測試完成\n")


async def main():
    """主測試函數"""
    print("\n" + "="*70)
    print(" "*20 + "AIVA 指令系統優化測試")
    print("="*70)
    
    try:
        # 測試 1: 回調機制
        await test_callback_mechanism()
        
        # 測試 2: DuckDuckGo 搜索
        await test_search_duckduckgo()
        
        # 測試 3: GitHub 搜索
        await test_search_github()
        
        # 測試 4: CVE 搜索
        await test_search_cve()
        
        # 測試 5: 並行命令執行
        await test_command_with_callback()
        
        print("\n" + "="*70)
        print(" "*25 + "所有測試完成！")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷\n")
    except Exception as e:
        print(f"\n\n❌ 測試失敗: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
