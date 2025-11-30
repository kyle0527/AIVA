#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA CLI - 統一命令行入口

這是 AIVA 的主要命令行接口,整合了所有 AI 分析功能。

使用方式:
    python aiva_cli.py                    # 啟動交互式選單
    python aiva_cli.py --query "..."      # 直接查詢能力
    python aiva_cli.py --attack "..."     # AI 執行攻擊
    python aiva_cli.py --stats            # 顯示統計資訊
    python aiva_cli.py --sync             # 同步能力到 RAG
    python aiva_cli.py --test             # 運行 AI 分析測試
    python aiva_cli.py --workflow "..."   # 獲取工作流推薦

示例:
    # 查詢攻擊相關能力
    python aiva_cli.py --query "攻擊路徑分析"
    
    # AI 執行攻擊
    python aiva_cli.py --attack "幫我跑 http://localhost:8080/WebGoat 的掃描"
    
    # 獲取滲透測試工作流
    python aiva_cli.py --workflow "web 應用滲透測試"
    
    # 查看系統統計
    python aiva_cli.py --stats
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# 常量定義
ERROR_MSG_CORE_NOT_AVAILABLE = "[Error] Core modules not available"

# 確保路徑正確
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 導入核心模組
try:
    from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery
    from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
    from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
    from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
    from services.core.aiva_core.core_capabilities.dialog.assistant import AIVADialogAssistant
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Core modules not fully available: {e}")
    CORE_AVAILABLE = False

# Rich UI (可選)
console = None  # 全局變量聲明
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


def print_banner():
    """顯示歡迎橫幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     █████╗ ██╗██╗   ██╗ █████╗     ██████╗██╗     ██╗      ║
    ║    ██╔══██╗██║██║   ██║██╔══██╗   ██╔════╝██║     ██║      ║
    ║    ███████║██║██║   ██║███████║   ██║     ██║     ██║      ║
    ║    ██╔══██║██║╚██╗ ██╔╝██╔══██║   ██║     ██║     ██║      ║
    ║    ██║  ██║██║ ╚████╔╝ ██║  ██║   ╚██████╗███████╗██║      ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝    ╚═════╝╚══════╝╚═╝      ║
    ║                                                              ║
    ║        AI-Driven Vulnerability Assessment System            ║
    ║              高級人工智慧漏洞評估平台                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    if RICH_AVAILABLE and console:
        console.print(Panel(banner, border_style="cyan", title="[bold]Welcome[/bold]"))  # type: ignore[union-attr]
    else:
        print(banner)


async def handle_query(query: str, top_k: int = 5):
    """處理查詢請求"""
    if not CORE_AVAILABLE:
        print(ERROR_MSG_CORE_NOT_AVAILABLE)
        return
    
    if RICH_AVAILABLE and console:
        console.print(f"\n[cyan]Querying:[/cyan] [bold]{query}[/bold]\n")  # type: ignore[union-attr]
    else:
        print(f"\nQuerying: {query}\n")
    
    query_system = AICapabilityQuery()
    results = await query_system.query(query, top_k=top_k)
    query_system.display_results(results, title=f"Query: {query}")


async def handle_attack(command: str):
    """處理攻擊執行請求"""
    if not CORE_AVAILABLE:
        print(ERROR_MSG_CORE_NOT_AVAILABLE)
        return
    
    if RICH_AVAILABLE and console:
        console.print(f"\n[red]AI Attack Command:[/red] [bold]{command}[/bold]\n")  # type: ignore[union-attr]
    else:
        print(f"\nAI 攻擊指令: {command}\n")
    
    # 初始化 AI 對話助理
    assistant = AIVADialogAssistant()
    
    try:
        # AI 處理指令並執行攻擊
        response = await assistant.process_user_input(command)
        
        if RICH_AVAILABLE and console:
            console.print("\n[cyan]AI Response:[/cyan]")  # type: ignore[union-attr]
            console.print(f"  Intent: {response['intent']}")  # type: ignore[union-attr]
            console.print(f"  Executable: {response['executable']}\n")  # type: ignore[union-attr]
            console.print(Panel(response['message'], border_style="green", title="AI Reply"))  # type: ignore[union-attr]
        else:
            print("\nAI 響應:")
            print(f"  意圖: {response['intent']}")
            print(f"  可執行: {response['executable']}")
            print(f"\n{response['message']}\n")
        
        # 顯示執行結果
        if response.get('data') and response['data'].get('result'):
            result = response['data']['result']
            
            if RICH_AVAILABLE and console:
                table = Table(title="Execution Result", box=box.ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Success", str(result['success']))
                table.add_row("Execution Time", f"{result['execution_time']:.2f}s")
                if result.get('result'):
                    table.add_row("Result", str(result['result'])[:200] + "...")
                if result.get('error'):
                    table.add_row("Error", result['error'])
                console.print(table)  # type: ignore[union-attr]
            else:
                print("\n執行結果:")
                print(f"  成功: {result['success']}")
                print(f"  執行時間: {result['execution_time']:.2f}s")
                if result.get('result'):
                    print(f"  結果: {str(result['result'])[:200]}...")
                if result.get('error'):
                    print(f"  錯誤: {result['error']}")
    
    except Exception as e:
        if RICH_AVAILABLE and console:
            console.print(f"\n[red]Error:[/red] {e}")  # type: ignore[union-attr]
        else:
            print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()


async def handle_stats():
    """顯示統計資訊"""
    if not CORE_AVAILABLE:
        print(ERROR_MSG_CORE_NOT_AVAILABLE)
        return
    
    if RICH_AVAILABLE and console:
        console.print("\n[cyan]Loading statistics...[/cyan]\n")  # type: ignore[union-attr]
    else:
        print("\nLoading statistics...\n")
    
    query_system = AICapabilityQuery()
    await query_system.show_statistics()


async def handle_sync(force_refresh: bool = False):
    """同步能力資料到 RAG"""
    if not CORE_AVAILABLE:
        print(ERROR_MSG_CORE_NOT_AVAILABLE)
        return
    
    print("\n[Sync] Starting capability synchronization...")
    print(f"Force refresh: {force_refresh}")
    
    try:
        persist_dir = Path("data/vector_db/chroma")
        vector_store = VectorStore(backend="chroma", persist_directory=persist_dir)
        kb = KnowledgeBase(vector_store=vector_store)
        connector = InternalLoopConnector(rag_knowledge_base=kb)
        
        result = await connector.sync_capabilities_to_rag(force_refresh=force_refresh)
        
        if result.success:
            print("\n[OK] Sync completed successfully!")
            print(f"  - Modules scanned: {result.modules_scanned}")
            print(f"  - Capabilities found: {result.capabilities_found}")
            print(f"  - Documents added: {result.documents_added}")
        else:
            print(f"\n[Error] Sync failed: {result.error or 'Unknown error'}")
    
    except Exception as e:
        print(f"[Error] Sync failed: {e}")


def handle_test():
    """運行 AI 分析測試"""
    print("\n[Test] Running AI analysis capability test...\n")
    
    test_file = project_root / "test_ai_analysis_final.py"
    
    if not test_file.exists():
        print(f"[Error] Test file not found: {test_file}")
        return
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=project_root,
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n[OK] Test completed successfully")
    else:
        print(f"\n[Error] Test failed with code {result.returncode}")


async def handle_workflow(task: str):
    """獲取工作流推薦"""
    if not CORE_AVAILABLE:
        print(ERROR_MSG_CORE_NOT_AVAILABLE)
        return
    
    if RICH_AVAILABLE and console:
        console.print(f"\n[cyan]Analyzing workflow for:[/cyan] [bold]{task}[/bold]\n")  # type: ignore[union-attr]
    else:
        print(f"\nAnalyzing workflow for: {task}\n")
    
    query_system = AICapabilityQuery()
    workflow = await query_system.get_workflow_recommendation(task, max_capabilities=10)
    
    if RICH_AVAILABLE and console:
        # Rich 顯示
        table = Table(title=f"Workflow: {task}", box=box.ROUNDED)
        table.add_column("#", justify="center", style="cyan", width=4)
        table.add_column("Capability", style="bold green")
        table.add_column("Module", style="yellow")
        table.add_column("Language", style="magenta")
        
        for i, cap in enumerate(workflow["capabilities"], 1):
            table.add_row(
                str(i),
                cap["name"],
                cap["module"],
                cap["language"]
            )
        
        console.print(table)  # type: ignore[union-attr]
        console.print(f"\n[bold]Total found:[/bold] {workflow['total_found']} capabilities\n")  # type: ignore[union-attr]
    else:
        # 純文本顯示
        print(f"Workflow: {task}")
        print("=" * 60)
        
        for i, cap in enumerate(workflow["capabilities"], 1):
            print(f"{i}. {cap['name']}")
            print(f"   Module: {cap['module']}")
            print(f"   Language: {cap['language']}")
            print()
        
        print(f"Total found: {workflow['total_found']} capabilities\n")


async def interactive_menu():
    """互動式選單"""
    if not CORE_AVAILABLE:
        print(ERROR_MSG_CORE_NOT_AVAILABLE)
        return
    
    print_banner()
    
    # \u6ce8\u610f\uff1aAICapabilityQuery \u5df2\u96c6\u6210\u5230\u5404\u500b handler \u4e2d\uff0c\u9019\u88e1\u4e0d\u9700\u8981\u5be6\u4f8b\u5316
    
    menu_options = [
        ("1", "快速查詢能力", "query"),
        ("2", "AI 執行攻擊 (實戰)", "attack"),
        ("3", "查看系統統計", "stats"),
        ("4", "獲取工作流推薦", "workflow"),
        ("5", "同步能力資料", "sync"),
        ("6", "運行測試驗證", "test"),
        ("0", "退出", "quit")
    ]
    
    while True:
        print("\n" + "=" * 60)
        print("AIVA AI 能力查詢系統")
        print("=" * 60)
        
        for key, name, _ in menu_options:
            print(f"  [{key}] {name}")
        
        print("=" * 60)
        
        try:
            if RICH_AVAILABLE:
                choice = Prompt.ask("選擇功能", choices=[o[0] for o in menu_options])
            else:
                choice = input("選擇功能 > ").strip()
            if choice == "0":
                print("\nGoodbye!")
                break
            elif choice == "1":
                query = input("\n請輸入查詢問題 > ").strip()
                if query:
                    await handle_query(query)
            elif choice == "2":
                command = input("\n請輸入攻擊指令 (如: 幫我跑 http://localhost:8080/WebGoat 的掃描) > ").strip()
                if command:
                    await handle_attack(command)
            elif choice == "3":
                await handle_stats()
            elif choice == "4":
                task = input("\n請輸入任務描述 (如: 滲透測試, 漏洞修復) > ").strip()
                if task:
                    await handle_workflow(task)
            elif choice == "5":
                confirm = input("\n確定要同步能力資料嗎? (y/n) > ").strip().lower()
                if confirm == "y":
                    await handle_sync(force_refresh=False)
            elif choice == "6":
                handle_test()  # 同步調用
            else:
                print("[Warning] 無效選項")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
        
        input("\n按 Enter 繼續...")


async def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AIVA AI-Driven Vulnerability Assessment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 啟動交互式選單
  python aiva_cli.py
  
  # AI 實際執行攻擊 (重點功能)
  python aiva_cli.py --attack "幫我跑 http://localhost:8080/WebGoat 的掃描"
  python aiva_cli.py --attack "對 http://example.com 執行 SQL 注入測試"
  
  # 查詢能力
  python aiva_cli.py --query "攻擊路徑分析"
  
  # 查看統計
  python aiva_cli.py --stats
  
  # 獲取工作流推薦
  python aiva_cli.py --workflow "web 應用滲透測試"
  
  # 同步能力資料
  python aiva_cli.py --sync
  
  # 運行測試
  python aiva_cli.py --test
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        help="直接查詢能力 (自然語言)",
        type=str
    )
    
    parser.add_argument(
        "--attack", "-a",
        help="AI 執行攻擊 (自然語言指令)",
        type=str
    )
    
    parser.add_argument(
        "--stats", "-s",
        help="顯示系統統計資訊",
        action="store_true"
    )
    
    parser.add_argument(
        "--sync",
        help="同步能力到 RAG 知識庫",
        action="store_true"
    )
    
    parser.add_argument(
        "--test", "-t",
        help="運行 AI 分析測試",
        action="store_true"
    )
    
    parser.add_argument(
        "--workflow", "-w",
        help="獲取任務工作流推薦",
        type=str
    )
    
    parser.add_argument(
        "--top-k", "-k",
        help="查詢返回結果數量 (default: 5)",
        type=int,
        default=5
    )
    
    args = parser.parse_args()
    
    # 根據參數執行對應功能
    if args.test:
        handle_test()  # 同步調用
    elif args.sync:
        await handle_sync(force_refresh=False)
    elif args.stats:
        await handle_stats()
    elif args.query:
        await handle_query(args.query, top_k=args.top_k)
    elif args.attack:
        await handle_attack(args.attack)
    elif args.workflow:
        await handle_workflow(args.workflow)
    else:
        # 啟動交互式選單
        await interactive_menu()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
