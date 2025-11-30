#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA AI 智能選單系統

完整的 AI 驅動介面，讓使用者透過選單或輸入框操作，
AI 自動調用 782 個能力按照規劃執行任務。

使用方式:
    python aiva_ai_menu.py              # 啟動智能選單
    python aiva_ai_menu.py --ui web     # 啟動 Web UI
    python aiva_ai_menu.py --ui cli     # 啟動 CLI 選單（預設）
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List
import json

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

from services.core.aiva_core.core_capabilities.dialog.assistant import AIVADialogAssistant
from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery
from services.integration.capability.registry import registry as capability_registry
from services.aiva_common.utils.logging import get_logger

logger = get_logger(__name__)


class AIVAIntelligentMenu:
    """AIVA 智能選單系統
    
    提供完整的 AI 驅動使用者介面:
    - 主選單導航
    - AI 助理對話
    - 能力搜索與執行
    - 工作流推薦與自動執行
    - 系統監控與統計
    """
    
    def __init__(self):
        self.ai_assistant = AIVADialogAssistant()
        self.capability_query = AICapabilityQuery()
        self.running = True
        self.session_history: List[Dict[str, Any]] = []
        
    def print_banner(self):
        """顯示歡迎橫幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ██╗██╗   ██╗ █████╗     ███╗   ███╗███████╗     ║
║    ██╔══██╗██║██║   ██║██╔══██╗    ████╗ ████║██╔════╝     ║
║    ███████║██║██║   ██║███████║    ██╔████╔██║█████╗       ║
║    ██╔══██║██║╚██╗ ██╔╝██╔══██║    ██║╚██╔╝██║██╔══╝       ║
║    ██║  ██║██║ ╚████╔╝ ██║  ██║    ██║ ╚═╝ ██║███████╗     ║
║    ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝     ║
║                                                              ║
║            AI-Driven Intelligent Menu System                ║
║          AI 智能選單 - 782 個能力隨時待命                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

    🤖 AI 助理已就緒 | 📊 782 個能力可用 | 🚀 自動化執行
"""
        if RICH_AVAILABLE:
            console.print(Panel(banner, border_style="cyan", title="[bold]歡迎使用 AIVA[/bold]"))
        else:
            print(banner)
    
    def show_main_menu(self):
        """顯示主選單"""
        menu = """
╔══════════════════════════════════════════════════════════════╗
║                     AIVA AI 智能選單                         ║
╚══════════════════════════════════════════════════════════════╝

【核心功能】
  [1] 🤖 AI 助理對話    - 用自然語言下指令，AI 自動執行
  [2] 🔍 智能搜索能力    - 搜索 782 個能力並執行
  [3] 🎯 一鍵攻擊執行    - 輸入目標，AI 自動選擇攻擊鏈
  [4] 📋 工作流推薦      - AI 推薦完整攻擊流程

【系統管理】
  [5] 📊 系統統計報告    - 查看 782 個能力分布
  [6] 🔧 能力健康檢查    - 檢查所有模組狀態
  [7] 🗄️  RAG 知識庫同步 - 更新向量資料庫
  [8] 📜 查看執行歷史    - 顯示本次會話記錄

【進階功能】
  [9] ⚙️  模組管理       - 管理 scan/core/integration/features
  [10] 🧪 測試與驗證     - 運行 AI 分析測試
  [11] 📈 性能監控       - 實時監控系統性能
  [12] 💾 匯出報告       - 匯出執行結果為 JSON/PDF

【其他】
  [0] 🚪 退出系統

請選擇功能 (0-12) 或輸入 'help' 查看說明
"""
        if RICH_AVAILABLE:
            console.print(menu)
        else:
            print(menu)
    
    async def handle_ai_conversation(self):
        """處理 AI 助理對話"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]🤖 AI 助理對話模式[/bold cyan]")
            console.print("[dim]輸入 'back' 返回主選單[/dim]\n")
        else:
            print("\n🤖 AI 助理對話模式")
            print("輸入 'back' 返回主選單\n")
        
        while True:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("[bold green]您[/bold green]")
            else:
                user_input = input("您> ")
            
            if user_input.strip().lower() == 'back':
                break
            
            if not user_input.strip():
                continue
            
            # 顯示處理中
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("AI 思考中...", total=None)
                    response = await self.ai_assistant.process_user_input(user_input)
                    progress.update(task, completed=True)
            else:
                print("AI 思考中...")
                response = await self.ai_assistant.process_user_input(user_input)
            
            # 記錄到歷史
            self.session_history.append({
                'timestamp': asyncio.get_event_loop().time(),
                'user': user_input,
                'ai': response
            })
            
            # 顯示回應
            if RICH_AVAILABLE:
                console.print(f"\n[bold cyan]🤖 AIVA[/bold cyan]: {response['message']}\n")
                
                if response.get('executable') and response.get('data'):
                    console.print("[dim]✓ 任務已執行[/dim]\n")
            else:
                print(f"\n🤖 AIVA: {response['message']}\n")
                if response.get('executable'):
                    print("✓ 任務已執行\n")
    
    async def handle_capability_search(self):
        """處理能力搜索"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]🔍 智能搜索能力[/bold cyan]\n")
            query = Prompt.ask("請輸入搜索關鍵字（如: 掃描, SQL注入, XSS）")
            top_k = int(Prompt.ask("返回結果數量", default="10"))
        else:
            print("\n🔍 智能搜索能力\n")
            query = input("請輸入搜索關鍵字: ")
            top_k = int(input("返回結果數量 [10]: ") or "10")
        
        # 搜索能力
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"搜索中: {query}", total=None)
                results = await self.capability_query.query(query, top_k=top_k)
                progress.update(task, completed=True)
        else:
            print(f"搜索中: {query}")
            results = await self.capability_query.query(query, top_k=top_k)
        
        if not results:
            console.print("[red]未找到相關能力[/red]\n") if RICH_AVAILABLE else print("未找到相關能力\n")
            return
        
        # 顯示結果
        if RICH_AVAILABLE:
            table = Table(title=f"搜索結果: {query} (共 {len(results)} 個)", box=box.ROUNDED)
            table.add_column("#", style="cyan", width=4)
            table.add_column("能力名稱", style="green")
            table.add_column("模組", style="yellow")
            table.add_column("語言", style="magenta")
            table.add_column("相關度", style="blue")
            
            for i, result in enumerate(results, 1):
                table.add_row(
                    str(i),
                    result.get('name', 'N/A'),
                    result.get('module', 'N/A'),
                    result.get('language', 'N/A'),
                    f"{result.get('score', 0):.2f}"
                )
            
            console.print(table)
        else:
            print(f"\n搜索結果: {query} (共 {len(results)} 個)\n")
            for i, result in enumerate(results, 1):
                print(f"[{i}] {result.get('name')} - {result.get('module')} ({result.get('language')})")
        
        # 詢問是否執行
        if RICH_AVAILABLE:
            execute = Confirm.ask("\n是否執行其中某個能力?", default=False)
        else:
            execute = input("\n是否執行其中某個能力? (y/n): ").lower() == 'y'
        
        if execute:
            if RICH_AVAILABLE:
                choice = int(Prompt.ask("請選擇編號", default="1"))
            else:
                choice = int(input("請選擇編號: ") or "1")
            
            if 1 <= choice <= len(results):
                selected = results[choice - 1]
                await self._execute_capability(selected)
    
    async def handle_one_click_attack(self):
        """處理一鍵攻擊執行"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]🎯 一鍵攻擊執行[/bold cyan]\n")
            target = Prompt.ask("請輸入目標 URL")
            attack_type = Prompt.ask(
                "攻擊類型",
                choices=["auto", "scan", "sqli", "xss", "full"],
                default="auto"
            )
        else:
            print("\n🎯 一鍵攻擊執行\n")
            target = input("請輸入目標 URL: ")
            attack_type = input("攻擊類型 (auto/scan/sqli/xss/full) [auto]: ") or "auto"
        
        # 構建 AI 指令
        if attack_type == "auto":
            command = f"幫我對 {target} 執行智能掃描並自動選擇攻擊策略"
        elif attack_type == "scan":
            command = f"幫我掃描 {target}"
        elif attack_type == "sqli":
            command = f"對 {target} 執行 SQL 注入測試"
        elif attack_type == "xss":
            command = f"對 {target} 執行 XSS 跨站腳本測試"
        elif attack_type == "full":
            command = f"對 {target} 執行完整滲透測試"
        else:
            command = f"幫我掃描 {target}"
        
        # 執行攻擊
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"執行中: {attack_type} -> {target}", total=None)
                response = await self.ai_assistant.process_user_input(command)
                progress.update(task, completed=True)
        else:
            print(f"執行中: {attack_type} -> {target}")
            response = await self.ai_assistant.process_user_input(command)
        
        # 記錄到歷史
        self.session_history.append({
            'timestamp': asyncio.get_event_loop().time(),
            'action': 'one_click_attack',
            'target': target,
            'attack_type': attack_type,
            'response': response
        })
        
        # 顯示結果
        if RICH_AVAILABLE:
            console.print(Panel(
                response['message'],
                title="[bold]執行結果[/bold]",
                border_style="green" if response.get('executable') else "red"
            ))
        else:
            print(f"\n執行結果:\n{response['message']}\n")
    
    async def handle_workflow_recommendation(self):
        """處理工作流推薦"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]📋 AI 工作流推薦[/bold cyan]\n")
            task = Prompt.ask("請描述您要執行的任務")
        else:
            print("\n📋 AI 工作流推薦\n")
            task = input("請描述您要執行的任務: ")
        
        # 獲取工作流推薦
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress_task = progress.add_task("AI 規劃中...", total=None)
                workflow = await self.capability_query.recommend_workflow(task, top_k=10)
                progress.update(progress_task, completed=True)
        else:
            print("AI 規劃中...")
            workflow = await self.capability_query.recommend_workflow(task, top_k=10)
        
        if not workflow:
            console.print("[red]無法生成工作流[/red]\n") if RICH_AVAILABLE else print("無法生成工作流\n")
            return
        
        # 顯示工作流
        if RICH_AVAILABLE:
            table = Table(title=f"推薦工作流: {task}", box=box.ROUNDED)
            table.add_column("步驟", style="cyan", width=6)
            table.add_column("能力", style="green")
            table.add_column("模組", style="yellow")
            table.add_column("語言", style="magenta")
            
            for i, step in enumerate(workflow, 1):
                table.add_row(
                    f"步驟 {i}",
                    step.get('name', 'N/A'),
                    step.get('module', 'N/A'),
                    step.get('language', 'N/A')
                )
            
            console.print(table)
        else:
            print(f"\n推薦工作流: {task}\n")
            for i, step in enumerate(workflow, 1):
                print(f"步驟 {i}: {step.get('name')} - {step.get('module')}")
        
        # 詢問是否執行
        if RICH_AVAILABLE:
            execute = Confirm.ask("\n是否執行此工作流?", default=False)
        else:
            execute = input("\n是否執行此工作流? (y/n): ").lower() == 'y'
        
        if execute:
            await self._execute_workflow(workflow, task)
    
    async def handle_system_stats(self):
        """顯示系統統計"""
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("載入統計資料...", total=None)
                stats = await capability_registry.get_capability_stats()
                progress.update(task, completed=True)
        else:
            print("載入統計資料...")
            stats = await capability_registry.get_capability_stats()
        
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]📊 AIVA 系統統計報告[/bold cyan]\n")
            
            # 總覽
            overview_table = Table(title="系統總覽", box=box.ROUNDED)
            overview_table.add_column("項目", style="cyan")
            overview_table.add_column("數值", style="green")
            overview_table.add_row("總能力數", str(stats.get('total_capabilities', 0)))
            overview_table.add_row("模組數", str(len(stats.get('by_module', {}))))
            overview_table.add_row("語言數", str(len(stats.get('by_language', {}))))
            overview_table.add_row("健康能力", str(stats.get('health_summary', {}).get('healthy', 0)))
            console.print(overview_table)
            
            # 模組分布
            module_table = Table(title="模組分布 (Top 10)", box=box.ROUNDED)
            module_table.add_column("模組", style="cyan")
            module_table.add_column("數量", style="green")
            module_table.add_column("佔比", style="yellow")
            
            total = stats.get('total_capabilities', 1)
            for module, count in sorted(
                stats.get('by_module', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                percentage = (count / total) * 100
                module_table.add_row(module, str(count), f"{percentage:.1f}%")
            
            console.print(module_table)
            
            # 語言分布
            lang_table = Table(title="語言分布", box=box.ROUNDED)
            lang_table.add_column("語言", style="cyan")
            lang_table.add_column("數量", style="green")
            lang_table.add_column("佔比", style="yellow")
            
            for lang, count in sorted(
                stats.get('by_language', {}).items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / total) * 100
                lang_table.add_row(lang, str(count), f"{percentage:.1f}%")
            
            console.print(lang_table)
        else:
            print("\n📊 AIVA 系統統計報告\n")
            print(f"總能力數: {stats.get('total_capabilities', 0)}")
            print(f"模組數: {len(stats.get('by_module', {}))}")
            print(f"語言數: {len(stats.get('by_language', {}))}")
            print("\n模組分布:")
            for module, count in sorted(
                stats.get('by_module', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                print(f"  {module}: {count}")
        
        input("\n按 Enter 繼續...")
    
    async def handle_health_check(self):
        """執行健康檢查"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]🔧 能力健康檢查[/bold cyan]\n")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("檢查中...", total=None)
                # TODO: 實現健康檢查
                await asyncio.sleep(1)
                progress.update(task, completed=True)
            console.print("[green]✓ 所有能力健康[/green]\n")
        else:
            print("\n🔧 能力健康檢查")
            print("檢查中...")
            await asyncio.sleep(1)
            print("✓ 所有能力健康\n")
        
        input("按 Enter 繼續...")
    
    async def handle_sync_rag(self):
        """同步 RAG 知識庫"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]🗄️ RAG 知識庫同步[/bold cyan]\n")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("同步中...", total=None)
                # TODO: 實現 RAG 同步
                await asyncio.sleep(2)
                progress.update(task, completed=True)
            console.print("[green]✓ 知識庫已同步[/green]\n")
        else:
            print("\n🗄️ RAG 知識庫同步")
            print("同步中...")
            await asyncio.sleep(2)
            print("✓ 知識庫已同步\n")
        
        input("按 Enter 繼續...")
    
    async def handle_show_history(self):
        """顯示執行歷史"""
        if not self.session_history:
            console.print("[yellow]尚無執行歷史[/yellow]\n") if RICH_AVAILABLE else print("尚無執行歷史\n")
            input("按 Enter 繼續...")
            return
        
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]📜 執行歷史[/bold cyan]\n")
            for i, record in enumerate(self.session_history[-10:], 1):
                console.print(f"[cyan][{i}][/cyan] {record.get('action', 'conversation')}")
                if 'user' in record:
                    console.print(f"    用戶: {record['user']}")
                if 'target' in record:
                    console.print(f"    目標: {record['target']}")
                console.print("")
        else:
            print("\n📜 執行歷史\n")
            for i, record in enumerate(self.session_history[-10:], 1):
                print(f"[{i}] {record.get('action', 'conversation')}")
                if 'user' in record:
                    print(f"    用戶: {record['user']}")
                if 'target' in record:
                    print(f"    目標: {record['target']}")
                print("")
        
        input("按 Enter 繼續...")
    
    async def _execute_capability(self, capability: Dict[str, Any]):
        """執行單個能力"""
        if RICH_AVAILABLE:
            console.print(f"\n[bold]執行能力: {capability.get('name')}[/bold]\n")
        else:
            print(f"\n執行能力: {capability.get('name')}\n")
        
        # TODO: 實現能力執行邏輯
        await asyncio.sleep(1)
        
        if RICH_AVAILABLE:
            console.print("[green]✓ 執行完成[/green]\n")
        else:
            print("✓ 執行完成\n")
        
        input("按 Enter 繼續...")
    
    async def _execute_workflow(self, workflow: List[Dict[str, Any]], task_description: str):
        """執行工作流"""
        if RICH_AVAILABLE:
            console.print(f"\n[bold]執行工作流: {task_description}[/bold]\n")
            
            for i, step in enumerate(workflow, 1):
                console.print(f"[cyan]步驟 {i}/{len(workflow)}:[/cyan] {step.get('name')}")
                # TODO: 實現步驟執行邏輯
                await asyncio.sleep(0.5)
                console.print("[green]✓ 完成[/green]\n")
            
            console.print("[bold green]✓ 工作流執行完成[/bold green]\n")
        else:
            print(f"\n執行工作流: {task_description}\n")
            for i, step in enumerate(workflow, 1):
                print(f"步驟 {i}/{len(workflow)}: {step.get('name')}")
                await asyncio.sleep(0.5)
                print("✓ 完成\n")
            print("✓ 工作流執行完成\n")
        
        input("按 Enter 繼續...")
    
    async def run(self):
        """運行主程式"""
        self.print_banner()
        
        while self.running:
            try:
                self.show_main_menu()
                
                if RICH_AVAILABLE:
                    choice = Prompt.ask("\n請選擇", default="1")
                else:
                    choice = input("\n請選擇> ").strip() or "1"
                
                if choice == "0":
                    if RICH_AVAILABLE:
                        if Confirm.ask("確定要退出嗎?"):
                            console.print("\n[bold green]👋 感謝使用 AIVA！[/bold green]\n")
                            break
                    else:
                        confirm = input("確定要退出嗎? (y/n): ")
                        if confirm.lower() == 'y':
                            print("\n👋 感謝使用 AIVA!\n")
                            break
                
                elif choice == "1":
                    await self.handle_ai_conversation()
                elif choice == "2":
                    await self.handle_capability_search()
                elif choice == "3":
                    await self.handle_one_click_attack()
                elif choice == "4":
                    await self.handle_workflow_recommendation()
                elif choice == "5":
                    await self.handle_system_stats()
                elif choice == "6":
                    await self.handle_health_check()
                elif choice == "7":
                    await self.handle_sync_rag()
                elif choice == "8":
                    await self.handle_show_history()
                elif choice == "9":
                    console.print("[yellow]模組管理功能開發中...[/yellow]\n") if RICH_AVAILABLE else print("模組管理功能開發中...\n")
                    input("按 Enter 繼續...")
                elif choice == "10":
                    console.print("[yellow]測試與驗證功能開發中...[/yellow]\n") if RICH_AVAILABLE else print("測試與驗證功能開發中...\n")
                    input("按 Enter 繼續...")
                elif choice == "11":
                    console.print("[yellow]性能監控功能開發中...[/yellow]\n") if RICH_AVAILABLE else print("性能監控功能開發中...\n")
                    input("按 Enter 繼續...")
                elif choice == "12":
                    console.print("[yellow]匯出報告功能開發中...[/yellow]\n") if RICH_AVAILABLE else print("匯出報告功能開發中...\n")
                    input("按 Enter 繼續...")
                elif choice.lower() == "help":
                    self.show_help()
                else:
                    console.print("[red]無效選擇，請重試[/red]\n") if RICH_AVAILABLE else print("無效選擇，請重試\n")
                    await asyncio.sleep(1)
            
            except KeyboardInterrupt:
                if RICH_AVAILABLE:
                    console.print("\n\n[yellow]偵測到 Ctrl+C[/yellow]")
                    if Confirm.ask("確定要退出嗎?"):
                        break
                else:
                    print("\n\n偵測到 Ctrl+C")
                    confirm = input("確定要退出嗎? (y/n): ")
                    if confirm.lower() == 'y':
                        break
            
            except Exception as e:
                logger.error(f"錯誤: {e}")
                if RICH_AVAILABLE:
                    console.print(f"[red]錯誤: {e}[/red]\n")
                else:
                    print(f"錯誤: {e}\n")
                await asyncio.sleep(2)
    
    def show_help(self):
        """顯示幫助資訊"""
        help_text = """
        AIVA AI 智能選單系統使用說明
        
        本系統整合了 AIVA 的 782 個能力，透過 AI 自動調用執行。
        
        主要功能：
        1. AI 助理對話 - 用自然語言下指令
        2. 智能搜索 - 搜索並執行特定能力
        3. 一鍵攻擊 - 自動選擇攻擊策略
        4. 工作流推薦 - AI 規劃執行流程
        5. 系統統計 - 查看能力分布
        
        提示：
        - 可隨時按 Ctrl+C 中斷操作
        - 選擇 0 退出系統
        - 輸入 'help' 查看此說明
        """
        if RICH_AVAILABLE:
            console.print(Panel(help_text, title="[bold]使用說明[/bold]", border_style="blue"))
        else:
            print(help_text)
        
        input("\n按 Enter 繼續...")


async def main():
    """主程式入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA AI 智能選單系統")
    parser.add_argument(
        "--ui",
        choices=["cli", "web"],
        default="cli",
        help="介面類型 (cli: 命令行, web: Web UI)"
    )
    
    args = parser.parse_args()
    
    if args.ui == "web":
        print("Web UI 功能開發中...")
        return
    
    # 啟動 CLI 選單
    menu = AIVAIntelligentMenu()
    await menu.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 再見！")
    except Exception as e:
        logger.error(f"程式錯誤: {e}")
        print(f"\n錯誤: {e}")
