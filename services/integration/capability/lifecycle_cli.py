#!/usr/bin/env python3
"""
AIVA 工具生命週期管理 CLI
基於 HackingTool 的 Rich UI 設計模式

提供互動式命令行介面來管理工具的安裝、更新、卸載等生命週期操作
"""

import asyncio
import argparse
import sys
from typing import Optional, List
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.text import Text
from rich.live import Live

from aiva_common.utils.logging import get_logger

from .lifecycle import ToolLifecycleManager, InstallationResult, ToolLifecycleEvent
from .models import CapabilityRecord


logger = get_logger(__name__)
console = Console()

# 常數定義
PROGRESS_TEXT_FORMAT = "[progress.description]{task.description}"
TOOL_ID_PROMPT = "請輸入工具 ID"
TOOL_ID_COLUMN = "工具 ID"


class LifecycleCLI:
    """工具生命週期管理 CLI 介面"""
    
    def __init__(self):
        self.lifecycle_manager = ToolLifecycleManager()
    
    async def install_tool(self, capability_id: str, force: bool = False) -> None:
        """安裝工具"""
        console.print(f"\n🔧 安裝工具: [bold blue]{capability_id}[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_FORMAT),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("正在安裝...", total=None)
            
            result = await self.lifecycle_manager.install_tool(capability_id, force)
            
            progress.update(task, completed=True)
        
        if result.success:
            console.print("✅ 安裝成功!")
            
            # 顯示安裝詳情
            table = Table(show_header=False, box=None)
            table.add_row("📁 安裝路徑:", result.installation_path or "未知")
            table.add_row("📦 版本:", result.installed_version or "未知")
            table.add_row("⏱️ 安裝時間:", f"{result.installation_time_seconds:.2f} 秒")
            
            if result.dependencies_installed:
                table.add_row("🔗 已安裝依賴:", ", ".join(result.dependencies_installed))
            
            console.print(table)
        else:
            console.print(f"❌ 安裝失敗: {result.error_message}")
    
    async def update_tool(self, capability_id: str) -> None:
        """更新工具"""
        console.print(f"\n🔄 更新工具: [bold blue]{capability_id}[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_FORMAT),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("正在更新...", total=None)
            
            success = await self.lifecycle_manager.update_tool(capability_id)
            
            progress.update(task, completed=True)
        
        if success:
            console.print("✅ 更新成功!")
        else:
            console.print("❌ 更新失敗")
    
    async def uninstall_tool(self, capability_id: str, remove_deps: bool = False) -> None:
        """卸載工具"""
        console.print(f"\n🗑️ 卸載工具: [bold blue]{capability_id}[/bold blue]")
        
        # 確認卸載
        if not Confirm.ask("確定要卸載此工具嗎?"):
            console.print("已取消卸載")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_FORMAT),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("正在卸載...", total=None)
            
            success = await self.lifecycle_manager.uninstall_tool(capability_id, remove_deps)
            
            progress.update(task, completed=True)
        
        if success:
            console.print("✅ 卸載成功!")
        else:
            console.print("❌ 卸載失敗")
    
    async def health_check(self, capability_id: Optional[str] = None) -> None:
        """健康檢查"""
        if capability_id:
            # 單個工具健康檢查
            console.print(f"\n🩺 檢查工具健康狀態: [bold blue]{capability_id}[/bold blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_TEXT_FORMAT),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("正在檢查...", total=None)
                
                health_info = await self.lifecycle_manager.health_check_tool(capability_id)
                
                progress.update(task, completed=True)
            
            self._display_health_info(health_info)
        else:
            # 批量健康檢查
            console.print("\n🩺 批量健康檢查")
            
            with Progress(
                SpinnerColumn(),
                TextColumn(PROGRESS_TEXT_FORMAT),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("正在檢查所有工具...", total=None)
                
                summary = await self.lifecycle_manager.batch_health_check()
                
                progress.update(task, completed=True)
            
            self._display_health_summary(summary)
    
    def _display_health_info(self, health_info: dict) -> None:
        """顯示單個工具健康資訊"""
        if health_info["success"]:
            status_icon = "✅"
            status_color = "green"
        else:
            status_icon = "❌"
            status_color = "red"
        
        panel_content = []
        panel_content.append(f"{status_icon} 狀態: [{status_color}]{health_info.get('status', 'unknown')}[/{status_color}]")
        panel_content.append(f"🏠 安裝路徑: {health_info.get('installation_path', 'N/A')}")
        panel_content.append(f"💾 已安裝: {health_info.get('is_installed', False)}")
        panel_content.append(f"⚡ 延遲: {health_info.get('latency_ms', 0)} ms")
        panel_content.append(f"🕐 檢查時間: {health_info.get('last_check', 'N/A')}")
        
        if health_info.get("error_message"):
            panel_content.append(f"❗ 錯誤: {health_info['error_message']}")
        
        console.print(Panel(
            "\n".join(panel_content),
            title=f"工具健康狀態 - {health_info.get('capability_name', 'Unknown')}",
            border_style=status_color,
            padding=(1, 2)
        ))
    
    def _display_health_summary(self, summary: dict) -> None:
        """顯示健康檢查摘要"""
        # 摘要資訊
        health_rate = summary["health_rate"]
        if health_rate >= 0.8:
            rate_color = "green"
        elif health_rate >= 0.5:
            rate_color = "yellow"
        else:
            rate_color = "red"
        
        summary_text = []
        summary_text.append(f"📊 總工具數: {summary['total_tools']}")
        summary_text.append(f"✅ 健康工具: {summary['healthy_tools']}")
        summary_text.append(f"❌ 異常工具: {summary['unhealthy_tools']}")
        summary_text.append(f"📈 健康率: [{rate_color}]{health_rate:.1%}[/{rate_color}]")
        
        console.print(Panel(
            "\n".join(summary_text),
            title="健康檢查摘要",
            border_style=rate_color,
            padding=(1, 2)
        ))
        
        # 詳細結果表格
        if summary["results"]:
            table = Table(title="詳細檢查結果")
            table.add_column(TOOL_ID_COLUMN, style="cyan")
            table.add_column("狀態", justify="center")
            table.add_column("已安裝", justify="center")
            table.add_column("延遲 (ms)", justify="right", style="magenta")
            table.add_column("錯誤訊息", style="red", max_width=50)
            
            for cap_id, result in summary["results"].items():
                status_icon = "✅" if result.get("success") else "❌"
                installed_icon = "✅" if result.get("is_installed") else "❌"
                latency = str(result.get("latency_ms", 0))
                error = result.get("error", "")[:50] if result.get("error") else ""
                
                table.add_row(cap_id, status_icon, installed_icon, latency, error)
            
            console.print(table)
    
    def show_events(
        self, 
        capability_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 20
    ) -> None:
        """顯示生命週期事件歷史"""
        console.print("\n📜 生命週期事件歷史")
        
        events = self.lifecycle_manager.get_lifecycle_events(
            capability_id=capability_id,
            event_type=event_type,
            limit=limit
        )
        
        if not events:
            console.print("沒有找到符合條件的事件")
            return
        
        table = Table(title=f"最近 {len(events)} 個事件")
        table.add_column("時間", style="cyan", width=20)
        table.add_column(TOOL_ID_COLUMN, style="yellow", width=25)
        table.add_column("事件類型", justify="center", width=15)
        table.add_column("狀態", justify="center", width=10)
        table.add_column("詳情", max_width=40)
        
        for event in events:
            timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # 狀態圖示
            if event.status == "success":
                status_icon = "✅"
            elif event.status == "failed":
                status_icon = "❌"
            elif event.status == "in_progress":
                status_icon = "🔄"
            else:
                status_icon = "❓"
            
            # 事件類型圖示
            if event.event_type == "install":
                type_icon = "📦 安裝"
            elif event.event_type == "update":
                type_icon = "🔄 更新"
            elif event.event_type == "uninstall":
                type_icon = "🗑️ 卸載"
            elif event.event_type == "health_check":
                type_icon = "🩺 健檢"
            else:
                type_icon = event.event_type
            
            details = ""
            if event.error_message:
                details = f"錯誤: {event.error_message[:35]}..."
            elif event.details:
                details = str(event.details)[:35] + "..." if len(str(event.details)) > 35 else str(event.details)
            
            table.add_row(
                timestamp,
                event.capability_id,
                type_icon,
                status_icon,
                details
            )
        
        console.print(table)
    
    async def interactive_menu(self) -> None:
        """互動式選單 - 基於 HackingTool 的 Rich UI 設計"""
        while True:
            console.clear()
            
            # 顯示標題
            title = Text("AIVA 工具生命週期管理器", style="bold cyan")
            console.print(Panel(title, expand=False))
            
            # 顯示選項
            options = [
                "1. 安裝工具",
                "2. 更新工具", 
                "3. 卸載工具",
                "4. 工具健康檢查",
                "5. 批量健康檢查",
                "6. 查看事件歷史",
                "7. 列出所有工具",
                "0. 退出"
            ]
            
            for option in options:
                console.print(f"  {option}")
            
            choice = Prompt.ask("\n請選擇操作", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            
            try:
                if choice == "0":
                    console.print("👋 再見!")
                    break
                elif choice == "1":
                    capability_id = Prompt.ask(TOOL_ID_PROMPT)
                    force = Confirm.ask("強制重新安裝?", default=False)
                    await self.install_tool(capability_id, force)
                elif choice == "2":
                    capability_id = Prompt.ask(TOOL_ID_PROMPT)
                    await self.update_tool(capability_id)
                elif choice == "3":
                    capability_id = Prompt.ask(TOOL_ID_PROMPT)
                    remove_deps = Confirm.ask("同時移除依賴?", default=False)
                    await self.uninstall_tool(capability_id, remove_deps)
                elif choice == "4":
                    capability_id = Prompt.ask(TOOL_ID_PROMPT)
                    await self.health_check(capability_id)
                elif choice == "5":
                    await self.health_check()
                elif choice == "6":
                    capability_id = Prompt.ask(f"{TOOL_ID_COLUMN} (留空查看全部)", default="")
                    self.show_events(capability_id if capability_id else None)
                elif choice == "7":
                    await self.list_tools()
                
                if choice != "0":
                    Prompt.ask("\n按 Enter 鍵繼續...")
                    
            except KeyboardInterrupt:
                console.print("\n\n👋 操作已取消，再見!")
                break
            except Exception as e:
                console.print(f"❌ 操作失敗: {str(e)}")
                Prompt.ask("\n按 Enter 鍵繼續...")
    
    async def list_tools(self) -> None:
        """顯示所有已註冊的工具"""
        console.print("\n📋 已註冊的工具")
        
        capabilities = await self.lifecycle_manager.registry.list_capabilities()
        
        if not capabilities:
            console.print("沒有找到已註冊的工具")
            return
        
        table = Table(title=f"共 {len(capabilities)} 個工具")
        table.add_column("ID", style="cyan", width=25)
        table.add_column("名稱", style="yellow", width=25)
        table.add_column("語言", justify="center", width=10)
        table.add_column("狀態", justify="center", width=10)
        table.add_column("描述", max_width=40)
        
        for cap in capabilities:
            # 狀態圖示
            if cap.status.value == "healthy":
                status_icon = "✅"
            elif cap.status.value == "degraded":
                status_icon = "⚠️"
            elif cap.status.value == "unhealthy":
                status_icon = "❌"
            else:
                status_icon = "❓"
            
            table.add_row(
                cap.id,
                cap.name,
                cap.language.value,
                status_icon,
                cap.description[:40] + "..." if len(cap.description) > 40 else cap.description
            )
        
        console.print(table)


async def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="AIVA 工具生命週期管理器")
    parser.add_argument("--interactive", "-i", action="store_true", help="啟動互動模式")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 安裝命令
    install_parser = subparsers.add_parser("install", help="安裝工具")
    install_parser.add_argument("capability_id", help="工具 ID")
    install_parser.add_argument("--force", "-f", action="store_true", help="強制重新安裝")
    
    # 更新命令
    update_parser = subparsers.add_parser("update", help="更新工具")
    update_parser.add_argument("capability_id", help="工具 ID")
    
    # 卸載命令
    uninstall_parser = subparsers.add_parser("uninstall", help="卸載工具")
    uninstall_parser.add_argument("capability_id", help="工具 ID")
    uninstall_parser.add_argument("--remove-deps", action="store_true", help="同時移除依賴")
    
    # 健康檢查命令
    health_parser = subparsers.add_parser("health", help="健康檢查")
    health_parser.add_argument("capability_id", nargs="?", help="工具 ID (留空檢查所有)")
    
    # 事件歷史命令
    events_parser = subparsers.add_parser("events", help="查看事件歷史")
    events_parser.add_argument("--capability-id", help="過濾特定工具")
    events_parser.add_argument("--event-type", help="過濾事件類型")
    events_parser.add_argument("--limit", type=int, default=20, help="顯示數量限制")
    
    # 列表命令
    subparsers.add_parser("list", help="列出所有工具")
    
    args = parser.parse_args()
    
    cli = LifecycleCLI()
    
    try:
        if args.interactive or not args.command:
            await cli.interactive_menu()
        elif args.command == "install":
            await cli.install_tool(args.capability_id, args.force)
        elif args.command == "update":
            await cli.update_tool(args.capability_id)
        elif args.command == "uninstall":
            await cli.uninstall_tool(args.capability_id, args.remove_deps)
        elif args.command == "health":
            await cli.health_check(args.capability_id)
        elif args.command == "events":
            cli.show_events(
                capability_id=args.capability_id,
                event_type=args.event_type,
                limit=args.limit
            )
        elif args.command == "list":
            await cli.list_tools()
    except KeyboardInterrupt:
        console.print("\n\n👋 程式已中斷")
    except Exception as e:
        console.print(f"❌ 執行錯誤: {str(e)}")
        logger.error("CLI 執行錯誤", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())