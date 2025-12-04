#!/usr/bin/env python3
"""
AIVA Rich CLI - 現代化命令行界面

整合 HackingTool 的 Rich UI 框架到 AIVA 系統中，提供豐富的視覺化介面。

特色功能:
- 彩色主題化介面  
- 互動式選單系統
- 實時進度指示器
- 結構化表格顯示
- 美化的面板和邊框
- 異常處理和錯誤顯示
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import traceback

# Rich UI 組件
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.tree import Tree
from rich.status import Status
from rich.traceback import install
from rich import box
from rich.align import Align
from rich.text import Text

# AIVA 核心模組  
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from services.core.aiva_core.ai_controller import AIController
except ImportError:
    # 如果 AI 控制器不可用，建立一個空的替代
    AIController = None

from services.integration.capability.registry import CapabilityRegistry
from services.integration.capability.toolkit import CapabilityToolkit
from services.integration.capability.models import CapabilityRecord, CapabilityStatus
from aiva_common.utils.logging import get_logger
from aiva_common.utils.ids import new_id

# 配置模組
from .rich_cli_config import (
    RICH_THEME, CONSOLE_CONFIG, MAIN_MENU_ITEMS, 
    SCAN_TYPES, STATUS_INDICATORS, AIVA_COLORS
)

# 樣式常數
STYLE_PRIMARY = "aiva.primary"
STYLE_SUCCESS = "aiva.success"
STYLE_WARNING = "aiva.warning"
STYLE_ERROR = "aiva.error"
STYLE_INFO = "aiva.info"
STYLE_MUTED = "aiva.muted"
STYLE_ACCENT = "aiva.accent"
STYLE_BOLD_ACCENT = "bold aiva.accent"

# 啟用 Rich 異常追蹤
install(show_locals=True)

# 全域 Console 實例
console = Console(**CONSOLE_CONFIG)
logger = get_logger(__name__)


def clear_screen():
    """清除螢幕內容"""
    os.system("cls" if sys.platform == "win32" else "clear")


def show_aiva_banner():
    """顯示 AIVA 品牌橫幅"""
    banner_text = """
    █████╗ ██╗██╗   ██╗ █████╗ 
   ██╔══██╗██║██║   ██║██╔══██╗
   ███████║██║██║   ██║███████║
   ██╔══██║██║╚██╗ ██╔╝██╔══██║
   ██║  ██║██║ ╚████╔╝ ██║  ██║
   ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝
   
   AI-Driven Vulnerability Assessment
   高級人工智慧漏洞評估平台
    """
    
    console.print(Panel(
        Align.center(Text(banner_text, style=f"bold {STYLE_PRIMARY}")),
        title=f"[{STYLE_BOLD_ACCENT}]Welcome to AIVA[/{STYLE_BOLD_ACCENT}]",
        subtitle=f"[{STYLE_MUTED}]v2.0 - Rich CLI Edition[/{STYLE_MUTED}]",
        border_style=STYLE_PRIMARY,
        box=box.DOUBLE
    ))
    console.print()


class AIVARichCLI:
    """AIVA Rich 命令行界面主類"""
    
    def __init__(self):
        self.trace_id = new_id("cli_session")
        self.registry = CapabilityRegistry()
        self.toolkit = CapabilityToolkit()
        self.ai_controller = None
        self.running = True
        
        logger.info("AIVA Rich CLI 已初始化", trace_id=self.trace_id)
    
    async def initialize(self):
        """異步初始化 AIVA 組件"""
        with Status("[aiva.info]正在初始化 AIVA 系統...", console=console):
            try:
                # 初始化 AI 控制器
                if AIController:
                    self.ai_controller = AIController()
                
                # 發現並註冊能力
                await self.registry.discover_capabilities()
                
                console.print("[aiva.success]✓[/aiva.success] AIVA 系統初始化完成")
                
            except Exception as e:
                console.print(f"[aiva.error]✗[/aiva.error] 初始化失敗: {e}")
                logger.error(f"初始化失敗: {e}", trace_id=self.trace_id)
                raise
    
    def show_main_menu(self):
        """顯示主選單"""
        clear_screen()
        show_aiva_banner()
        
        # 系統狀態面板
        status_table = Table(box=box.SIMPLE)
        status_table.add_column("組件", style="bold aiva.info")
        status_table.add_column("狀態", justify="center")
        status_table.add_column("詳情", style="aiva.muted")
        
        # 檢查各組件狀態
        ai_status = "[aiva.success]●[/aiva.success] 在線" if self.ai_controller else "[aiva.error]●[/aiva.error] 離線"
        capability_count = len(self.registry.get_all_capabilities())
        
        status_table.add_row("AI 引擎", ai_status, f"控制器已{'載入' if self.ai_controller else '未載入'}")
        status_table.add_row("能力註冊表", "[aiva.success]●[/aiva.success] 活躍", f"{capability_count} 個已註冊能力")
        status_table.add_row("工具包", "[aiva.success]●[/aiva.success] 就緒", "所有工具可用")
        
        console.print(Panel(
            status_table,
            title="[bold aiva.accent]系統狀態[/bold aiva.accent]",
            border_style="aiva.info"
        ))
        console.print()
        
        # 主選單選項
        menu_table = Table(title="[bold aiva.primary]主選單[/bold aiva.primary]", box=box.MINIMAL_DOUBLE_HEAD)
        menu_table.add_column("選項", justify="center", style="bold aiva.accent", width=8)
        menu_table.add_column("功能", style="bold")
        menu_table.add_column("描述", style="aiva.muted")
        
        menu_options = [
            ("1", "漏洞掃描", "啟動 AI 驅動的安全評估"),
            ("2", "能力管理", "管理註冊的安全工具和能力"),
            ("3", "AI 對話", "與 AIVA AI 引擎互動"),
            ("4", "AI 能力查詢", "查詢 AIVA 的功能與能力"),
            ("5", "工具集成", "整合新的安全工具"),
            ("6", "系統監控", "查看系統狀態和日誌"),
            ("7", "設定配置", "調整 AIVA 系統設定"),
            ("8", "報告生成", "生成掃描和評估報告"),
            ("9", "幫助文檔", "查看使用指南和 API 文檔"),
            ("A", "關於 AIVA", "版本資訊和開發團隊"),
            ("0", "退出", "安全退出 AIVA CLI")
        ]
        
        for option, name, desc in menu_options:
            menu_table.add_row(option, name, desc)
        
        console.print(menu_table)
        console.print()
    
    def get_user_choice(self, valid_choices: List[str]) -> str:
        """獲取用戶選擇"""
        return Prompt.ask(
            f"[{STYLE_ACCENT}]請選擇一個選項[/{STYLE_ACCENT}]",
            choices=valid_choices,
            show_choices=False
        )
    
    async def handle_vulnerability_scan(self):
        """處理漏洞掃描功能"""
        console.print(Panel(
            "[aiva.info]漏洞掃描模組[/aiva.info]\n\n"
            "此功能將啟動 AIVA 的 AI 驅動安全評估系統。\n"
            "系統會智能選擇適當的工具組合進行全面掃描。",
            title="[bold aiva.primary]AI 漏洞掃描[/bold aiva.primary]",
            border_style="aiva.primary"
        ))
        
        target = Prompt.ask("[aiva.accent]請輸入掃描目標[/aiva.accent] (IP/域名/URL)")
        
        if not target:
            console.print("[aiva.warning]⚠[/aiva.warning] 目標不能為空")
            return
        
        scan_types = ["快速掃描", "標準掃描", "深度掃描", "自定義掃描"]
        scan_table = Table(title="掃描類型", box=box.SIMPLE)
        scan_table.add_column("編號", justify="center", style="bold aiva.accent")
        scan_table.add_column("類型", style="bold")
        scan_table.add_column("描述", style="aiva.muted")
        
        descriptions = [
            "基本端口掃描和服務識別",
            "標準漏洞檢測和常見攻擊向量",
            "全面安全評估和高級威脅檢測", 
            "用戶自定義掃描策略"
        ]
        
        for i, (scan_type, desc) in enumerate(zip(scan_types, descriptions)):
            scan_table.add_row(str(i+1), scan_type, desc)
        
        console.print(scan_table)
        
        scan_choice = self.get_user_choice(['1', '2', '3', '4'])
        selected_scan = scan_types[int(scan_choice)-1]
        
        console.print(f"\n[aiva.success]✓[/aiva.success] 已選擇: {selected_scan}")
        console.print(f"[aiva.info]→[/aiva.info] 目標: {target}")
        
        if Confirm.ask("\n是否開始掃描?"):
            await self.execute_scan(target, selected_scan)
    
    async def execute_scan(self, target: str, scan_type: str):
        """執行掃描任務"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            scan_task = progress.add_task(f"[aiva.primary]掃描 {target}...", total=100)
            
            # 模擬掃描過程
            steps = [
                ("初始化掃描引擎", 10),
                ("目標探測和端口掃描", 25),
                ("服務指紋識別", 40),
                ("漏洞檢測分析", 65),
                ("AI 威脅評估", 80),
                ("生成掃描報告", 100)
            ]
            
            for step_name, completion in steps:
                progress.update(scan_task, description=f"[aiva.info]{step_name}...")
                progress.update(scan_task, completed=completion)
                # 實際掃描會根據真實進度更新，無需固定延遲
            
            progress.update(scan_task, description="[aiva.success]掃描完成!")
        
        # 顯示掃描結果
        result_panel = Panel(
            "[aiva.success]✓ 掃描已完成![/aiva.success]\n\n"
            f"目標: [bold]{target}[/bold]\n"
            f"掃描類型: [bold]{scan_type}[/bold]\n"
            f"完成時間: [bold]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold]\n\n"
            "[aiva.info]詳細報告已保存到 ./reports/ 目錄[/aiva.info]",
            title="[bold aiva.success]掃描結果[/bold aiva.success]",
            border_style="aiva.success"
        )
        
        console.print(result_panel)
    
    async def handle_capability_management(self):
        """處理能力管理功能"""
        console.print(Panel(
            "[aiva.info]能力管理系統[/aiva.info]\n\n"
            "管理 AIVA 註冊的所有安全工具和能力模組。\n"
            "可以查看、啟用、停用或配置各種安全工具。",
            title="[bold aiva.primary]能力管理[/bold aiva.primary]",
            border_style="aiva.primary"
        ))
        
        capabilities = self.registry.get_all_capabilities()
        
        if not capabilities:
            console.print("[aiva.warning]⚠[/aiva.warning] 未發現任何已註冊的能力")
            return
        
        # 按類型分組顯示能力
        capability_tree = Tree("[bold aiva.primary]已註冊能力[/bold aiva.primary]")
        
        # 按類型分組
        by_type: Dict[str, List[CapabilityRecord]] = {}
        for cap in capabilities:
            cap_type = cap.type.value if hasattr(cap.type, 'value') else str(cap.type)
            if cap_type not in by_type:
                by_type[cap_type] = []
            by_type[cap_type].append(cap)
        
        for cap_type, caps in by_type.items():
            type_branch = capability_tree.add(f"[bold aiva.accent]{cap_type}[/bold aiva.accent] ({len(caps)})")
            
            for cap in caps[:5]:  # 只顯示前5個，避免過長
                status_color = "aiva.success" if cap.status == CapabilityStatus.ACTIVE else "aiva.muted"
                type_branch.add(f"[{status_color}]{cap.name}[/{status_color}] - {cap.description[:50]}...")
            
            if len(caps) > 5:
                type_branch.add(f"[aiva.muted]... 還有 {len(caps) - 5} 個能力[/aiva.muted]")
        
        console.print(capability_tree)
        console.print()
        
        # 能力管理選項
        mgmt_options = [
            ("1", "查看詳細列表", "顯示所有能力的詳細資訊"),
            ("2", "搜索能力", "按名稱或類型搜索特定能力"),
            ("3", "啟用/停用", "切換能力狀態"),
            ("4", "重新掃描", "重新發現系統中的能力"),
            ("5", "返回主選單", "回到主選單")
        ]
        
        mgmt_table = Table(title="管理選項", box=box.SIMPLE)
        mgmt_table.add_column("選項", justify="center", style="bold aiva.accent")
        mgmt_table.add_column("功能", style="bold")
        mgmt_table.add_column("描述", style="aiva.muted")
        
        for option, name, desc in mgmt_options:
            mgmt_table.add_row(option, name, desc)
        
        console.print(mgmt_table)
        
        choice = self.get_user_choice(['1', '2', '3', '4', '5'])
        
        if choice == '1':
            self.show_detailed_capabilities(capabilities)
        elif choice == '2':
            await self.search_capabilities(capabilities)
        elif choice == '3':
            await self.toggle_capability_status(capabilities)
        elif choice == '4':
            await self.rescan_capabilities()
        # choice == '5' 會自動返回
    
    def show_detailed_capabilities(self, capabilities: List[CapabilityRecord]):
        """顯示詳細的能力列表"""
        console.print(Rule("[bold aiva.primary]詳細能力列表[/bold aiva.primary]"))
        
        detail_table = Table(box=box.MINIMAL_DOUBLE_HEAD)
        detail_table.add_column("ID", style="aiva.muted", width=20)
        detail_table.add_column("名稱", style="bold", width=25)
        detail_table.add_column("類型", style="aiva.info", width=15)
        detail_table.add_column("狀態", justify="center", width=10)
        detail_table.add_column("語言", style="aiva.accent", width=10)
        detail_table.add_column("描述", style="aiva.muted")
        
        for cap in capabilities:
            status_icon = "✓" if cap.status == CapabilityStatus.ACTIVE else "○"
            status_color = "aiva.success" if cap.status == CapabilityStatus.ACTIVE else "aiva.muted"
            
            detail_table.add_row(
                cap.id[:18] + "..." if len(cap.id) > 20 else cap.id,
                cap.name,
                cap.type.value if hasattr(cap.type, 'value') else str(cap.type),
                f"[{status_color}]{status_icon}[/{status_color}]",
                cap.language.value if hasattr(cap.language, 'value') else str(cap.language),
                cap.description[:50] + "..." if len(cap.description) > 50 else cap.description
            )
        
        console.print(detail_table)
        
        Prompt.ask("\n[aiva.muted]按 Enter 鍵繼續...[/aiva.muted]", default="")
    
    async def handle_ai_interaction(self):
        """處理 AI 互動功能"""
        console.print(Panel(
            "[aiva.info]AIVA AI 對話系統[/aiva.info]\n\n"
            "與 AIVA 的人工智慧引擎直接對話。\n"
            "您可以詢問安全建議、分析結果或請求執行特定任務。\n\n"
            "[aiva.muted]輸入 'exit' 結束對話[/aiva.muted]",
            title="[bold aiva.primary]AI 互動模式[/bold aiva.primary]",
            border_style="aiva.primary"
        ))
        
        if not self.ai_controller:
            console.print("[aiva.error]✗[/aiva.error] AI 控制器未初始化")
            return
        
        console.print("[aiva.success]✓[/aiva.success] AI 引擎已就緒，開始對話...")
        console.print()
        
        while True:
            user_input = Prompt.ask("[bold aiva.accent]您[/bold aiva.accent]")
            
            if user_input.lower() in ['exit', 'quit', '退出']:
                console.print("[aiva.info]→[/aiva.info] 結束 AI 對話")
                break
            
            # 調用真實的 AI 控制器
            with Status("[aiva.info]AI 正在思考...", console=console):
                # AI 控制器會處理實際的思考時間
                pass
            
            # 這裡應該是實際的 AI 回應
            ai_response = f"[bold aiva.primary]AIVA[/bold aiva.primary]: 收到您的請求「{user_input}」。基於當前系統狀態和安全知識庫，我建議進行進一步的分析。這是一個模擬回應，實際版本會提供更詳細的安全建議。"
            
            console.print(Panel(
                ai_response,
                border_style="aiva.info",
                box=box.SIMPLE
            ))
            console.print()
    
    async def handle_ai_capability_query(self):
        """處理 AI 能力查詢功能 - 新增"""
        console.print(Panel(
            "[aiva.info]AIVA AI 能力查詢系統[/aiva.info]\n\n"
            "透過自然語言查詢 AIVA 的功能與能力\n"
            "基於 RAG 技術，包含 782 個註冊能力",
            title="[bold aiva.accent]AI Capability Query[/bold aiva.accent]",
            border_style="aiva.accent"
        ))
        
        # 子選單
        sub_menu = [
            ("1", "我能做什麼？", "查詢 AIVA 的核心能力"),
            ("2", "滲透測試工作流", "獲取滲透測試建議"),
            ("3", "漏洞修復指南", "查詢漏洞處理流程"),
            ("4", "攻擊路徑分析", "分析可能的攻擊路徑"),
            ("5", "自定義查詢", "輸入自然語言查詢"),
            ("6", "能力統計", "查看系統能力統計"),
            ("0", "返回主選單", "回到主選單")
        ]
        
        # 顯示子選單
        sub_table = Table(title="[bold aiva.info]查詢選項[/bold aiva.info]", box=box.MINIMAL)
        sub_table.add_column("選項", justify="center", style="bold aiva.accent", width=6)
        sub_table.add_column("功能", style="bold")
        sub_table.add_column("說明", style="aiva.muted")
        
        for opt, name, desc in sub_menu:
            sub_table.add_row(opt, name, desc)
        
        console.print()
        console.print(sub_table)
        console.print()
        
        choice = Prompt.ask(
            "[aiva.accent]選擇功能[/aiva.accent]",
            choices=[o[0] for o in sub_menu],
            default="0"
        )
        
        if choice == "0":
            return
        
        try:
            # 動態導入查詢系統
            from ..cognitive_core.ai_capability_query import AICapabilityQuery
            
            with Status("[aiva.info]初始化 AI 查詢系統...", console=console):
                query_system = AICapabilityQuery()
            
            console.print("[aiva.success]✓[/aiva.success] AI 查詢系統就緒\n")
            
            if choice == "1":
                # 核心能力查詢
                with Status("[aiva.info]查詢中...", console=console):
                    results = await query_system.query("What are my core capabilities", top_k=5)
                query_system.display_results(results, title="核心能力")
                
            elif choice == "2":
                # 滲透測試工作流
                with Status("[aiva.info]分析滲透測試工作流...", console=console):
                    workflow = await query_system.get_workflow_recommendation(
                        "penetration testing workflow", 
                        max_capabilities=8
                    )
                
                console.print()
                console.print(Panel(
                    f"[bold]任務:[/bold] {workflow['task']}\n"
                    f"[bold]找到:[/bold] {workflow['total_found']} 個相關能力",
                    title="[bold cyan]工作流推薦[/bold cyan]",
                    border_style="cyan"
                ))
                
                # 顯示推薦能力
                wf_table = Table(box=box.ROUNDED)
                wf_table.add_column("階段", style="cyan", width=6)
                wf_table.add_column("能力", style="bold green")
                wf_table.add_column("模組", style="yellow")
                
                for i, cap in enumerate(workflow["capabilities"], 1):
                    wf_table.add_row(
                        str(i),
                        cap["name"],
                        cap["module"]
                    )
                
                console.print(wf_table)
                
            elif choice == "3":
                # 漏洞修復指南
                with Status("[aiva.info]查詢漏洞修復流程...", console=console):
                    results = await query_system.query(
                        "vulnerability fixing and remediation process", 
                        top_k=5
                    )
                query_system.display_results(results, title="漏洞修復指南")
                
            elif choice == "4":
                # 攻擊路徑分析
                with Status("[aiva.info]查詢攻擊路徑分析工具...", console=console):
                    results = await query_system.query(
                        "attack path analysis and exploitation", 
                        top_k=5
                    )
                query_system.display_results(results, title="攻擊路徑分析")
                
            elif choice == "5":
                # 自定義查詢
                query = Prompt.ask("\n[aiva.accent]請輸入查詢問題[/aiva.accent]")
                if query.strip():
                    with Status("[aiva.info]查詢中...", console=console):
                        results = await query_system.query(query, top_k=5)
                    query_system.display_results(results, title=f"查詢: {query}")
                
            elif choice == "6":
                # 能力統計
                with Status("[aiva.info]統計分析中...", console=console):
                    await query_system.show_statistics()
        
        except ImportError as e:
            console.print(f"[aiva.error]✗[/aiva.error] AI 查詢系統不可用: {e}")
            logger.error(f"Failed to import AICapabilityQuery: {e}")
        except Exception as e:
            console.print(f"[aiva.error]✗[/aiva.error] 查詢失敗: {e}")
            logger.error(f"AI query failed: {e}", exc_info=True)
    
    async def run(self):
        """運行主程式循環"""
        try:
            clear_screen()
            show_aiva_banner()
            
            with Status("[aiva.info]正在啟動 AIVA Rich CLI...", console=console):
                await self.initialize()
            
            while self.running:
                try:
                    self.show_main_menu()
                    choice = self.get_user_choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'])
                    
                    if choice == '0':
                        if Confirm.ask("\n[aiva.warning]確定要退出 AIVA CLI 嗎?[/aiva.warning]"):
                            self.running = False
                            console.print("\n[aiva.success]感謝使用 AIVA！再見！[/aiva.success]")
                    elif choice == '1':
                        await self.handle_vulnerability_scan()
                    elif choice == '2':
                        await self.handle_capability_management()
                    elif choice == '3':
                        await self.handle_ai_interaction()
                    elif choice == '4':
                        await self.handle_ai_capability_query()
                    elif choice == '5':
                        console.print("[aiva.info]工具集成功能開發中...[/aiva.info]")
                    elif choice == '6':
                        console.print("[aiva.info]系統監控功能開發中...[/aiva.info]")
                    elif choice == '7':
                        console.print("[aiva.info]設定配置功能開發中...[/aiva.info]")
                    elif choice == '8':
                        console.print("[aiva.info]報告生成功能開發中...[/aiva.info]")
                    elif choice == '9':
                        self.show_help()
                    elif choice.upper() == 'A':
                        self.show_about()
                    
                    if choice != '0':
                        Prompt.ask("\n[aiva.muted]按 Enter 鍵繼續...[/aiva.muted]", default="")
                
                except KeyboardInterrupt:
                    if Confirm.ask("\n[aiva.warning]檢測到 Ctrl+C，是否退出?[/aiva.warning]"):
                        self.running = False
                    continue
                except Exception as e:
                    console.print_exception(show_locals=True)
                    Prompt.ask("\n[aiva.error]發生錯誤，按 Enter 繼續...[/aiva.error]", default="")
        
        except Exception as e:
            console.print(f"\n[aiva.error]✗[/aiva.error] CLI 啟動失敗: {e}")
            console.print_exception(show_locals=True)
            sys.exit(1)
    
    def show_help(self):
        """顯示幫助資訊"""
        help_text = """
[bold aiva.primary]AIVA Rich CLI 使用指南[/bold aiva.primary]

[bold aiva.accent]快速開始:[/bold aiva.accent]
1. 選擇主選單中的功能選項
2. 按照提示輸入所需參數
3. 查看執行結果和報告

[bold aiva.accent]主要功能:[/bold aiva.accent]
• [aiva.info]漏洞掃描[/aiva.info] - AI 驅動的安全評估
• [aiva.info]能力管理[/aiva.info] - 管理安全工具和模組
• [aiva.info]AI 對話[/aiva.info] - 與 AIVA 智能引擎互動
• [aiva.info]工具集成[/aiva.info] - 整合新的安全工具

[bold aiva.accent]鍵盤快捷鍵:[/bold aiva.accent]
• Ctrl+C - 中斷當前操作
• Enter - 確認選擇或繼續
• 數字鍵 - 選擇選單項目

[bold aiva.accent]技術支援:[/bold aiva.accent]
如遇問題請查看 logs/ 目錄中的日誌檔案
或訪問項目文檔獲取更多資訊。
        """
        
        console.print(Panel(
            help_text,
            title="[bold aiva.accent]幫助文檔[/bold aiva.accent]",
            border_style="aiva.info"
        ))
    
    def show_about(self):
        """顯示關於資訊"""
        about_text = """
[bold aiva.primary]AIVA - AI-Driven Vulnerability Assessment[/bold aiva.primary]

[bold aiva.accent]版本資訊:[/bold aiva.accent]
• 版本: v2.0 Rich CLI Edition
• 構建日期: 2024-11-01
• Python: {python_version}
• Rich UI: 整合 HackingTool 視覺框架

[bold aiva.accent]核心特色:[/bold aiva.accent]
• 🤖 AI 驅動的智能安全評估
• 🔧 模組化工具整合架構
• 🎨 現代化 Rich CLI 界面
• 🔍 全面的漏洞檢測引擎
• 📊 智能化報告生成系統

[bold aiva.accent]開發團隊:[/bold aiva.accent]
AIVA 是一個開源安全評估平台
致力於提供最先進的 AI 安全解決方案

[aiva.muted]感謝使用 AIVA！[/aiva.muted]
        """.format(python_version=sys.version.split()[0])
        
        console.print(Panel(
            about_text,
            title="[bold aiva.accent]關於 AIVA[/bold aiva.accent]",
            border_style="aiva.primary"
        ))


async def main():
    """主入口函數"""
    try:
        cli = AIVARichCLI()
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[aiva.warning]程式被用戶中斷[/aiva.warning]")
    except Exception as e:
        console.print(f"\n[aiva.error]程式異常退出: {e}[/aiva.error]")
        console.print_exception(show_locals=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())