#!/usr/bin/env python3
"""
HackingTool SQL 工具管理 CLI
提供命令行介面來管理和使用 HackingTool SQL 注入工具

使用方式:
    python hackingtool_sql_cli.py status                    # 查看工具狀態
    python hackingtool_sql_cli.py install <tool_name>       # 安裝指定工具
    python hackingtool_sql_cli.py install-all              # 安裝所有工具
    python hackingtool_sql_cli.py test <tool_name> <url>   # 測試工具
    python hackingtool_sql_cli.py report                   # 生成狀態報告
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加 AIVA 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.aiva_common.utils.logging import get_logger
from services.features.function_sqli.hackingtool_manager import sql_tool_manager
from services.features.function_sqli.hackingtool_config import HACKINGTOOL_SQL_CONFIGS

logger = get_logger(__name__)


class HackingToolSQLCLI:
    """HackingTool SQL 工具 CLI"""
    
    def __init__(self):
        self.manager = sql_tool_manager
    
    async def show_status(self):
        """顯示所有工具狀態"""
        print("🔍 檢查 HackingTool SQL 工具狀態...\n")
        
        status_data = await self.manager.check_all_tools_status()
        
        # 統計
        total = len(status_data)
        installed = sum(1 for s in status_data.values() if s["installed"])
        executable = sum(1 for s in status_data.values() if s["executable"])
        
        print(f"📊 統計: {installed}/{total} 已安裝, {executable}/{total} 可執行\n")
        
        # 詳細狀態
        for tool_name, status in status_data.items():
            config = HACKINGTOOL_SQL_CONFIGS[tool_name]
            
            if status["executable"]:
                status_icon = "✅"
            elif status["installed"]:
                status_icon = "❌"
            else:
                status_icon = "⚪"
            print(f"{status_icon} {config.title}")
            print(f"   類型: {config.tool_type.value}")
            if status['executable']:
                status_text = "可執行"
            elif status['installed']:
                status_text = "已安裝"
            else:
                status_text = "未安裝"
            print(f"   狀態: {status_text}")
            
            if status["missing_dependencies"]:
                print(f"   缺少依賴: {', '.join(status['missing_dependencies'])}")
            
            if status["error"]:
                print(f"   錯誤: {status['error']}")
            
            print(f"   項目: {config.project_url}")
            print()
    
    async def install_tool(self, tool_name: str):
        """安裝指定工具"""
        if tool_name not in HACKINGTOOL_SQL_CONFIGS:
            print(f"❌ 未知工具: {tool_name}")
            print(f"可用工具: {', '.join(HACKINGTOOL_SQL_CONFIGS.keys())}")
            return
        
        config = HACKINGTOOL_SQL_CONFIGS[tool_name]
        print(f"🔧 正在安裝 {config.title}...")
        
        result = await self.manager.install_tool(tool_name)
        
        if result["success"]:
            print(f"✅ {tool_name} 安裝成功！")
            print(f"   安裝路徑: {result['install_path']}")
            print(f"   耗時: {result['duration']:.1f} 秒")
        else:
            print(f"❌ {tool_name} 安裝失敗")
            print(f"   錯誤: {result.get('error', 'Unknown error')}")
            if result.get('steps_completed'):
                print(f"   已完成步驟: {', '.join(result['steps_completed'])}")
    
    async def install_all_tools(self):
        """安裝所有工具"""
        print("🚀 開始批量安裝所有 HackingTool SQL 工具...\n")
        
        results = await self.manager.install_all_tools()
        
        # 顯示結果
        successful = []
        failed = []
        
        for tool_name, result in results.items():
            if result["success"]:
                successful.append(tool_name)
                print(f"✅ {tool_name}: 安裝成功 ({result['duration']:.1f}s)")
            else:
                failed.append(tool_name)
                print(f"❌ {tool_name}: 安裝失敗 - {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 安裝完成: {len(successful)} 成功, {len(failed)} 失敗")
        
        if failed:
            print(f"\n失敗的工具: {', '.join(failed)}")
    
    async def test_tool(self, tool_name: str, target_url: str):
        """測試工具對指定目標的檢測"""
        if tool_name not in HACKINGTOOL_SQL_CONFIGS:
            print(f"❌ 未知工具: {tool_name}")
            return
        
        config = HACKINGTOOL_SQL_CONFIGS[tool_name]
        print(f"🧪 使用 {config.title} 測試 {target_url}...")
        
        # 檢查工具狀態
        status = await self.manager._check_tool_status(tool_name, config)
        if not status["executable"]:
            print(f"❌ 工具 {tool_name} 不可執行，請先安裝")
            return
        
        # 模擬測試（實際應該調用檢測引擎）
        print("⚠️  注意：這是測試功能，實際檢測需要通過 AIVA 引擎執行")
        print(f"   工具: {config.title}")
        print(f"   目標: {target_url}")
        print(f"   類型: {config.tool_type.value}")
        print(f"   支援: GET={config.supports_get}, POST={config.supports_post}")
    
    async def generate_report(self):
        """生成詳細報告"""
        print("📄 生成 HackingTool SQL 工具狀態報告...\n")
        
        report = await self.manager.generate_status_report()
        
        # 顯示摘要
        summary = report["summary"]
        print("📊 摘要統計:")
        print(f"   總工具數: {summary['total_tools']}")
        print(f"   安裝率: {summary['installation_rate']}")
        print(f"   可執行率: {summary['executable_rate']}")
        print()
        
        # 按類型顯示
        print("🔧 按類型分類:")
        for tool_type, tools in report["by_type"].items():
            print(f"   {tool_type}:")
            for tool in tools:
                status = "✅" if tool["status"]["executable"] else "❌"
                print(f"     {status} {tool['name']}")
        print()
        
        # 保存報告到檔案
        report_file = Path("hackingtool_sql_report.json")
        report_content = json.dumps(report, indent=2, ensure_ascii=False)
        report_file.write_text(report_content, encoding='utf-8')
        
        print(f"📁 詳細報告已保存到: {report_file}")
    
    def list_tools(self):
        """列出所有可用工具"""
        print("📋 可用的 HackingTool SQL 工具:\n")
        
        for tool_name, config in HACKINGTOOL_SQL_CONFIGS.items():
            print(f"🔧 {tool_name}")
            print(f"   標題: {config.title}")
            print(f"   描述: {config.description}")
            print(f"   類型: {config.tool_type.value}")
            print(f"   優先級: {config.priority}")
            print(f"   項目: {config.project_url}")
            print()
    
    async def get_recommendations(self, target_type: str = "web"):
        """獲取工具推薦"""
        print(f"💡 針對 {target_type} 類型目標的工具推薦:\n")
        
        recommendations = await self.manager.get_tool_recommendations(target_type)
        
        for i, rec in enumerate(recommendations, 1):
            status = "✅ 可用" if rec["available"] else "❌ 需要安裝"
            print(f"{i}. {rec['tool']} - {status}")
            print(f"   推薦理由: {rec['reason']}")
            print(f"   優先級: {rec['priority']}")
            print()


async def main():
    parser = argparse.ArgumentParser(description="HackingTool SQL 工具管理 CLI")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # status 命令
    subparsers.add_parser('status', help='顯示所有工具狀態')
    
    # install 命令
    install_parser = subparsers.add_parser('install', help='安裝指定工具')
    install_parser.add_argument('tool_name', help='工具名稱')
    
    # install-all 命令
    subparsers.add_parser('install-all', help='安裝所有工具')
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='測試工具')
    test_parser.add_argument('tool_name', help='工具名稱')
    test_parser.add_argument('url', help='測試目標 URL')
    
    # report 命令
    subparsers.add_parser('report', help='生成狀態報告')
    
    # list 命令
    subparsers.add_parser('list', help='列出所有可用工具')
    
    # recommend 命令
    rec_parser = subparsers.add_parser('recommend', help='獲取工具推薦')
    rec_parser.add_argument('--type', default='web', help='目標類型 (web, api, etc.)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = HackingToolSQLCLI()
    
    try:
        if args.command == 'status':
            await cli.show_status()
        elif args.command == 'install':
            await cli.install_tool(args.tool_name)
        elif args.command == 'install-all':
            await cli.install_all_tools()
        elif args.command == 'test':
            await cli.test_tool(args.tool_name, args.url)
        elif args.command == 'report':
            await cli.generate_report()
        elif args.command == 'list':
            cli.list_tools()
        elif args.command == 'recommend':
            await cli.get_recommendations(args.type)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️  操作被用戶中斷")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        logger.error(f"CLI 執行錯誤: {e}")


if __name__ == "__main__":
    asyncio.run(main())