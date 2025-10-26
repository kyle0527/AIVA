#!/usr/bin/env python3
"""
AIVA 能力管理命令行工具
整合 aiva_common 的所有工具和插件功能

功能特色:
- 使用 aiva_common 的 schema 工具進行驗證和程式碼產生
- 利用現有的連接性測試和模組管理工具
- 遵循統一的日誌和追蹤標準
- 支援豐富的互動式操作和報告產生
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yaml

# 加入 AIVA 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aiva_common.utils.logging import get_logger
from aiva_common.utils.ids import new_id
from aiva_common.enums import ProgrammingLanguage, Severity, Confidence

from .registry import CapabilityRegistry
from .models import CapabilityRecord, CapabilityType, CapabilityStatus
from .toolkit import CapabilityToolkit

# 設定結構化日誌
logger = get_logger(__name__)


class CapabilityManager:
    """AIVA 能力管理器 - 命令行介面"""
    
    def __init__(self):
        self.registry = CapabilityRegistry()
        self.toolkit = CapabilityToolkit()
        self.trace_id = new_id("trace")
        
        logger.info("AIVA 能力管理器已初始化", trace_id=self.trace_id)
    
    async def discover_and_register(self, auto_register: bool = False) -> Dict[str, Any]:
        """發現並註冊系統中的能力"""
        
        logger.info("開始能力發現過程", trace_id=self.trace_id)
        
        # 使用註冊中心的發現功能
        discovery_stats = await self.registry.discover_capabilities()
        
        print(f"🔍 發現統計:")
        print(f"   總共發現: {discovery_stats['discovered_count']} 個能力")
        
        for lang, count in discovery_stats.get('languages', {}).items():
            print(f"   {lang}: {count} 個")
        
        print(f"\n📦 模組分布:")
        for module, count in discovery_stats.get('modules', {}).items():
            print(f"   {module}: {count} 個能力")
        
        if discovery_stats.get('errors'):
            print(f"\n❌ 發現過程中的錯誤:")
            for error in discovery_stats['errors']:
                print(f"   {error}")
        
        # 如果啟用自動註冊，則將發現的能力註冊到系統中
        if auto_register:
            print(f"\n🔄 自動註冊中...")
            registered_count = 0
            
            # 這裡需要獲取發現的能力並逐一註冊
            # 目前的實現需要進一步完善
            print(f"✅ 已註冊 {registered_count} 個能力")
        
        return discovery_stats
    
    async def list_capabilities(
        self, 
        language: Optional[str] = None,
        capability_type: Optional[str] = None,
        status: Optional[str] = None,
        output_format: str = "table"
    ) -> None:
        """列出能力"""
        
        # 轉換篩選參數
        lang_filter = ProgrammingLanguage(language) if language else None
        type_filter = CapabilityType(capability_type) if capability_type else None
        status_filter = CapabilityStatus(status) if status else None
        
        capabilities = await self.registry.list_capabilities(
            language=lang_filter,
            capability_type=type_filter,
            status=status_filter
        )
        
        if not capabilities:
            print("未找到符合條件的能力。")
            return
        
        if output_format == "table":
            self._print_capabilities_table(capabilities)
        elif output_format == "json":
            self._print_capabilities_json(capabilities)
        elif output_format == "yaml":
            self._print_capabilities_yaml(capabilities)
        else:
            print(f"不支援的輸出格式: {output_format}")
    
    def _print_capabilities_table(self, capabilities: List[CapabilityRecord]) -> None:
        """以表格形式顯示能力"""
        
        print(f"\n📋 能力列表 ({len(capabilities)} 個):")
        print("=" * 120)
        print(f"{'ID':<30} {'名稱':<25} {'語言':<10} {'類型':<12} {'狀態':<10} {'模組':<20}")
        print("-" * 120)
        
        for cap in capabilities:
            print(f"{cap.id:<30} {cap.name[:24]:<25} {cap.language.value:<10} "
                  f"{cap.capability_type.value:<12} {cap.status.value:<10} {cap.module:<20}")
        
        print("=" * 120)
    
    def _print_capabilities_json(self, capabilities: List[CapabilityRecord]) -> None:
        """以 JSON 形式顯示能力"""
        
        capabilities_data = [cap.model_dump() for cap in capabilities]
        print(json.dumps(capabilities_data, indent=2, ensure_ascii=False, default=str))
    
    def _print_capabilities_yaml(self, capabilities: List[CapabilityRecord]) -> None:
        """以 YAML 形式顯示能力"""
        
        capabilities_data = [cap.model_dump() for cap in capabilities]
        print(yaml.dump(capabilities_data, default_flow_style=False, allow_unicode=True))
    
    async def inspect_capability(self, capability_id: str) -> None:
        """詳細檢查能力"""
        
        capability = await self.registry.get_capability(capability_id)
        
        if not capability:
            print(f"❌ 能力 '{capability_id}' 不存在")
            return
        
        print(f"\n🔍 能力詳細資訊: {capability.name}")
        print("=" * 80)
        
        # 基本資訊
        print(f"📋 基本資訊:")
        print(f"   ID: {capability.id}")
        print(f"   名稱: {capability.name}")
        print(f"   描述: {capability.description}")
        print(f"   版本: {capability.version}")
        print(f"   模組: {capability.module}")
        print(f"   語言: {capability.language.value}")
        print(f"   類型: {capability.capability_type.value}")
        print(f"   狀態: {capability.status.value}")
        
        # 入口點和配置
        print(f"\n⚙️  執行配置:")
        print(f"   入口點: {capability.entrypoint}")
        print(f"   超時時間: {capability.timeout_seconds} 秒")
        print(f"   重試次數: {capability.retry_count} 次")
        print(f"   優先級: {capability.priority}/100")
        
        # 輸入參數
        if capability.inputs:
            print(f"\n📥 輸入參數 ({len(capability.inputs)} 個):")
            for param in capability.inputs:
                required = "必需" if param.required else "可選"
                default = f" (默認: {param.default})" if param.default is not None else ""
                print(f"   - {param.name} ({param.type}) [{required}]{default}")
                print(f"     {param.description}")
        
        # 輸出參數
        if capability.outputs:
            print(f"\n📤 輸出參數 ({len(capability.outputs)} 個):")
            for output in capability.outputs:
                print(f"   - {output.name} ({output.type})")
                print(f"     {output.description}")
        
        # 依賴關係
        if capability.dependencies:
            print(f"\n🔗 依賴關係 ({len(capability.dependencies)} 個):")
            for dep in capability.dependencies:
                print(f"   - {dep}")
        
        # 前置條件
        if capability.prerequisites:
            print(f"\n✅ 前置條件 ({len(capability.prerequisites)} 個):")
            for prereq in capability.prerequisites:
                print(f"   - {prereq}")
        
        # 標籤
        if capability.tags:
            print(f"\n🏷️  標籤: {', '.join(capability.tags)}")
        
        # 時間戳
        print(f"\n🕒 時間資訊:")
        print(f"   創建時間: {capability.created_at.isoformat()}")
        print(f"   更新時間: {capability.updated_at.isoformat()}")
        if capability.last_probe:
            print(f"   最後探測: {capability.last_probe.isoformat()}")
        if capability.last_success:
            print(f"   最後成功: {capability.last_success.isoformat()}")
        
        print("=" * 80)
    
    async def test_capability(self, capability_id: str, verbose: bool = False) -> None:
        """測試能力連接性"""
        
        capability = await self.registry.get_capability(capability_id)
        
        if not capability:
            print(f"❌ 能力 '{capability_id}' 不存在")
            return
        
        print(f"🧪 測試能力: {capability.name}")
        print("=" * 60)
        
        # 使用工具集進行連接性測試
        evidence = await self.toolkit.test_capability_connectivity(capability)
        
        # 顯示測試結果
        status_icon = "✅" if evidence.success else "❌"
        print(f"{status_icon} 測試結果: {'成功' if evidence.success else '失敗'}")
        print(f"⏱️  延遲時間: {evidence.latency_ms} 毫秒")
        print(f"🔍 探測類型: {evidence.probe_type}")
        print(f"📅 測試時間: {evidence.timestamp.isoformat()}")
        
        if evidence.trace_id:
            print(f"🔗 追蹤ID: {evidence.trace_id}")
        
        if evidence.error_message:
            print(f"❗ 錯誤訊息: {evidence.error_message}")
        
        if verbose and evidence.metadata:
            print(f"\n📊 詳細資訊:")
            for key, value in evidence.metadata.items():
                print(f"   {key}: {value}")
        
        print("=" * 60)
    
    async def validate_capability_schema(self, capability_file: str) -> None:
        """驗證能力模式定義"""
        
        try:
            # 讀取能力定義檔案
            capability_path = Path(capability_file)
            if not capability_path.exists():
                print(f"❌ 檔案不存在: {capability_file}")
                return
            
            # 根據檔案格式解析
            if capability_path.suffix.lower() == '.json':
                with open(capability_path, 'r', encoding='utf-8') as f:
                    capability_data = json.load(f)
            elif capability_path.suffix.lower() in ['.yaml', '.yml']:
                with open(capability_path, 'r', encoding='utf-8') as f:
                    capability_data = yaml.safe_load(f)
            else:
                print(f"❌ 不支援的檔案格式: {capability_path.suffix}")
                return
            
            # 創建能力物件
            capability = CapabilityRecord.model_validate(capability_data)
            
            print(f"🔍 驗證能力定義: {capability.name}")
            print("=" * 60)
            
            # 使用工具集進行模式驗證
            is_valid, errors = await self.toolkit.validate_capability_schema(capability)
            
            if is_valid:
                print("✅ 模式驗證通過")
            else:
                print("❌ 模式驗證失敗")
                for error in errors:
                    print(f"   • {error}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 驗證過程中出現錯誤: {str(e)}")
            logger.error("能力模式驗證失敗", error=str(e), exc_info=True)
    
    async def generate_documentation(
        self, 
        capability_id: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> None:
        """產生能力文件"""
        
        if capability_id:
            # 為單個能力產生文件
            capability = await self.registry.get_capability(capability_id)
            if not capability:
                print(f"❌ 能力 '{capability_id}' 不存在")
                return
            
            doc = await self.toolkit.generate_capability_documentation(capability)
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(doc)
                print(f"📄 文件已儲存至: {output_file}")
            else:
                print(doc)
        else:
            # 為所有能力產生摘要文件
            capabilities = await self.registry.list_capabilities()
            summary = await self.toolkit.export_capabilities_summary(capabilities)
            
            # 產生摘要報告
            summary_md = self._generate_summary_report(summary)
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary_md)
                print(f"📊 摘要報告已儲存至: {output_file}")
            else:
                print(summary_md)
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> str:
        """產生摘要報告"""
        
        report = f"""# AIVA 能力註冊中心摘要報告

**生成時間**: {summary['generated_at']}
**總能力數**: {summary['total_count']}

## 語言分布

"""
        
        for lang, count in summary['by_language'].items():
            percentage = (count / summary['total_count']) * 100
            report += f"- **{lang}**: {count} 個 ({percentage:.1f}%)\n"
        
        report += "\n## 能力類型分布\n\n"
        
        for cap_type, count in summary['by_type'].items():
            percentage = (count / summary['total_count']) * 100
            report += f"- **{cap_type}**: {count} 個 ({percentage:.1f}%)\n"
        
        report += "\n## 健康狀態概覽\n\n"
        
        health = summary['health_overview']
        total = sum(health.values())
        for status, count in health.items():
            percentage = (count / total) * 100 if total > 0 else 0
            status_icon = {"healthy": "✅", "issues": "⚠️", "unknown": "❓"}.get(status, "❓")
            report += f"- {status_icon} **{status}**: {count} 個 ({percentage:.1f}%)\n"
        
        report += "\n## 最近更新的能力\n\n"
        
        for update in summary['recent_updates']:
            report += f"- `{update['id']}` - {update['name']} ({update['status']}) - {update['updated_at']}\n"
        
        if summary['top_dependencies']:
            report += "\n## 熱門依賴\n\n"
            
            for dep, count in list(summary['top_dependencies'].items())[:10]:
                report += f"- `{dep}`: {count} 個能力依賴\n"
        
        return report
    
    async def generate_bindings(
        self, 
        capability_id: str,
        target_languages: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """產生跨語言綁定"""
        
        capability = await self.registry.get_capability(capability_id)
        if not capability:
            print(f"❌ 能力 '{capability_id}' 不存在")
            return
        
        print(f"🔧 為能力 '{capability.name}' 產生跨語言綁定...")
        
        # 使用工具集產生綁定
        bindings = await self.toolkit.generate_cross_language_bindings(capability)
        
        if not bindings:
            print("❌ 無法產生任何語言綁定")
            return
        
        # 篩選目標語言
        if target_languages:
            bindings = {
                lang: code for lang, code in bindings.items() 
                if lang in target_languages
            }
        
        # 輸出綁定
        for lang, code in bindings.items():
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # 根據語言選擇檔案擴展名
                ext_map = {
                    "python": ".py",
                    "go": ".go", 
                    "rust": ".rs",
                    "typescript": ".ts"
                }
                
                filename = f"{capability_id.replace('.', '_')}_binding{ext_map.get(lang, '.txt')}"
                file_path = output_path / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                print(f"✅ {lang} 綁定已儲存至: {file_path}")
            else:
                print(f"\n--- {lang.upper()} 綁定 ---")
                print(code)
                print("-" * 50)
    
    async def show_stats(self) -> None:
        """顯示系統統計資訊"""
        
        stats = await self.registry.get_capability_stats()
        
        print("\n📊 AIVA 能力註冊中心統計")
        print("=" * 50)
        
        print(f"📦 總能力數: {stats['total_capabilities']}")
        
        print(f"\n🔤 語言分布:")
        for lang, count in stats['by_language'].items():
            print(f"   {lang}: {count}")
        
        print(f"\n🎯 類型分布:")
        for cap_type, count in stats['by_type'].items():
            print(f"   {cap_type}: {count}")
        
        print(f"\n💚 健康狀態:")
        for status, count in stats['health_summary'].items():
            status_icon = {
                "healthy": "✅",
                "degraded": "⚠️", 
                "failed": "❌",
                "unknown": "❓"
            }.get(status, "❓")
            print(f"   {status_icon} {status}: {count}")
        
        print("=" * 50)


async def main():
    """主程式入口"""
    
    parser = argparse.ArgumentParser(
        description="AIVA 能力管理命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s discover --auto-register     # 發現並自動註冊能力
  %(prog)s list --language python       # 列出 Python 能力
  %(prog)s inspect security.sqli.scan   # 檢查特定能力
  %(prog)s test security.sqli.scan      # 測試能力連接性
  %(prog)s validate capability.yaml     # 驗證能力定義
  %(prog)s docs --all --output report.md # 產生完整報告
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # discover 命令
    discover_parser = subparsers.add_parser('discover', help='發現系統中的能力')
    discover_parser.add_argument('--auto-register', action='store_true', 
                                help='自動註冊發現的能力')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出已註冊的能力')
    list_parser.add_argument('--language', choices=['python', 'go', 'rust', 'typescript'],
                           help='按語言篩選')
    list_parser.add_argument('--type', choices=['scanner', 'analyzer', 'remediation', 'integration', 'utility'],
                           help='按類型篩選')
    list_parser.add_argument('--status', choices=['healthy', 'degraded', 'failed', 'unknown'],
                           help='按狀態篩選')
    list_parser.add_argument('--format', choices=['table', 'json', 'yaml'], default='table',
                           help='輸出格式')
    
    # inspect 命令
    inspect_parser = subparsers.add_parser('inspect', help='詳細檢查能力')
    inspect_parser.add_argument('capability_id', help='能力ID')
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='測試能力連接性')
    test_parser.add_argument('capability_id', help='能力ID')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='顯示詳細資訊')
    
    # validate 命令
    validate_parser = subparsers.add_parser('validate', help='驗證能力定義')
    validate_parser.add_argument('file', help='能力定義檔案 (JSON/YAML)')
    
    # docs 命令
    docs_parser = subparsers.add_parser('docs', help='產生能力文件')
    docs_parser.add_argument('capability_id', nargs='?', help='特定能力ID')
    docs_parser.add_argument('--all', action='store_true', help='產生所有能力的摘要')
    docs_parser.add_argument('--output', '-o', help='輸出檔案路徑')
    
    # bindings 命令
    bindings_parser = subparsers.add_parser('bindings', help='產生跨語言綁定')
    bindings_parser.add_argument('capability_id', help='能力ID')
    bindings_parser.add_argument('--languages', nargs='+', 
                                choices=['python', 'go', 'rust', 'typescript'],
                                help='目標語言')
    bindings_parser.add_argument('--output-dir', help='輸出目錄')
    
    # stats 命令
    stats_parser = subparsers.add_parser('stats', help='顯示統計資訊')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 創建管理器實例
    manager = CapabilityManager()
    
    try:
        if args.command == 'discover':
            await manager.discover_and_register(auto_register=args.auto_register)
        
        elif args.command == 'list':
            await manager.list_capabilities(
                language=args.language,
                capability_type=args.type,
                status=args.status,
                output_format=args.format
            )
        
        elif args.command == 'inspect':
            await manager.inspect_capability(args.capability_id)
        
        elif args.command == 'test':
            await manager.test_capability(args.capability_id, verbose=args.verbose)
        
        elif args.command == 'validate':
            await manager.validate_capability_schema(args.file)
        
        elif args.command == 'docs':
            if args.all:
                await manager.generate_documentation(output_file=args.output)
            else:
                await manager.generate_documentation(
                    capability_id=args.capability_id,
                    output_file=args.output
                )
        
        elif args.command == 'bindings':
            await manager.generate_bindings(
                capability_id=args.capability_id,
                target_languages=args.languages,
                output_dir=args.output_dir
            )
        
        elif args.command == 'stats':
            await manager.show_stats()
            
    except KeyboardInterrupt:
        print("\n⚠️  操作已取消")
    except Exception as e:
        print(f"❌ 操作失敗: {str(e)}")
        logger.error("命令執行失敗", command=args.command, error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())