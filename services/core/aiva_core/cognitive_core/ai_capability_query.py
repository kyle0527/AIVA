"""AI 能力查詢系統 - 用戶友好的 AI 分析接口（v2.0 六大模組支持）

整合 AI 自我分析能力到 AIVA 核心架構，提供簡化的查詢接口。
支持六大模組分類、過濾、統計和分類報告生成。

使用方式:
    from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery
    
    query_system = AICapabilityQuery()
    results = await query_system.query("如何進行滲透測試", top_k=5)
    query_system.display_results(results)
    
    # 按模組過濾
    results = await query_system.query_with_filters(
        question="掃描能力",
        aiva_module="core_capabilities",
        entry_point="AICommander"
    )

CLI 使用:
    # 查詢
    python -m services.core.aiva_core.cognitive_core.ai_capability_query "XSS檢測"
    
    # 統計
    python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats
    
    # 按模組過濾
    python -m services.core.aiva_core.cognitive_core.ai_capability_query --module cognitive_core
    
    # 生成分類報告
    python -m services.core.aiva_core.cognitive_core.ai_capability_query --classify
"""

import asyncio
import argparse
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .internal_loop_connector import InternalLoopConnector
from .rag.knowledge_base import KnowledgeBase
from .rag.vector_store import VectorStore


logger = logging.getLogger(__name__)

# ============================================================================
# 六大模組常數定義
# ============================================================================

AIVA_SIX_MODULES = [
    "cognitive_core",      # 認知核心
    "internal_exploration", # 內部探索
    "task_planning",        # 任務規劃
    "external_learning",    # 外部學習
    "core_capabilities",    # 核心能力
    "service_backbone"      # 服務骨幹
]

AIVA_ENTRY_POINTS = [
    "app.py",                    # FastAPI HTTP 入口
    "AICommander",               # AI 指揮官
    "CapabilityOrchestrator",    # 能力編排器
    "InternalLoopConnector",     # 內閉環連接器
    "ExternalLoopConnector",     # 外閉環連接器
    "BackgroundTask"             # 後台任務
]

if RICH_AVAILABLE:
    console = Console()


class AICapabilityQuery:
    """AI 能力查詢系統
    
    提供簡化的 AI 分析接口，整合自我認知能力到 AIVA 核心。
    
    特性:
    - 自然語言查詢
    - 能力統計分析
    - 工作流推薦
    - Rich UI 顯示 (可選)
    
    示例:
        >>> query_system = AICapabilityQuery()
        >>> results = await query_system.query("掃描工具")
        >>> query_system.display_results(results)
    """
    
    def __init__(self, persist_dir: Optional[Path] = None):
        """初始化查詢系統
        
        Args:
            persist_dir: ChromaDB 持久化目錄，默認為 data/vector_db/chroma
        """
        if persist_dir is None:
            persist_dir = Path("data/vector_db/chroma")
        
        self.persist_dir = persist_dir
        
        # 延遲初始化，只在需要時加載
        self._vector_store = None
        self._kb = None
        self._connector = None
        
        logger.info(f"AICapabilityQuery initialized with persist_dir: {persist_dir}")
    
    @property
    def vector_store(self):
        """延遲加載 VectorStore"""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                backend="chroma", 
                persist_directory=self.persist_dir
            )
        return self._vector_store
    
    @property
    def kb(self):
        """延遲加載 KnowledgeBase"""
        if self._kb is None:
            self._kb = KnowledgeBase(vector_store=self.vector_store)
        return self._kb
    
    @property
    def connector(self):
        """延遲加載 InternalLoopConnector"""
        if self._connector is None:
            self._connector = InternalLoopConnector(rag_knowledge_base=self.kb)
        return self._connector
    
    async def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """查詢能力 - async保留供未來異步查詢擴展
        
        Args:
            question: 自然語言問題
                例如: "如何進行滲透測試", "掃描工具有哪些", "修復漏洞的方法"
            top_k: 返回結果數量
            
        Returns:
            能力列表，每個能力包含 metadata (capability_name, module, language 等)
            
        示例:
            >>> results = await query_system.query("攻擊路徑分析", top_k=3)
            >>> print(results[0]["metadata"]["capability_name"])
            'find_attack_paths'
        """
        logger.info(f"Querying: '{question}' (top_k={top_k})")
        
        try:
            results = await asyncio.to_thread(
                self.connector.query_self_awareness, question, top_k=top_k
            )
            logger.info(f"Found {len(results.results)} results")
            return results.results
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return []
    
    def display_results(self, results: List[Dict[str, Any]], title: str = "查詢結果"):
        """顯示查詢結果
        
        Args:
            results: query() 返回的結果列表
            title: 顯示標題
        """
        if not results:
            if RICH_AVAILABLE:
                console.print("[yellow]未找到相關能力[/yellow]")
            else:
                print("未找到相關能力")
            return
        
        if RICH_AVAILABLE:
            self._display_results_rich(results, title)
        else:
            self._display_results_plain(results, title)
    
    def _display_results_rich(self, results: List[Dict[str, Any]], title: str):
        """Rich UI 顯示"""
        table = Table(title=f"[bold cyan]{title}[/bold cyan]", box=box.ROUNDED)
        table.add_column("#", justify="center", style="cyan", width=4)
        table.add_column("能力名稱", style="bold green", no_wrap=True)
        table.add_column("模組", style="yellow")
        table.add_column("語言", justify="center", style="magenta", width=10)
        
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            table.add_row(
                str(i),
                meta.get("capability_name", "Unknown"),
                meta.get("module", "Unknown"),
                meta.get("language", "Unknown")
            )
        
        console.print(table)
    
    def _display_results_plain(self, results: List[Dict[str, Any]], title: str):
        """純文本顯示"""
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}\n")
        
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            print(f"{i}. {meta.get('capability_name', 'Unknown')}")
            print(f"   Module: {meta.get('module', 'Unknown')}")
            print(f"   Language: {meta.get('language', 'Unknown')}")
            print()
    
    async def show_statistics(self) -> Dict[str, Any]:
        """顯示能力統計
        
        Returns:
            統計數據字典，包含 total, modules, languages 等
        """
        try:
            import chromadb
            
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            collection = await asyncio.to_thread(client.get_collection, 'aiva_capabilities')
            all_data = await asyncio.to_thread(collection.get, include=['metadatas'])
            
            # 統計分析
            if all_data['metadatas'] is not None:
                modules = Counter([m.get('module', 'unknown') for m in all_data['metadatas']])
                languages = Counter([m.get('language', 'unknown') for m in all_data['metadatas']])
                total = len(all_data['metadatas'])
            else:
                modules = Counter()
                languages = Counter()
                total = 0
            
            stats = {
                "total": total,
                "modules": dict(modules.most_common()),
                "languages": dict(languages.most_common())
            }
            
            # 顯示
            if RICH_AVAILABLE:
                self._display_statistics_rich(stats)
            else:
                self._display_statistics_plain(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {"total": 0, "modules": {}, "languages": {}}
    
    def _display_statistics_rich(self, stats: Dict[str, Any]):
        """Rich UI 顯示統計"""
        total = stats["total"]
        
        # 模組分布
        module_table = Table(title="[bold]模組分布 (Top 10)[/bold]", box=box.SIMPLE)
        module_table.add_column("模組", style="cyan")
        module_table.add_column("數量", justify="right", style="green")
        module_table.add_column("佔比", justify="right", style="yellow")
        
        for module, count in list(stats["modules"].items())[:10]:
            percentage = count / total * 100
            module_table.add_row(module, str(count), f"{percentage:.1f}%")
        
        console.print(module_table)
        console.print()
        
        # 語言分布
        lang_table = Table(title="[bold]語言分布[/bold]", box=box.SIMPLE)
        lang_table.add_column("語言", style="cyan")
        lang_table.add_column("數量", justify="right", style="green")
        lang_table.add_column("佔比", justify="right", style="yellow")
        
        for lang, count in stats["languages"].items():
            percentage = count / total * 100
            lang_table.add_row(lang, str(count), f"{percentage:.1f}%")
        
        console.print(lang_table)
        
        # 總結面板
        summary = Panel(
            f"[bold]總計:[/bold] {total} 個能力\n"
            f"[bold]模組數:[/bold] {len(stats['modules'])}\n"
            f"[bold]語言數:[/bold] {len(stats['languages'])}",
            title="[bold cyan]系統摘要[/bold cyan]",
            border_style="cyan"
        )
        console.print()
        console.print(summary)
    
    def _display_statistics_plain(self, stats: Dict[str, Any]):
        """純文本顯示統計"""
        total = stats["total"]
        
        print("\n" + "=" * 60)
        print("模組分布 (Top 10)")
        print("=" * 60)
        for module, count in list(stats["modules"].items())[:10]:
            percentage = count / total * 100
            print(f"{module:30} {count:5} ({percentage:5.1f}%)")
        
        print("\n" + "=" * 60)
        print("語言分布")
        print("=" * 60)
        for lang, count in stats["languages"].items():
            percentage = count / total * 100
            print(f"{lang:30} {count:5} ({percentage:5.1f}%)")
        
        print("\n" + "=" * 60)
        print("系統摘要")
        print("=" * 60)
        print(f"總計: {total} 個能力")
        print(f"模組數: {len(stats['modules'])}")
        print(f"語言數: {len(stats['languages'])}")
        print()
    
    async def get_workflow_recommendation(
        self, 
        task: str, 
        max_capabilities: int = 10
    ) -> Dict[str, Any]:
        """獲取工作流推薦
        
        Args:
            task: 任務描述，如 "滲透測試", "漏洞修復", "攻擊路徑分析"
            max_capabilities: 最多推薦的能力數量
            
        Returns:
            推薦的工作流字典，包含:
                - task: 任務名稱
                - capabilities: 推薦的能力列表
                - total_found: 找到的能力總數
                
        示例:
            >>> workflow = await query_system.get_workflow_recommendation("滲透測試")
            >>> for cap in workflow["capabilities"]:
            ...     print(cap["name"], "-", cap["module"])
        """
        logger.info(f"Getting workflow recommendation for: {task}")
        
        capabilities = await self.query(task, top_k=max_capabilities)
        
        workflow = {
            "task": task,
            "capabilities": [],
            "total_found": len(capabilities)
        }
        
        for cap in capabilities:
            meta = cap.get("metadata", {})
            workflow["capabilities"].append({
                "name": meta.get("capability_name", "Unknown"),
                "module": meta.get("module", "Unknown"),
                "language": meta.get("language", "Unknown"),
                "is_async": meta.get("is_async", False)
            })
        
        logger.info(f"Recommended {len(workflow['capabilities'])} capabilities for '{task}'")
        return workflow
    
    async def query_by_module(self, module: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """按模組查詢能力
        
        Args:
            module: 模組名稱，如 "scan", "core/aiva_core", "integration"
            top_k: 返回結果數量
            
        Returns:
            該模組的能力列表
        """
        query_text = f"module {module} capabilities"
        results = await self.query(query_text, top_k=top_k * 2)  # 查多一些再過濾
        
        # 過濾出匹配模組的結果
        filtered = [
            r for r in results 
            if r.get("metadata", {}).get("module") == module
        ]
        
        return filtered[:top_k]
    
    async def query_by_language(self, language: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """按語言查詢能力
        
        Args:
            language: 語言名稱，如 "python", "rust", "typescript", "go"
            top_k: 返回結果數量
            
        Returns:
            該語言的能力列表
        """
        query_text = f"{language} programming language implementations"
        results = await self.query(query_text, top_k=top_k * 2)
        
        # 過濾出匹配語言的結果
        filtered = [
            r for r in results 
            if r.get("metadata", {}).get("language", "").lower() == language.lower()
        ]
        
        return filtered[:top_k]
    
    async def query_with_filters(
        self,
        question: str,
        aiva_module: Optional[str] = None,
        entry_point: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """帶過濾條件的查詢（六大模組支持）
        
        Args:
            question: 自然語言問題
            aiva_module: AIVA 六大模組之一，如 "cognitive_core", "internal_exploration"
            entry_point: 入口點名稱，如 "AICommander", "app.py"
            top_k: 返回結果數量
            
        Returns:
            過濾後的能力列表
            
        示例:
            >>> results = await query_system.query_with_filters(
            ...     "掃描能力",
            ...     aiva_module="core_capabilities",
            ...     entry_point="AICommander"
            ... )
        """
        # 先查詢獲取較多結果
        results = await self.query(question, top_k=top_k * 3)
        
        # 應用過濾條件
        filtered = results
        
        if aiva_module:
            filtered = [
                r for r in filtered
                if r.get("metadata", {}).get("aiva_module") == aiva_module
            ]
            logger.info(f"Filtered by aiva_module={aiva_module}: {len(filtered)} results")
        
        if entry_point:
            filtered = [
                r for r in filtered
                if r.get("metadata", {}).get("entry_point") == entry_point
            ]
            logger.info(f"Filtered by entry_point={entry_point}: {len(filtered)} results")
        
        return filtered[:top_k]
    
    async def get_classification_report(self) -> Dict[str, Any]:
        """生成完整的能力分類報告（六大模組）
        
        Returns:
            分類報告字典，包含:
                - total: 總能力數
                - by_module: 按六大模組分類的統計
                - by_entry_point: 按入口點分類的統計
                - by_sub_module: 按子模組分類的統計
                - report_time: 生成時間
        """
        logger.info("Generating classification report...")
        
        try:
            import chromadb
            
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            collection = await asyncio.to_thread(client.get_collection, 'aiva_capabilities')
            all_data = await asyncio.to_thread(collection.get, include=['metadatas'])
            
            if not all_data['metadatas']:
                return self._empty_classification_report()
            
            # 統計分析
            by_module = defaultdict(list)
            by_entry_point = defaultdict(list)
            by_sub_module = defaultdict(list)
            
            for meta in all_data['metadatas']:
                cap_name = meta.get('capability_name', 'Unknown')
                
                # 按六大模組分類
                aiva_module = meta.get('aiva_module')
                if aiva_module:
                    by_module[aiva_module].append(cap_name)
                else:
                    by_module['unclassified'].append(cap_name)
                
                # 按入口點分類
                entry_point = meta.get('entry_point')
                if entry_point:
                    by_entry_point[entry_point].append(cap_name)
                else:
                    by_entry_point['unknown'].append(cap_name)
                
                # 按子模組分類
                sub_module = meta.get('sub_module')
                if sub_module:
                    by_sub_module[sub_module].append(cap_name)
            
            report = {
                "total": len(all_data['metadatas']),
                "by_module": {k: len(v) for k, v in by_module.items()},
                "by_entry_point": {k: len(v) for k, v in by_entry_point.items()},
                "by_sub_module": {k: len(v) for k, v in by_sub_module.items()},
                "details": {
                    "module_capabilities": dict(by_module),
                    "entry_point_capabilities": dict(by_entry_point),
                    "sub_module_capabilities": dict(by_sub_module)
                },
                "report_time": datetime.now().isoformat()
            }
            
            logger.info(f"Classification report generated: {report['total']} capabilities")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate classification report: {e}", exc_info=True)
            return self._empty_classification_report()
    
    def _empty_classification_report(self) -> Dict[str, Any]:
        """返回空的分類報告"""
        return {
            "total": 0,
            "by_module": {},
            "by_entry_point": {},
            "by_sub_module": {},
            "details": {},
            "report_time": datetime.now().isoformat()
        }
    
    def display_classification_report(self, report: Dict[str, Any]):
        """顯示分類報告
        
        Args:
            report: get_classification_report() 返回的報告
        """
        if RICH_AVAILABLE:
            self._display_classification_report_rich(report)
        else:
            self._display_classification_report_plain(report)
    
    def _display_classification_report_rich(self, report: Dict[str, Any]):
        """Rich UI 顯示分類報告"""
        console.print("\n[bold cyan]AIVA 能力分類報告[/bold cyan]")
        console.print(f"生成時間: {report['report_time']}\n")
        
        # 總覽
        console.print(Panel(
            f"[bold]總計:[/bold] {report['total']} 個能力",
            title="[bold yellow]總覽[/bold yellow]",
            border_style="yellow"
        ))
        console.print()
        
        # 按六大模組分類
        if report['by_module']:
            module_table = Table(title="[bold]按六大模組分類[/bold]", box=box.ROUNDED)
            module_table.add_column("模組", style="cyan")
            module_table.add_column("能力數量", justify="right", style="green")
            module_table.add_column("佔比", justify="right", style="yellow")
            
            for module in AIVA_SIX_MODULES:
                count = report['by_module'].get(module, 0)
                if count > 0:
                    percentage = count / report['total'] * 100 if report['total'] > 0 else 0
                    module_table.add_row(module, str(count), f"{percentage:.1f}%")
            
            # 未分類
            unclassified = report['by_module'].get('unclassified', 0)
            if unclassified > 0:
                percentage = unclassified / report['total'] * 100 if report['total'] > 0 else 0
                module_table.add_row("[red]unclassified[/red]", str(unclassified), f"{percentage:.1f}%")
            
            console.print(module_table)
            console.print()
        
        # 按入口點分類
        if report['by_entry_point']:
            entry_table = Table(title="[bold]按入口點分類[/bold]", box=box.ROUNDED)
            entry_table.add_column("入口點", style="cyan")
            entry_table.add_column("能力數量", justify="right", style="green")
            entry_table.add_column("佔比", justify="right", style="yellow")
            
            sorted_entries = sorted(report['by_entry_point'].items(), key=lambda x: x[1], reverse=True)
            for entry, count in sorted_entries[:10]:  # 只顯示前 10
                percentage = count / report['total'] * 100 if report['total'] > 0 else 0
                entry_table.add_row(entry, str(count), f"{percentage:.1f}%")
            
            console.print(entry_table)
            console.print()
    
    def _display_classification_report_plain(self, report: Dict[str, Any]):
        """純文本顯示分類報告"""
        print("\n" + "=" * 60)
        print("AIVA 能力分類報告")
        print("=" * 60)
        print(f"生成時間: {report['report_time']}")
        print(f"總計: {report['total']} 個能力\n")
        
        # 按六大模組分類
        if report['by_module']:
            print("=" * 60)
            print("按六大模組分類")
            print("=" * 60)
            for module in AIVA_SIX_MODULES:
                count = report['by_module'].get(module, 0)
                if count > 0:
                    percentage = count / report['total'] * 100 if report['total'] > 0 else 0
                    print(f"{module:30} {count:5} ({percentage:5.1f}%)")
            
            unclassified = report['by_module'].get('unclassified', 0)
            if unclassified > 0:
                percentage = unclassified / report['total'] * 100 if report['total'] > 0 else 0
                print(f"{'unclassified':30} {unclassified:5} ({percentage:5.1f}%)")
            print()
        
        # 按入口點分類
        if report['by_entry_point']:
            print("=" * 60)
            print("按入口點分類 (Top 10)")
            print("=" * 60)
            sorted_entries = sorted(report['by_entry_point'].items(), key=lambda x: x[1], reverse=True)
            for entry, count in sorted_entries[:10]:
                percentage = count / report['total'] * 100 if report['total'] > 0 else 0
                print(f"{entry:30} {count:5} ({percentage:5.1f}%)")
            print()
    
    def save_classification_report(self, report: Dict[str, Any], output_path: Path):
        """保存分類報告到 JSON 檔案
        
        Args:
            report: 分類報告
            output_path: 輸出檔案路徑
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Classification report saved to: {output_path}")
            
            if RICH_AVAILABLE:
                console.print(f"[green]✓[/green] 報告已保存到: {output_path}")
            else:
                print(f"✓ 報告已保存到: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            if RICH_AVAILABLE:
                console.print(f"[red]✗[/red] 保存失敗: {e}")
            else:
                print(f"✗ 保存失敗: {e}")


async def quick_query(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """快速查詢函數 (便捷接口)
    
    Args:
        question: 自然語言問題
        top_k: 返回結果數量
        
    Returns:
        能力列表
        
    示例:
        >>> from services.core.aiva_core.cognitive_core.ai_capability_query import quick_query
        >>> results = await quick_query("攻擊工具")
        >>> print([r["metadata"]["capability_name"] for r in results])
    """
    query_system = AICapabilityQuery()
    return await query_system.query(question, top_k)


async def quick_stats():
    """快速顯示統計 (便捷接口)"""
    query_system = AICapabilityQuery()
    return await query_system.show_statistics()


# ============================================================================
# CLI 入口（v2.0 六大模組支持）
# ============================================================================

if __name__ == "__main__":
    
    async def _handle_cli_args(args: argparse.Namespace) -> None:
        """處理 CLI 參數（v2.0）"""
        query_system = AICapabilityQuery()
        
        # 統計模式
        if args.stats:
            await query_system.show_statistics()
            return
        
        # 分類報告模式
        if args.classify:
            report = await query_system.get_classification_report()
            query_system.display_classification_report(report)
            
            # 如果指定了輸出路徑，保存到檔案
            if args.output:
                output_path = Path(args.output)
                query_system.save_classification_report(report, output_path)
            else:
                # 預設保存到 reports 目錄
                default_output = Path("reports") / f"capability_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                query_system.save_classification_report(report, default_output)
            return
        
        # 列出模組
        if args.list_modules:
            print("\nAIVA 六大模組:")
            for i, module in enumerate(AIVA_SIX_MODULES, 1):
                print(f"  {i}. {module}")
            print("\n入口點:")
            for i, entry in enumerate(AIVA_ENTRY_POINTS, 1):
                print(f"  {i}. {entry}")
            return
        
        # 按模組查詢
        if args.module:
            if args.module not in AIVA_SIX_MODULES:
                print(f"錯誤: 無效的模組名稱 '{args.module}'")
                print(f"可用模組: {', '.join(AIVA_SIX_MODULES)}")
                return
            
            results = await query_system.query_with_filters(
                question=args.query or f"{args.module} capabilities",
                aiva_module=args.module,
                entry_point=args.entry_point,
                top_k=args.top_k
            )
            query_system.display_results(results, title=f"模組 [{args.module}] 的能力")
            return
        
        # 按入口點查詢
        if args.entry_point:
            results = await query_system.query_with_filters(
                question=args.query or f"{args.entry_point} capabilities",
                entry_point=args.entry_point,
                top_k=args.top_k
            )
            query_system.display_results(results, title=f"入口點 [{args.entry_point}] 的能力")
            return
        
        # 一般查詢
        if args.query:
            results = await query_system.query(args.query, top_k=args.top_k)
            query_system.display_results(results)
            return
        
        # 交互式模式（無參數時）
        await _handle_interactive_mode(query_system)

    async def _handle_interactive_mode(query_system: "AICapabilityQuery") -> None:
        """處理交互式模式"""
        print("\nAIVA AI Capability Query System (v2.0 六大模組支持)")
        print("=" * 60)
        print("指令:")
        print("  - 輸入問題以搜尋能力")
        print("  - 'stats' 顯示統計")
        print("  - 'classify' 顯示分類報告")
        print("  - 'modules' 列出六大模組")
        print("  - 'quit' 退出")
        print("=" * 60)
        
        while True:
            try:
                question = (await asyncio.to_thread(input, "\n[Query] > ")).strip()
                
                if question.lower() in ["quit", "exit", "q"]:
                    break
                elif question.lower() in ["stats", "statistics"]:
                    await query_system.show_statistics()
                elif question.lower() in ["classify", "classification"]:
                    report = await query_system.get_classification_report()
                    query_system.display_classification_report(report)
                elif question.lower() in ["modules", "list"]:
                    print("\nAIVA 六大模組:")
                    for i, module in enumerate(AIVA_SIX_MODULES, 1):
                        print(f"  {i}. {module}")
                elif question:
                    results = await query_system.query(question, top_k=5)
                    query_system.display_results(results)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Interactive mode error: {e}", exc_info=True)

    def main():
        """主程序入口（v2.0 六大模組支持）"""
        parser = argparse.ArgumentParser(
            description="AIVA AI 能力查詢系統 (v2.0 六大模組支持)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""範例:
  # 查詢能力
  python -m services.core.aiva_core.cognitive_core.ai_capability_query "XSS檢測工具"
  
  # 顯示統計
  python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats
  
  # 按模組過濾
  python -m services.core.aiva_core.cognitive_core.ai_capability_query --module cognitive_core
  
  # 生成分類報告
  python -m services.core.aiva_core.cognitive_core.ai_capability_query --classify
  
  # 生成分類報告並保存
  python -m services.core.aiva_core.cognitive_core.ai_capability_query --classify --output report.json
  
  # 按入口點查詢
  python -m services.core.aiva_core.cognitive_core.ai_capability_query --entry-point AICommander
  
  # 組合過濾
  python -m services.core.aiva_core.cognitive_core.ai_capability_query --module cognitive_core --entry-point CapabilityOrchestrator "決策能力"
            """
        )
        
        parser.add_argument(
            "query",
            nargs="*",
            help="查詢問題（自然語言）"
        )
        
        parser.add_argument(
            "--stats", "-s",
            action="store_true",
            help="顯示能力統計"
        )
        
        parser.add_argument(
            "--classify", "-c",
            action="store_true",
            help="生成並顯示能力分類報告（按六大模組）"
        )
        
        parser.add_argument(
            "--module", "-m",
            choices=AIVA_SIX_MODULES,
            help="按 AIVA 六大模組過濾"
        )
        
        parser.add_argument(
            "--entry-point", "-e",
            help="按入口點過濾 (例如: AICommander, app.py, CapabilityOrchestrator)"
        )
        
        parser.add_argument(
            "--list-modules", "-l",
            action="store_true",
            help="列出所有可用的模組和入口點"
        )
        
        parser.add_argument(
            "--top-k", "-k",
            type=int,
            default=10,
            help="返回結果數量 (預設: 10)"
        )
        
        parser.add_argument(
            "--output", "-o",
            help="分類報告輸出路徑 (配合 --classify 使用)"
        )
        
        args = parser.parse_args()
        
        # 處理 query 參數（可能是列表）
        if args.query:
            args.query = " ".join(args.query)
        else:
            args.query = None
        
        # 執行
        asyncio.run(_handle_cli_args(args))
    
    main()
