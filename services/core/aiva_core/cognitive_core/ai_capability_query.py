"""AI 能力查詢系統 - 用戶友好的 AI 分析接口

整合 AI 自我分析能力到 AIVA 核心架構，提供簡化的查詢接口。

使用方式:
    from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery
    
    query_system = AICapabilityQuery()
    results = await query_system.query("如何進行滲透測試", top_k=5)
    query_system.display_results(results)
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import Counter

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


# 命令行入口
if __name__ == "__main__":
    import sys
    
    async def _handle_command_line(query_system: "AICapabilityQuery") -> None:
        """處理命令行參數"""
        question = " ".join(sys.argv[1:])
        
        if question in ["--stats", "-s"]:
            await query_system.show_statistics()
        else:
            results = await query_system.query(question, top_k=5)
            query_system.display_results(results)

    async def _handle_interactive_mode(query_system: "AICapabilityQuery") -> None:
        """處理交互式模式"""
        print("\nAIVA AI Capability Query System")
        print("=" * 60)
        print("Commands:")
        print("  - Type your question to search capabilities")
        print("  - Type 'stats' to show statistics")
        print("  - Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                question = (await asyncio.to_thread(input, "\n[Query] > ")).strip()
                
                if question.lower() in ["quit", "exit", "q"]:
                    break
                elif question.lower() in ["stats", "statistics"]:
                    await query_system.show_statistics()
                elif question:
                    results = await query_system.query(question, top_k=5)
                    query_system.display_results(results)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    async def main():
        """主程序入口 - 降低認知複雜度"""
        query_system = AICapabilityQuery()
        
        if len(sys.argv) > 1:
            # 命令行查詢
            await _handle_command_line(query_system)
        else:
            # 交互式模式
            await _handle_interactive_mode(query_system)
    
    asyncio.run(main())
