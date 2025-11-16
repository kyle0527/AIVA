"""Internal Loop Connector - 內部閉環連接器

將 internal_exploration 的能力分析結果注入到 cognitive_core RAG，實現 AI 自我認知

數據流：
internal_exploration (能力分析) → InternalLoopConnector → RAG Knowledge Base

遵循 aiva_common 修復規範:
- 使用統一的日誌記錄
- 使用統一的錯誤處理
- 使用統一的數據格式
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class InternalLoopConnector:
    """內部閉環連接器
    
    職責：
    1. 從 internal_exploration 獲取能力分析結果
    2. 轉換為 RAG 知識庫可接受的格式
    3. 注入到 cognitive_core/rag 知識庫
    4. 建立 AI 自我認知能力
    
    這是 AI 自我優化雙重閉環中「對內探索閉環」的關鍵組件
    """
    
    def __init__(self, rag_knowledge_base=None):
        """初始化內部閉環連接器
        
        Args:
            rag_knowledge_base: RAG 知識庫實例，如果為 None 則延遲初始化
        """
        self.rag_kb = rag_knowledge_base
        self._module_explorer = None
        self._capability_analyzer = None
        
        logger.info("InternalLoopConnector initialized")
    
    @property
    def module_explorer(self):
        """延遲加載 ModuleExplorer"""
        if self._module_explorer is None:
            from ..internal_exploration.module_explorer import ModuleExplorer
            self._module_explorer = ModuleExplorer()
        return self._module_explorer
    
    @property
    def capability_analyzer(self):
        """延遲加載 CapabilityAnalyzer"""
        if self._capability_analyzer is None:
            from ..internal_exploration.capability_analyzer import CapabilityAnalyzer
            self._capability_analyzer = CapabilityAnalyzer()
        return self._capability_analyzer
    
    async def sync_capabilities_to_rag(self, force_refresh: bool = False) -> dict[str, Any]:
        """同步能力到 RAG 知識庫
        
        這是內部閉環的核心方法，將系統能力注入 AI 的認知體系
        
        Args:
            force_refresh: 是否強制刷新（清空舊數據）
            
        Returns:
            同步統計: {
                "modules_scanned": int,
                "capabilities_found": int,
                "documents_added": int,
                "timestamp": str,
                "success": bool
            }
        """
        logger.info("🔄 Starting internal loop synchronization...")
        
        try:
            # 步驟 1: 掃描模組
            logger.info("  Step 1: Scanning modules...")
            modules = await self.module_explorer.explore_all_modules()
            
            # 步驟 2: 分析能力
            logger.info("  Step 2: Analyzing capabilities...")
            capabilities = await self.capability_analyzer.analyze_capabilities(modules)
            
            # 步驟 3: 轉換為文檔
            logger.info("  Step 3: Converting to documents...")
            documents = self._convert_to_documents(capabilities)
            
            # 步驟 4: 注入 RAG
            logger.info("  Step 4: Injecting to RAG...")
            documents_added = await self._inject_to_rag(documents, force_refresh)
            
            result = {
                "modules_scanned": len(modules),
                "capabilities_found": len(capabilities),
                "documents_added": documents_added,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
            
            logger.info(f"✅ Internal loop sync completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Internal loop sync failed: {e}", exc_info=True)
            return {
                "modules_scanned": 0,
                "capabilities_found": 0,
                "documents_added": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "error": str(e)
            }
    
    def _convert_to_documents(self, capabilities: list[dict]) -> list[dict]:
        """將能力轉換為 RAG 文檔格式
        
        Args:
            capabilities: 能力列表
            
        Returns:
            RAG 文檔列表
        """
        documents = []
        
        for cap in capabilities:
            # 構建可讀的能力描述
            params_str = ", ".join(
                f"{p['name']}: {p.get('annotation', 'Any')}" 
                for p in cap["parameters"]
            )
            
            content_parts = [
                f"# Capability: {cap['name']}",
                f"\nModule: {cap['module']}",
                f"Function: {cap['name']}({params_str})",
            ]
            
            if cap.get("return_type"):
                content_parts.append(f"Returns: {cap['return_type']}")
            
            if cap.get("description"):
                content_parts.append(f"\nDescription: {cap['description']}")
            
            if cap.get("docstring"):
                content_parts.append(f"\nDocumentation:\n{cap['docstring']}")
            
            content = "\n".join(content_parts)
            
            doc = {
                "content": content,
                "metadata": {
                    "type": "capability",
                    "capability_name": cap["name"],
                    "module": cap["module"],
                    "file_path": cap["file_path"],
                    "is_async": cap.get("is_async", False),
                    "parameters_count": len(cap["parameters"]),
                    "source": "internal_exploration",
                    "sync_timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            documents.append(doc)
        
        return documents
    
    async def _inject_to_rag(self, documents: list[dict], force_refresh: bool) -> int:
        """注入文檔到 RAG 知識庫
        
        Args:
            documents: 文檔列表
            force_refresh: 是否清空舊數據
            
        Returns:
            成功添加的文檔數量
        """
        if self.rag_kb is None:
            logger.warning("RAG Knowledge Base not initialized, skipping injection")
            return 0
        
        added_count = 0
        
        # TODO: 如果 force_refresh，清空舊的自我認知數據
        # if force_refresh:
        #     await self.rag_kb.clear_namespace("self_awareness")
        
        for i, doc in enumerate(documents):
            try:
                # 確保 metadata 是字典類型
                metadata_dict = {}
                for key, value in doc["metadata"].items():
                    # 確保所有值都是可序列化的基本類型
                    if isinstance(value, (str, int, float, bool)):
                        metadata_dict[key] = value
                    elif value is None:
                        metadata_dict[key] = None
                    else:
                        # 複雜類型轉為字串
                        metadata_dict[key] = str(value)
                
                # 添加命名空間
                metadata_dict["namespace"] = "self_awareness"
                
                # 使用 RAG 知識庫的 add_knowledge 方法
                success = self.rag_kb.add_knowledge(
                    content=doc["content"],
                    metadata=metadata_dict
                )
                
                if success:
                    added_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to add document {i}: {e}")
                logger.debug(f"Document: {doc}")  # 調試用
        
        logger.info(f"  Injected {added_count}/{len(documents)} documents to RAG")
        return added_count
    
    async def query_self_awareness(self, query: str, top_k: int = 5) -> list[dict]:
        """查詢自我認知知識
        
        測試方法：驗證 AI 能否回答「我有什麼能力」
        
        Args:
            query: 查詢字串（如 "我有哪些攻擊能力"）
            top_k: 返回結果數量
            
        Returns:
            相關能力列表
        """
        if self.rag_kb is None:
            logger.warning("RAG Knowledge Base not initialized")
            return []
        
        try:
            results = self.rag_kb.search(query, top_k=top_k)
            
            # 過濾自我認知數據
            self_awareness_results = [
                r for r in results 
                if r.get("metadata", {}).get("namespace") == "self_awareness"
            ]
            
            return self_awareness_results
            
        except Exception as e:
            logger.error(f"Self-awareness query failed: {e}")
            return []
    
    def get_sync_status(self) -> dict[str, Any]:
        """獲取同步狀態
        
        Returns:
            狀態資訊
        """
        return {
            "connector": "InternalLoopConnector",
            "status": "active" if self.rag_kb else "inactive",
            "rag_initialized": self.rag_kb is not None,
            "last_sync": None  # TODO: 實現最後同步時間追蹤
        }
