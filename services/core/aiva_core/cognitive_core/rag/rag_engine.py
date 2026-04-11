"""RAG Engine - 檢索增強生成引擎

將向量檢索與 AI 決策結合，增強攻擊計畫生成

優化更新 (2026-01-11):
- 添加查詢緩存機制 (TTL 5 分鐘)
- 支援批量並行查詢
- 去語意化環境特徵匹配
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, TYPE_CHECKING, Optional
from enum import Enum

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

from aiva_common.schemas import (
    AttackPlan,
    AttackStep,
    AttackTarget,
    ExperienceSample,
)

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """知識類型"""
    VULNERABILITY = "vulnerability"
    ATTACK_TECHNIQUE = "attack_technique"
    BEST_PRACTICE = "best_practice"
    EXPERIENCE = "experience"
    MITIGATION = "mitigation"


class QueryCache:
    """查詢緩存
    
    實現 TTL 緩存以減少重複 RAG 查詢
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """初始化緩存
        
        Args:
            max_size: 最大緩存條目數
            default_ttl: 預設 TTL（秒）
        """
        self._cache: dict[str, tuple[float, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, query: str, query_type: str, top_k: int) -> str:
        """生成緩存鍵"""
        content = f"{query_type}:{query}:{top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, query_type: str, top_k: int) -> Optional[Any]:
        """獲取緩存"""
        key = self._make_key(query, query_type, top_k)
        
        if key in self._cache:
            timestamp, result = self._cache[key]
            if time.time() - timestamp < self._default_ttl:
                self._hits += 1
                return result
            else:
                # 過期，刪除
                del self._cache[key]
        
        self._misses += 1
        return None
    
    def set(self, query: str, query_type: str, top_k: int, result: Any) -> None:
        """設置緩存"""
        # 清理過期條目
        self._cleanup()
        
        # 檢查容量
        if len(self._cache) >= self._max_size:
            # 刪除最舊的條目
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        
        key = self._make_key(query, query_type, top_k)
        self._cache[key] = (time.time(), result)
    
    def _cleanup(self) -> None:
        """清理過期條目"""
        current = time.time()
        expired = [
            k for k, (ts, _) in self._cache.items()
            if current - ts > self._default_ttl
        ]
        for k in expired:
            del self._cache[k]
    
    def clear(self) -> None:
        """清空緩存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> dict[str, Any]:
        """獲取緩存統計"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0
        }


class RAGEngine:
    """RAG 引擎

    結合向量檢索和 AI 生成，提供上下文增強的決策
    
    優化特性:
    - 查詢緩存 (TTL 5 分鐘)
    - 批量並行查詢
    - 去語意化環境特徵匹配
    """

    def __init__(
        self, 
        knowledge_base: "KnowledgeBase",
        cache_size: int = 1000,
        cache_ttl: float = 300.0
    ) -> None:
        """初始化 RAG 引擎

        Args:
            knowledge_base: 知識庫實例
            cache_size: 緩存大小
            cache_ttl: 緩存 TTL（秒）
        """
        self.knowledge_base = knowledge_base
        self._cache = QueryCache(max_size=cache_size, default_ttl=cache_ttl)
        logger.info(f"RAG Engine initialized with cache (size={cache_size}, ttl={cache_ttl}s)")

    async def _search_with_cache(
        self,
        query: str,
        query_type: str,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """帶緩存的搜索
        
        Args:
            query: 查詢字符串
            query_type: 查詢類型
            top_k: 返回數量
            
        Returns:
            搜索結果列表
        """
        # 檢查緩存
        cached = self._cache.get(query, query_type, top_k)
        if cached is not None:
            logger.debug(f"Cache hit for {query_type}: {query[:50]}...")
            return cached
        
        # 執行實際搜索
        results = await self.knowledge_base.search(
            query=f"{query_type} {query}",
            top_k=top_k
        )
        
        # 更新緩存
        self._cache.set(query, query_type, top_k, results)
        
        return results

    async def enhance_attack_plan(
        self,
        target: AttackTarget,
        objective: str,
    ) -> dict[str, Any]:
        """增強攻擊計畫

        使用 RAG 檢索相關經驗和技術，增強計畫生成

        Args:
            target: 攻擊目標
            objective: 攻擊目標描述
            base_plan: 基礎計畫（可選）

        Returns:
            增強上下文字典，包含相關經驗和技術
        """
        # 構建查詢
        query = f"{objective} {target.target_type} {target.target_url}"

        # 優化：批量並行查詢（使用緩存）
        techniques_task = self._search_with_cache(query, "attack_technique", 5)
        experiences_task = self._search_with_cache(query, "experience success", 5)
        practices_task = self._search_with_cache(query, "best_practice", 3)
        
        techniques, successful_experiences, best_practices = await asyncio.gather(
            techniques_task, experiences_task, practices_task
        )

        # 構建增強上下文
        context = {
            "target": {
                "url": target.target_url,
                "type": target.target_type,
                "description": target.description,
            },
            "objective": objective,
            "similar_techniques": [
                {
                    "content": entry.get("content", ""),
                    "relevance_score": entry.get("relevance_score", 0.0),
                    "metadata": entry.get("metadata", {}),
                }
                for entry in techniques
            ],
            "successful_experiences": [
                {
                    "content": entry.get("content", ""),
                    "relevance_score": entry.get("relevance_score", 0.0),
                    "metadata": entry.get("metadata", {}),
                }
                for entry in successful_experiences
            ],
            "best_practices": [
                {
                    "content": entry.get("content", ""),
                    "relevance_score": entry.get("relevance_score", 0.0),
                }
                for entry in best_practices
            ],
        }

        logger.info(
            f"Enhanced attack plan with {len(techniques)} techniques, "
            f"{len(successful_experiences)} experiences, "
            f"{len(best_practices)} best practices"
        )

        return context

    async def suggest_next_step(
        self,
        current_state: dict[str, Any],
        previous_steps: list[AttackStep],
    ) -> dict[str, Any]:
        """建議下一步驟

        基於當前狀態和歷史步驟，使用 RAG 建議最佳下一步

        Args:
            current_state: 當前狀態
            previous_steps: 之前的步驟列表

        Returns:
            建議上下文字典
        """
        # 構建查詢
        steps_summary = " -> ".join([step.tool_type for step in previous_steps])
        query = f"{current_state.get('vulnerability_type', 'unknown')} {steps_summary}"

        # 檢索類似執行序列
        similar_experiences = await self.knowledge_base.search(
            query=f"experience success {query}",
            top_k=5,
        )

        # 分析成功模式
        tool_suggestions: dict[str, int] = {}
        for entry in similar_experiences:
            metadata = entry.get("metadata", {})
            # 假設元數據中有 next_tool 信息
            next_tool = metadata.get("next_tool")
            if next_tool:
                tool_suggestions[next_tool] = tool_suggestions.get(next_tool, 0) + 1

        # 排序建議
        sorted_suggestions = sorted(
            tool_suggestions.items(), key=lambda x: x[1], reverse=True
        )

        context = {
            "current_state": current_state,
            "steps_count": len(previous_steps),
            "similar_cases": len(similar_experiences),
            "suggested_tools": [
                {"tool": tool, "frequency": count}
                for tool, count in sorted_suggestions[:3]
            ],
            "reference_experiences": [
                {
                    "content": entry.get("content", ""),
                    "metadata": entry.get("metadata", {}),
                    "relevance_score": entry.get("relevance_score", 0.0),
                }
                for entry in similar_experiences[:3]
            ],
        }

        logger.info(
            f"Generated step suggestions based on {len(similar_experiences)} cases"
        )

        return context

    async def analyze_failure(
        self,
        failed_step: AttackStep,
        error_message: str,
    ) -> dict[str, Any]:
        """分析失敗原因並建議修正

        Args:
            failed_step: 失敗的步驟
            error_message: 錯誤信息

        Returns:
            分析結果和建議
        """
        # 構建查詢
        query = f"{failed_step.tool_type} {failed_step.action} {error_message}"

        # 檢索類似的失敗案例
        similar_failures = await self.knowledge_base.search(
            query=f"experience failed {query}",
            top_k=3,
        )

        # 檢索緩解措施
        mitigations = await self.knowledge_base.search(
            query=f"mitigation {query}",
            top_k=3,
        )

        context = {
            "failed_step": {
                "tool": failed_step.tool_type,
                "description": failed_step.action,
                "parameters": failed_step.parameters,
            },
            "error_message": error_message,
            "similar_failures": [
                {
                    "content": entry.get("content", ""),
                    "metadata": entry.get("metadata", {}),
                }
                for entry in similar_failures
            ],
            "suggested_mitigations": [
                {"content": entry.get("content", "")}
                for entry in mitigations
            ],
        }

        logger.info(
            f"Analyzed failure for {failed_step.tool_type}, "
            f"found {len(similar_failures)} similar cases"
        )

        return context

    # 外部武器庫 payload collection 對應表
    _ARSENAL_COLLECTION_MAP: dict[str, str] = {
        "xss": "knowledge_xss_payloads",
        "xss_reflected": "knowledge_xss_payloads",
        "xss_stored": "knowledge_xss_payloads",
        "xss_dom": "knowledge_xss_payloads",
        "sqli": "knowledge_sqli_payloads",
        "sql_injection": "knowledge_sqli_payloads",
        "sql_injection_blind": "knowledge_sqli_payloads",
        "sql_injection_union": "knowledge_sqli_payloads",
        "sql_injection_time_based": "knowledge_sqli_payloads",
    }

    async def get_relevant_payloads(
        self,
        vulnerability_type: str,
        target_info: dict[str, Any],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """獲取相關有效載荷

        優先從外部武器庫的專屬 ChromaDB collection 語義檢索；
        若找不到對應 collection，則降級到既有知識庫搜尋。

        Args:
            vulnerability_type: 漏洞類型 (如 "xss", "sql_injection")
            target_info: 目標信息 (technology, framework, version 等)
            top_k: 返回數量

        Returns:
            有效載荷列表，每項包含 payload / relevance_score / metadata / source
        """
        target_desc = " ".join(
            filter(None, [
                target_info.get("technology", ""),
                target_info.get("framework", ""),
                target_info.get("version", ""),
            ])
        )
        query = f"{vulnerability_type} {target_desc}".strip()

        # 查找是否有對應的外部武器庫 collection
        vuln_key = vulnerability_type.lower().replace("-", "_").replace(" ", "_")
        arsenal_collection = self._ARSENAL_COLLECTION_MAP.get(vuln_key)

        arsenal_results: list[dict[str, Any]] = []
        if arsenal_collection:
            try:
                import chromadb
                from pathlib import Path

                # 外部武器庫 ChromaDB 路徑（由 _dev_tools/ingest_payloads_to_rag.py 寫入）
                db_path = Path(__file__).resolve().parents[7] / "data" / "chroma"
                if db_path.exists():
                    client = chromadb.PersistentClient(path=str(db_path))
                    existing = {c.name for c in client.list_collections()}
                    if arsenal_collection in existing:
                        collection = client.get_collection(name=arsenal_collection)
                        raw = collection.query(
                            query_texts=[query],
                            n_results=min(top_k, collection.count()),
                        )
                        ids = (raw.get("ids") or [[]])[0]
                        docs = (raw.get("documents") or [[]])[0]
                        dists = (raw.get("distances") or [[]])[0]
                        metas = (raw.get("metadatas") or [[]])[0]

                        for i, doc_id in enumerate(ids):
                            dist = dists[i] if i < len(dists) else 1.0
                            score = round(1.0 / (1.0 + dist), 4)
                            arsenal_results.append({
                                "payload": docs[i] if i < len(docs) else "",
                                "relevance_score": score,
                                "metadata": {
                                    **(metas[i] if i < len(metas) else {}),
                                    "doc_id": doc_id,
                                },
                                "source": "external_arsenal",
                            })
                        logger.info(
                            f"Arsenal RAG: retrieved {len(arsenal_results)} payloads "
                            f"from '{arsenal_collection}' for '{vulnerability_type}'"
                        )
            except Exception as e:
                logger.warning(f"Arsenal payload retrieval failed, falling back: {e}")

        # 降級：從既有知識庫搜索
        kb_results: list[dict[str, Any]] = []
        if not arsenal_results:
            payloads = await self.knowledge_base.search(
                query=f"payload {query}",
                top_k=top_k,
            )
            kb_results = [
                {
                    "payload": entry.get("content", ""),
                    "relevance_score": entry.get("relevance_score", 0.0),
                    "metadata": entry.get("metadata", {}),
                    "source": "knowledge_base",
                }
                for entry in payloads
            ]

        results = arsenal_results or kb_results
        logger.info(
            f"get_relevant_payloads: {len(results)} payloads for '{vulnerability_type}' "
            f"(source: {'arsenal' if arsenal_results else 'knowledge_base'})"
        )
        return results

    def learn_from_experience(self, sample: ExperienceSample) -> None:
        """從經驗樣本學習

        將成功或失敗的經驗添加到知識庫

        Args:
            sample: 經驗樣本
        """
        # 將經驗樣本添加到知識庫
        import asyncio
        task = asyncio.create_task(self.knowledge_base.add_knowledge(
            content=f"Experience: {sample.action_taken.get('type', 'unknown')} - {sample.action_taken.get('action', '')}",
            metadata={
                "sample_id": sample.sample_id,
                "session_id": sample.session_id,
                "plan_id": sample.plan_id,
                "reward": sample.reward,
                "is_positive": sample.is_positive,
                "quality_score": sample.quality_score or 0.0,
                "type": "experience"
            }
        ))
        # 不等待以避免阻塞,但保持引用以防止過早回收
        self._pending_tasks = getattr(self, '_pending_tasks', [])
        self._pending_tasks.append(task)

        # 如果有特定的有效載荷或模式，提取並存儲
        if sample.is_positive and sample.action_taken:
            self._extract_successful_pattern(sample)

        logger.info(
            f"Learned from experience: session={sample.session_id}, "
            f"is_positive={sample.is_positive}, quality={sample.quality_score or 0.0:.2f}"
        )

    def _extract_successful_pattern(self, sample: ExperienceSample) -> None:
        """提取成功模式

        Args:
            sample: 經驗樣本
        """
        # 提取有效載荷
        payload = sample.action_taken.get("payload")
        if payload:
            vuln_type = sample.state_before.get("vulnerability_type", "unknown")
            tool_type = sample.action_taken.get("tool_type", "unknown")
            entry_id = f"payload_{vuln_type}_{hash(str(payload)) % 10000}"

            # 使用 add_knowledge 而不是 add_entry
            import asyncio
            task = asyncio.create_task(self.knowledge_base.add_knowledge(
                content=payload,
                metadata={
                    "entry_id": entry_id,
                    "vulnerability_type": vuln_type,
                    "tool_type": tool_type,
                    "source_session": sample.session_id,
                    "reward": sample.reward,
                    "target_url": sample.target_info.get("target_url", "unknown"),
                    "type": "payload",
                    "verified": True
                }
            ))
            # 不等待以避免阻塞,但保持引用以防止過早回收
            self._pending_tasks = getattr(self, '_pending_tasks', [])
            self._pending_tasks.append(task)
            logger.debug(f"Extracted payload pattern: {entry_id}")

    async def retrieve_similar_cases(
        self,
        objective: str,
        target_info: dict[str, Any],
        top_k: int = 5
    ) -> dict[str, Any]:
        """檢索相似攻擊案例（UnifiedAttackExecutor 需要）
        
        Args:
            objective: 攻擊目標描述（如 "檢查 XSS 漏洞"）
            target_info: 目標信息字典
            top_k: 返回最相似的 k 個案例
        
        Returns:
            {
                "similar_cases": [...],
                "best_practices": [...],
                "recommended_tools": [...]
            }
        """
        # 構建查詢
        target_url = target_info.get("target_url", target_info.get("url", ""))
        vulnerability_type = objective.split("檢查")[-1].strip() if "檢查" in objective else objective
        
        query = f"{objective} {target_url} {vulnerability_type}"
        
        # 檢索成功案例
        successful_cases = await self.knowledge_base.search(
            query=f"experience success {query}",
            top_k=top_k
        )
        
        # 檢索最佳實踐
        best_practices = await self.knowledge_base.search(
            query=f"best_practice {vulnerability_type}",
            top_k=3
        )
        
        # 分析推薦工具（基於成功案例）
        tool_counts: dict[str, int] = {}
        for case in successful_cases:
            metadata = case.get("metadata", {})
            tool = metadata.get("tool_type") or metadata.get("selected_tool")
            if tool:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        recommended_tools = sorted(
            [{"tool": tool, "frequency": count} for tool, count in tool_counts.items()],
            key=lambda x: x["frequency"],
            reverse=True
        )[:3]
        
        return {
            "similar_cases": [
                {
                    "content": case.get("content", ""),
                    "metadata": case.get("metadata", {}),
                    "relevance_score": case.get("relevance_score", 0.0)
                }
                for case in successful_cases
            ],
            "best_practices": [
                {
                    "content": bp.get("content", ""),
                    "relevance_score": bp.get("relevance_score", 0.0)
                }
                for bp in best_practices
            ],
            "recommended_tools": recommended_tools
        }
    
    async def search_capabilities_by_environment(
        self,
        environment_features: dict[str, float | int],
        top_k: int = 5,
        filter_type: str | None = None
    ) -> list[dict[str, Any]]:
        """根據環境特徵搜索最匹配的能力（去語意化檢索）
        
        v2.1 (2026-01-08): 新增方法 - HackOne 戰略規劃
        
        使用去語意化反射引擎：
        - 不依賴語義理解，純向量相似度
        - 毫秒級檢索（< 5ms）
        - 確定性結果
        
        使用場景：
        1. 偵察階段發現環境特徵（http_403, waf_cloudflare）
        2. 快速匹配適用的攻擊能力
        3. 返回預排序的能力列表給 EnhancedDecisionAgent
        
        Args:
            environment_features: 環境特徵字典
                範例: {
                    'http_403': 3,          # 403 狀態出現 3 次
                    'waf_cloudflare': 1,    # 檢測到 Cloudflare WAF
                    'db_error_mysql': 2,    # MySQL 錯誤出現 2 次
                }
            top_k: 返回前 k 個最匹配的能力
            filter_type: 能力類型過濾器（如 'scanner', 'analyzer'）
        
        Returns:
            匹配能力列表，格式:
            [
                {
                    'capability_id': 'security.sqli.time_based',
                    'match_score': 0.87,  # 相似度分數（0-1）
                    'metadata': {...},
                    'document': '時間盲注檢測'
                },
                ...
            ]
        """
        # 將整數權重轉換為浮點數
        normalized_features = {k: float(v) for k, v in environment_features.items()}
        
        # 使用 VectorStore 的去語意化檢索
        results = self.knowledge_base.vector_store.search_by_environment(
            environment_features=normalized_features,
            top_k=top_k,
            filter_type=filter_type
        )
        
        logger.info(
            f"Environment-based capability search: {len(environment_features)} features "
            f"→ {len(results)} matches (top_{top_k})"
        )
        
        return results
    
    async def load_capabilities_from_registry(
        self,
        capability_records: list[dict[str, Any]]
    ) -> int:
        """從 integration/capability 註冊表批量加載能力
        
        v2.1 (2026-01-08): 新增方法
        
        Args:
            capability_records: CapabilityRecord 格式的能力列表
            
        Returns:
            成功載入的數量
        """
        count = 0
        for record in capability_records:
            try:
                self.knowledge_base.vector_store.add_capability_from_registry(
                    capability=record,
                    capability_id=record.get('id')
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load capability {record.get('id')}: {e}")
        
        logger.info(f"Loaded {count} capabilities from registry to RAG Engine")
        return count
    
    async def index_new_experience(
        self,
        experience
    ) -> None:
        """將新經驗索引到向量庫（UnifiedAttackExecutor 需要）
        
        Args:
            experience: ExperienceSample 對象或字典
        """
        try:
            # 提取經驗數據
            if hasattr(experience, "to_dict"):
                exp_dict = experience.to_dict()
            elif isinstance(experience, dict):
                exp_dict = experience
            else:
                exp_dict = {
                    "sample_id": getattr(experience, "sample_id", "unknown"),
                    "state": getattr(experience, "state", {}),
                    "action": getattr(experience, "action", {}),
                    "reward": getattr(experience, "reward", 0.0)
                }
            
            # 構建索引內容
            state = exp_dict.get("state", {})
            action = exp_dict.get("action", {})
            reward = exp_dict.get("reward", 0.0)
            
            # 生成文本描述
            content = f"Experience: {action.get('type', 'attack')} on {state.get('target', 'unknown')}, reward: {reward}"
            
            # 添加到知識庫
            await self.knowledge_base.add_knowledge(
                content=content,
                metadata={
                    "type": "experience",
                    "sample_id": exp_dict.get("sample_id"),
                    "reward": reward,
                    "action_type": action.get("type"),
                    "target": state.get("target"),
                    "success": reward > 0.5,
                    "timestamp": exp_dict.get("timestamp", ""),
                    "knowledge_type": KnowledgeType.EXPERIENCE.value  # 改用 metadata 傳遞類型
                }
            )
            
            logger.debug(f"Indexed experience: {exp_dict.get('sample_id')}")
            
        except Exception as e:
            logger.error(f"Failed to index experience: {e}")
            # 不拋出異常，避免阻塞執行流程

    def save_knowledge(self) -> None:
        """保存知識庫"""
        # KnowledgeBase 使用向量存儲,不需要顯式保存
        logger.info("Knowledge base persisted through vector store")

    def get_statistics(self) -> dict[str, Any]:
        """獲取 RAG 引擎統計信息

        Returns:
            統計信息字典
        """
        return {
            "knowledge_base": "active",
            "rag_engine": "active",
            "vector_store": type(self.knowledge_base.vector_store).__name__
        }

