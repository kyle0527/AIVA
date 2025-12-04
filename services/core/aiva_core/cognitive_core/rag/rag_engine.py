"""RAG Engine - 檢索增強生成引擎

將向量檢索與 AI 決策結合，增強攻擊計畫生成
"""

import logging
from typing import Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

from services.aiva_common.schemas import (
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

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 引擎

    結合向量檢索和 AI 生成，提供上下文增強的決策
    """

    def __init__(self, knowledge_base: "KnowledgeBase") -> None:
        """初始化 RAG 引擎

        Args:
            knowledge_base: 知識庫實例
        """
        self.knowledge_base = knowledge_base
        logger.info("RAG Engine initialized")

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

        # 檢索相關攻擊技術
        techniques = await self.knowledge_base.search(
            query=f"attack_technique {query}",
            top_k=5,
        )

        # 檢索成功經驗
        successful_experiences = await self.knowledge_base.search(
            query=f"experience success {query}",
            top_k=5,
        )

        # 檢索最佳實踐
        best_practices = await self.knowledge_base.search(
            query=f"best_practice {query}",
            top_k=3,
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

    async def get_relevant_payloads(
        self,
        vulnerability_type: str,
        target_info: dict[str, Any],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """獲取相關有效載荷

        Args:
            vulnerability_type: 漏洞類型
            target_info: 目標信息
            top_k: 返回數量

        Returns:
            有效載荷列表
        """
        # 構建查詢
        target_desc = " ".join(
            [
                target_info.get("technology", ""),
                target_info.get("framework", ""),
                target_info.get("version", ""),
            ]
        )
        query = f"{vulnerability_type} {target_desc}"

        # 檢索相關載荷
        payloads = await self.knowledge_base.search(
            query=f"payload {query}",
            top_k=top_k,
        )

        # 按相關性排序（知識庫返回的已經是按相關性排序）
        results = [
            {
                "payload": entry.get("content", ""),
                "relevance_score": entry.get("relevance_score", 0.0),
                "metadata": entry.get("metadata", {}),
            }
            for entry in payloads
        ]

        logger.info(f"Retrieved {len(results)} payloads for {vulnerability_type}")

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
