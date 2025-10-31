"""RAG Engine - 檢索增強生成引擎

將知識庫檢索與 AI 決策結合，增強攻擊計畫生成
"""

import logging
from typing import Any

from services.aiva_common.schemas import (
    AttackPlan,
    AttackStep,
    AttackTarget,
    ExperienceSample,
)

from .knowledge_base import KnowledgeBase, KnowledgeType

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 引擎

    結合向量檢索和 AI 生成，提供上下文增強的決策
    """

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        """初始化 RAG 引擎

        Args:
            knowledge_base: 知識庫實例
        """
        self.knowledge_base = knowledge_base
        logger.info("RAG Engine initialized")

    def enhance_attack_plan(
        self,
        target: AttackTarget,
        objective: str,
        base_plan: AttackPlan | None = None,
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
        query = f"{objective} {target.type} {target.url}"

        # 檢索相關攻擊技術
        attack_techniques = self.knowledge_base.search(
            query=query,
            entry_type=KnowledgeType.ATTACK_TECHNIQUE,
            top_k=3,
        )

        # 檢索成功經驗
        successful_experiences = self.knowledge_base.search(
            query=query,
            entry_type=KnowledgeType.EXPERIENCE,
            tags=["success"],
            top_k=5,
        )

        # 檢索最佳實踐
        best_practices = self.knowledge_base.search(
            query=query,
            entry_type=KnowledgeType.BEST_PRACTICE,
            top_k=3,
        )

        # 構建增強上下文
        context = {
            "target": {
                "url": target.url,
                "type": target.type,
                "description": target.description,
            },
            "objective": objective,
            "similar_techniques": [
                {
                    "title": entry.title,
                    "content": entry.content,
                    "success_rate": entry.success_rate,
                    "usage_count": entry.usage_count,
                }
                for entry in attack_techniques
            ],
            "successful_experiences": [
                {
                    "title": entry.title,
                    "content": entry.content,
                    "metadata": entry.metadata,
                }
                for entry in successful_experiences
            ],
            "best_practices": [
                {"title": entry.title, "content": entry.content}
                for entry in best_practices
            ],
        }

        logger.info(
            f"Enhanced attack plan with {len(attack_techniques)} techniques, "
            f"{len(successful_experiences)} experiences, "
            f"{len(best_practices)} best practices"
        )

        return context

    def suggest_next_step(
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
        steps_summary = " -> ".join([step.tool for step in previous_steps])
        query = f"{current_state.get('vulnerability_type', 'unknown')} {steps_summary}"

        # 檢索類似執行序列
        similar_experiences = self.knowledge_base.search(
            query=query,
            entry_type=KnowledgeType.EXPERIENCE,
            tags=["success"],
            top_k=5,
        )

        # 分析成功模式
        tool_suggestions: dict[str, int] = {}
        for entry in similar_experiences:
            metadata = entry.metadata
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
                    "title": entry.title,
                    "metadata": entry.metadata,
                    "success_rate": entry.success_rate,
                }
                for entry in similar_experiences[:3]
            ],
        }

        logger.info(
            f"Generated step suggestions based on {len(similar_experiences)} cases"
        )

        return context

    def analyze_failure(
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
        query = f"{failed_step.tool} {failed_step.description} {error_message}"

        # 檢索類似失敗案例
        similar_failures = self.knowledge_base.search(
            query=query,
            entry_type=KnowledgeType.EXPERIENCE,
            tags=["failed"],
            top_k=5,
        )

        # 檢索緩解措施
        mitigations = self.knowledge_base.search(
            query=query,
            entry_type=KnowledgeType.MITIGATION,
            top_k=3,
        )

        context = {
            "failed_step": {
                "tool": failed_step.tool,
                "description": failed_step.description,
                "parameters": failed_step.parameters,
            },
            "error_message": error_message,
            "similar_failures": [
                {
                    "title": entry.title,
                    "content": entry.content,
                    "metadata": entry.metadata,
                }
                for entry in similar_failures
            ],
            "suggested_mitigations": [
                {"title": entry.title, "content": entry.content}
                for entry in mitigations
            ],
        }

        logger.info(
            f"Analyzed failure for {failed_step.tool}, "
            f"found {len(similar_failures)} similar cases"
        )

        return context

    def get_relevant_payloads(
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
        payloads = self.knowledge_base.search(
            query=query,
            entry_type=KnowledgeType.PAYLOAD,
            top_k=top_k,
        )

        # 按成功率排序
        sorted_payloads = sorted(payloads, key=lambda e: e.success_rate, reverse=True)

        results = [
            {
                "payload": entry.content,
                "title": entry.title,
                "success_rate": entry.success_rate,
                "usage_count": entry.usage_count,
                "metadata": entry.metadata,
            }
            for entry in sorted_payloads
        ]

        logger.info(f"Retrieved {len(results)} payloads for {vulnerability_type}")

        return results

    def learn_from_experience(self, sample: ExperienceSample) -> None:
        """從經驗樣本學習

        將成功或失敗的經驗添加到知識庫

        Args:
            sample: 經驗樣本
        """
        # 添加到知識庫
        self.knowledge_base.add_experience_sample(sample)

        # 如果有特定的有效載荷或模式，提取並存儲
        if sample.reward.success and sample.action.parameters:
            self._extract_successful_pattern(sample)

        logger.info(
            f"Learned from experience: session={sample.session_id}, "
            f"success={sample.reward.success}, quality={sample.quality_score:.2f}"
        )

    def _extract_successful_pattern(self, sample: ExperienceSample) -> None:
        """提取成功模式

        Args:
            sample: 經驗樣本
        """
        # 提取有效載荷
        payload = sample.action.parameters.get("payload")
        if payload:
            entry_id = (
                f"payload_{sample.state.vulnerability_type.value}_"
                f"{hash(payload) % 10000}"
            )

            self.knowledge_base.add_entry(
                entry_id=entry_id,
                entry_type=KnowledgeType.PAYLOAD,
                title=f"{sample.state.vulnerability_type.value} Payload",
                content=payload,
                tags=[
                    sample.state.vulnerability_type.value,
                    sample.action.tool,
                    "verified",
                ],
                metadata={
                    "source_session": sample.session_id,
                    "reward": sample.reward.total_score,
                    "target_type": sample.state.target_url,
                },
            )

    def save_knowledge(self) -> None:
        """保存知識庫"""
        self.knowledge_base.save_knowledge_base()
        logger.info("Knowledge base saved")

    def get_statistics(self) -> dict[str, Any]:
        """獲取 RAG 引擎統計信息

        Returns:
            統計信息字典
        """
        return {
            "knowledge_base": self.knowledge_base.get_statistics(),
        }
