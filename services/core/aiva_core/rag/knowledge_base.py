"""
Knowledge Base - 知識庫管理

管理漏洞知識、攻擊技術、最佳實踐等知識
"""



from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

from services.aiva_common.schemas import (
    AttackPlan,
    ExperienceSample,
)

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """知識類型"""

    VULNERABILITY = "vulnerability"  # 漏洞知識
    ATTACK_TECHNIQUE = "attack_technique"  # 攻擊技術
    BEST_PRACTICE = "best_practice"  # 最佳實踐
    EXPERIENCE = "experience"  # 經驗樣本
    MITIGATION = "mitigation"  # 緩解措施
    PAYLOAD = "payload"  # 有效載荷
    EXPLOIT_PATTERN = "exploit_pattern"  # 利用模式


@dataclass
class KnowledgeEntry:
    """知識條目"""

    id: str
    type: KnowledgeType
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEntry:
        """從字典創建"""
        return cls(
            id=data["id"],
            type=KnowledgeType(data["type"]),
            title=data["title"],
            content=data["content"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.0),
        )


class KnowledgeBase:
    """知識庫管理器

    管理和檢索各類安全知識
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        data_directory: Path | None = None,
    ) -> None:
        """初始化知識庫

        Args:
            vector_store: 向量存儲實例
            data_directory: 數據目錄
        """
        self.vector_store = vector_store or VectorStore()
        self.data_directory = data_directory or Path("./data/knowledge")
        self.data_directory.mkdir(parents=True, exist_ok=True)

        self.entries: dict[str, KnowledgeEntry] = {}

        self._load_knowledge_base()

        logger.info(f"KnowledgeBase initialized with {len(self.entries)} entries")

    def _load_knowledge_base(self) -> None:
        """從磁盤加載知識庫"""
        # 加載知識條目
        entries_file = self.data_directory / "entries.json"
        if entries_file.exists():
            with open(entries_file, encoding="utf-8") as f:
                entries_data = json.load(f)
                for entry_data in entries_data:
                    entry = KnowledgeEntry.from_dict(entry_data)
                    self.entries[entry.id] = entry

        # 加載向量存儲
        self.vector_store.load(self.data_directory / "vectors")

        logger.info(f"Loaded {len(self.entries)} knowledge entries")

    def save_knowledge_base(self) -> None:
        """保存知識庫到磁盤"""
        # 保存知識條目
        entries_file = self.data_directory / "entries.json"
        with open(entries_file, "w", encoding="utf-8") as f:
            entries_data = [entry.to_dict() for entry in self.entries.values()]
            json.dump(entries_data, f, indent=2)

        # 保存向量存儲
        self.vector_store.save(self.data_directory / "vectors")

        logger.info(f"Saved {len(self.entries)} knowledge entries")

    def add_entry(
        self,
        entry_id: str,
        entry_type: KnowledgeType,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEntry:
        """添加知識條目

        Args:
            entry_id: 條目 ID
            entry_type: 知識類型
            title: 標題
            content: 內容
            tags: 標籤列表
            metadata: 元數據

        Returns:
            創建的知識條目
        """
        entry = KnowledgeEntry(
            id=entry_id,
            type=entry_type,
            title=title,
            content=content,
            tags=tags or [],
            metadata=metadata or {},
        )

        self.entries[entry_id] = entry

        # 添加到向量存儲
        searchable_text = f"{title}\n{content}\n{' '.join(tags or [])}"
        self.vector_store.add_document(
            doc_id=entry_id,
            text=searchable_text,
            metadata={
                "type": entry_type.value,
                "title": title,
                "tags": tags or [],
            },
        )

        logger.info(f"Added knowledge entry: {entry_id} ({entry_type.value})")

        return entry

    def add_experience_sample(self, sample: ExperienceSample) -> None:
        """添加經驗樣本到知識庫

        Args:
            sample: 經驗樣本
        """
        entry_id = f"exp_{sample.session_id}_{sample.step_index}"

        # 構建內容
        content = f"""
Target: {sample.state.target_url}
Vulnerability: {sample.state.vulnerability_type.value}
Tool: {sample.action.tool}
Parameters: {json.dumps(sample.action.parameters, indent=2)}
Success: {sample.reward.success}
Score: {sample.reward.total_score:.2f}
"""

        self.add_entry(
            entry_id=entry_id,
            entry_type=KnowledgeType.EXPERIENCE,
            title=f"Experience: {sample.state.vulnerability_type.value}",
            content=content,
            tags=[
                sample.state.vulnerability_type.value,
                sample.action.tool,
                "success" if sample.reward.success else "failed",
            ],
            metadata={
                "session_id": sample.session_id,
                "step_index": sample.step_index,
                "quality_score": sample.quality_score,
                "reward": sample.reward.total_score,
            },
        )

    def add_attack_plan(self, plan: AttackPlan, plan_id: str | None = None) -> None:
        """添加攻擊計畫到知識庫

        Args:
            plan: 攻擊計畫
            plan_id: 計畫 ID（可選）
        """
        entry_id = plan_id or f"plan_{plan.target.url.replace('/', '_')}"

        # 構建內容
        steps_text = "\n".join(
            [
                f"{i+1}. {step.tool} - {step.description}"
                for i, step in enumerate(plan.steps)
            ]
        )

        content = f"""
Target: {plan.target.url}
Objective: {plan.objective}
Priority: {plan.priority}

Steps:
{steps_text}

Expected Results:
{', '.join(plan.expected_results)}
"""

        self.add_entry(
            entry_id=entry_id,
            entry_type=KnowledgeType.ATTACK_TECHNIQUE,
            title=f"Attack Plan: {plan.objective}",
            content=content,
            tags=[
                plan.target.type,
                f"priority_{plan.priority}",
                *plan.expected_results[:3],  # 前 3 個期望結果作為標籤
            ],
            metadata={
                "target_url": plan.target.url,
                "target_type": plan.target.type,
                "priority": plan.priority,
                "steps_count": len(plan.steps),
            },
        )

    def search(
        self,
        query: str,
        entry_type: KnowledgeType | None = None,
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeEntry]:
        """搜索知識條目

        Args:
            query: 查詢文本
            entry_type: 知識類型過濾
            tags: 標籤過濾
            top_k: 返回結果數量

        Returns:
            知識條目列表
        """
        # 構建過濾條件
        filter_metadata: dict[str, Any] = {}
        if entry_type:
            filter_metadata["type"] = entry_type.value

        # 向量搜索
        search_results = self.vector_store.search(
            query=query, top_k=top_k * 2, filter_metadata=filter_metadata
        )

        # 應用標籤過濾
        filtered_results = []
        for result in search_results:
            entry = self.entries.get(result["doc_id"])
            if entry is None:
                continue

            # 標籤過濾
            if tags and not any(tag in entry.tags for tag in tags):
                continue

            filtered_results.append(entry)

            if len(filtered_results) >= top_k:
                break

        logger.debug(f"Search for '{query}' returned {len(filtered_results)} results")

        return filtered_results

    def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        """獲取知識條目

        Args:
            entry_id: 條目 ID

        Returns:
            知識條目，不存在則返回 None
        """
        return self.entries.get(entry_id)

    def update_usage_stats(self, entry_id: str, success: bool = True) -> None:
        """更新知識條目使用統計

        Args:
            entry_id: 條目 ID
            success: 是否成功使用
        """
        entry = self.entries.get(entry_id)
        if entry is None:
            return

        entry.usage_count += 1
        entry.updated_at = datetime.now()

        # 更新成功率（使用指數移動平均）
        alpha = 0.1
        success_value = 1.0 if success else 0.0
        entry.success_rate = alpha * success_value + (1 - alpha) * entry.success_rate

        logger.debug(
            f"Updated usage stats for {entry_id}: "
            f"count={entry.usage_count}, success_rate={entry.success_rate:.2f}"
        )

    def get_top_entries(
        self,
        entry_type: KnowledgeType | None = None,
        metric: str = "usage_count",
        top_k: int = 10,
    ) -> list[KnowledgeEntry]:
        """獲取排名最高的知識條目

        Args:
            entry_type: 知識類型過濾
            metric: 排序指標（usage_count 或 success_rate）
            top_k: 返回結果數量

        Returns:
            知識條目列表
        """
        filtered_entries = [
            entry
            for entry in self.entries.values()
            if entry_type is None or entry.type == entry_type
        ]

        sorted_entries = sorted(
            filtered_entries,
            key=lambda e: getattr(e, metric, 0),
            reverse=True,
        )

        return sorted_entries[:top_k]

    def get_statistics(self) -> dict[str, Any]:
        """獲取知識庫統計信息

        Returns:
            統計信息字典
        """
        stats_by_type = {}
        for entry in self.entries.values():
            type_name = entry.type.value
            if type_name not in stats_by_type:
                stats_by_type[type_name] = {
                    "count": 0,
                    "total_usage": 0,
                    "avg_success_rate": 0.0,
                }

            stats_by_type[type_name]["count"] += 1
            stats_by_type[type_name]["total_usage"] += entry.usage_count
            stats_by_type[type_name]["avg_success_rate"] += entry.success_rate

        # 計算平均值
        for stats in stats_by_type.values():
            if stats["count"] > 0:
                stats["avg_success_rate"] /= stats["count"]

        return {
            "total_entries": len(self.entries),
            "by_type": stats_by_type,
            "vector_store": self.vector_store.get_statistics(),
        }
