"""
Experience Manager - 經驗管理器

基於強化學習的經驗重放機制 (Experience Replay Memory)，
用於訓練編排器的攻擊執行經驗管理和訓練資料集建立。

參考實現:
- PyTorch DQN Tutorial (Experience Replay Buffer)
- Integration Module Experience Repository (資料庫持久化)
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any, Deque
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExperienceTransition:
    """經驗轉換 (Transition)
    
    表示單一攻擊執行的狀態轉換，對應 RL 中的 (state, action, next_state, reward)
    
    v2.0 擴展：支援統一反饋架構的環境標記
    
    Attributes:
        experience_id: 經驗唯一識別碼
        state: 當前狀態 (AST圖、目標資訊、上下文)
        action: 執行的攻擊動作 (攻擊類型、參數、配置)
        next_state: 下一狀態 (執行結果、新發現)
        reward: 獎勵值 (基於成功率、完成度、評分)
        metadata: 額外元資料 (時間戳、執行軌跡)
        environment: 環境標記 (sandbox/production) - v2.0 新增
    """
    
    def __init__(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        next_state: dict[str, Any],
        reward: float,
        metadata: dict[str, Any] | None = None,
        environment: str | None = None,  # v2.0: 環境標記
    ):
        self.experience_id = f"exp_{uuid4().hex[:8]}"
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.metadata = metadata or {}
        self.environment = environment  # v2.0: 保存環境資訊
        self.timestamp = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "experience_id": self.experience_id,
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "metadata": self.metadata,
            "environment": self.environment,  # v2.0
            "timestamp": self.timestamp.isoformat(),
        }


class ExperienceManager:
    """經驗管理器
    
    管理攻擊執行經驗的記錄、存儲、採樣和訓練資料集建立。
    採用循環緩衝區 (Circular Buffer) 設計，自動淘汰舊經驗。
    
    核心功能:
    1. 經驗記錄 (push): 保存攻擊執行經驗
    2. 隨機採樣 (sample): 為訓練提供批次資料
    3. 優先級採樣 (prioritized_sample): 基於獎勵值的優先採樣
    4. 資料集建立 (create_dataset): 生成訓練資料集
    5. 統計分析 (get_statistics): 經驗庫分析
    
    整合點:
    - 與 ExperienceRepository 整合進行持久化存儲
    - 與 UnifiedAttackExecutor 整合提供訓練資料（自動學習）
    - 與 RAG 整合進行經驗檢索和相似度匹配
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        auto_persist: bool = True,
        persist_batch_size: int = 100,
        unified_manager=None,
    ):
        """初始化經驗管理器
        
        Args:
            capacity: 經驗緩衝區容量 (預設 10000)
            auto_persist: 是否自動持久化到整合模組資料庫
            persist_batch_size: 自動持久化的批次大小
            unified_manager: UnifiedDataManager 實例（用於持久化）
        """
        self.capacity = capacity
        self.memory: Deque[ExperienceTransition] = deque(maxlen=capacity)
        # 添加 buffer 別名以兼容外部調用（如 UnifiedExecutor）
        self.buffer = self.memory
        self._total_experiences = 0
        self._total_reward = 0.0
        
        # 整合模組連接
        self.auto_persist = auto_persist
        self.persist_batch_size = persist_batch_size
        self.unified_manager = unified_manager
        self._pending_persist_count = 0
        
        logger.info(
            f"ExperienceManager initialized with capacity={capacity}, "
            f"auto_persist={auto_persist}"
        )
    
    def push(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        next_state: dict[str, Any],
        reward: float,
        metadata: dict[str, Any] | None = None,
        environment: str | None = None,  # v2.0: 環境標記
    ) -> str:
        """保存經驗轉換
        
        Args:
            state: 當前狀態 (包含 AST 圖、目標資訊)
            action: 執行的攻擊動作
            next_state: 下一狀態 (執行結果)
            reward: 獎勵值 (0.0-1.0)
            metadata: 額外元資料
            environment: 環境標記 ("sandbox" 或 "production") - v2.0 新增
        
        Returns:
            experience_id: 經驗唯一識別碼
        
        Example:
            >>> manager = ExperienceManager()
            >>> exp_id = manager.push(
            ...     state={"ast": {...}, "target": "http://example.com"},
            ...     action={"type": "sqli", "params": {...}},
            ...     next_state={"success": True, "findings": [...]},
            ...     reward=0.85,
            ...     metadata={"execution_time": 2.5},
            ...     environment="sandbox"  # v2.0
            ... )
        """
        transition = ExperienceTransition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            metadata=metadata,
            environment=environment,  # v2.0
        )
        
        self.memory.append(transition)
        self._total_experiences += 1
        self._total_reward += reward
        self._pending_persist_count += 1
        
        logger.debug(
            f"Saved experience {transition.experience_id} "
            f"(reward: {reward:.2f}, buffer size: {len(self.memory)})"
        )
        
        # 自動持久化到整合模組
        if (
            self.auto_persist
            and self.unified_manager
            and self._pending_persist_count >= self.persist_batch_size
        ):
            self._persist_to_integration()
        
        return transition.experience_id
    
    def _persist_to_integration(self) -> None:
        """持久化經驗到整合模組資料庫
        
        將記憶體中的高品質經驗批次保存到 UnifiedDataManager
        """
        if not self.unified_manager:
            logger.warning("UnifiedDataManager not configured, skipping persist")
            return
        
        try:
            # 採樣高品質經驗進行持久化
            high_quality = [
                exp for exp in self.memory
                if exp.reward >= 0.6  # 最低品質閾值
            ]
            
            # 批次保存到整合模組
            saved_count = 0
            for exp in high_quality[-self.persist_batch_size:]:
                try:
                    self.unified_manager.save_attack_experience(
                        plan_id=exp.metadata.get("plan_id", f"plan_{exp.experience_id}"),
                        attack_type=exp.action.get("type", "unknown"),
                        ast_graph=exp.state.get("ast", {}),
                        execution_trace={
                            "state_before": exp.state,
                            "action": exp.action,
                            "state_after": exp.next_state,
                        },
                        metrics={"overall_score": exp.reward},
                        feedback={"reward": exp.reward, "metadata": exp.metadata},
                        target_info=exp.state.get("target_info"),
                        metadata=exp.metadata,
                    )
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Failed to persist experience {exp.experience_id}: {e}")
            
            self._pending_persist_count = 0
            logger.info(f"Persisted {saved_count}/{len(high_quality)} experiences to integration module")
            
        except Exception as e:
            logger.error(f"Failed to persist batch: {e}")
    
    def load_from_integration(
        self,
        attack_type: str | None = None,
        min_score: float = 0.7,
        limit: int = 100,
    ) -> int:
        """從整合模組載入歷史經驗到記憶體
        
        用於跨會話共享經驗或初始化訓練
        
        Args:
            attack_type: 攻擊類型過濾
            min_score: 最低分數閾值
            limit: 載入數量限制
        
        Returns:
            實際載入的經驗數量
        """
        if not self.unified_manager:
            logger.warning("UnifiedDataManager not configured, cannot load")
            return 0
        
        try:
            # 從整合模組查詢高品質經驗
            records = self.unified_manager.query_high_quality_experiences(
                attack_type=attack_type,
                min_score=min_score,
                limit=limit,
            )
            
            # 轉換並加載到記憶體
            loaded_count = 0
            for record in records:
                try:
                    # 重建 ExperienceTransition
                    trace = record.execution_trace or {}
                    self.push(
                        state=trace.get("state_before", {}),
                        action=trace.get("action", {}),
                        next_state=trace.get("state_after", {}),
                        reward=record.overall_score or 0.0,
                        metadata={
                            "plan_id": record.plan_id,
                            "attack_type": record.attack_type,
                            "loaded_from_db": True,
                            "original_created_at": record.created_at.isoformat() if record.created_at else None,
                        },
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load experience {record.id}: {e}")
            
            logger.info(
                f"Loaded {loaded_count}/{len(records)} experiences from integration module "
                f"(type={attack_type}, min_score={min_score})"
            )
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load from integration: {e}")
            return 0
    
    def get_experiences_by_environment(
        self,
        environment: str,
        limit: int | None = None
    ) -> list[ExperienceTransition]:
        """按環境類型過濾經驗（v2.0 新增）
        
        用於統一反饋架構：分別獲取 sandbox 或 production 經驗
        
        Args:
            environment: 環境類型 ("sandbox" 或 "production")
            limit: 返回數量限制（None 表示返回全部）
        
        Returns:
            匹配環境的經驗列表
        
        Example:
            >>> manager = ExperienceManager()
            >>> sandbox_exps = manager.get_experiences_by_environment("sandbox")
            >>> production_exps = manager.get_experiences_by_environment("production", limit=100)
        """
        filtered = [
            exp for exp in self.memory
            if exp.environment == environment
        ]
        
        if limit is not None:
            filtered = filtered[-limit:]  # 返回最近的 N 個
        
        logger.debug(
            f"Filtered {len(filtered)} experiences by environment={environment}"
        )
        
        return filtered
    
    def sample(self, batch_size: int) -> list[ExperienceTransition]:
        """隨機採樣經驗批次
        
        Args:
            batch_size: 批次大小
        
        Returns:
            隨機採樣的經驗轉換列表
        
        Raises:
            ValueError: 如果緩衝區經驗數不足
        """
        import random
        
        if len(self.memory) < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer: "
                f"{len(self.memory)} < {batch_size}"
            )
        
        return random.sample(list(self.memory), batch_size)
    
    def prioritized_sample(
        self,
        batch_size: int,
        min_reward: float = 0.5,
    ) -> list[ExperienceTransition]:
        """優先級採樣 (基於獎勵值)
        
        優先採樣高獎勵經驗，用於提升訓練品質。
        
        Args:
            batch_size: 批次大小
            min_reward: 最低獎勵閾值
        
        Returns:
            優先級排序的經驗轉換列表
        """
        # 過濾高品質經驗
        high_quality_experiences = [
            exp for exp in self.memory
            if exp.reward >= min_reward
        ]
        
        if len(high_quality_experiences) < batch_size:
            logger.warning(
                f"Not enough high-quality experiences "
                f"(>= {min_reward}): {len(high_quality_experiences)}"
            )
            # 降級為隨機採樣
            return self.sample(batch_size)
        
        # 按獎勵值降序排序
        sorted_experiences = sorted(
            high_quality_experiences,
            key=lambda x: x.reward,
            reverse=True,
        )
        
        return sorted_experiences[:batch_size]
    
    def create_dataset(
        self,
        name: str,
        min_reward: float = 0.5,
        max_samples: int = 1000,
    ) -> dict[str, Any]:
        """建立訓練資料集
        
        Args:
            name: 資料集名稱
            min_reward: 最低獎勵閾值
            max_samples: 最大樣本數
        
        Returns:
            訓練資料集字典
        
        Example:
            >>> dataset = manager.create_dataset(
            ...     name="sqli_training_v1",
            ...     min_reward=0.7,
            ...     max_samples=500
            ... )
        """
        # 過濾符合條件的經驗
        filtered_experiences = [
            exp for exp in self.memory
            if exp.reward >= min_reward
        ]
        
        # 限制樣本數
        selected_experiences = filtered_experiences[:max_samples]
        
        dataset = {
            "dataset_id": f"dataset_{uuid4().hex[:8]}",
            "name": name,
            "description": f"Training dataset with {len(selected_experiences)} samples",
            "min_reward_threshold": min_reward,
            "max_samples": max_samples,
            "actual_samples": len(selected_experiences),
            "experiences": [exp.to_dict() for exp in selected_experiences],
            "created_at": datetime.now().isoformat(),
            "statistics": {
                "avg_reward": sum(e.reward for e in selected_experiences) / len(selected_experiences) if selected_experiences else 0.0,
                "min_reward": min(e.reward for e in selected_experiences) if selected_experiences else 0.0,
                "max_reward": max(e.reward for e in selected_experiences) if selected_experiences else 0.0,
            },
        }
        
        logger.info(
            f"Created dataset '{name}' with {len(selected_experiences)} samples "
            f"(avg reward: {dataset['statistics']['avg_reward']:.2f})"
        )
        
        return dataset
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取經驗庫統計資訊
        
        Returns:
            統計資訊字典
        """
        if not self.memory:
            return {
                "total_experiences": 0,
                "current_buffer_size": 0,
                "avg_reward": 0.0,
                "capacity_usage": "0%",
            }
        
        rewards = [exp.reward for exp in self.memory]
        
        return {
            "total_experiences": self._total_experiences,
            "current_buffer_size": len(self.memory),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "capacity": self.capacity,
            "capacity_usage": f"{len(self.memory) / self.capacity * 100:.1f}%",
            "lifetime_avg_reward": self._total_reward / self._total_experiences if self._total_experiences > 0 else 0.0,
        }
    
    def clear(self) -> None:
        """清空經驗緩衝區"""
        self.memory.clear()
        logger.info("Experience buffer cleared")
    
    def __len__(self) -> int:
        """返回當前緩衝區大小"""
        return len(self.memory)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ExperienceManager("
            f"capacity={self.capacity}, "
            f"buffer_size={len(self.memory)}, "
            f"avg_reward={stats['avg_reward']:.2f})"
        )
    
    def add_sample(self, sample: Any) -> str:
        """添加經驗樣本（UnifiedAttackExecutor 整合點）
        
        這是與 UnifiedAttackExecutor 的橋接方法，接收 ExperienceSample 對象
        並將其轉換為內部的 ExperienceTransition 格式。
        
        Args:
            sample: ExperienceSample 對象（來自 UnifiedAttackExecutor）
        
        Returns:
            experience_id: 經驗唯一識別碼
        
        Example:
            >>> # 在 UnifiedAttackExecutor 中使用
            >>> for sample in samples:
            >>>     self.experience_manager.add_sample(sample)
        """
        from services.aiva_common.schemas import ExperienceSample
        
        # 檢查類型
        if not isinstance(sample, ExperienceSample):
            logger.warning(f"Expected ExperienceSample, got {type(sample).__name__}")
            # 嘗試轉換
            if hasattr(sample, 'state_before') and hasattr(sample, 'action_taken'):
                pass  # 繼續處理
            else:
                raise TypeError(f"Cannot convert {type(sample).__name__} to ExperienceTransition")
        
        # 轉換 ExperienceSample → ExperienceTransition
        experience_id = self.push(
            state=sample.state_before,
            action=sample.action_taken,
            next_state=sample.state_after,
            reward=sample.reward,
            metadata={
                "sample_id": sample.sample_id,
                "session_id": sample.session_id,
                "plan_id": sample.plan_id,
                "quality_score": sample.quality_score,
                "is_positive": sample.is_positive,
                "confidence": sample.confidence,
                "learning_tags": sample.learning_tags,
                "difficulty_level": sample.difficulty_level,
                "timestamp": sample.timestamp.isoformat() if hasattr(sample.timestamp, 'isoformat') else str(sample.timestamp),
                "duration_ms": sample.duration_ms,
                "reward_breakdown": sample.reward_breakdown,
                "context": sample.context,
                "target_info": sample.target_info,
            }
        )
        
        logger.debug(f"Converted ExperienceSample {sample.sample_id} → ExperienceTransition {experience_id}")
        
        return experience_id
    
    def get_high_quality_samples(
        self,
        min_quality: float = 0.6,
        limit: int = 1000,
    ) -> list[ExperienceTransition]:
        """獲取高質量樣本（用於訓練）
        
        Args:
            min_quality: 最低質量閾值
            limit: 最大返回數量
        
        Returns:
            高質量經驗轉換列表
        """
        # 過濾高質量經驗（基於獎勵值）
        high_quality = [
            exp for exp in self.memory
            if exp.reward >= min_quality
        ]
        
        # 按獎勵值降序排序
        sorted_experiences = sorted(
            high_quality,
            key=lambda x: x.reward,
            reverse=True,
        )
        
        result = sorted_experiences[:limit]
        
        logger.info(
            f"Retrieved {len(result)} high-quality samples "
            f"(min_quality={min_quality}, total_available={len(high_quality)})"
        )
        
        return result


# ==================== 整合示例 ====================

def integrate_with_repository_example():
    """整合範例: ExperienceManager + ExperienceRepository
    
    展示如何將記憶體經驗持久化到資料庫。
    """
    from services.integration.aiva_integration.reception import ExperienceRepository
    
    # 初始化
    manager = ExperienceManager(capacity=10000)
    repository = ExperienceRepository(
        database_url="sqlite:///data/integration/experiences/experience.db"
    )
    
    # 記錄經驗 (記憶體)
    _ = manager.push(
        state={"ast": {...}, "target": "http://example.com"},
        action={"type": "sqli", "params": {...}},
        next_state={"success": True},
        reward=0.85,
    )
    
    # 持久化到資料庫 (定期批次保存)
    if len(manager) >= 100:
        # 採樣高品質經驗
        high_quality = manager.prioritized_sample(batch_size=50, min_reward=0.7)
        
        for exp in high_quality:
            repository.save_experience(
                plan_id=exp.metadata.get("plan_id", "unknown"),
                attack_type=exp.action.get("type", "unknown"),
                ast_graph=exp.state.get("ast", {}),
                execution_trace=exp.next_state,
                metrics={"overall_score": exp.reward},
                feedback={"reward": exp.reward},
            )
        
        logger.info(f"Persisted {len(high_quality)} experiences to database")


if __name__ == "__main__":
    # 快速測試
    logging.basicConfig(level=logging.INFO)
    
    manager = ExperienceManager(capacity=100)
    
    # 模擬經驗記錄
    for i in range(10):
        exp_id = manager.push(
            state={"iteration": i, "target": f"test_{i}"},
            action={"type": "test_attack"},
            next_state={"success": i % 2 == 0},
            reward=0.5 + (i / 20),
        )
    
    print(manager)
    print("\n統計資訊:")
    import json
    print(json.dumps(manager.get_statistics(), indent=2, ensure_ascii=False))
    
    # 測試採樣
    samples = manager.sample(batch_size=3)
    print(f"\n隨機採樣 {len(samples)} 個經驗")
    
    # 測試資料集建立
    dataset = manager.create_dataset(name="test_dataset", min_reward=0.6, max_samples=5)
    print(f"\n資料集樣本數: {dataset['actual_samples']}")
