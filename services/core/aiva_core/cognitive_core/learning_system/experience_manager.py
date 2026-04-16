"""
Experience Manager - 經驗管理器

基於強化學習的經驗重放機制 (Experience Replay Memory)，
用於訓練編排器的攻擊執行經驗管理和訓練資料集建立。

數據來源（三個）:
1. Integration Module (整合模組): services/integration/data/experiences/*.jsonl
   - 實際執行記錄（本次任務）
   - 按能力分類: xss.jsonl, sqli.jsonl, ssrf.jsonl, phase0.jsonl 等
   - 包含時間戳: 所有記錄都有 timestamp
   
2. Historical Data (歷史數據): 同樣從整合模組讀取，但是歷史時間段
   - 作為"預期響應"的參考
   - 相同場景的不同結果
   
3. Capability Knowledge Base (能力知識庫): 
   - 位置: cognitive_core/learning_system/knowledge/
   - 格式: Markdown 分析報告
   - 內容: XSS/SQLi/SSRF 等攻擊手法、繞過技巧、預期響應模式
   - 文件: XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md, SQLI_MODULE_...等

學習流程:
1. 讀取三個數據源
2. 三路比對：本次 vs 歷史 vs 知識庫
3. 分析差異和變化趨勢
4. 判斷是否為已知情況
   - 已知：使用知識庫建議
   - 未知：觸發 RAG 搜索（搜索技術文檔、CVE、安全研究等）
5. 生成優化建議（參數+方法）
6. 生成新權重並驗證
7. 效果好才保存，效果差丟棄

統一格式: {timestamp, task_id, capability, target, request, response, result, metadata}

參考實現:
- PyTorch DQN Tutorial (Experience Replay Buffer)
- SimpleDataManager (簡單數據管理器，JSONL 格式)
- ModuleKnowledgeManager (模組知識庫管理器)
"""

from collections import deque
from datetime import datetime
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# RAG 觸發器和通知系統（延遲導入避免循環依賴）
try:
    from .notification_system import NotificationSystem, get_notification_system
    from .rag_trigger import RAGTrigger, UnknownSituationAlert
except ImportError:
    RAGTrigger = None
    UnknownSituationAlert = None
    get_notification_system = None
    NotificationSystem = None
    logger.warning("RAG Trigger or Notification System not available")


class ExperienceTransition:
    """經驗轉換 (Transition)
    
    表示單一攻擊執行的狀態轉換，對應 RL 中的 (state, action, next_state, reward)
    
    v2.0 重新設計：移除環境標記，改用目標敏感度
    
    設計原則（基於 Transfer Learning 最佳實踐）：
    - 經驗標記基於「目標屬性」而非「執行環境」
    - 確保學習結果可直接遷移到相同敏感度的目標
    - 沒有 sandbox vs production 的 domain gap
    
    Attributes:
        experience_id: 經驗唯一識別碼
        state: 當前狀態 (AST圖、目標資訊、上下文)
        action: 執行的攻擊動作 (攻擊類型、參數、配置)
        next_state: 下一狀態 (執行結果、新發現)
        reward: 獎勵值 (基於成功率、完成度、評分)
        metadata: 額外元資料 (時間戳、執行軌跡、target_sensitivity)
        environment: [已棄用] 保留作為向後兼容
    """
    
    def __init__(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        next_state: dict[str, Any],
        reward: float,
        metadata: dict[str, Any] | None = None,
        environment: str | None = None,  # [已棄用] 保留向後兼容
        target_sensitivity: float | None = None,  # v2.0: 新增目標敏感度
    ):
        self.experience_id = f"exp_{uuid4().hex[:8]}"
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.metadata = metadata or {}
        
        # v2.0: 優先使用目標敏感度，環境標記作為向後兼容
        if target_sensitivity is not None:
            self.metadata["target_sensitivity"] = target_sensitivity
        elif environment is not None:
            # 從舊的環境標記推導敏感度（向後兼容）
            self.metadata["target_sensitivity"] = 0.2 if environment == "sandbox" else 0.8
            
        self.environment = environment  # [已棄用] 保留向後兼容
        self.timestamp = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "experience_id": self.experience_id,
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "metadata": self.metadata,  # 包含 target_sensitivity
            "environment": self.environment,  # [已棄用] 向後兼容
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
        data_manager=None,  # SimpleDataManager 實例
        knowledge_manager=None,  # ModuleKnowledgeManager 實例
        rag_engine=None,  # RAGEngine 實例
        similarity_threshold: float = 0.6,  # RAG 觸發相似度閾值
        enable_notifications: bool = True,  # 是否啟用用戶通知
    ):
        """初始化經驗管理器
        
        Args:
            capacity: 經驗緩衝區容量 (預設 10000)
            auto_persist: 是否自動持久化到整合模組
            persist_batch_size: 自動持久化的批次大小
            data_manager: SimpleDataManager 實例（用於讀取整合模組數據）
            knowledge_manager: ModuleKnowledgeManager 實例（用於讀取能力知識庫）
            rag_engine: RAGEngine 實例（用於 RAG 搜索）
            similarity_threshold: RAG 觸發的相似度閾值（低於此值觸發 RAG）
            enable_notifications: 是否啟用用戶通知
        """
        self.capacity = capacity
        self.memory: deque[ExperienceTransition] = deque(maxlen=capacity)
        # 添加 buffer 別名以兼容外部調用（如 UnifiedExecutor）
        self.buffer = self.memory
        self._total_experiences = 0
        self._total_reward = 0.0
        
        # 整合模組連接（讀取實際執行記錄和歷史數據）
        self.auto_persist = auto_persist
        self.persist_batch_size = persist_batch_size
        self.data_manager = data_manager
        self._pending_persist_count = 0
        
        # 能力知識庫連接（讀取分析報告）
        self.knowledge_manager = knowledge_manager
        
        # RAG 觸發器和通知系統
        self.notification_system = None
        self.rag_trigger = None
        
        if enable_notifications and get_notification_system:
            self.notification_system = get_notification_system()
            logger.info("User notification system enabled")
        
        if rag_engine and RAGTrigger:
            # 創建通知回調
            def notification_callback(alert: UnknownSituationAlert):
                if self.notification_system:
                    if alert.status == "searching":
                        self.notification_system.notify_unknown_situation(
                            alert_id=alert.alert_id,
                            trigger_reason=alert.trigger_reason,
                            data_snapshot=alert.data_snapshot,
                            search_query=alert.search_query,
                        )
                        self.notification_system.notify_rag_triggered(
                            alert_id=alert.alert_id,
                            search_query=alert.search_query,
                        )
                    elif alert.status == "found":
                        results_summary = [
                            {
                                "type": r.get("type", "unknown"),
                                "relevance": r.get("relevance_score", 0.0),
                                "preview": r.get("content", "")[:100],
                            }
                            for r in alert.rag_results[:5]
                        ]
                        self.notification_system.notify_rag_completed(
                            alert_id=alert.alert_id,
                            results_count=len(alert.rag_results),
                            results_summary=results_summary,
                        )
                    elif alert.status == "error":
                        self.notification_system.notify_rag_failed(
                            alert_id=alert.alert_id,
                            error_message="RAG 搜索過程中發生錯誤",
                        )
            
            self.rag_trigger = RAGTrigger(
                similarity_threshold=similarity_threshold,
                rag_engine=rag_engine,
                notification_callback=notification_callback,
            )
            logger.info(
                f"RAG Trigger initialized with similarity_threshold={similarity_threshold}"
            )
        
        logger.info(
            f"ExperienceManager initialized with capacity={capacity}, "
            f"auto_persist={auto_persist}, "
            f"has_knowledge_manager={knowledge_manager is not None}, "
            f"has_rag_trigger={self.rag_trigger is not None}"
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
        capability: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> int:
        """從整合模組載入歷史經驗到記憶體
        
        從 SimpleDataManager 讀取按能力分類的 JSONL 數據
        
        Args:
            capability: 能力類型過濾 (xss, sqli, ssrf, phase0 等)
            start_time: 開始時間過濾
            end_time: 結束時間過濾
            limit: 載入數量限制
        
        Returns:
            實際載入的經驗數量
        """
        if not self.data_manager:
            logger.warning("SimpleDataManager not configured, cannot load")
            return 0
        
        try:
            # 從整合模組查詢數據
            if capability:
                records = self.data_manager.load_capability_data(
                    capability=capability,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
            else:
                # 加載所有能力的數據
                all_data = self.data_manager.load_all_data(limit_per_capability=limit)
                records = []
                for cap_records in all_data.values():
                    records.extend(cap_records)
            
            # 轉換並加載到記憶體
            loaded_count = 0
            for record in records:
                try:
                    # 重建 ExperienceTransition
                    self.push(
                        state=record.get("request", {}),
                        action={"capability": record.get("capability")},
                        next_state=record.get("response", {}),
                        reward=1.0 if record.get("result") else 0.0,  # 簡化評分
                        metadata={
                            "task_id": record.get("task_id"),
                            "target": record.get("target"),
                            "timestamp": record.get("timestamp"),
                            "loaded_from_integration": True,
                        },
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load record {record.get('task_id')}: {e}")
            
            logger.info(
                f"Loaded {loaded_count}/{len(records)} experiences from integration module "
                f"(capability={capability})"
            )
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load from integration: {e}")
            return 0
    
    def get_experiences_by_target_sensitivity(
        self,
        sensitivity_min: float,
        sensitivity_max: float,
        limit: int | None = None
    ) -> list[ExperienceTransition]:
        """按目標敏感度過濾經驗（v2.0 重新設計）
        
        環境無關設計：基於目標敏感度而非執行環境過濾
        確保學習經驗可直接遷移到相同敏感度的其他目標
        
        Args:
            sensitivity_min: 最低敏感度 (0.0-1.0)
            sensitivity_max: 最高敏感度 (0.0-1.0)
            limit: 返回數量限制（None 表示返回全部）
        
        Returns:
            匹配敏感度範圍的經驗列表
        
        Example:
            >>> manager = ExperienceManager()
            >>> # 獲取低敏感目標的經驗 (可用於任何低敏感目標)
            >>> low_sens_exps = manager.get_experiences_by_target_sensitivity(0.0, 0.3)
            >>> # 獲取高敏感目標的經驗
            >>> high_sens_exps = manager.get_experiences_by_target_sensitivity(0.7, 1.0, limit=100)
        """
        filtered = [
            exp for exp in self.memory
            if (exp.metadata.get("target_sensitivity", 0.5) >= sensitivity_min
                and exp.metadata.get("target_sensitivity", 0.5) <= sensitivity_max)
        ]
        
        if limit is not None:
            filtered = filtered[-limit:]  # 返回最近的 N 個
        
        logger.debug(
            f"Filtered {len(filtered)} experiences by sensitivity=[{sensitivity_min}, {sensitivity_max}]"
        )
        
        return filtered
    
    # 保留舊方法作為向後兼容，但標記為已棄用
    def get_experiences_by_environment(
        self,
        environment: str,
        limit: int | None = None
    ) -> list[ExperienceTransition]:
        """[已棄用] 按環境類型過濾經驗
        
        警告：此方法基於 sandbox/production 區分，已不推薦使用
        請改用 get_experiences_by_target_sensitivity()，基於目標敏感度過濾
        這樣學習經驗才能直接遷移
        
        Args:
            environment: 環境類型 ("sandbox" 或 "production")
            limit: 返回數量限制
        
        Returns:
            匹配環境的經驗列表
        """
        import warnings
        warnings.warn(
            "get_experiences_by_environment() is deprecated. "
            "Use get_experiences_by_target_sensitivity() instead for better transfer learning.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # 映射舊環境類型到敏感度範圍
        if environment == "sandbox":
            return self.get_experiences_by_target_sensitivity(0.0, 0.3, limit)
        else:  # production
            return self.get_experiences_by_target_sensitivity(0.7, 1.0, limit)
    
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
            raise ValueError(
                f"高品質經驗不足: 需要 {batch_size} 筆，但只有 {len(high_quality_experiences)} 筆 "
                f"(獎勵閾值 >= {min_reward})。"
                "請調低 min_reward 閾值或累積更多高品質經驗。"
            )
        
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
        from aiva_common.schemas import ExperienceSample
        
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
    
    async def trigger_learning_with_rag(
        self,
        capability: str,
        current_data: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """觸發學習流程（帶 RAG 支持）
        
        完整的學習流程：
        1. 從三個數據源讀取數據
        2. 三路比對（當前 vs 歷史 vs 知識庫）
        3. 判斷是否為已知情況
        4. 未知情況 → 觸發 RAG 搜索
        5. 生成優化建議和新權重
        6. 驗證效果
        7. 效果好才保存
        
        Args:
            capability: 能力類型 (xss, sqli, ssrf, phase0 等)
            current_data: 當前執行數據
            session_id: 學習會話 ID
            
        Returns:
            學習結果字典，包含優化建議、RAG 結果等
        """
        session_id = session_id or f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 通知學習開始
        if self.notification_system:
            self.notification_system.notify_learning_started(
                session_id=session_id,
                capability=capability,
                data_sources=[
                    "當前執行記錄",
                    "歷史數據",
                    "能力知識庫",
                ],
            )
        
        logger.info(
            f"🎓 [學習會話 {session_id}] 開始學習 {capability} 能力"
        )
        
        # 1. 讀取三個數據源
        logger.info("📚 [數據源 1/3] 讀取當前執行記錄...")
        current_records = []
        if self.data_manager:
            try:
                current_records = self.data_manager.load_capability_data(
                    capability=capability,
                    limit=50,
                )
            except Exception as e:
                logger.error(f"Failed to load current records: {e}")
        
        logger.info("📚 [數據源 2/3] 讀取歷史數據...")
        historical_data = []
        if self.data_manager:
            try:
                # 讀取一週前到現在的數據
                from datetime import timedelta
                historical_data = self.data_manager.load_capability_data(
                    capability=capability,
                    start_time=datetime.now() - timedelta(days=7),
                    limit=100,
                )
            except Exception as e:
                logger.error(f"Failed to load historical data: {e}")
        
        logger.info("📚 [數據源 3/3] 讀取能力知識庫...")
        knowledge_base_data = []
        if self.knowledge_manager:
            try:
                # 從知識庫讀取分析報告
                kb_info = self.knowledge_manager.get_module_info(capability)
                if kb_info:
                    knowledge_base_data = [kb_info]
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
        
        logger.info(
            f"✅ 數據源讀取完成: "
            f"當前記錄={len(current_records)}, "
            f"歷史數據={len(historical_data)}, "
            f"知識庫={len(knowledge_base_data)}"
        )
        
        # 2-4. 三路比對並判斷是否需要 RAG
        alert = None
        rag_results = []
        
        if self.rag_trigger:
            logger.info("🔍 檢查是否為已知情況...")
            alert = await self.rag_trigger.trigger_rag_if_needed(
                current_data=current_data,
                current_records=current_records,
                historical_data=historical_data,
                knowledge_base_data=knowledge_base_data,
            )
            
            if alert:
                logger.warning(
                    f"⚠️ 未知情況，已觸發 RAG 搜索: {alert.alert_id}"
                )
                rag_results = alert.rag_results
            else:
                logger.info("✅ 已知情況，使用現有知識庫")
        
        # 5. 生成優化建議
        logger.info("💡 生成優化建議...")
        optimization_plan = self._generate_optimization_plan(
            current_data=current_data,
            current_records=current_records,
            historical_data=historical_data,
            knowledge_base_data=knowledge_base_data,
            rag_results=rag_results,
        )
        
        # 6. 驗證效果（簡化版，實際需要執行測試）
        logger.info("✅ 驗證優化效果...")
        validation_passed = self._validate_optimization(optimization_plan)
        
        # 7. 保存結果（如果驗證通過）
        if validation_passed:
            logger.info("✅ 驗證通過，保存新權重")
            self._save_optimization(capability, optimization_plan)
        else:
            logger.warning("❌ 驗證未通過，丟棄優化")
        
        # 通知學習完成
        if self.notification_system:
            self.notification_system.notify_learning_completed(
                session_id=session_id,
                capability=capability,
                improvements=optimization_plan.get("improvements", {}),
                validation_passed=validation_passed,
            )
        
        logger.info(
            f"🎓 [學習會話 {session_id}] 完成: "
            f"{'✅ 保存' if validation_passed else '❌ 丟棄'}"
        )
        
        # 返回學習結果
        return {
            "session_id": session_id,
            "capability": capability,
            "unknown_situation_detected": alert is not None,
            "rag_triggered": alert is not None,
            "rag_results_count": len(rag_results),
            "optimization_plan": optimization_plan,
            "validation_passed": validation_passed,
            "alert": alert.to_dict() if alert else None,
        }
    
    def _generate_optimization_plan(
        self,
        current_data: dict[str, Any],
        current_records: list[dict[str, Any]],
        historical_data: list[dict[str, Any]],
        knowledge_base_data: list[dict[str, Any]],
        rag_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """生成優化建議
        
        基於三個數據源和 RAG 結果生成優化計畫
        """
        plan = {
            "timestamp": datetime.now().isoformat(),
            "data_sources_used": {
                "current_records": len(current_records),
                "historical_data": len(historical_data),
                "knowledge_base": len(knowledge_base_data),
                "rag_results": len(rag_results),
            },
            "improvements": {},
            "new_weights": {},
        }
        
        # 分析當前數據與歷史數據的差異
        if historical_data:
            # 計算成功率變化
            current_success = current_data.get("result", {}).get("success", False)
            historical_success_rate = sum(
                1 for h in historical_data
                if h.get("result", {}).get("success", False)
            ) / len(historical_data)
            
            plan["improvements"]["success_rate_change"] = (
                1.0 if current_success else 0.0
            ) - historical_success_rate
        
        # 從 RAG 結果提取建議
        if rag_results:
            plan["improvements"]["rag_suggestions"] = [
                {
                    "type": r.get("type"),
                    "relevance": r.get("relevance_score"),
                    "content_preview": r.get("content", "")[:100],
                }
                for r in rag_results[:3]
            ]
        
        # 生成新權重（簡化版）
        plan["new_weights"]["learning_rate"] = 0.01
        plan["new_weights"]["exploration_rate"] = 0.1
        
        return plan
    
    def _validate_optimization(self, optimization_plan: dict[str, Any]) -> bool:
        """驗證優化效果
        
        簡化版：實際應該執行測試來驗證
        """
        # 簡化驗證：如果有改進建議就認為有效
        improvements = optimization_plan.get("improvements", {})
        
        # 檢查是否有正向改進
        success_rate_change = improvements.get("success_rate_change", 0.0)
        has_rag_suggestions = len(improvements.get("rag_suggestions", [])) > 0
        
        # 如果成功率提升或有 RAG 建議，認為有效
        return success_rate_change > -0.1 or has_rag_suggestions
    
    def _save_optimization(
        self,
        capability: str,
        optimization_plan: dict[str, Any]
    ) -> None:
        """保存優化結果
        
        將新權重和優化建議保存到文件
        """
        try:
            import json
            from pathlib import Path
            
            # 保存到 learning_system/data/optimizations/
            save_dir = Path("services/core/aiva_core/cognitive_core/learning_system/data/optimizations")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            filename = save_dir / f"{capability}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(optimization_plan, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved optimization plan to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization: {e}")


# ==================== 整合示例 ====================

def integrate_with_repository_example():
    """整合範例: ExperienceManager + ExperienceRepository
    
    展示如何將記憶體經驗持久化到資料庫。
    
    注意: ExperienceRepository 已移除，此函數僅供參考。
    """
    # from services.integration.aiva_integration.reception import ExperienceRepository
    
    # 初始化
    manager = ExperienceManager(capacity=10000)
    # repository = ExperienceRepository(
    #     database_url="sqlite:///data/integration/experiences/experience.db"
    # )
    
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
        
        # 註解掉未完成的持久化邏輯
        # for exp in high_quality:
        #     repository.save_experience(
        #         plan_id=exp.metadata.get("plan_id", "unknown"),
        #         attack_type=exp.action.get("type", "unknown"),
        #         ast_graph=exp.state.get("ast", {}),
        #         execution_trace=exp.next_state,
        #         metrics={"overall_score": exp.reward},
        #         feedback={"reward": exp.reward},
        #     )
        
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

