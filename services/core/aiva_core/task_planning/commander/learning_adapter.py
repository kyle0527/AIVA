"""學習系統適配器

負責經驗學習、模型訓練和知識檢索
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LearningAdapter:
    """學習系統適配器
    
    連接經驗管理器、模型訓練器和 RAG 引擎
    使用 CLI 執行架構（subprocess），不直接依賴其他模組
    """

    def __init__(
        self,
        data_directory: Any = None,
        enabled: bool = True,
    ):
        """初始化學習適配器
        
        Args:
            data_directory: 數據目錄路徑（用於存儲學習數據）
            enabled: 是否啟用學習功能
        """
        self.data_directory = data_directory
        self.enabled = enabled
        # CLI 架構 - 透過 subprocess 調用學習服務

    def learn_from_experience(self, context: dict[str, Any]) -> dict[str, Any]:
        """從經驗中學習

        Args:
            context: 包含 experience_sample

        Returns:
            學習結果
        """
        logger.info("📚 Learning from experience...")

        sample = context.get("experience_sample")
        if not sample:
            return {"success": False, "error": "No experience sample provided"}

        # 1. 添加到經驗管理器
        self.experience_manager.add_sample(sample)

        # 2. 添加到 RAG 知識庫
        self.rag_engine.learn_from_experience(sample)

        return {
            "success": True,
            "sample_quality": sample.quality_score,
            "knowledge_updated": True,
        }

    async def train_model(self, context: dict[str, Any]) -> dict[str, Any]:
        """訓練模型

        Args:
            context: 訓練配置

        Returns:
            訓練結果
        """
        logger.info("🎓 Training AI model...")

        min_samples = context.get("min_samples", 100)
        model_type = context.get("model_type", "supervised")
        
        # 從經驗管理器獲取訓練數據
        training_data = self.experience_manager.get_high_quality_samples(
            min_quality=0.6,
            limit=min_samples
        )
        
        if len(training_data) < min_samples:
            return {
                "success": False,
                "error": f"Not enough training data: {len(training_data)} < {min_samples}",
                "samples_available": len(training_data)
            }
        
        # 使用 ModelTrainer 訓練
        from services.aiva_common.schemas import ModelTrainingConfig
        from uuid import uuid4
        
        config = ModelTrainingConfig(
            config_id=f"config_{uuid4().hex[:8]}",
            model_type=model_type,
            training_mode=model_type,
            batch_size=32,
            epochs=10
        )
        
        samples = [exp.to_dict() for exp in training_data]
        
        result = await self.model_trainer.train(
            samples=samples,
            config=config,
            mode=model_type
        )

        if hasattr(result, 'model_dump'):
            return result.model_dump()
        elif hasattr(result, '__dict__'):
            return vars(result)
        else:
            return {"success": True, "result": str(result)}

    async def retrieve_knowledge(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢索知識

        Args:
            context: 包含 query

        Returns:
            檢索結果
        """
        logger.info("🔎 Retrieving knowledge from RAG...")

        query = context.get("query", "")
        top_k = context.get("top_k", 5)

        results = await self.rag_engine.knowledge_base.search(
            query=query,
            top_k=top_k,
        )

        return {
            "success": True,
            "results_count": len(results),
            "results": [
                {
                    "content": entry.get("content", "")[:200],
                    "relevance_score": entry.get("relevance_score", 0.0),
                    "source": entry.get("source", "unknown"),
                }
                for entry in results
            ],
        }

    async def run_training_session(
        self,
        scenario_ids: list[str] | None = None,
        episodes_per_scenario: int = 10,
    ) -> dict[str, Any]:
        """運行訓練會話（已廢棄 - 由 UnifiedExecutor 自動管理）

        Args:
            scenario_ids: 場景 ID 列表
            episodes_per_scenario: 每個場景的回合數

        Returns:
            訓練結果
        """
        logger.warning("⚠️ run_training_session is deprecated. Training is now automatic in UnifiedExecutor.")
        
        status = self.unified_executor.get_learning_status()
        
        return {
            "success": True,
            "message": "Training is now automatic. Current learning status:",
            "learning_status": status
        }

    def get_learning_status(self) -> dict[str, Any]:
        """獲取學習狀態"""
        try:
            return self.unified_executor.get_learning_status()
        except Exception as e:
            logger.error(f"Failed to get learning status: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> dict[str, Any]:
        """獲取統計信息"""
        return {
            "knowledge_stats": self.rag_engine.get_statistics(),
            "experience_stats": self.experience_manager.get_statistics(),
            "learning_status": self.get_learning_status(),
        }

    def save_state(self) -> None:
        """保存狀態"""
        logger.info("💾 Saving learning state...")

        # 保存 RAG 知識庫
        self.rag_engine.save_knowledge()

        # 保存經驗數據
        experiences_path = self.data_directory / "experiences.jsonl"
        try:
            dataset = self.experience_manager.create_dataset(
                name="saved_experiences",
                min_reward=0.0,
                max_samples=10000
            )
            with open(experiences_path, "w", encoding="utf-8") as f:
                for exp in dataset.get("experiences", []):
                    f.write(json.dumps(exp, ensure_ascii=False) + "\n")
            logger.info(f"  Saved {len(dataset.get('experiences', []))} experiences")
        except Exception as e:
            logger.error(f"  Failed to save experiences: {e}")

        logger.info("✅ Learning state saved")
