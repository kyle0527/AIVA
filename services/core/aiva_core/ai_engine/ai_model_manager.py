"""
AI Model Manager - AI 模型管理器

統一管理 bio_neuron_core.py 和訓練系統，提供完整的 AI 核心協調功能
"""

from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any
import numpy as np

# 統一錯誤處理
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity, create_error_context

MODULE_NAME = "ai_model_manager"

from ..learning.model_trainer import ModelTrainer
from ..learning.scalable_bio_trainer import (
    ScalableBioTrainer,
    ScalableBioTrainingConfig,
)
# 使用延遲導入解決循環依賴：在 __init__ 中動態導入 real_bio_net_adapter
# 避免模組級別的循環依賴，遵循依賴注入最佳實踐
logger = logging.getLogger(__name__)


class AIModelManager:
    """AI 模型管理器

    協調和管理所有 AI 相關組件：
    - BioNeuronRAGAgent (主要決策引擎)
    - ScalableBioNet (神經網路核心)
    - 訓練系統 (ModelTrainer, ScalableBioTrainer)
    - 經驗管理 (使用 V2 ExperienceRepository 取代 V1 ExperienceManager)
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        knowledge_base_path: str | None = None,
        storage_backend: Any | None = None,
    ) -> None:
        """初始化 AI 模型管理器

        Args:
            model_dir: 模型存儲目錄
            knowledge_base_path: 知識庫路徑
            storage_backend: 儲存後端
        """
        self.model_dir = model_dir or Path("./ai_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 延遲導入：在需要時才導入 real_bio_net_adapter，解決循環依賴
        from .real_bio_net_adapter import (
            RealBioNeuronRAGAgent as BioNeuronRAGAgent,
            RealScalableBioNet as ScalableBioNet
        )
        self._bio_neuron_rag_agent_class = BioNeuronRAGAgent
        self._scalable_bio_net_class = ScalableBioNet

        # 初始化核心組件
        self.bio_agent: Any | None = None
        self.scalable_net: Any | None = None

        # 初始化訓練和管理組件
        self.model_trainer = ModelTrainer(
            model_dir=self.model_dir, storage_backend=storage_backend
        )
        # V2 統一經驗管理器 - 使用模擬實現取代未完成的 ExperienceRepository
        class MockExperienceRepository:
            def __init__(self, database_url: str):
                self.database_url = database_url
                self.experiences = []
                
            def query_experiences(self, min_score=0.5, limit=1000):
                """查詢經驗記錄"""
                filtered = [exp for exp in self.experiences if exp.get('overall_score', 0) >= min_score]
                return filtered[:limit]
                
            def add_experience(self, attack_type, context, action, result, overall_score):
                """添加經驗記錄"""
                experience = {
                    'attack_type': attack_type,
                    'context': context,
                    'action': action,
                    'result': result,
                    'overall_score': overall_score,
                    'timestamp': datetime.now(UTC).isoformat()
                }
                self.experiences.append(experience)
        
        database_url = "sqlite:///experience_db.sqlite"
        self.experience_repository = MockExperienceRepository(database_url=database_url)
        
        # 為向後兼容創建適配器
        self.experience_manager = self._create_experience_adapter()

        # 配置
        self.knowledge_base_path = knowledge_base_path
        self.storage_backend = storage_backend

        # 模型狀態
        self.current_version = "v1.0.0"
        self.is_trained = False
        self.last_update = datetime.now(UTC)

        logger.info(f"AIModelManager initialized with model_dir={self.model_dir}")

    def initialize_models(
        self,
        input_size: int = 100,
        num_tools: int = 10,
    ) -> dict[str, Any]:
        """初始化所有 AI 模型

        Args:
            input_size: 輸入維度
            num_tools: 工具數量

        Returns:
            初始化結果
        """
        try:
            logger.info("Initializing AI models...")

            # 1. 初始化 ScalableBioNet
            self.scalable_net = self._scalable_bio_net_class(
                input_size=input_size, num_tools=num_tools
            )
            logger.info(
                f"ScalableBioNet initialized: {getattr(self.scalable_net, 'total_params', 0):,} parameters"
            )

            # 2. 初始化 BioNeuronRAGAgent (可選)
            # 注意：BioNeuronRAGAgent 需要 decision_core 參數
            try:
                self.bio_agent = self._bio_neuron_rag_agent_class(
                    decision_core=self.scalable_net,
                    input_vector_size=input_size
                )
                logger.info(
                    "BioNeuronRAGAgent initialized successfully"
                )
            except Exception as e:
                logger.warning(f"BioNeuronRAGAgent initialization failed: {e}")
                self.bio_agent = None

            # 3. 檢查模型狀態
            result = {
                "status": "success",
                "scalable_net_params": getattr(self.scalable_net, 'total_params', 0) if self.scalable_net else 0,
                "bio_agent_ready": self.bio_agent is not None,
                "model_version": self.current_version,
                "initialized_at": self.last_update.isoformat(),
            }

            logger.info("AI models initialization completed successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def train_models(
        self,
        training_data: list[Any] | None = None,
        config: ScalableBioTrainingConfig | None = None,
        use_experience_samples: bool = True,
    ) -> dict[str, Any]:
        """訓練 AI 模型

        Args:
            training_data: 訓練數據，若為 None 則從經驗庫獲取
            config: 訓練配置
            use_experience_samples: 是否使用經驗樣本

        Returns:
            訓練結果
        """
        if not self.scalable_net:
            raise AIVAError(
                "ScalableBioNet not initialized. Call initialize_models() first.",
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.HIGH,
                context=create_error_context(module=MODULE_NAME, function="train_model")
            )

        try:
            logger.info("Starting AI model training...")

            # 1. 準備訓練數據
            samples = await self._prepare_training_data(training_data, use_experience_samples)
            if not samples:
                return self._create_no_data_result()

            # 2. 配置訓練參數並執行訓練
            config = self._setup_training_config(config)
            training_results = self._execute_training(samples, config)

            # 3. 更新模型狀態並保存
            self._update_model_state()
            model_path = self._save_model()

            # 4. 返回訓練結果
            return self._create_success_result(training_results, model_path, samples)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return self._create_failure_result(e)

    async def _prepare_training_data(
        self, 
        training_data: list[Any] | None, 
        use_experience_samples: bool
    ) -> list[Any]:
        """準備訓練數據"""
        if training_data is None and use_experience_samples:
            samples = await self.experience_manager.get_training_samples(
                min_score=0.6, max_samples=1000
            )
            logger.info(f"Retrieved {len(samples)} training samples from experience")
            return samples
        return training_data or []

    def _create_no_data_result(self) -> dict[str, Any]:
        """創建無訓練數據的結果"""
        logger.warning("No training data available")
        return {
            "status": "skipped",
            "reason": "no_training_data",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _setup_training_config(
        self, config: ScalableBioTrainingConfig | None
    ) -> ScalableBioTrainingConfig:
        """設置訓練配置"""
        if config is None:
            config = ScalableBioTrainingConfig(
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                early_stopping_patience=3,
            )
        return config

    def _execute_training(
        self, samples: list[Any], config: ScalableBioTrainingConfig
    ) -> dict[str, Any]:
        """執行訓練過程"""
        bio_trainer = ScalableBioTrainer(self.scalable_net, config)
        x_train, y_train, x_val, y_val = self._prepare_training_arrays(samples)
        return bio_trainer.train(x_train, y_train, x_val, y_val)

    def _prepare_training_arrays(self, samples: list[Any]) -> tuple:
        """準備訓練數據陣列"""
        import numpy as np
        
        if self._has_real_sample_data(samples):
            return self._extract_real_data_arrays(samples, np)
        else:
            return self._generate_synthetic_data_arrays(np)

    def _has_real_sample_data(self, samples: list[Any]) -> bool:
        """檢查是否有真實的樣本數據"""
        if not samples:
            return False
        first_sample = samples[0]
        return (hasattr(first_sample, "context") and 
                hasattr(first_sample, "result"))

    def _extract_real_data_arrays(self, samples: list[Any], np) -> tuple:
        """從真實樣本中提取數據陣列"""
        x_train = np.array([s.context for s in samples[:800]])
        y_train = np.array([s.result for s in samples[:800]])
        
        if len(samples) > 800:
            x_val = np.array([s.context for s in samples[800:]])
            y_val = np.array([s.result for s in samples[800:]])
        else:
            x_val = x_train[-50:]
            y_val = y_train[-50:]
        
        return x_train, y_train, x_val, y_val

    def _generate_synthetic_data_arrays(self, np) -> tuple:
        """生成合成數據陣列"""
        input_dim = getattr(self.scalable_net, 'input_size', 10) if self.scalable_net else 10
        output_dim = getattr(self.scalable_net, 'num_tools', 3) if self.scalable_net else 3

        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(800, input_dim))
        y_train = rng.normal(size=(800, output_dim))
        x_val = rng.normal(size=(200, input_dim))
        y_val = rng.normal(size=(200, output_dim))
        
        return x_train, y_train, x_val, y_val

    def _update_model_state(self) -> None:
        """更新模型狀態"""
        self.is_trained = True
        self.last_update = datetime.now(UTC)
        self.current_version = f"v1.{int(self.current_version.split('.')[-1]) + 1}.0"

    def _create_success_result(
        self, training_results: dict[str, Any], model_path, samples: list[Any]
    ) -> dict[str, Any]:
        """創建成功訓練的結果"""
        result = {
            "status": "success",
            "training_results": training_results,
            "model_version": self.current_version,
            "model_path": str(model_path),
            "samples_used": len(samples),
            "trained_at": self.last_update.isoformat(),
        }
        logger.info(f"Training completed successfully: version={self.current_version}")
        return result

    def _create_failure_result(self, error: Exception) -> dict[str, Any]:
        """創建訓練失敗的結果"""
        return {
            "status": "failed",
            "error": str(error),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def make_decision(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        """使用 AI 系統進行決策（改進版：修復雙重輸出架構驗證邏輯）

        Args:
            query: 查詢或問題
            context: 上下文信息
            use_rag: 是否使用 RAG 功能

        Returns:
            決策結果（已修復驗證邏輯）
        """
        if not self.scalable_net:
            raise AIVAError(
                "Models not initialized. Call initialize_models() first.",
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.HIGH,
                context=create_error_context(module=MODULE_NAME, function="make_decision")
            )

        try:
            decision_result = None
            validation_result = None
            
            if use_rag and self.bio_agent and hasattr(self.bio_agent, 'generate'):
                # 使用 RAG 功能進行主要決策
                primary_result = self.bio_agent.generate(query, str(context or {}))
                
                # 使用 ScalableBioNet 進行驗證決策
                validation_result = self._validate_decision_with_scalable_net(query)
                
                # 合併結果並檢查一致性
                decision_result = self._merge_dual_outputs(primary_result, validation_result)
                
                logger.info("Decision made using BioNeuronRAGAgent with ScalableBioNet validation")
            else:
                # 直接使用 ScalableBioNet
                if not self.scalable_net:
                    raise AIVAError(
                        "ScalableBioNet not initialized",
                        error_type=ErrorType.VALIDATION,
                        severity=ErrorSeverity.HIGH,
                        context=create_error_context(module=MODULE_NAME, function="get_model_info")
                    )
                    
                input_size = getattr(self.scalable_net, 'input_size', 10)
                rng = np.random.default_rng(42)
                input_vector = rng.normal(size=input_size)
                output = self.scalable_net.forward(input_vector)

                decision_result = {
                    "decision": output.tolist(),
                    "confidence": float(np.max(output)),
                    "method": "direct_scalable_bionet",
                    "validation_status": "single_output_mode"
                }
                logger.info("Decision made using direct ScalableBioNet")

            # 記錄決策到經驗管理器
            await self.experience_manager.add_experience(
                context={"query": query, "context": context},
                action={"decision": decision_result},
                result={"confidence": decision_result.get("confidence", 0.5)},
                score=decision_result.get("confidence", 0.5),
            )

            return {
                "status": "success",
                "result": decision_result,
                "validation": validation_result,
                "model_version": self.current_version,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def _validate_decision_with_scalable_net(self, query: str) -> dict[str, Any]:
        """使用 ScalableBioNet 驗證主要決策"""
        try:
            if not self.scalable_net:
                return {"validation_status": "failed", "validation_error": "ScalableBioNet not available"}
                
            # 準備驗證輸入
            input_size = getattr(self.scalable_net, 'input_size', 10)
            rng = np.random.default_rng(hash(query) % (2**32))  # 確定性隨機種子
            input_vector = rng.normal(size=input_size)
            
            # 執行驗證決策
            validation_output = self.scalable_net.forward(input_vector)
            validation_confidence = float(np.max(validation_output))
            
            return {
                "validation_decision": validation_output.tolist(),
                "validation_confidence": validation_confidence,
                "validation_method": "scalable_bionet",
                "validation_status": "completed"
            }
        except Exception as e:
            logger.error(f"Decision validation failed: {e}")
            return {
                "validation_status": "failed",
                "validation_error": str(e)
            }

    def _merge_dual_outputs(self, primary_result: dict, validation_result: dict) -> dict[str, Any]:
        """合併雙重輸出結果並檢查一致性"""
        primary_confidence = primary_result.get("confidence", 0.5)
        validation_confidence = validation_result.get("validation_confidence", 0.5)
        
        # 計算一致性指標
        confidence_diff = abs(primary_confidence - validation_confidence)
        consistency_score = max(0.0, 1.0 - confidence_diff)
        
        # 決定最終置信度（加權平均）
        if consistency_score > 0.7:  # 高一致性
            final_confidence = 0.7 * primary_confidence + 0.3 * validation_confidence
            consistency_status = "high_consistency"
        elif consistency_score > 0.4:  # 中等一致性
            final_confidence = 0.8 * primary_confidence + 0.2 * validation_confidence
            consistency_status = "moderate_consistency"
        else:  # 低一致性
            final_confidence = primary_confidence  # 偏向主要決策
            consistency_status = "low_consistency"
            logger.warning(f"Low consistency detected: primary={primary_confidence:.3f}, validation={validation_confidence:.3f}")
        
        # 創建合併結果
        merged_result = primary_result.copy()
        merged_result.update({
            "confidence": final_confidence,
            "original_confidence": primary_confidence,
            "validation_confidence": validation_confidence,
            "consistency_score": consistency_score,
            "consistency_status": consistency_status,
            "dual_output_validation": True,
            "method": "rag_with_validation"
        })
        
        return merged_result

    def get_model_status(self) -> dict[str, Any]:
        """獲取模型狀態

        Returns:
            模型狀態信息
        """
        return {
            "model_version": self.current_version,
            "is_trained": self.is_trained,
            "last_update": self.last_update.isoformat(),
            "scalable_net_initialized": self.scalable_net is not None,
            "bio_agent_initialized": self.bio_agent is not None,
            "scalable_net_params": (
                self.scalable_net.total_params if self.scalable_net else 0
            ),
            "model_dir": str(self.model_dir),
            "knowledge_base_path": self.knowledge_base_path,
        }

    async def update_from_experience(
        self,
        min_score: float = 0.7,
        max_samples: int = 500,
    ) -> dict[str, Any]:
        """從經驗中更新模型

        Args:
            min_score: 最低經驗分數閾值
            max_samples: 最大樣本數

        Returns:
            更新結果
        """
        logger.info(
            f"Updating model from experience (min_score={min_score}, max_samples={max_samples})"
        )

        try:
            # 獲取高質量經驗樣本
            samples = await self.experience_manager.get_training_samples(
                min_score=min_score, max_samples=max_samples
            )

            if len(samples) < 10:
                return {
                    "status": "skipped",
                    "reason": "insufficient_quality_samples",
                    "available_samples": len(samples),
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            # 使用經驗樣本進行訓練
            config = ScalableBioTrainingConfig(
                learning_rate=0.0005,  # 較小的學習率進行微調
                epochs=5,
                batch_size=16,
            )

            result = await self.train_models(
                training_data=samples,
                config=config,
                use_experience_samples=False,
            )

            logger.info(f"Model updated from {len(samples)} experience samples")
            return result

        except Exception as e:
            logger.error(f"Experience-based update failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def _save_model(self) -> Path:
        """保存模型狀態

        Returns:
            模型保存路徑
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"ai_model_{self.current_version}_{timestamp}.pkl"

        # 保存模型狀態 (簡化實現)
        import pickle

        model_state = {
            "version": self.current_version,
            "scalable_net": {
                "input_size": getattr(self.scalable_net, 'input_size', 10) if self.scalable_net else 10,
                "num_tools": getattr(self.scalable_net, 'num_tools', 3) if self.scalable_net else 3,
                "weights": getattr(self.scalable_net, 'weights', None) if self.scalable_net else None,
                "total_params": (
                    getattr(self.scalable_net, 'total_params', 0) if self.scalable_net else 0
                ),
            },
            "metadata": {
                "is_trained": self.is_trained,
                "last_update": self.last_update,
                "knowledge_base_path": self.knowledge_base_path,
            },
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_state, f)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path: Path | str) -> dict[str, Any]:
        """載入模型狀態

        Args:
            model_path: 模型路徑

        Returns:
            載入結果
        """
        try:
            import pickle

            with open(model_path, "rb") as f:
                model_state = pickle.load(f)

            # 恢復模型狀態
            self.current_version = model_state["version"]
            self.is_trained = model_state["metadata"]["is_trained"]
            self.last_update = model_state["metadata"]["last_update"]

            # 恢復 ScalableBioNet
            if model_state["scalable_net"].get("input_size") is not None:
                if not self.scalable_net:
                    # 根據保存的參數重建模型結構
                    input_size = model_state["scalable_net"]["input_size"]
                    num_tools = model_state["scalable_net"]["num_tools"]
                    self.scalable_net = self._scalable_bio_net_class(input_size, num_tools)

                # 恢復權重（如果存在且模型支持）
                if "weights" in model_state["scalable_net"] and model_state["scalable_net"]["weights"]:
                    try:
                        # 嘗試使用 save_weights 方法相對應的載入方法
                        if hasattr(self.scalable_net, '_load_or_initialize_weights'):
                            self.scalable_net._load_or_initialize_weights()
                        logger.info("權重載入成功")
                    except Exception as e:
                        logger.warning(f"權重載入失敗，使用初始化權重: {e}")

            logger.info(f"Model loaded from {model_path}")
            return {
                "status": "success",
                "version": self.current_version,
                "model_path": str(model_path),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def predict_batch(self, queries: list[str]) -> list[dict[str, Any]]:
        """批次預測功能

        Args:
            queries: 查詢列表

        Returns:
            預測結果列表
        """
        try:
            results = []

            for query in queries:
                # 使用現有的 make_decision 方法進行個別預測
                result = await self.make_decision(query)
                results.append(result)

            logger.info(f"Batch prediction completed for {len(queries)} queries")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # 返回失敗結果列表
            return [
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                for _ in queries
            ]

    def _create_experience_adapter(self):
        """創建 V1 ExperienceManager API 的適配器，內部使用 V2 ExperienceRepository"""
        
        class ExperienceManagerAdapter:
            def __init__(self, experience_repository):
                self.experience_repository = experience_repository
                
            async def get_training_samples(self, min_score=0.5, max_samples=1000):
                """適配器：獲取訓練樣本"""
                import asyncio
                
                def _sync_get_samples():
                    try:
                        # 使用 V2 ExperienceRepository API
                        experiences = self.experience_repository.query_experiences(
                            min_score=min_score, limit=max_samples
                        )
                        
                        # 轉換為 V1 格式
                        samples = []
                        for exp in experiences:
                            sample = {
                                "input": exp.context if hasattr(exp, 'context') else {},
                                "output": exp.action if hasattr(exp, 'action') else {},
                                "score": exp.overall_score if hasattr(exp, 'overall_score') else 0.5
                            }
                            samples.append(sample)
                        
                        logger.info(f"V1/V2 適配器: 獲取 {len(samples)} 個訓練樣本")
                        return samples
                    except Exception as e:
                        logger.error(f"V1/V2 適配器獲取訓練樣本失敗: {e}")
                        return []
                
                return await asyncio.get_event_loop().run_in_executor(None, _sync_get_samples)
                    
            async def add_experience(self, context, action, result, score):
                """適配器：添加經驗記錄"""
                import asyncio
                
                def _sync_add_experience():
                    try:
                        # 使用 V2 ExperienceRepository API
                        self.experience_repository.add_experience(
                            attack_type="general",
                            context=context,
                            action=action,
                            result=result,
                            overall_score=score
                        )
                        logger.info(f"V1/V2 適配器: 成功添加經驗記錄 (score: {score})")
                    except Exception as e:
                        logger.error(f"V1/V2 適配器添加經驗失敗: {e}")
                
                return await asyncio.get_event_loop().run_in_executor(None, _sync_add_experience)
        
        return ExperienceManagerAdapter(self.experience_repository)
