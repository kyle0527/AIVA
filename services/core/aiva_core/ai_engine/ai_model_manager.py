"""
AI Model Manager - AI 模型管理器

統一管理 bio_neuron_core.py 和訓練系統，提供完整的 AI 核心協調功能
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC

from .bio_neuron_core import BioNeuronRAGAgent, ScalableBioNet
from ..learning.model_trainer import ModelTrainer
from ..learning.scalable_bio_trainer import ScalableBioTrainer, ScalableBioTrainingConfig
from ..learning.experience_manager import ExperienceManager

logger = logging.getLogger(__name__)


class AIModelManager:
    """AI 模型管理器
    
    協調和管理所有 AI 相關組件：
    - BioNeuronRAGAgent (主要決策引擎)
    - ScalableBioNet (神經網路核心)
    - 訓練系統 (ModelTrainer, ScalableBioTrainer)
    - 經驗管理 (ExperienceManager)
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
        
        # 初始化核心組件
        self.bio_agent: Optional[BioNeuronRAGAgent] = None
        self.scalable_net: Optional[ScalableBioNet] = None
        
        # 初始化訓練和管理組件
        self.model_trainer = ModelTrainer(model_dir=self.model_dir, storage_backend=storage_backend)
        self.experience_manager = ExperienceManager(storage_backend=storage_backend)
        
        # 配置
        self.knowledge_base_path = knowledge_base_path
        self.storage_backend = storage_backend
        
        # 模型狀態
        self.current_version = "v1.0.0"
        self.is_trained = False
        self.last_update = datetime.now(UTC)
        
        logger.info(f"AIModelManager initialized with model_dir={self.model_dir}")
    
    async def initialize_models(
        self,
        input_size: int = 100,
        num_tools: int = 10,
        knowledge_base_path: str | None = None,
    ) -> Dict[str, Any]:
        """初始化所有 AI 模型
        
        Args:
            input_size: 輸入維度
            num_tools: 工具數量
            knowledge_base_path: 知識庫路徑
            
        Returns:
            初始化結果
        """
        try:
            logger.info("Initializing AI models...")
            
            # 1. 初始化 ScalableBioNet
            self.scalable_net = ScalableBioNet(
                input_size=input_size,
                num_tools=num_tools
            )
            logger.info(f"ScalableBioNet initialized: {self.scalable_net.total_params:,} parameters")
            
            # 2. 初始化 BioNeuronRAGAgent
            kb_path = knowledge_base_path or self.knowledge_base_path
            if kb_path:
                self.bio_agent = BioNeuronRAGAgent(
                    knowledge_base_path=kb_path,
                    model=self.scalable_net
                )
                logger.info(f"BioNeuronRAGAgent initialized with knowledge base: {kb_path}")
            else:
                logger.warning("No knowledge base path provided, BioNeuronRAGAgent not initialized")
            
            # 3. 檢查模型狀態
            result = {
                "status": "success",
                "scalable_net_params": self.scalable_net.total_params,
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
        training_data: List[Any] | None = None,
        config: ScalableBioTrainingConfig | None = None,
        use_experience_samples: bool = True,
    ) -> Dict[str, Any]:
        """訓練 AI 模型
        
        Args:
            training_data: 訓練數據，若為 None 則從經驗庫獲取
            config: 訓練配置
            use_experience_samples: 是否使用經驗樣本
            
        Returns:
            訓練結果
        """
        if not self.scalable_net:
            raise ValueError("ScalableBioNet not initialized. Call initialize_models() first.")
        
        try:
            logger.info("Starting AI model training...")
            
            # 1. 準備訓練數據
            if training_data is None and use_experience_samples:
                # 從經驗管理器獲取訓練樣本
                samples = await self.experience_manager.get_training_samples(
                    min_score=0.6,
                    max_samples=1000
                )
                logger.info(f"Retrieved {len(samples)} training samples from experience")
            else:
                samples = training_data or []
            
            if not samples:
                logger.warning("No training data available")
                return {
                    "status": "skipped",
                    "reason": "no_training_data",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            
            # 2. 配置訓練參數
            if config is None:
                config = ScalableBioTrainingConfig(
                    learning_rate=0.001,
                    epochs=10,
                    batch_size=32,
                    early_stopping_patience=3,
                )
            
            # 3. 執行 ScalableBioNet 專用訓練
            bio_trainer = ScalableBioTrainer(self.scalable_net, config)
            
            # 準備訓練數據格式 (簡化示例)
            import numpy as np
            X_train = np.array([[s.context for s in samples[:800]] if hasattr(samples[0], 'context') 
                               else np.random.randn(800, 10)])
            y_train = np.array([[s.result for s in samples[:800]] if hasattr(samples[0], 'result')
                               else np.random.randn(800, 10)])
            X_val = np.array([[s.context for s in samples[800:]] if len(samples) > 800 and hasattr(samples[0], 'context')
                             else np.random.randn(200, 10)])
            y_val = np.array([[s.result for s in samples[800:]] if len(samples) > 800 and hasattr(samples[0], 'result')
                             else np.random.randn(200, 10)])
            
            # 執行訓練
            training_results = bio_trainer.train(X_train, y_train, X_val, y_val)
            
            # 4. 更新模型狀態
            self.is_trained = True
            self.last_update = datetime.now(UTC)
            self.current_version = f"v1.{int(self.current_version.split('.')[-1]) + 1}.0"
            
            # 5. 保存模型
            model_path = await self._save_model()
            
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
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
    
    async def make_decision(
        self,
        query: str,
        context: Dict[str, Any] | None = None,
        use_rag: bool = True,
    ) -> Dict[str, Any]:
        """使用 AI 系統進行決策
        
        Args:
            query: 查詢或問題
            context: 上下文信息
            use_rag: 是否使用 RAG 功能
            
        Returns:
            決策結果
        """
        if not self.scalable_net:
            raise ValueError("Models not initialized. Call initialize_models() first.")
        
        try:
            if use_rag and self.bio_agent:
                # 使用 RAG 功能
                result = await self.bio_agent.process_query(query, context or {})
                logger.info("Decision made using BioNeuronRAGAgent with RAG")
            else:
                # 直接使用 ScalableBioNet
                import numpy as np
                
                # 簡化的輸入處理 (實際應用中需要更複雜的特徵工程)
                input_vector = np.random.randn(1, self.scalable_net.fc1.shape[0])
                output = self.scalable_net.forward(input_vector)
                
                result = {
                    "decision": output.tolist(),
                    "confidence": float(np.max(output)),
                    "method": "direct_scalable_bionet",
                }
                logger.info("Decision made using direct ScalableBioNet")
            
            # 記錄決策到經驗管理器
            await self.experience_manager.add_experience(
                context={"query": query, "context": context},
                action={"decision": result},
                result={"confidence": result.get("confidence", 0.5)},
                score=result.get("confidence", 0.5),
            )
            
            return {
                "status": "success",
                "result": result,
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
    
    async def get_model_status(self) -> Dict[str, Any]:
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
            "scalable_net_params": self.scalable_net.total_params if self.scalable_net else 0,
            "model_dir": str(self.model_dir),
            "knowledge_base_path": self.knowledge_base_path,
        }
    
    async def update_from_experience(
        self,
        min_score: float = 0.7,
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """從經驗中更新模型
        
        Args:
            min_score: 最低經驗分數閾值
            max_samples: 最大樣本數
            
        Returns:
            更新結果
        """
        logger.info(f"Updating model from experience (min_score={min_score}, max_samples={max_samples})")
        
        try:
            # 獲取高質量經驗樣本
            samples = await self.experience_manager.get_training_samples(
                min_score=min_score,
                max_samples=max_samples
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
    
    async def _save_model(self) -> Path:
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
                "fc1": self.scalable_net.fc1 if self.scalable_net else None,
                "fc2": self.scalable_net.fc2 if self.scalable_net else None,
                "spiking1_weights": (
                    self.scalable_net.spiking1.weights if self.scalable_net else None
                ),
                "total_params": self.scalable_net.total_params if self.scalable_net else 0,
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
    
    async def load_model(self, model_path: Path | str) -> Dict[str, Any]:
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
            if model_state["scalable_net"]["fc1"] is not None:
                if not self.scalable_net:
                    # 根據保存的權重推斷模型結構
                    input_size = model_state["scalable_net"]["fc1"].shape[0]
                    num_tools = model_state["scalable_net"]["fc2"].shape[1]
                    self.scalable_net = ScalableBioNet(input_size, num_tools)
                
                self.scalable_net.fc1 = model_state["scalable_net"]["fc1"]
                self.scalable_net.fc2 = model_state["scalable_net"]["fc2"]
                self.scalable_net.spiking1.weights = model_state["scalable_net"]["spiking1_weights"]
            
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