#!/usr/bin/env python3
"""
Training Coordinator

協調訓練相關任務
負責模型訓練、經驗學習、知識更新
"""

import logging
from typing import Dict, Any, List

from .base_coordinator import BaseCoordinator, CoordinatorTask

logger = logging.getLogger(__name__)


class TrainingCoordinator(BaseCoordinator):
    """訓練協調器
    
    負責協調：
    - 訓練數據準備 (DataHub)
    - 外部知識學習 (Learning)
    - 模型微調 (BioNeuron)
    - 經驗總結
    """
    
    def __init__(self, module_registry):
        super().__init__(module_registry, name="TrainingCoordinator")
    
    async def decompose_task(self, task: CoordinatorTask) -> List[Dict[str, Any]]:
        """分解訓練任務
        
        Args:
            task: 訓練任務
        
        Returns:
            子任務列表
        """
        training_type = task.parameters.get("training_type", "fine_tune")
        
        logger.info(f"Decomposing training task: {training_type}")
        
        subtasks = []
        
        if training_type == "fine_tune":
            # 模型微調流程
            subtasks.extend(self._create_fine_tune_subtasks(task.parameters))
        
        elif training_type == "knowledge_update":
            # 知識庫更新流程
            subtasks.extend(self._create_knowledge_update_subtasks(task.parameters))
        
        elif training_type == "experience_learning":
            # 經驗學習流程
            subtasks.extend(self._create_experience_learning_subtasks(task.parameters))
        
        else:
            # 通用訓練
            subtasks.append({
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "TRAIN",
                    "description": f"Execute {training_type} training",
                    "training_type": training_type
                }
            })
        
        return subtasks
    
    def _create_fine_tune_subtasks(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """創建模型微調子任務"""
        return [
            {
                "module_id": "data_hub",
                "parameters": {
                    "description": "Prepare training dataset",
                    "task_type": "prepare_dataset",
                    "dataset_type": parameters.get("dataset_type", "attack_traces"),
                    "min_samples": parameters.get("min_samples", 1000)
                }
            },
            {
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "TRAIN",
                    "description": "Fine-tune BioNeuron model",
                    "training_mode": "fine_tune",
                    "epochs": parameters.get("epochs", 5),
                    "learning_rate": parameters.get("learning_rate", 1e-5)
                }
            }
        ]
    
    def _create_knowledge_update_subtasks(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """創建知識庫更新子任務"""
        return [
            {
                "module_id": "learning",
                "parameters": {
                    "description": "Learn from external sources",
                    "source": parameters.get("source", ""),
                    "source_type": parameters.get("source_type", "url")
                }
            },
            {
                "module_id": "learning",
                "parameters": {
                    "description": "Update knowledge base",
                    "update_type": "incremental",
                    "verify": True
                }
            }
        ]
    
    def _create_experience_learning_subtasks(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """創建經驗學習子任務"""
        return [
            {
                "module_id": "data_hub",
                "parameters": {
                    "description": "Query recent experiences",
                    "query_type": "experiences",
                    "time_range": parameters.get("time_range", "last_7_days"),
                    "min_score": parameters.get("min_score", 0.7)
                }
            },
            {
                "module_id": "bio_neuron",
                "parameters": {
                    "task_type": "LEARN",
                    "description": "Learn from experiences",
                    "learning_mode": "experience_based",
                    "consolidation": True
                }
            }
        ]
    
    async def aggregate_results(
        self, 
        task: CoordinatorTask, 
        subtask_results: List[Any]
    ) -> Any:
        """聚合訓練結果
        
        Args:
            task: 原始任務
            subtask_results: 子任務結果列表
        
        Returns:
            聚合後的訓練報告
        """
        logger.info(f"Aggregating training results for task {task.task_id}")
        
        training_report = {
            "task_id": task.task_id,
            "training_type": task.parameters.get("training_type", "unknown"),
            "stages": [],
            "metrics": {},
            "success": True
        }
        
        # 解析子任務結果
        for i, result in enumerate(subtask_results):
            stage_data = {}
            
            if hasattr(result, 'data'):
                stage_data = {
                    "stage_index": i,
                    "success": result.success,
                    "data": result.data,
                    "execution_time": result.execution_time,
                    "metrics": result.metrics if hasattr(result, 'metrics') else {}
                }
                
                # 檢查是否失敗
                if not result.success:
                    training_report["success"] = False
            else:
                stage_data = {
                    "stage_index": i,
                    "success": True,
                    "data": result
                }
            
            training_report["stages"].append(stage_data)
        
        # 聚合指標
        training_report["metrics"] = self._aggregate_training_metrics(
            training_report["stages"]
        )
        
        return training_report
    
    def _aggregate_training_metrics(self, stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合訓練指標"""
        metrics = {
            "total_stages": len(stages),
            "successful_stages": sum(1 for s in stages if s.get("success", False)),
            "total_time": sum(s.get("execution_time", 0) for s in stages),
            "stage_metrics": []
        }
        
        # 提取各階段指標
        for stage in stages:
            stage_metrics = stage.get("metrics", {})
            if stage_metrics:
                metrics["stage_metrics"].append(stage_metrics)
        
        return metrics
