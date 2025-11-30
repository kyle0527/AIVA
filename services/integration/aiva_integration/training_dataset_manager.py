#!/usr/bin/env python3
"""
Training Dataset Manager

訓練數據集管理器
負責準備、管理和組織用於 AI 模型訓練的數據集
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil

logger = logging.getLogger(__name__)


class TrainingDatasetManager:
    """訓練數據集管理器
    
    功能：
    - 從經驗庫準備訓練數據
    - 數據集版本管理
    - 數據集格式轉換
    - 數據集統計和驗證
    
    使用方式：
        manager = TrainingDatasetManager(dataset_dir="data/training")
        dataset_path = await manager.prepare_dataset(
            task_type="vulnerability_detection",
            min_samples=1000,
            split_ratio={"train": 0.7, "val": 0.15, "test": 0.15}
        )
    """
    
    def __init__(self, dataset_dir: str = "data/training"):
        """
        Args:
            dataset_dir: 數據集存儲目錄
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 數據集元數據存儲
        self.metadata_file = self.dataset_dir / "datasets_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"TrainingDatasetManager initialized with directory: {self.dataset_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """載入數據集元數據"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return {"datasets": {}}
        return {"datasets": {}}
    
    def _save_metadata(self):
        """保存數據集元數據"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def prepare_dataset(
        self,
        task_type: str,
        min_samples: int = 1000,
        split_ratio: Dict[str, float] | None = None,
        data_source: str = "experience_repository",
        filters: Dict[str, Any] | None = None
    ) -> Path:
        """準備訓練數據集
        
        Args:
            task_type: 任務類型（如 "vulnerability_detection", "attack_planning"）
            min_samples: 最小樣本數
            split_ratio: 數據集分割比例 {"train": 0.7, "val": 0.15, "test": 0.15}
            data_source: 數據來源
            filters: 數據過濾條件
        
        Returns:
            數據集目錄路徑
        """
        if split_ratio is None:
            split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}
        if filters is None:
            filters = {}
            split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}
        
        if filters is None:
            filters = {}
        
        logger.info(f"Preparing dataset for task: {task_type}, min_samples: {min_samples}")
        
        # 創建數據集目錄
        dataset_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = self.dataset_dir / dataset_id
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 收集數據
            samples = await self._collect_samples(
                task_type=task_type,
                min_samples=min_samples,
                data_source=data_source,
                filters=filters
            )
            
            logger.info(f"Collected {len(samples)} samples")
            
            if len(samples) < min_samples:
                logger.warning(f"Only {len(samples)} samples collected, less than minimum {min_samples}")
            
            # 2. 數據預處理
            processed_samples = await self._preprocess_samples(samples, task_type)
            
            # 3. 分割數據集
            splits = self._split_dataset(processed_samples, split_ratio)
            
            # 4. 保存數據集
            for split_name, split_data in splits.items():
                split_file = dataset_path / f"{split_name}.jsonl"
                self._save_jsonl(split_data, split_file)
                logger.info(f"Saved {split_name} split: {len(split_data)} samples")
            
            # 5. 保存配置和統計信息
            config = {
                "dataset_id": dataset_id,
                "task_type": task_type,
                "created_at": datetime.now().isoformat(),
                "total_samples": len(processed_samples),
                "split_ratio": split_ratio,
                "splits": {name: len(data) for name, data in splits.items()},
                "data_source": data_source,
                "filters": filters
            }
            
            config_file = dataset_path / "config.json"
            # 使用同步 I/O 因為 JSON 文件很小且處理很快
            with open(config_file, 'w', encoding='utf-8') as f:  # noqa: ASYNC230
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 6. 更新元數據
            self.metadata["datasets"][dataset_id] = {
                "task_type": task_type,
                "path": str(dataset_path),
                "created_at": config["created_at"],
                "total_samples": config["total_samples"],
                "status": "ready"
            }
            self._save_metadata()
            
            logger.info(f"✅ Dataset prepared successfully: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}", exc_info=True)
            # 清理失敗的數據集
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            raise
    
    async def _collect_samples(
        self,
        task_type: str,
        min_samples: int,
        data_source: str,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """從數據源收集樣本"""
        samples = []
        
        if data_source == "experience_repository":
            # 從經驗庫收集
            try:
                from .reception.experience_repository import ExperienceRepository
                
                repo = ExperienceRepository("sqlite:///aiva_operations.sqlite")
                experiences = await repo.query_high_quality_experiences(  # type: ignore[attr-defined]
                    min_score=filters.get("min_score", 0.0),
                    limit=min_samples * 2  # 收集更多以便後續過濾
                )
                
                # 轉換為訓練樣本格式
                for exp in experiences:
                    sample = self._experience_to_sample(exp, task_type)
                    if sample:
                        samples.append(sample)
                
            except ImportError:
                logger.warning("ExperienceRepository not available")
        
        elif data_source == "synthetic":
            # 生成合成數據（用於測試）
            samples = self._generate_synthetic_samples(task_type, min_samples)
        
        return samples[:min_samples * 2]  # 返回足夠的樣本
    
    def _experience_to_sample(
        self,
        experience: Dict[str, Any],
        task_type: str
    ) -> Optional[Dict[str, Any]]:
        """將經驗記錄轉換為訓練樣本"""
        try:
            # 根據任務類型提取相關字段
            if task_type == "vulnerability_detection":
                return {
                    "input": {
                        "code": experience.get("ast_graph", {}),
                        "context": experience.get("target_info", {})
                    },
                    "output": {
                        "vulnerabilities": experience.get("metrics", {}).get("vulnerabilities", [])
                    },
                    "metadata": {
                        "experience_id": experience.get("id"),
                        "attack_type": experience.get("attack_type"),
                        "score": experience.get("score", 0.0)
                    }
                }
            
            elif task_type == "attack_planning":
                return {
                    "input": {
                        "target": experience.get("target_info", {}),
                        "constraints": experience.get("metadata", {})
                    },
                    "output": {
                        "plan": experience.get("execution_trace", {})
                    },
                    "metadata": {
                        "experience_id": experience.get("id"),
                        "success": experience.get("feedback", {}).get("success", False)
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert experience to sample: {e}")
            return None
    
    def _generate_synthetic_samples(
        self,
        task_type: str,
        count: int
    ) -> List[Dict[str, Any]]:
        """生成合成訓練樣本（用於測試）"""
        samples = []
        
        for i in range(count):
            sample = {
                "input": {"synthetic": True, "id": i, "task_type": task_type},
                "output": {"result": f"synthetic_{task_type}_{i}"},
                "metadata": {"generated": True, "task_type": task_type}
            }
            samples.append(sample)
        
        return samples
    
    async def _preprocess_samples(  # type: ignore[misc]
        self,
        samples: List[Dict[str, Any]],
        task_type: str
    ) -> List[Dict[str, Any]]:
        """預處理樣本數據 - async保留供未來異步處理擴展"""
        processed = []
        
        for sample in samples:
            try:
                # 數據清洗
                if not sample.get("input") or not sample.get("output"):
                    continue
                
                # 數據標準化
                processed_sample = {
                    "input": sample["input"],
                    "output": sample["output"],
                    "task_type": task_type,  # 添加 task_type 到樣本
                    "metadata": sample.get("metadata", {})
                }
                
                processed.append(processed_sample)
                
            except Exception as e:
                logger.warning(f"Failed to preprocess sample: {e}")
                continue
        
        return processed
    
    def _split_dataset(
        self,
        samples: List[Dict[str, Any]],
        split_ratio: Dict[str, float]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """分割數據集為訓練、驗證、測試集"""
        import random
        
        # 打亂樣本
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        splits = {}
        
        start_idx = 0
        for split_name, ratio in split_ratio.items():
            count = int(total * ratio)
            end_idx = start_idx + count
            
            if split_name == list(split_ratio.keys())[-1]:
                # 最後一個分割包含所有剩餘樣本
                end_idx = total
            
            splits[split_name] = shuffled[start_idx:end_idx]
            start_idx = end_idx
        
        return splits
    
    def _save_jsonl(self, data: List[Dict[str, Any]], file_path: Path):
        """保存為 JSONL 格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def list_datasets(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有數據集
        
        Args:
            task_type: 可選，篩選特定任務類型的數據集
        
        Returns:
            數據集信息列表
        """
        datasets = []
        
        for dataset_id, info in self.metadata["datasets"].items():
            if task_type is None or info.get("task_type") == task_type:
                datasets.append({
                    "dataset_id": dataset_id,
                    **info
                })
        
        return datasets
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """獲取數據集詳細信息
        
        Args:
            dataset_id: 數據集 ID
        
        Returns:
            數據集信息或 None
        """
        return self.metadata["datasets"].get(dataset_id)
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """刪除數據集
        
        Args:
            dataset_id: 數據集 ID
        
        Returns:
            是否成功刪除
        """
        if dataset_id not in self.metadata["datasets"]:
            logger.warning(f"Dataset {dataset_id} not found")
            return False
        
        try:
            # 刪除數據集目錄
            dataset_info = self.metadata["datasets"][dataset_id]
            dataset_path = Path(dataset_info["path"])
            
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                logger.info(f"Deleted dataset directory: {dataset_path}")
            
            # 更新元數據
            del self.metadata["datasets"][dataset_id]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            return False
