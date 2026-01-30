"""
Online Learner - 線上學習器

實現即時權重更新的線上學習機制，用於統一反饋架構。
支援單步梯度下降和小批次更新。

設計理念：
- 即時反饋：每次執行後立即更新模型
- 小步更新：使用小學習率避免災難性遺忘
- 適用場景：靶場環境的探索式學習

日期: 2025-12-26
版本: 2.0.0
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from aiva_common.utils import get_logger

logger = get_logger(__name__)


class OnlineLearner:
    """線上學習器
    
    提供即時權重更新能力，適用於：
    1. 靶場環境的探索式學習
    2. 快速適應新環境
    3. 增量學習場景
    
    特點：
    - 單樣本更新：每個經驗立即學習
    - 小學習率：防止災難性遺忘（lr=0.0001）
    - 梯度裁剪：避免梯度爆炸
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.0001,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        """初始化線上學習器
        
        Args:
            model: PyTorch 神經網絡模型
            learning_rate: 學習率（預設 0.0001，避免過度更新）
            max_grad_norm: 最大梯度範數（用於梯度裁剪）
            device: 運算裝置 ("cpu" 或 "cuda")
        """
        self.model = model
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # 優化器：使用 Adam（適合線上學習）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0001  # 添加權重衰減以防止過擬合
        )
        
        # 損失函數（根據任務類型可調整）
        self.criterion = nn.MSELoss()  # 回歸任務
        
        # 統計資訊
        self._total_updates = 0
        self._total_loss = 0.0
        
        logger.info(
            f"✅ OnlineLearner initialized: lr={learning_rate}, "
            f"device={device}"
        )
    
    def update_from_experience(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        weight: float = 1.0
    ) -> dict[str, Any]:
        """從單個經驗更新模型
        
        Args:
            state: 輸入狀態張量 (batch_size=1)
            target: 目標輸出張量
            weight: 樣本權重（用於優先級學習）
        
        Returns:
            更新資訊字典
        """
        self.model.train()
        
        # 移動到裝置
        state = state.to(self.device)
        target = target.to(self.device)
        
        # 前向傳播
        prediction = self.model(state)
        
        # 計算損失
        loss = self.criterion(prediction, target) * weight
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        
        # 更新權重
        self.optimizer.step()
        
        # 統計
        self._total_updates += 1
        self._total_loss += loss.item()
        
        logger.debug(
            f"Online update #{self._total_updates}: loss={loss.item():.4f}"
        )
        
        return {
            "loss": loss.item(),
            "total_updates": self._total_updates,
            "avg_loss": self._total_loss / self._total_updates
        }
    
    def update_from_batch(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        """從小批次更新模型
        
        Args:
            states: 輸入狀態張量 (batch_size, *)
            targets: 目標輸出張量 (batch_size, *)
            weights: 樣本權重張量 (batch_size,)
        
        Returns:
            更新資訊字典
        """
        self.model.train()
        
        # 移動到裝置
        states = states.to(self.device)
        targets = targets.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        
        # 前向傳播
        predictions = self.model(states)
        
        # 計算損失
        loss = self.criterion(predictions, targets)
        
        # 應用權重
        if weights is not None:
            loss = (loss * weights).mean()
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        
        # 更新權重
        self.optimizer.step()
        
        # 統計
        batch_size = states.size(0)
        self._total_updates += batch_size
        self._total_loss += loss.item() * batch_size
        
        logger.debug(
            f"Batch update ({batch_size} samples): loss={loss.item():.4f}"
        )
        
        return {
            "loss": loss.item(),
            "batch_size": batch_size,
            "total_updates": self._total_updates,
            "avg_loss": self._total_loss / self._total_updates
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取學習統計資訊
        
        Returns:
            統計資訊字典
        """
        return {
            "total_updates": self._total_updates,
            "total_loss": self._total_loss,
            "avg_loss": self._total_loss / max(self._total_updates, 1),
            "learning_rate": self.learning_rate,
            "device": self.device
        }
    
    def reset_statistics(self) -> None:
        """重置統計資訊"""
        self._total_updates = 0
        self._total_loss = 0.0
        logger.info("Statistics reset")


def create_online_learner(
    model: nn.Module,
    learning_rate: float = 0.0001,
    device: str = "cpu"
) -> OnlineLearner:
    """創建線上學習器的便捷函數
    
    Args:
        model: PyTorch 模型
        learning_rate: 學習率
        device: 運算裝置
    
    Returns:
        OnlineLearner 實例
    """
    return OnlineLearner(
        model=model,
        learning_rate=learning_rate,
        device=device
    )

