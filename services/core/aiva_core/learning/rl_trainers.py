"""
強化學習訓練器

實現 DQN 和 PPO 算法的訓練流程
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .rl_models import ActorCritic, DQNNetwork, ReplayBuffer, RolloutBuffer

logger = logging.getLogger(__name__)


class DQNTrainer:
    """DQN (Deep Q-Network) 訓練器
    
    實現 DQN 算法的訓練邏輯
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str | None = None,
    ) -> None:
        """初始化 DQN 訓練器
        
        Args:
            state_dim: 狀態維度
            action_dim: 動作維度
            learning_rate: 學習率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最終探索率
            epsilon_decay: 探索率衰減
            buffer_capacity: 經驗回放緩衝區容量
            batch_size: 批次大小
            target_update_freq: 目標網絡更新頻率
            device: 計算設備 (cpu/cuda)
        """
        # 設備
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"DQN Trainer using device: {self.device}")
        
        # 網絡
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 優化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 經驗回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 超參數
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 統計
        self.steps = 0
        self.losses: list[float] = []
        
        logger.info(
            f"DQN initialized: state_dim={state_dim}, action_dim={action_dim}, "
            f"lr={learning_rate}, gamma={gamma}"
        )
    
    def select_action(self, state: np.ndarray) -> int:
        """選擇動作 (ε-greedy)
        
        Args:
            state: 當前狀態
        
        Returns:
            選中的動作
        """
        action, _ = self.policy_net.select_action(state, epsilon=self.epsilon)
        return action
    
    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        """訓練一步
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            done: 是否結束
        
        Returns:
            損失值（如果執行了訓練）
        """
        # 存儲經驗
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # 檢查是否可以訓練
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 採樣批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # 移動到設備
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 計算當前 Q 值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 計算目標 Q 值 (Double DQN)
        with torch.no_grad():
            # 使用 policy net 選擇動作
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # 使用 target net 評估價值
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (
                1 - dones.unsqueeze(1)
            )
        
        # 計算損失 (Huber Loss)
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新統計
        self.steps += 1
        self.losses.append(loss.item())
        
        # 更新 target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(f"Target network updated at step {self.steps}")
        
        # 更新 epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def get_metrics(self) -> dict[str, Any]:
        """獲取訓練指標
        
        Returns:
            訓練指標字典
        """
        return {
            "steps": self.steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0.0,
            "recent_losses": self.losses[-10:],
        }
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路徑
        """
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "epsilon": self.epsilon,
            },
            path,
        )
        logger.info(f"DQN model saved to {path}")
    
    def load(self, path: str) -> None:
        """加載模型
        
        Args:
            path: 模型路徑
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint.get("steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        logger.info(f"DQN model loaded from {path}")


class PPOTrainer:
    """PPO (Proximal Policy Optimization) 訓練器
    
    實現 PPO 算法的訓練邏輯
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        """初始化 PPO 訓練器
        
        Args:
            state_dim: 狀態維度
            action_dim: 動作維度
            learning_rate: 學習率
            gamma: 折扣因子
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip 參數
            value_coef: 價值損失係數
            entropy_coef: 熵正則化係數
            max_grad_norm: 梯度裁剪
            ppo_epochs: PPO 更新次數
            mini_batch_size: mini-batch 大小
            device: 計算設備
        """
        # 設備
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"PPO Trainer using device: {self.device}")
        
        # 網絡
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        
        # 優化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()
        
        # 超參數
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # 統計
        self.updates = 0
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropies: list[float] = []
        
        logger.info(
            f"PPO initialized: state_dim={state_dim}, action_dim={action_dim}, "
            f"lr={learning_rate}, gamma={gamma}, clip_eps={clip_epsilon}"
        )
    
    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """選擇動作
        
        Args:
            state: 當前狀態
            deterministic: 是否確定性選擇
        
        Returns:
            (動作, log 概率, 狀態價值)
        """
        return self.actor_critic.select_action(state, deterministic)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """存儲轉換
        
        Args:
            state: 狀態
            action: 動作
            reward: 獎勵
            log_prob: log 概率
            value: 狀態價值
            done: 是否結束
        """
        self.rollout_buffer.push(state, action, reward, log_prob, value, done)
    
    def update(self) -> dict[str, float]:
        """PPO 更新
        
        Returns:
            更新指標
        """
        if len(self.rollout_buffer) == 0:
            logger.warning("Rollout buffer is empty, skipping update")
            return {}
        
        # 獲取數據
        states, actions, rewards, old_log_probs, old_values = self.rollout_buffer.get()
        
        # 計算回報和優勢
        returns, advantages = self.rollout_buffer.compute_returns(
            self.gamma, self.gae_lambda
        )
        
        # 標準化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 移動到設備
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # PPO 更新多次
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.ppo_epochs):
            # Mini-batch 訓練
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                # 評估動作
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    states[batch_indices], actions[batch_indices]
                )
                
                # 計算比率
                ratios = torch.exp(log_probs - old_log_probs[batch_indices])
                
                # Clipped surrogate objective
                surr1 = ratios * advantages[batch_indices]
                surr2 = (
                    torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages[batch_indices]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 價值損失
                value_loss = nn.functional.mse_loss(values, returns[batch_indices])
                
                # 總損失
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )
                
                # 反向傳播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                
                # 累計統計
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # 平均損失
        num_updates = self.ppo_epochs * (len(states) // self.mini_batch_size + 1)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        # 更新統計
        self.updates += 1
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)
        
        # 清空 buffer
        self.rollout_buffer.clear()
        
        logger.debug(
            f"PPO update #{self.updates}: "
            f"policy_loss={avg_policy_loss:.4f}, "
            f"value_loss={avg_value_loss:.4f}, "
            f"entropy={avg_entropy:.4f}"
        )
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "updates": self.updates,
        }
    
    def get_metrics(self) -> dict[str, Any]:
        """獲取訓練指標
        
        Returns:
            訓練指標字典
        """
        return {
            "updates": self.updates,
            "buffer_size": len(self.rollout_buffer),
            "avg_policy_loss": (
                np.mean(self.policy_losses[-10:]) if self.policy_losses else 0.0
            ),
            "avg_value_loss": (
                np.mean(self.value_losses[-10:]) if self.value_losses else 0.0
            ),
            "avg_entropy": np.mean(self.entropies[-10:]) if self.entropies else 0.0,
            "recent_policy_losses": self.policy_losses[-5:],
            "recent_value_losses": self.value_losses[-5:],
        }
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路徑
        """
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": self.updates,
            },
            path,
        )
        logger.info(f"PPO model saved to {path}")
    
    def load(self, path: str) -> None:
        """加載模型
        
        Args:
            path: 模型路徑
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.updates = checkpoint.get("updates", 0)
        logger.info(f"PPO model loaded from {path}")
