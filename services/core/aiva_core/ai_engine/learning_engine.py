"""
Learning Engine - 學習引擎
實現各種機器學習演算法和訓練策略

這個模組提供了：
- 監督學習演算法
- 強化學習機制
- 在線學習能力
- 經驗重播系統
- 遷移學習支援
"""

from __future__ import annotations

import logging
import time
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from .neural_network import FeedForwardNetwork

logger = logging.getLogger(__name__)


class LossFunction:
    """損失函數集合"""
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方誤差"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """交叉熵損失"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """二元交叉熵損失"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class Optimizer:
    """優化器基類"""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
    
    def update(self, network, gradients: Dict[str, np.ndarray]):
        """更新網路參數"""
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    """隨機梯度下降優化器"""
    
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, network, gradients: Dict[str, np.ndarray]):
        """更新網路參數"""
        for layer_idx, layer in enumerate(network.layers):
            if layer_idx not in self.velocity:
                self.velocity[layer_idx] = {
                    'weights': np.zeros_like(layer.weights),
                    'biases': np.zeros_like(layer.biases)
                }
            
            # 動量更新
            self.velocity[layer_idx]['weights'] = (
                self.momentum * self.velocity[layer_idx]['weights'] - 
                self.learning_rate * layer.weights_grad
            )
            self.velocity[layer_idx]['biases'] = (
                self.momentum * self.velocity[layer_idx]['biases'] - 
                self.learning_rate * layer.biases_grad
            )
            
            # 更新參數
            layer.weights += self.velocity[layer_idx]['weights']
            layer.biases += self.velocity[layer_idx]['biases']


class AdamOptimizer(Optimizer):
    """Adam 優化器"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一階動量
        self.v = {}  # 二階動量
        self.t = 0   # 時間步
    
    def update(self, network, gradients: Dict[str, np.ndarray]):
        """更新網路參數"""
        self.t += 1
        
        for layer_idx, layer in enumerate(network.layers):
            if layer_idx not in self.m:
                self.m[layer_idx] = {
                    'weights': np.zeros_like(layer.weights),
                    'biases': np.zeros_like(layer.biases)
                }
                self.v[layer_idx] = {
                    'weights': np.zeros_like(layer.weights),
                    'biases': np.zeros_like(layer.biases)
                }
            
            # 更新權重的動量
            self.m[layer_idx]['weights'] = self.beta1 * self.m[layer_idx]['weights'] + (1 - self.beta1) * layer.weights_grad
            self.v[layer_idx]['weights'] = self.beta2 * self.v[layer_idx]['weights'] + (1 - self.beta2) * (layer.weights_grad ** 2)
            
            # 更新偏置的動量
            self.m[layer_idx]['biases'] = self.beta1 * self.m[layer_idx]['biases'] + (1 - self.beta1) * layer.biases_grad
            self.v[layer_idx]['biases'] = self.beta2 * self.v[layer_idx]['biases'] + (1 - self.beta2) * (layer.biases_grad ** 2)
            
            # 偏差修正
            m_hat_weights = self.m[layer_idx]['weights'] / (1 - self.beta1 ** self.t)
            v_hat_weights = self.v[layer_idx]['weights'] / (1 - self.beta2 ** self.t)
            
            m_hat_biases = self.m[layer_idx]['biases'] / (1 - self.beta1 ** self.t)
            v_hat_biases = self.v[layer_idx]['biases'] / (1 - self.beta2 ** self.t)
            
            # 更新參數
            layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
            layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)


class ExperienceBuffer:
    """經驗重播緩衝區"""
    
    def __init__(self, capacity: int = 10000):
        """
        初始化經驗緩衝區
        
        Args:
            capacity: 緩衝區容量
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: Any, reward: float, next_state: np.ndarray, done: bool):
        """添加經驗"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """隨機採樣經驗"""
        if len(self.buffer) < batch_size:
            return self.buffer
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class SupervisedLearner:
    """監督學習器"""
    
    def __init__(self, network, optimizer: Optimizer, loss_function: str = 'mse'):
        """
        初始化監督學習器
        
        Args:
            network: 神經網路
            optimizer: 優化器
            loss_function: 損失函數類型
        """
        self.network = network
        self.optimizer = optimizer
        
        # 損失函數映射
        self.loss_functions = {
            'mse': LossFunction.mean_squared_error,
            'cross_entropy': LossFunction.cross_entropy,
            'binary_cross_entropy': LossFunction.binary_cross_entropy
        }
        self.loss_function = self.loss_functions.get(loss_function, LossFunction.mean_squared_error)
        
        # 訓練歷史
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
    
    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """訓練一個批次"""
        # 前向傳播
        predictions = self.network.forward(x_batch)
        
        # 計算損失
        loss = self.loss_function(y_batch, predictions)
        
        # 反向傳播
        grad_output = predictions - y_batch  # 簡化的梯度計算
        self.network.backward(grad_output)
        
        # 更新參數
        self.optimizer.update(self.network, {})
        
        return loss
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, verbose: bool = True) -> Dict[str, List[float]]:
        """
        訓練模型
        
        Args:
            x_train: 訓練資料
            y_train: 訓練標籤
            x_val: 驗證資料
            y_val: 驗證標籤
            epochs: 訓練世代數
            batch_size: 批次大小
            verbose: 是否顯示訓練過程
            
        Returns:
            訓練歷史
        """
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # 隨機打亂資料
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss = self.train_batch(x_batch, y_batch)
                epoch_loss += batch_loss
            
            avg_loss = epoch_loss / n_batches
            self.training_history['loss'].append(avg_loss)
            
            # 驗證
            if x_val is not None and y_val is not None:
                val_predictions = self.network.predict(x_val)
                val_loss = self.loss_function(y_val, val_predictions)
                
                # 計算準確率（分類任務）
                if len(y_val.shape) > 1:
                    val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == np.argmax(y_val, axis=1))
                    self.training_history['accuracy'].append(val_accuracy)
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.training_history


class ReinforcementLearner:
    """強化學習器"""
    
    def __init__(self, network, optimizer: Optimizer, gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        """
        初始化強化學習器
        
        Args:
            network: 神經網路
            optimizer: 優化器
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰減
        """
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        self.experience_buffer = ExperienceBuffer()
        
        # 訓練統計
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'average_reward': 0
        }
    
    def select_action(self, state: np.ndarray, num_actions: int) -> int:
        """選擇動作（ε-greedy策略）"""
        if np.random.random() < self.epsilon:
            return np.random.randint(num_actions)
        else:
            q_values = self.network.predict(state.reshape(1, -1))
            return np.argmax(q_values[0])
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """存儲經驗"""
        self.experience_buffer.push(state, action, reward, next_state, done)
    
    def replay_train(self, batch_size: int = 32):
        """經驗重播訓練"""
        if len(self.experience_buffer) < batch_size:
            return
        
        batch = self.experience_buffer.sample(batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # 計算目標Q值
        current_q_values = self.network.predict(states)
        next_q_values = self.network.predict(next_states)
        
        target_q_values = current_q_values.copy()
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # 訓練網路
        grad_output = current_q_values - target_q_values
        self.network.backward(grad_output)
        self.optimizer.update(self.network, {})
        
        # 衰減探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class OnlineLearner:
    """在線學習器"""
    
    def __init__(self, network, optimizer: Optimizer, forgetting_factor: float = 0.95):
        """
        初始化在線學習器
        
        Args:
            network: 神經網路
            optimizer: 優化器
            forgetting_factor: 遺忘因子
        """
        self.network = network
        self.optimizer = optimizer
        self.forgetting_factor = forgetting_factor
        
        # 在線統計
        self.sample_count = 0
        self.running_mean_loss = 0
    
    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        """在線更新"""
        # 前向傳播
        prediction = self.network.forward(x.reshape(1, -1))
        
        # 計算損失
        loss = np.mean((y.reshape(1, -1) - prediction) ** 2)
        
        # 更新運行平均損失
        self.sample_count += 1
        alpha = 1 / self.sample_count if self.sample_count < 100 else 0.01
        self.running_mean_loss = (1 - alpha) * self.running_mean_loss + alpha * loss
        
        # 反向傳播和更新
        grad_output = prediction - y.reshape(1, -1)
        self.network.backward(grad_output)
        self.optimizer.update(self.network, {})
        
        return loss
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """獲取性能指標"""
        return {
            'sample_count': self.sample_count,
            'running_mean_loss': self.running_mean_loss,
            'learning_rate': self.optimizer.learning_rate
        }


class TransferLearner:
    """遷移學習器"""
    
    def __init__(self, source_network, target_network):
        """
        初始化遷移學習器
        
        Args:
            source_network: 源網路
            target_network: 目標網路
        """
        self.source_network = source_network
        self.target_network = target_network
    
    def transfer_weights(self, layer_indices: Optional[List[int]] = None, freeze_layers: bool = True):
        """
        遷移權重
        
        Args:
            layer_indices: 要遷移的層索引
            freeze_layers: 是否凍結遷移的層
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.source_network.layers) - 1))  # 除了最後一層
        
        for idx in layer_indices:
            if idx < len(self.target_network.layers):
                # 複製權重
                self.target_network.layers[idx].weights = self.source_network.layers[idx].weights.copy()
                self.target_network.layers[idx].biases = self.source_network.layers[idx].biases.copy()
                
                # 凍結層（將學習率設為0）
                if freeze_layers:
                    self.target_network.layers[idx].frozen = True
    
    def fine_tune(self, x_train: np.ndarray, y_train: np.ndarray, 
                  epochs: int = 50, learning_rate: float = 0.0001):
        """
        微調網路
        
        Args:
            x_train: 訓練資料
            y_train: 訓練標籤
            epochs: 訓練世代數
            learning_rate: 學習率
        """
        optimizer = AdamOptimizer(learning_rate)
        learner = SupervisedLearner(self.target_network, optimizer)
        
        return learner.train(x_train, y_train, epochs=epochs, verbose=True)


class LearningEngineManager:
    """學習引擎管理器"""
    
    def __init__(self):
        """初始化學習引擎管理器"""
        self.learners = {}
        self.training_sessions = {}
    
    def create_supervised_learner(self, name: str, network, optimizer_type: str = 'adam', 
                                loss_function: str = 'mse', **optimizer_kwargs) -> SupervisedLearner:
        """創建監督學習器"""
        if optimizer_type == 'adam':
            optimizer = AdamOptimizer(**optimizer_kwargs)
        elif optimizer_type == 'sgd':
            optimizer = SGDOptimizer(**optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        learner = SupervisedLearner(network, optimizer, loss_function)
        self.learners[name] = learner
        return learner
    
    def create_reinforcement_learner(self, name: str, network, optimizer_type: str = 'adam', 
                                   **rl_kwargs) -> ReinforcementLearner:
        """創建強化學習器"""
        if optimizer_type == 'adam':
            optimizer = AdamOptimizer()
        elif optimizer_type == 'sgd':
            optimizer = SGDOptimizer()
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        learner = ReinforcementLearner(network, optimizer, **rl_kwargs)
        self.learners[name] = learner
        return learner
    
    def create_online_learner(self, name: str, network, optimizer_type: str = 'adam', 
                            **online_kwargs) -> OnlineLearner:
        """創建在線學習器"""
        if optimizer_type == 'adam':
            optimizer = AdamOptimizer()
        elif optimizer_type == 'sgd':
            optimizer = SGDOptimizer()
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        learner = OnlineLearner(network, optimizer, **online_kwargs)
        self.learners[name] = learner
        return learner
    
    def get_learner(self, name: str):
        """獲取學習器"""
        return self.learners.get(name)
    
    def save_training_session(self, name: str, session_data: Dict[str, Any]):
        """保存訓練會話"""
        session_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        session_data['session_id'] = str(uuid4())
        self.training_sessions[name] = session_data
    
    def get_training_history(self, name: str) -> Optional[Dict[str, Any]]:
        """獲取訓練歷史"""
        return self.training_sessions.get(name)


# 匯出的類別和函數
__all__ = [
    'LossFunction',
    'Optimizer', 
    'SGDOptimizer',
    'AdamOptimizer',
    'ExperienceBuffer',
    'SupervisedLearner',
    'ReinforcementLearner', 
    'OnlineLearner',
    'TransferLearner',
    'LearningEngineManager'
]