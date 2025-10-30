"""
強化學習神經網絡模型

支持 DQN (Deep Q-Network) 和 PPO (Proximal Policy Optimization)
"""



import logging



import torch


from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Deep Q-Network 模型
    
    用於 DQN 算法的神經網絡架構
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
    ) -> None:
        """初始化 DQN 網絡
        
        Args:
            state_dim: 狀態維度
            action_dim: 動作維度
            hidden_dims: 隱藏層維度元組
            activation: 激活函數 (relu, tanh, leaky_relu)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_name = activation
        
        # 構建網絡層
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
        
        # 輸出層：Q 值（每個動作一個 Q 值）
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self.apply(self._init_weights)
        
        logger.info(
            f"DQN Network initialized: state_dim={state_dim}, "
            f"action_dim={action_dim}, hidden={hidden_dims}"
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            state: 狀態張量 [batch_size, state_dim]
        
        Returns:
            Q 值張量 [batch_size, action_dim]
        """
        return self.network(state)
    
    def select_action(
        self, state: np.ndarray, epsilon: float = 0.1
    ) -> tuple[int, torch.Tensor]:
        """ε-greedy 策略選擇動作
        
        Args:
            state: 狀態向量
            epsilon: 探索概率
        
        Returns:
            (選中的動作, Q 值)
        """
        # ε-greedy
        if np.random.random() < epsilon:
            # 探索：隨機選擇
            action = np.random.randint(0, self.action_dim)
            q_values = torch.zeros(self.action_dim)
        else:
            # 利用：選擇最大 Q 值的動作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor).squeeze(0)
                action = int(q_values.argmax().item())
        
        return action, q_values
    
    def _get_activation(self, activation: str) -> nn.Module:
        """獲取激活函數
        
        Args:
            activation: 激活函數名稱
        
        Returns:
            激活函數模組
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()  # 默認
    
    def _init_weights(self, module: nn.Module) -> None:
        """初始化權重
        
        Args:
            module: 神經網絡模組
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


class ActorCritic(nn.Module):
    """Actor-Critic 網絡 (用於 PPO)
    
    包含 Actor (策略網絡) 和 Critic (價值網絡)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        activation: str = "tanh",
    ) -> None:
        """初始化 Actor-Critic 網絡
        
        Args:
            state_dim: 狀態維度
            action_dim: 動作維度
            hidden_dims: 隱藏層維度
            activation: 激活函數
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特徵提取器
        self.feature_extractor = self._build_feature_extractor(
            state_dim, hidden_dims[0], activation
        )
        
        # Actor 網絡 (策略)
        self.actor = self._build_actor(hidden_dims, action_dim, activation)
        
        # Critic 網絡 (價值)
        self.critic = self._build_critic(hidden_dims, activation)
        
        # 初始化權重
        self.apply(self._init_weights)
        
        logger.info(
            f"Actor-Critic Network initialized: state_dim={state_dim}, "
            f"action_dim={action_dim}, hidden={hidden_dims}"
        )
    
    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向傳播
        
        Args:
            state: 狀態張量 [batch_size, state_dim]
        
        Returns:
            (動作概率分佈, 狀態價值)
        """
        # 特徵提取
        features = self.feature_extractor(state)
        
        # Actor 輸出：動作概率
        action_probs = F.softmax(self.actor(features), dim=-1)
        
        # Critic 輸出：狀態價值
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """根據策略選擇動作
        
        Args:
            state: 狀態向量
            deterministic: 是否確定性選擇 (選擇概率最高的)
        
        Returns:
            (選中的動作, log 概率, 狀態價值)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = self.forward(state_tensor)
            
            if deterministic:
                # 確定性：選擇概率最高的動作
                action = int(action_probs.argmax().item())
                log_prob = torch.log(action_probs[0, action] + 1e-10)
            else:
                # 隨機採樣
                dist = Categorical(action_probs)
                action_tensor = dist.sample()
                action = int(action_tensor.item())
                log_prob = dist.log_prob(action_tensor)
        
        return action, log_prob, state_value.squeeze()
    
    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """評估給定狀態-動作對
        
        Args:
            states: 狀態批次 [batch_size, state_dim]
            actions: 動作批次 [batch_size]
        
        Returns:
            (log 概率, 狀態價值, 熵)
        """
        action_probs, state_values = self.forward(states)
        
        # 計算 log 概率
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # 計算熵（用於鼓勵探索）
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(), entropy
    
    def _build_feature_extractor(
        self, input_dim: int, output_dim: int, activation: str
    ) -> nn.Sequential:
        """構建特徵提取器
        
        Args:
            input_dim: 輸入維度
            output_dim: 輸出維度
            activation: 激活函數
        
        Returns:
            特徵提取器
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            self._get_activation(activation),
        )
    
    def _build_actor(
        self, hidden_dims: tuple[int, ...], action_dim: int, activation: str
    ) -> nn.Sequential:
        """構建 Actor 網絡
        
        Args:
            hidden_dims: 隱藏層維度
            action_dim: 動作維度
            activation: 激活函數
        
        Returns:
            Actor 網絡
        """
        layers = []
        input_dim = hidden_dims[0]
        
        # 隱藏層
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dims[i]
        
        # 輸出層（動作 logits）
        layers.append(nn.Linear(input_dim, action_dim))
        
        return nn.Sequential(*layers)
    
    def _build_critic(
        self, hidden_dims: tuple[int, ...], activation: str
    ) -> nn.Sequential:
        """構建 Critic 網絡
        
        Args:
            hidden_dims: 隱藏層維度
            activation: 激活函數
        
        Returns:
            Critic 網絡
        """
        layers = []
        input_dim = hidden_dims[0]
        
        # 隱藏層
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dims[i]
        
        # 輸出層（標量價值）
        layers.append(nn.Linear(input_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """獲取激活函數"""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            return nn.Tanh()  # PPO 默認用 tanh
    
    def _init_weights(self, module: nn.Module) -> None:
        """初始化權重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


class ReplayBuffer:
    """Experience Replay Buffer
    
    用於 DQN 算法的經驗回放緩衝區
    """
    
    def __init__(self, capacity: int = 10000) -> None:
        """初始化緩衝區
        
        Args:
            capacity: 緩衝區容量
        """
        self.capacity = capacity
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0
        
        logger.info(f"ReplayBuffer initialized with capacity={capacity}")
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """添加經驗
        
        Args:
            state: 當前狀態
            action: 執行的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            done: 是否結束
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """隨機採樣批次
        
        Args:
            batch_size: 批次大小
        
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )
    
    def __len__(self) -> int:
        """緩衝區大小"""
        return len(self.buffer)


class RolloutBuffer:
    """Rollout Buffer (用於 PPO)
    
    存儲軌跡數據用於 PPO 訓練
    """
    
    def __init__(self) -> None:
        """初始化 Buffer"""
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """添加經驗
        
        Args:
            state: 狀態
            action: 動作
            reward: 獎勵
            log_prob: log 概率
            value: 狀態價值
            done: 是否結束
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """獲取所有數據
        
        Returns:
            (states, actions, rewards, log_probs, values)
        """
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.stack(self.log_probs),
            torch.stack(self.values),
        )
    
    def compute_returns(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> torch.Tensor:
        """計算 GAE (Generalized Advantage Estimation) 回報
        
        Args:
            gamma: 折扣因子
            gae_lambda: GAE lambda
        
        Returns:
            回報張量
        """
        returns = []
        advantages = []
        gae = 0.0
        
        # 反向計算 GAE
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1].item()
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * (1 - int(self.dones[t])) - self.values[t].item()
            
            # GAE
            gae = delta + gamma * gae_lambda * (1 - int(self.dones[t])) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t].item())
        
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)
    
    def clear(self) -> None:
        """清空 Buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        """Buffer 大小"""
        return len(self.states)
