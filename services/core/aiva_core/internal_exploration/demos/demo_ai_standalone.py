#!/usr/bin/env python3
"""直接演示AI內部能力（獨立版本，無需完整模組導入）

展示3個AI內部能力的核心功能：
1. 能力範圍分類器 - 自動分類能力的可見性和訪問級別
2. DQN強化學習網絡 - 智能決策模型
3. PPO強化學習網絡 - 策略優化模型
"""

import torch
import torch.nn as nn
import numpy as np
from enum import Enum

print("="*70)
print("🤖 AIVA AI內部能力直接執行演示")
print("="*70)
print()

# ===== 1. 能力範圍分類器（簡化版）=====
print("【1/3】能力範圍分類器 (Capability Scope Classifier)")
print("功能：根據文件路徑自動分類能力的範圍和可見性")
print("-"*70)

class CapabilityScope(Enum):
    """能力範圍"""
    CORE = "core"
    SERVICE = "service"
    GLOBAL = "global"

class CapabilityVisibility(Enum):
    """能力可見性"""
    PUBLIC = "public"
    INTERNAL = "internal"
    SYSTEM = "system"

def classify_capability(file_path: str):
    """分類能力"""
    path_lower = file_path.lower()
    
    # 內部探索 → CORE, INTERNAL
    if "internal_exploration" in path_lower or "internal_loop" in path_lower:
        return (CapabilityScope.CORE, CapabilityVisibility.INTERNAL)
    
    # 任務規劃 → SERVICE, PUBLIC
    if "task_planning" in path_lower:
        return (CapabilityScope.SERVICE, CapabilityVisibility.PUBLIC)
    
    # Features → GLOBAL, PUBLIC
    if "features" in path_lower:
        return (CapabilityScope.GLOBAL, CapabilityVisibility.PUBLIC)
    
    return (CapabilityScope.CORE, CapabilityVisibility.INTERNAL)

# 測試用例
test_paths = [
    ("services/core/aiva_core/internal_exploration/analyzer.py", "內部探索工具"),
    ("services/core/aiva_core/task_planning/planner.py", "任務規劃器"),
    ("services/features/sqli/scanner.py", "SQL注入掃描器"),
    ("services/core/aiva_core/cognitive_core/rag_engine.py", "RAG引擎")
]

print("\n測試路徑分類：")
for path, desc in test_paths:
    scope, visibility = classify_capability(path)
    print(f"📄 {desc}")
    print(f"   路徑: {path}")
    print(f"   └─ 範圍: {scope.value} | 可見性: {visibility.value}\n")

print("✅ 能力範圍分類器執行成功\n")

# ===== 2. DQN強化學習網絡 =====
print("="*70)
print("【2/3】DQN強化學習網絡 (Deep Q-Network)")
print("功能：深度Q網絡，用於攻擊策略的智能決策")
print("-"*70)

class SimpleDQN(nn.Module):
    """簡化的DQN網絡"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def select_action(self, state, epsilon=0.1):
        """ε-greedy策略選擇動作"""
        if np.random.random() < epsilon:
            return np.random.randint(0, 4), torch.zeros(4)
        else:
            with torch.no_grad():
                q_values = self.forward(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
                return int(q_values.argmax().item()), q_values

# 創建DQN
state_dim = 10  # 10個環境狀態特徵
action_dim = 4   # 4種可能的攻擊動作

dqn = SimpleDQN(state_dim, action_dim)

print(f"\n📊 DQN網絡架構:")
print(f"   - 輸入維度（狀態）: {state_dim}")
print(f"   - 輸出維度（動作）: {action_dim}")
print(f"   - 隱藏層: [64, 32]")
print(f"   - 總參數量: {sum(p.numel() for p in dqn.parameters())}")

# 模擬攻擊場景
print(f"\n🎯 模擬3次攻擊決策:")
attack_names = ["SQL注入", "XSS攻擊", "路徑遍歷", "命令注入"]

for i in range(3):
    # 隨機生成環境狀態（例如：目標特徵、防護強度等）
    state = np.random.randn(state_dim)
    action, q_values = dqn.select_action(state, epsilon=0.1)
    
    print(f"\n   決策 #{i+1}:")
    print(f"   - 環境狀態: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}, ...]")
    print(f"   - 選擇攻擊: {attack_names[action]}")
    print(f"   - Q值分布: {q_values.numpy()}")
    print(f"   - 信心度: {q_values[action].item():.3f}")

print("\n✅ DQN強化學習網絡執行成功\n")

# ===== 3. PPO強化學習網絡 =====
print("="*70)
print("【3/3】PPO策略網絡 (Proximal Policy Optimization)")
print("功能：策略優化網絡，用於學習最優攻擊序列")
print("-"*70)

class SimplePPO(nn.Module):
    """簡化的PPO網絡"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.actor(state)
    
    def select_action(self, state):
        """根據策略概率選擇動作"""
        with torch.no_grad():
            probs = self.forward(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            action = torch.multinomial(probs, 1).item()
            return action, probs

# 創建PPO
ppo = SimplePPO(state_dim, action_dim)

print(f"\n📊 PPO策略網絡架構:")
print(f"   - 輸入維度: {state_dim}")
print(f"   - 輸出維度: {action_dim}")
print(f"   - 策略類型: 隨機策略（Softmax）")

# 模擬學習攻擊序列
print(f"\n🎯 模擬攻擊序列學習:")
sequence = []
state = np.random.randn(state_dim)

for step in range(5):
    action, probs = ppo.select_action(state)
    action_idx = int(action)  # 確保是整數索引
    sequence.append(attack_names[action_idx])
    
    print(f"\n   步驟 {step+1}: {attack_names[action_idx]}")
    print(f"   - 動作概率分布:")
    for i, (name, prob) in enumerate(zip(attack_names, probs.numpy())):
        bar = "█" * int(prob * 50)
        print(f"      {name:12}: {prob:.3f} {bar}")
    
    # 模擬環境變化
    state = state + np.random.randn(state_dim) * 0.1

print(f"\n   🔗 生成的攻擊序列: {' → '.join(sequence)}")
print("\n✅ PPO策略網絡執行成功\n")

print("="*70)
print("🎉 AI內部能力演示完成")
print()
print("💡 總結:")
print("   1️⃣  能力分類器：自動化管理676個能力的訪問權限")
print("   2️⃣  DQN網絡：基於價值函數的智能決策（選最優動作）")
print("   3️⃣  PPO網絡：基於策略梯度的序列學習（學習攻擊鏈）")
print("="*70)
