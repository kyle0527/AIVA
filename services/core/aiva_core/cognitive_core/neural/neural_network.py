"""Neural Network - 神經網路基礎架構
與 BioNeuron Core 配合的通用神經網路實現

這個模組提供了基礎的神經網路架構，包括：
- 前饋神經網路
- 循環神經網路 (RNN)
- 長短期記憶網路 (LSTM)
- 注意力機制
"""

import logging
from typing import Union, List, Any

import numpy as np

NDArray = Union[np.ndarray, List, Any]

logger = logging.getLogger(__name__)


class ActivationFunctions:
    """常用激活函數集合"""

    @staticmethod
    def relu(x: NDArray) -> NDArray:
        """ReLU 激活函數"""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: NDArray) -> NDArray:
        """ReLU 導數"""
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: NDArray) -> NDArray:
        """Sigmoid 激活函數"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    @staticmethod
    def sigmoid_derivative(x: NDArray) -> NDArray:
        """Sigmoid 導數"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x: NDArray) -> NDArray:
        """Tanh 激活函數"""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: NDArray) -> NDArray:
        """Tanh 導數"""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x: NDArray) -> NDArray:
        """Softmax 激活函數"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class DenseLayer:
    """全連接層"""

    def __init__(self, input_size: int, output_size: int, activation: str = "relu", seed: int | None = None):
        """初始化全連接層

        Args:
            input_size: 輸入維度
            output_size: 輸出維度
            activation: 激活函數類型
            seed: 隨機種子 (None 表示使用隨機種子)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Xavier/He 初始化使用現代 Generator API
        rng = np.random.default_rng(seed)
        if activation == "relu":
            # He 初始化適用於 ReLU
            self.weights = rng.standard_normal((input_size, output_size)) * np.sqrt(2.0 / input_size)
        else:
            # Xavier 初始化適用於 tanh/sigmoid
            self.weights = rng.standard_normal((input_size, output_size)) * np.sqrt(1.0 / input_size)
        self.biases = np.zeros(output_size)

        # 梯度
        self.weights_grad = np.zeros_like(self.weights)
        self.biases_grad = np.zeros_like(self.biases)

        # 快取
        self.last_input = None
        self.last_output = None

        # 激活函數映射
        self.activation_funcs = {
            "relu": (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            "sigmoid": (
                ActivationFunctions.sigmoid,
                ActivationFunctions.sigmoid_derivative,
            ),
            "tanh": (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            "softmax": (ActivationFunctions.softmax, None),
        }

    def forward(self, x: NDArray) -> NDArray:
        """前向傳播"""
        self.last_input = x
        z = np.dot(x, self.weights) + self.biases

        if self.activation in self.activation_funcs:
            activation_func, _ = self.activation_funcs[self.activation]
            self.last_output = activation_func(z)
        else:
            self.last_output = z

        return self.last_output

    def backward(self, grad_output: NDArray) -> NDArray:
        """反向傳播"""
        if self.activation in self.activation_funcs:
            _, derivative_func = self.activation_funcs[self.activation]
            if derivative_func:
                grad_output = grad_output * derivative_func(self.last_output)

        # 計算梯度
        if self.last_input is not None:
            self.weights_grad = np.dot(self.last_input.T, grad_output)
            self.biases_grad = np.sum(grad_output, axis=0)

        # 返回輸入梯度
        return np.dot(grad_output, self.weights.T)

    def update_weights(self, learning_rate: float):
        """更新權重"""
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad


class FeedForwardNetwork:
    """前饋神經網路"""

    def __init__(self, layer_sizes: list[int], activations: list[str] | None = None):
        """初始化前饋神經網路

        Args:
            layer_sizes: 各層的神經元數量
            activations: 各層的激活函數
        """
        self.layer_sizes = layer_sizes
        self.layers = []

        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["softmax"]

        # 創建網路層
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activations[i] if i < len(activations) else "relu",
            )
            self.layers.append(layer)

    def forward(self, x: NDArray) -> NDArray:
        """前向傳播"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: NDArray):
        """反向傳播"""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def update_weights(self, learning_rate: float):
        """更新所有層的權重"""
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def predict(self, x: NDArray) -> NDArray:
        """預測"""
        return self.forward(x)


class RecurrentLayer:
    """循環神經網路層"""

    def __init__(self, input_size: int, hidden_size: int, activation: str = "tanh", seed: int | None = None):
        """初始化RNN層

        Args:
            input_size: 輸入維度
            hidden_size: 隱藏層維度
            activation: 激活函數
            seed: 隨機種子 (None 表示使用隨機種子)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        # 權重初始化使用現代 Generator API (遵循 PEP8 命名規範)
        rng = np.random.default_rng(seed)
        # Xavier 初始化適用於 RNN
        scale = np.sqrt(1.0 / (input_size + hidden_size))
        self.wxh = rng.standard_normal((input_size, hidden_size)) * scale
        self.whh = rng.standard_normal((hidden_size, hidden_size)) * scale
        self.bh = np.zeros(hidden_size)

        # 隱藏狀態
        self.hidden_state = np.zeros(hidden_size)

        # 激活函數
        if activation == "tanh":
            self.activation_func = ActivationFunctions.tanh
        elif activation == "relu":
            self.activation_func = ActivationFunctions.relu
        else:
            self.activation_func = lambda x: x

    def forward(self, x: NDArray) -> NDArray:
        """前向傳播"""
        self.hidden_state = self.activation_func(
            np.dot(x, self.wxh) + np.dot(self.hidden_state, self.whh) + self.bh
        )
        return self.hidden_state

    def reset_hidden_state(self):
        """重置隱藏狀態"""
        self.hidden_state = np.zeros(self.hidden_size)


class LSTMLayer:
    """長短期記憶網路層"""

    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        """初始化LSTM層

        Args:
            input_size: 輸入維度
            hidden_size: 隱藏層維度
            seed: 隨機種子 (None 表示使用隨機種子)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 使用現代 Generator API 和 Xavier 初始化 (遵循 PEP8 命名規範)
        rng = np.random.default_rng(seed)
        combined_size = input_size + hidden_size
        scale = np.sqrt(1.0 / combined_size)

        # 遺忘門權重
        self.wf = rng.standard_normal((combined_size, hidden_size)) * scale
        self.bf = np.zeros(hidden_size)

        # 輸入門權重
        self.wi = rng.standard_normal((combined_size, hidden_size)) * scale
        self.bi = np.zeros(hidden_size)

        # 候選值權重
        self.wc = rng.standard_normal((combined_size, hidden_size)) * scale
        self.bc = np.zeros(hidden_size)

        # 輸出門權重
        self.wo = rng.standard_normal((combined_size, hidden_size)) * scale
        self.bo = np.zeros(hidden_size)

        # 狀態
        self.hidden_state = np.zeros(hidden_size)
        self.cell_state = np.zeros(hidden_size)

    def forward(self, x: NDArray) -> NDArray:
        """前向傳播"""
        # 合併輸入和隱藏狀態
        combined = np.concatenate([x, self.hidden_state])

        # 遺忘門
        forget_gate = ActivationFunctions.sigmoid(np.dot(combined, self.wf) + self.bf)

        # 輸入門
        input_gate = ActivationFunctions.sigmoid(np.dot(combined, self.wi) + self.bi)

        # 候選值
        candidate = ActivationFunctions.tanh(np.dot(combined, self.wc) + self.bc)

        # 更新細胞狀態
        self.cell_state = forget_gate * self.cell_state + input_gate * candidate

        # 輸出門
        output_gate = ActivationFunctions.sigmoid(np.dot(combined, self.wo) + self.bo)

        # 更新隱藏狀態
        self.hidden_state = output_gate * ActivationFunctions.tanh(self.cell_state)

        return self.hidden_state

    def reset_states(self):
        """重置狀態"""
        self.hidden_state = np.zeros(self.hidden_size)
        self.cell_state = np.zeros(self.hidden_size)


class AttentionMechanism:
    """注意力機制"""

    def __init__(self, hidden_size: int, seed: int | None = None):
        """初始化注意力機制

        Args:
            hidden_size: 隱藏層維度
            seed: 隨機種子 (None 表示使用隨機種子)
        """
        self.hidden_size = hidden_size
        # 使用現代 Generator API 和 Xavier 初始化 (遵循 PEP8 命名規範)
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / hidden_size)
        self.w_attention = rng.standard_normal((hidden_size, hidden_size)) * scale
        self.v_attention = rng.standard_normal(hidden_size) * scale

    def forward(
        self, hidden_states: NDArray, query: NDArray
    ) -> tuple[NDArray, NDArray]:
        """計算注意力權重和加權輸出

        Args:
            hidden_states: 隱藏狀態序列 [seq_len, hidden_size]
            query: 查詢向量 [hidden_size]

        Returns:
            context_vector: 上下文向量
            attention_weights: 注意力權重
        """
        # 計算注意力分數
        scores = np.dot(hidden_states, self.w_attention)
        scores = np.dot(scores, query)

        # 計算注意力權重
        attention_weights = ActivationFunctions.softmax(scores)

        # 計算上下文向量
        context_vector = np.sum(
            hidden_states * attention_weights[:, np.newaxis], axis=0
        )

        return context_vector, attention_weights


class NeuralNetworkBuilder:
    """神經網路建構器"""

    @staticmethod
    def create_classifier(
        input_size: int, num_classes: int, hidden_sizes: list[int] | None = None
    ) -> FeedForwardNetwork:
        """創建分類器

        Args:
            input_size: 輸入維度
            num_classes: 類別數量
            hidden_sizes: 隱藏層維度列表

        Returns:
            分類神經網路
        """
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layer_sizes = [input_size] + hidden_sizes + [num_classes]
        activations = ["relu"] * len(hidden_sizes) + ["softmax"]

        return FeedForwardNetwork(layer_sizes, activations)

    @staticmethod
    def create_regressor(
        input_size: int, output_size: int, hidden_sizes: list[int] | None = None
    ) -> FeedForwardNetwork:
        """創建回歸器

        Args:
            input_size: 輸入維度
            output_size: 輸出維度
            hidden_sizes: 隱藏層維度列表

        Returns:
            回歸神經網路
        """
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        activations = ["relu"] * len(hidden_sizes) + ["linear"]

        return FeedForwardNetwork(layer_sizes, activations)


# 匯出的類別和函數
__all__ = [
    "ActivationFunctions",
    "DenseLayer",
    "FeedForwardNetwork",
    "RecurrentLayer",
    "LSTMLayer",
    "AttentionMechanism",
    "NeuralNetworkBuilder",
]
