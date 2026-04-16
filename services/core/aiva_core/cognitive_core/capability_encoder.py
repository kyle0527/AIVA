"""
AIVA 結構化能力編碼器 (Capability Encoder)
==========================================
將能力記錄轉換為 512 維向量，作為 5M Decision Engine 的輸入。

設計原則：
1. 結構化特徵 - 不使用自然語言，直接編碼結構化屬性
2. 固定維度 - 輸出 512 維向量，匹配 5M 模型輸入
3. 確定性 - 相同輸入產生相同輸出（無隨機性）
4. 高效 - 純 NumPy 運算，無需 GPU

向量結構（512 維）：
------------------
[0:64]   - 模組編碼 (one-hot + 擴展)
[64:128] - 組件類型編碼
[128:256] - 參數特徵編碼
[256:384] - 標籤特徵編碼
[384:448] - 路徑長度和結構特徵
[448:512] - 預留擴展空間

版本: 1.0.0
日期: 2026-01-04
"""

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

# 模組編碼映射（固定順序）
MODULE_ENCODING = {
    'cognitive_core': 0,
    'internal_exploration': 1,
    'task_planning': 2,
    'core_capabilities': 3,
    'service_backbone': 4,
    'learning_system': 5,  # cognitive_core 子系統
    'external_learning': 5,  # 向後相容，映射到 learning_system
    'unknown': 6
}

# 組件類型編碼
COMPONENT_TYPE_ENCODING = {
    'AI': 0,
    'AI組件': 0,
    '程式': 1,
    '程式組件': 1,
    '混合': 2,
    '混合組件': 2,
    'unknown': 3
}

# 參數類型編碼
PARAM_TYPE_ENCODING = {
    'str': 0,
    'string': 0,
    'int': 1,
    'integer': 1,
    'float': 2,
    'bool': 3,
    'boolean': 3,
    'list': 4,
    'List': 4,
    'dict': 5,
    'Dict': 5,
    'any': 6,
    'unknown': 7
}


@dataclass
class EncodingConfig:
    """編碼配置"""
    output_dim: int = 512
    module_dim: int = 64
    component_dim: int = 64
    param_dim: int = 128
    tag_dim: int = 128
    structure_dim: int = 64
    reserved_dim: int = 64
    
    def validate(self) -> bool:
        """驗證維度配置總和是否為 output_dim"""
        total = (self.module_dim + self.component_dim + self.param_dim + 
                 self.tag_dim + self.structure_dim + self.reserved_dim)
        return total == self.output_dim


class CapabilityEncoder:
    """結構化能力編碼器
    
    將能力記錄（Flow）轉換為 512 維向量，供 5M Decision Engine 使用。
    
    使用方式：
    ```python
    encoder = CapabilityEncoder()
    
    flow = {
        'id': 1,
        'primary_module': 'cognitive_core',
        'primary_component_type': 'AI組件',
        'parameters': [{'name': 'query', 'type': 'str'}],
        'structured_tags': ['module:cognitive_core', 'type:AI'],
        'length': 5
    }
    
    vector = encoder.encode(flow)
    # vector.shape == (512,)
    ```
    """
    
    def __init__(self, config: EncodingConfig | None = None):
        """初始化編碼器
        
        Args:
            config: 編碼配置（可選）
        """
        self.config = config or EncodingConfig()
        
        if not self.config.validate():
            raise ValueError(f"維度配置錯誤：總和必須為 {self.config.output_dim}")
        
        # 預計算位置索引
        self._module_start = 0
        self._module_end = self.config.module_dim
        
        self._component_start = self._module_end
        self._component_end = self._component_start + self.config.component_dim
        
        self._param_start = self._component_end
        self._param_end = self._param_start + self.config.param_dim
        
        self._tag_start = self._param_end
        self._tag_end = self._tag_start + self.config.tag_dim
        
        self._structure_start = self._tag_end
        self._structure_end = self._structure_start + self.config.structure_dim
        
        self._reserved_start = self._structure_end
        self._reserved_end = self.config.output_dim
    
    def encode(self, flow: dict[str, Any]) -> np.ndarray:
        """將單個 Flow 編碼為向量
        
        Args:
            flow: 能力記錄（來自 latest_classification.json）
            
        Returns:
            512 維 float32 向量
        """
        vector = np.zeros(self.config.output_dim, dtype=np.float32)
        
        # 1. 編碼模組
        self._encode_module(vector, flow.get('primary_module', 'unknown'))
        
        # 2. 編碼組件類型
        self._encode_component_type(vector, flow.get('primary_component_type', 'unknown'))
        
        # 3. 編碼參數
        self._encode_parameters(vector, flow.get('parameters', []))
        
        # 4. 編碼標籤
        self._encode_tags(vector, flow.get('structured_tags', []))
        
        # 5. 編碼結構特徵
        self._encode_structure(vector, flow)
        
        # 6. 歸一化（L2 範數）
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def encode_batch(self, flows: list[dict[str, Any]]) -> np.ndarray:
        """批量編碼多個 Flow
        
        Args:
            flows: 能力記錄列表
            
        Returns:
            形狀為 (N, 512) 的向量矩陣
        """
        vectors = np.zeros((len(flows), self.config.output_dim), dtype=np.float32)
        
        for i, flow in enumerate(flows):
            vectors[i] = self.encode(flow)
        
        return vectors
    
    def _encode_module(self, vector: np.ndarray, module: str) -> None:
        """編碼模組（One-Hot + 擴展）"""
        module_idx = MODULE_ENCODING.get(module, MODULE_ENCODING['unknown'])
        
        # One-hot 編碼（前 8 維）
        if module_idx < 8:
            vector[self._module_start + module_idx] = 1.0
        
        # 擴展編碼（用於增加區分度）
        # 使用模組名稱的哈希來填充剩餘維度
        hash_val = int(hashlib.md5(module.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val % (2**31))
        
        # 填充 [8:64] 維度
        extension = np.random.randn(self.config.module_dim - 8).astype(np.float32)
        extension = extension / (np.linalg.norm(extension) + 1e-8) * 0.5
        vector[self._module_start + 8:self._module_end] = extension
    
    def _encode_component_type(self, vector: np.ndarray, comp_type: str) -> None:
        """編碼組件類型"""
        type_idx = COMPONENT_TYPE_ENCODING.get(comp_type, COMPONENT_TYPE_ENCODING['unknown'])
        
        # One-hot 編碼（前 4 維）
        if type_idx < 4:
            vector[self._component_start + type_idx] = 1.0
        
        # 擴展編碼
        hash_val = int(hashlib.md5(comp_type.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val % (2**31))
        
        extension = np.random.randn(self.config.component_dim - 4).astype(np.float32)
        extension = extension / (np.linalg.norm(extension) + 1e-8) * 0.3
        vector[self._component_start + 4:self._component_end] = extension
    
    def _encode_parameters(self, vector: np.ndarray, parameters: list[dict]) -> None:
        """編碼參數特徵"""
        if not parameters:
            return
        
        # 參數數量（歸一化到 [0, 1]）
        param_count = min(len(parameters), 20) / 20.0
        vector[self._param_start] = param_count
        
        # 參數類型分布
        type_counts = np.zeros(8, dtype=np.float32)
        for param in parameters:
            param_type = param.get('type', 'any')
            # 處理複合類型
            base_type = param_type.split('[')[0].split('|')[0].strip()
            type_idx = PARAM_TYPE_ENCODING.get(base_type, PARAM_TYPE_ENCODING['unknown'])
            type_counts[type_idx] += 1
        
        # 歸一化類型分布
        if type_counts.sum() > 0:
            type_counts = type_counts / type_counts.sum()
        
        vector[self._param_start + 1:self._param_start + 9] = type_counts
        
        # 必填參數比例
        required_count = sum(1 for p in parameters if p.get('default') is None and not p.get('name', '').startswith('*'))
        required_ratio = required_count / max(len(parameters), 1)
        vector[self._param_start + 9] = required_ratio
        
        # 參數名稱哈希特徵（用於區分不同參數組合）
        param_names = sorted([p.get('name', '') for p in parameters])
        param_hash = hashlib.md5('|'.join(param_names).encode()).hexdigest()[:16]
        hash_int = int(param_hash, 16)
        
        np.random.seed(hash_int % (2**31))
        hash_features = np.random.randn(self.config.param_dim - 10).astype(np.float32)
        hash_features = hash_features / (np.linalg.norm(hash_features) + 1e-8) * 0.4
        vector[self._param_start + 10:self._param_end] = hash_features
    
    def _encode_tags(self, vector: np.ndarray, tags: list[str]) -> None:
        """編碼結構化標籤"""
        if not tags:
            return
        
        # 標籤數量
        vector[self._tag_start] = min(len(tags), 20) / 20.0
        
        # 解析標籤
        tag_features = np.zeros(32, dtype=np.float32)
        
        for tag in tags:
            if ':' in tag:
                key, value = tag.split(':', 1)
                
                # 模組標籤
                if key == 'module':
                    idx = MODULE_ENCODING.get(value, 6)
                    tag_features[idx] = 1.0
                
                # 類型標籤
                elif key == 'type':
                    idx = COMPONENT_TYPE_ENCODING.get(value, 3)
                    tag_features[8 + idx] = 1.0
                
                # 長度標籤
                elif key == 'length':
                    if value == 'short':
                        tag_features[12] = 1.0
                    elif value == 'medium':
                        tag_features[13] = 1.0
                    elif value == 'long':
                        tag_features[14] = 1.0
                
                # 異步標籤
                elif key == 'async':
                    tag_features[15] = 1.0 if value == 'true' else 0.0
        
        vector[self._tag_start + 1:self._tag_start + 33] = tag_features
        
        # 標籤哈希特徵
        tag_str = '|'.join(sorted(tags))
        tag_hash = hashlib.md5(tag_str.encode()).hexdigest()[:16]
        hash_int = int(tag_hash, 16)
        
        np.random.seed(hash_int % (2**31))
        hash_features = np.random.randn(self.config.tag_dim - 33).astype(np.float32)
        hash_features = hash_features / (np.linalg.norm(hash_features) + 1e-8) * 0.3
        vector[self._tag_start + 33:self._tag_end] = hash_features
    
    def _encode_structure(self, vector: np.ndarray, flow: dict) -> None:
        """編碼結構特徵"""
        # 路徑長度（歸一化）
        length = flow.get('length', 1)
        vector[self._structure_start] = min(length, 50) / 50.0
        
        # 是否為入口點
        is_entry = flow.get('id', 0) <= 10  # 前 10 個 flow 通常是主要入口
        vector[self._structure_start + 1] = 1.0 if is_entry else 0.0
        
        # 涉及的模組數量
        modules = flow.get('modules', [])
        unique_modules = len(set(modules)) if modules else 0
        vector[self._structure_start + 2] = min(unique_modules, 6) / 6.0
        
        # 模組跨越（是否涉及多個模組）
        vector[self._structure_start + 3] = 1.0 if unique_modules > 1 else 0.0
        
        # 返回類型編碼
        return_type = flow.get('return_type', 'unknown')
        base_return = return_type.split('[')[0].split('|')[0].strip()
        return_idx = PARAM_TYPE_ENCODING.get(base_return, PARAM_TYPE_ENCODING['unknown'])
        
        # One-hot 返回類型
        return_onehot = np.zeros(8, dtype=np.float32)
        return_onehot[return_idx] = 1.0
        vector[self._structure_start + 4:self._structure_start + 12] = return_onehot
        
        # Flow ID 哈希（增加唯一性）
        flow_id = flow.get('id', 0)
        np.random.seed(flow_id % (2**31))
        id_features = np.random.randn(self.config.structure_dim - 12).astype(np.float32)
        id_features = id_features / (np.linalg.norm(id_features) + 1e-8) * 0.2
        vector[self._structure_start + 12:self._structure_end] = id_features
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """計算兩個向量的餘弦相似度
        
        Args:
            vec1: 向量 1
            vec2: 向量 2
            
        Returns:
            相似度分數（-1 到 1）
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        """找到最相似的向量
        
        Args:
            query_vector: 查詢向量
            candidate_vectors: 候選向量矩陣 (N, 512)
            top_k: 返回前 k 個結果
            
        Returns:
            [(index, similarity_score), ...]
        """
        # 計算所有相似度
        similarities = np.dot(candidate_vectors, query_vector)
        
        # 找到 top_k
        if top_k >= len(similarities):
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]
        
        return [(int(idx), float(similarities[idx])) for idx in indices[:top_k]]


# 便捷函數
def encode_capability(flow: dict[str, Any]) -> np.ndarray:
    """便捷函數：編碼單個能力
    
    Args:
        flow: 能力記錄
        
    Returns:
        512 維向量
    """
    encoder = CapabilityEncoder()
    return encoder.encode(flow)


def encode_capabilities(flows: list[dict[str, Any]]) -> np.ndarray:
    """便捷函數：批量編碼能力
    
    Args:
        flows: 能力記錄列表
        
    Returns:
        (N, 512) 向量矩陣
    """
    encoder = CapabilityEncoder()
    return encoder.encode_batch(flows)


# 測試代碼
if __name__ == "__main__":
    # 測試編碼器
    encoder = CapabilityEncoder()
    
    # 測試 flow
    test_flow = {
        'id': 1,
        'primary_module': 'cognitive_core',
        'primary_component_type': 'AI組件',
        'parameters': [
            {'name': 'query', 'type': 'str', 'default': None},
            {'name': 'top_k', 'type': 'int', 'default': 5}
        ],
        'structured_tags': ['module:cognitive_core', 'type:AI', 'length:medium', 'async:false'],
        'length': 5,
        'return_type': 'List[Dict]',
        'modules': ['cognitive_core', 'service_backbone']
    }
    
    vector = encoder.encode(test_flow)
    
    print(f"向量維度: {vector.shape}")
    print(f"向量範數: {np.linalg.norm(vector):.4f}")
    print(f"非零元素: {np.count_nonzero(vector)}")
    print(f"向量摘要: min={vector.min():.4f}, max={vector.max():.4f}, mean={vector.mean():.4f}")
