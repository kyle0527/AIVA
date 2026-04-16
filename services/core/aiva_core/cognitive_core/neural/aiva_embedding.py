#!/usr/bin/env python3
"""
AIVA 自研 Embedding 層
使用 all-MiniLM-L6-v2 的預訓練權重，但用 AIVA 自己的架構實現
這樣可以完全掌控模型，不依賴外部庫
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class AIVAEmbedding(nn.Module):
    """
    AIVA 自主可控的 Embedding 模型
    
    設計理念：
    1. 使用 all-MiniLM-L6-v2 的預訓練權重（站在巨人肩膀上）
    2. 用 AIVA 自己的架構包裝（完全自主可控）
    3. 保持接口簡潔（方便替換和升級）
    
    架構說明：
    - Transformer: all-MiniLM-L6-v2 的 BERT 骨幹（從 Hugging Face 加載）
    - Pooling: AIVA 自定義的 mean pooling（不依賴 sentence-transformers）
    - Normalize: AIVA 自己實現的歸一化
    """
    
    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length: int = 256,
        device: str | None = None
    ):
        """
        初始化 AIVA Embedding 模型
        
        Args:
            model_name_or_path: Hugging Face 模型名稱或本地路徑
            max_seq_length: 最大序列長度
            device: 運算設備 (cuda/cpu)
        """
        super().__init__()
        
        self.max_seq_length = max_seq_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 載入預訓練的 Transformer（all-MiniLM-L6-v2 的權重）
        logger.info(f"🔧 載入預訓練權重: {model_name_or_path}")
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # 移到指定設備
        self.transformer.to(self.device)
        self.transformer.eval()  # 預設為推理模式
        
        # 記錄模型信息
        embedding_dim = self.transformer.config.hidden_size
        num_params = sum(p.numel() for p in self.transformer.parameters())
        
        logger.info("✅ AIVA Embedding 模型初始化完成:")
        logger.info(f"   - 預訓練權重: {model_name_or_path}")
        logger.info(f"   - 輸出維度: {embedding_dim}")
        logger.info(f"   - 參數量: {num_params:,} ({num_params/1e6:.1f}M)")
        logger.info(f"   - 最大長度: {max_seq_length}")
        logger.info(f"   - 設備: {self.device}")
    
    def _mean_pooling(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        AIVA 自定義的 Mean Pooling
        
        原理：對所有 token 的 embeddings 取加權平均（attention_mask 作為權重）
        這樣可以聚合整個句子的信息到一個向量
        
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            sentence_embeddings: [batch_size, hidden_size]
        """
        # 擴展 attention_mask 維度以匹配 token_embeddings
        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 加權求和
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # 計算實際 token 數量（避免除以零）
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # 平均
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            embeddings: 句子向量 [batch_size, hidden_size]
        """
        # 通過 Transformer 獲取 token embeddings
        with torch.no_grad():  # 推理時不需要梯度
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # 取出最後一層的 hidden states
        token_embeddings = outputs.last_hidden_state
        
        # 應用 mean pooling
        sentence_embeddings = self._mean_pooling(token_embeddings, attention_mask)
        
        # L2 歸一化（這樣餘弦相似度 = 點積）
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
    
    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = True
    ) -> np.ndarray | torch.Tensor:
        """
        編碼句子為向量
        
        這個接口與 SentenceTransformer 兼容，方便替換
        
        Args:
            sentences: 單個句子或句子列表
            batch_size: 批次大小
            show_progress_bar: 是否顯示進度條（保留接口，暫不實現）
            convert_to_numpy: 是否轉換為 numpy
            convert_to_tensor: 是否轉換為 tensor
            device: 設備（None 則使用初始化時的設備）
            normalize_embeddings: 是否歸一化
        
        Returns:
            embeddings: 向量 [num_sentences, hidden_size]
        """
        # 統一處理為列表
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # 使用指定設備
        target_device = device or self.device
        
        all_embeddings = []
        
        # 分批處理
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            
            # Tokenization
            encoded = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            
            # 移到目標設備
            input_ids = encoded['input_ids'].to(target_device)
            attention_mask = encoded['attention_mask'].to(target_device)
            
            # 前向傳播
            with torch.no_grad():
                batch_embeddings = self.forward(input_ids, attention_mask)
            
            all_embeddings.append(batch_embeddings)
        
        # 合併所有批次
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # 可選：再次歸一化
        if normalize_embeddings and not convert_to_tensor:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 轉換輸出格式
        if convert_to_numpy:
            return embeddings.cpu().numpy()
        elif convert_to_tensor:
            return embeddings
        else:
            return embeddings.cpu().numpy()
    
    def similarity(
        self,
        embeddings1: torch.Tensor | np.ndarray,
        embeddings2: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """
        計算向量相似度（餘弦相似度）
        
        Args:
            embeddings1: 第一組向量
            embeddings2: 第二組向量
        
        Returns:
            similarity: 相似度矩陣
        """
        # 轉換為 tensor
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.tensor(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.tensor(embeddings2)
        
        # 移到同一設備
        embeddings1 = embeddings1.to(self.device)
        embeddings2 = embeddings2.to(self.device)
        
        # 計算餘弦相似度（已歸一化的向量可以直接點積）
        similarity = torch.mm(embeddings1, embeddings2.transpose(0, 1))
        
        return similarity
    
    def save(self, save_path: str):
        """
        保存模型（包括 Transformer 和 Tokenizer）
        
        Args:
            save_path: 保存路徑
        """
        logger.info(f"💾 保存 AIVA Embedding 模型到: {save_path}")
        self.transformer.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("✅ 模型保存完成")
    
    @classmethod
    def load(cls, load_path: str, device: str | None = None):
        """
        加載模型
        
        Args:
            load_path: 模型路徑
            device: 運算設備
        
        Returns:
            model: AIVAEmbedding 實例
        """
        logger.info(f"📂 從本地加載 AIVA Embedding 模型: {load_path}")
        return cls(model_name_or_path=load_path, device=device)


# 向後兼容：提供與 SentenceTransformer 相同的接口
def SentenceTransformer(model_name_or_path: str, device: str | None = None):
    """
    AIVA 版本的 SentenceTransformer
    
    這個函數提供與原始 SentenceTransformer 相同的接口
    但實際使用 AIVA 自己的實現
    
    Args:
        model_name_or_path: 模型名稱或路徑
        device: 設備
    
    Returns:
        AIVAEmbedding 實例
    """
    logger.info("🔄 使用 AIVA 自研 Embedding 層（兼容 SentenceTransformer 接口）")
    return AIVAEmbedding(model_name_or_path, device=device)


if __name__ == "__main__":
    # 測試代碼
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("AIVA Embedding 模型測試")
    print("=" * 60)
    
    # 初始化模型
    model = AIVAEmbedding()
    
    # 測試編碼
    sentences = [
        "SQL injection vulnerability detected",
        "Cross-site scripting attack found",
        "Server-side request forgery vulnerability"
    ]
    
    print("\n測試句子:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")
    
    # 生成 embeddings
    embeddings = model.encode(sentences)
    
    print(f"\nEmbedding 維度: {embeddings.shape}")
    print(f"第一個向量的前 10 維: {embeddings[0][:10]}")
    
    # 測試相似度
    similarity = model.similarity(embeddings[[0]], embeddings)
    
    print("\n第一個句子與所有句子的相似度:")
    for i, sim in enumerate(similarity[0]):
        print(f"  與句子 {i+1} 的相似度: {sim:.4f}")
    
    print("\n✅ 測試完成！")
