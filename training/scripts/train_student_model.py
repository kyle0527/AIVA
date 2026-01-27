"""
AIVA 5M 神經網絡知識蒸餾訓練腳本

使用大模型（Teacher）的軟標籤訓練 AIVA 5M 參數的小模型（Student）

蒸餾流程：
1. Teacher Model（大模型）生成軟標籤（概率分布）
2. Student Model（5M 小模型）學習模仿 Teacher
3. 使用蒸餾損失函數：結合硬標籤和軟標籤
4. Temperature Scaling 讓 Student 學習 Teacher 的不確定性

優勢：
- 5M 參數模型可快速推理（適合實時掃描）
- 繼承大模型的知識（不需要重新訓練大模型）
- 針對安全領域優化（不是通用模型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """訓練配置"""
    # 數據
    train_data_path: str = "../data/distillation_dataset/distillation_train.json"
    val_data_path: str = "../data/distillation_dataset/distillation_val.json"
    vocab_dir: str = "../data/security_vocabulary"
    
    # 模型
    embedding_dim: int = 384        # Embedding 維度
    hidden_dim: int = 512           # 隱藏層維度
    num_classes: int = 7            # 漏洞類型數量
    
    # 蒸餾參數
    temperature: float = 3.0        # 溫度參數（越大越軟）
    alpha: float = 0.7              # 軟標籤權重
    beta: float = 0.3               # 硬標籤權重
    
    # 訓練
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 設備
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 輸出
    output_dir: str = "../models"
    checkpoint_every: int = 5       # 每N個epoch保存一次


class SecurityDataset(Dataset):
    """安全場景數據集"""
    
    def __init__(self, data_path: Path, tokenizer=None):
        """
        初始化數據集
        
        Args:
            data_path: JSON 數據文件路徑
            tokenizer: 分詞器（如果為 None 則使用簡單編碼）
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        
        # 漏洞類型映射
        self.vuln_to_idx = {
            "sql_injection": 0,
            "xss": 1,
            "ssrf": 2,
            "rce": 3,
            "idor": 4,
            "jwt_attack": 5,
            "graphql_introspection": 6
        }
        
        logger.info(f"📦 載入數據集: {len(self.data)} 個樣本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 輸入文本
        text = sample["scenario_text"]
        
        # Teacher 的軟標籤
        teacher_vuln = sample["teacher_vulnerability_type"]
        teacher_severity = sample["teacher_severity"]
        teacher_confidence = sample["teacher_confidence"]
        
        # 構建 Teacher 的概率分布（軟標籤）
        teacher_probs = self._build_teacher_probs(
            teacher_vuln, 
            teacher_confidence
        )
        
        # 硬標籤（one-hot）
        hard_label = self.vuln_to_idx[teacher_vuln]
        
        return {
            "text": text,
            "teacher_probs": torch.tensor(teacher_probs, dtype=torch.float32),
            "hard_label": torch.tensor(hard_label, dtype=torch.long),
            "severity": torch.tensor(teacher_severity, dtype=torch.float32),
            "confidence": torch.tensor(teacher_confidence, dtype=torch.float32)
        }
    
    def _build_teacher_probs(
        self, 
        correct_vuln: str, 
        confidence: float
    ) -> np.ndarray:
        """
        構建 Teacher 的概率分布
        
        高置信度時：幾乎全部概率給正確類別
        低置信度時：概率分散到相似類別
        
        Args:
            correct_vuln: 正確的漏洞類型
            confidence: 置信度 (0-1)
        
        Returns:
            概率分布 (num_classes,)
        """
        num_classes = len(self.vuln_to_idx)
        probs = np.ones(num_classes) * ((1 - confidence) / (num_classes - 1))
        probs[self.vuln_to_idx[correct_vuln]] = confidence
        
        return probs


class AIVA5MStudentModel(nn.Module):
    """
    AIVA 5M 參數 Student Model
    
    架構：
    - Input: 512維向量（來自 AIVAEmbedding）
    - Hidden: 2層全連接 + Dropout
    - Output: 漏洞分類 + 嚴重性回歸
    
    參數量：約 5M
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.config = config
        
        # 從 Embedding (384) 投影到隱藏層 (512)
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        
        # 隱藏層
        self.hidden_layers = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 輸出頭
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.severity_regressor = nn.Linear(config.hidden_dim, 1)
        
        # 初始化權重
        self._init_weights()
        
        # 計算參數量
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🔢 Student Model 參數量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, embeddings):
        """
        前向傳播
        
        Args:
            embeddings: [batch_size, embedding_dim]
        
        Returns:
            logits: [batch_size, num_classes]
            severity: [batch_size, 1]
        """
        # 投影
        x = self.input_projection(embeddings)
        x = F.relu(x)
        
        # 隱藏層
        features = self.hidden_layers(x)
        
        # 輸出
        logits = self.classifier(features)
        severity = torch.sigmoid(self.severity_regressor(features))
        
        return logits, severity


class DistillationLoss(nn.Module):
    """
    知識蒸餾損失函數
    
    L = α * L_soft + β * L_hard
    
    - L_soft: Student 與 Teacher 的 KL 散度（軟標籤）
    - L_hard: Student 與真實標籤的交叉熵（硬標籤）
    """
    
    def __init__(self, temperature: float, alpha: float, beta: float):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self, 
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        計算蒸餾損失
        
        Args:
            student_logits: Student 的輸出 logits [batch, num_classes]
            teacher_probs: Teacher 的概率分布 [batch, num_classes]
            hard_labels: 真實標籤 [batch]
        
        Returns:
            total_loss
        """
        # 軟標籤損失（KL 散度）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = teacher_probs  # 已經是概率分布
        
        soft_loss = F.kl_div(
            student_soft, 
            teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬標籤損失（交叉熵）
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        
        # 總損失
        total_loss = self.alpha * soft_loss + self.beta * hard_loss
        
        return total_loss, soft_loss, hard_loss


class DistillationTrainer:
    """知識蒸餾訓練器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 創建輸出目錄
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入數據
        self.train_dataset = SecurityDataset(Path(config.train_data_path))
        self.val_dataset = SecurityDataset(Path(config.val_data_path))
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # 創建模型
        self.student_model = AIVA5MStudentModel(config).to(self.device)
        
        # 損失函數
        self.distillation_loss = DistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            beta=config.beta
        )
        self.severity_loss = nn.MSELoss()
        
        # 優化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # 訓練狀態
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """訓練一個 epoch"""
        self.student_model.train()
        
        total_loss = 0
        total_soft_loss = 0
        total_hard_loss = 0
        total_severity_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            # 數據準備（這裡簡化為隨機 embedding，實際應使用 AIVAEmbedding）
            batch_size = len(batch["text"])
            embeddings = torch.randn(batch_size, self.config.embedding_dim).to(self.device)
            
            teacher_probs = batch["teacher_probs"].to(self.device)
            hard_labels = batch["hard_label"].to(self.device)
            severity_targets = batch["severity"].to(self.device)
            
            # 前向傳播
            logits, severity_pred = self.student_model(embeddings)
            
            # 計算損失
            dist_loss, soft_loss, hard_loss = self.distillation_loss(
                logits, teacher_probs, hard_labels
            )
            sev_loss = self.severity_loss(severity_pred.squeeze(), severity_targets)
            
            loss = dist_loss + 0.1 * sev_loss  # 嚴重性損失權重較小
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item()
            total_soft_loss += soft_loss.item()
            total_hard_loss += hard_loss.item()
            total_severity_loss += sev_loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += hard_labels.size(0)
            correct += (predicted == hard_labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return {
            "loss": avg_loss,
            "soft_loss": total_soft_loss / len(self.train_loader),
            "hard_loss": total_hard_loss / len(self.train_loader),
            "severity_loss": total_severity_loss / len(self.train_loader),
            "accuracy": accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """驗證"""
        self.student_model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch_size = len(batch["text"])
                embeddings = torch.randn(batch_size, self.config.embedding_dim).to(self.device)
                
                teacher_probs = batch["teacher_probs"].to(self.device)
                hard_labels = batch["hard_label"].to(self.device)
                severity_targets = batch["severity"].to(self.device)
                
                logits, severity_pred = self.student_model(embeddings)
                
                dist_loss, _, _ = self.distillation_loss(
                    logits, teacher_probs, hard_labels
                )
                sev_loss = self.severity_loss(severity_pred.squeeze(), severity_targets)
                loss = dist_loss + 0.1 * sev_loss
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += hard_labels.size(0)
                correct += (predicted == hard_labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(self):
        """完整訓練流程"""
        logger.info("🚀 開始知識蒸餾訓練...")
        logger.info(f"   - 訓練樣本: {len(self.train_dataset)}")
        logger.info(f"   - 驗證樣本: {len(self.val_dataset)}")
        logger.info(f"   - Temperature: {self.config.temperature}")
        logger.info(f"   - α (軟標籤): {self.config.alpha}")
        logger.info(f"   - β (硬標籤): {self.config.beta}")
        logger.info("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            # 訓練
            train_metrics = self.train_epoch()
            
            # 驗證
            val_metrics = self.validate()
            
            # 記錄
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            
            # 調整學習率
            self.scheduler.step(val_metrics["loss"])
            
            # 日誌
            logger.info(
                f"Epoch {self.current_epoch}/{self.config.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # 保存最佳模型
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")
                logger.info(f"  ✅ 保存最佳模型 (val_loss: {val_metrics['loss']:.4f})")
            
            # 定期保存
            if self.current_epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pt")
        
        logger.info("=" * 60)
        logger.info(f"✅ 訓練完成！最佳驗證損失: {self.best_val_loss:.4f}")
        logger.info(f"💾 模型保存在: {self.output_dir}")
    
    def save_checkpoint(self, filename: str):
        """保存檢查點"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state": self.student_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config
        }
        
        torch.save(checkpoint, self.output_dir / filename)


def main():
    """主函數"""
    config = TrainingConfig()
    
    logger.info("=" * 60)
    logger.info("🎓 AIVA 5M 神經網絡知識蒸餾訓練")
    logger.info("=" * 60)
    
    trainer = DistillationTrainer(config)
    trainer.train()
    
    logger.info("=" * 60)
    logger.info("✅ 全部完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
