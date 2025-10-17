"""
🤖 攻擊模式訓練器 - Attack Pattern Trainer
使用真實攻擊數據訓練 AI 模型識別安全威脅

功能:
1. 基於真實 OWASP 攻擊日誌訓練模型
2. 學習 8 種主要攻擊模式
3. 實時威脅檢測和分類
4. 生成防禦建議
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class AttackPatternTrainer:
    """攻擊模式訓練器"""

    # 攻擊類型常數
    SQL_INJECTION = 'SQL Injection'
    XSS_ATTACK = 'XSS Attack'
    AUTH_BYPASS = 'Authentication Bypass'
    PATH_TRAVERSAL = 'Path Traversal'
    FILE_UPLOAD_ATTACK = 'File Upload Attack'
    ERROR_BASED_ATTACK = 'Error-Based Attack'
    PARAM_POLLUTION = 'Parameter Pollution'
    BLOCKED_ACTIVITY = 'Blocked Activity'

    # 攻擊類型到 ID 的映射
    ATTACK_TYPES = {
        SQL_INJECTION: 0,
        XSS_ATTACK: 1,
        AUTH_BYPASS: 2,
        PATH_TRAVERSAL: 3,
        FILE_UPLOAD_ATTACK: 4,
        ERROR_BASED_ATTACK: 5,
        PARAM_POLLUTION: 6,
        BLOCKED_ACTIVITY: 7
    }
    
    def __init__(self):
        """初始化訓練器"""
        self.training_data = None
        self.model_weights = None
        self.attack_vectors = []
        self.labels = []
        self.training_history = []
        self.evaluation_metrics = {}
        self.test_size = 0.2  # 測試集比例
        
    def load_training_data(self, data_file: str = "_out/attack_training_data.json") -> bool:
        """載入訓練數據"""
        try:
            with open(data_file, encoding='utf-8') as f:
                self.training_data = json.load(f)
            
            logger.info(f"✓ 載入訓練數據: {data_file}")
            logger.info(f"  - 攻擊類型: {self.training_data['metadata']['attack_types']}")
            logger.info(f"  - 總攻擊數: {self.training_data['metadata']['total_attacks']}")
            logger.info(f"  - 成功率: {self.training_data['metadata']['success_rate']:.2f}%")
            
            return True
        except Exception as e:
            logger.error(f"✗ 載入失敗: {e}")
            return False
    
    def prepare_features(self) -> np.ndarray:
        """準備特徵向量"""
        logger.info("🔧 準備特徵向量...")
        
        # 為每種攻擊類型創建特徵向量
        for attack_type, type_id in self.ATTACK_TYPES.items():
            if attack_type in self.training_data['attack_patterns']:
                pattern_data = self.training_data['attack_patterns'][attack_type]
                
                # 特徵: [頻率, 數量歸一化, 樣本數量, 類型ID]
                features = [
                    pattern_data['frequency'],
                    pattern_data['count'] / self.training_data['metadata']['total_attacks'],
                    len(pattern_data['samples']) / 10.0,  # 歸一化到 0-1
                    type_id / len(self.ATTACK_TYPES)
                ]
                
                self.attack_vectors.append(features)
                self.labels.append(type_id)
        
        logger.info(f"✓ 準備了 {len(self.attack_vectors)} 個特徵向量")
        return np.array(self.attack_vectors)
    
    def train_model(self, epochs: int = 100, learning_rate: float = 0.01) -> dict:
        """訓練簡單的攻擊檢測模型"""
        logger.info(f"🎓 開始訓練模型 (epochs={epochs}, lr={learning_rate})...")
        
        features = np.array(self.attack_vectors)
        labels = np.array(self.labels)
        
        # 初始化權重 (簡單線性模型)
        n_features = features.shape[1]
        rng = np.random.default_rng(42)  # 固定種子確保可重現性
        self.model_weights = rng.normal(0, 0.01, (n_features, len(self.ATTACK_TYPES)))
        bias = np.zeros(len(self.ATTACK_TYPES))
        
        # 訓練循環
        for epoch in range(epochs):
            # 前向傳播
            logits = np.dot(features, self.model_weights) + bias
            predictions = self._softmax(logits)
            
            # 計算損失 (交叉熵)
            loss = -np.mean(np.log(predictions[range(len(labels)), labels] + 1e-10))
            
            # 計算準確率
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == labels)
            
            # 記錄訓練歷史
            if epoch % 10 == 0:
                self.training_history.append({
                    'epoch': epoch,
                    'loss': float(loss),
                    'accuracy': float(accuracy)
                })
                logger.info(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {accuracy:.2%}")
            
            # 簡單梯度下降 (實際應使用反向傳播)
            grad = (predictions - self._one_hot(labels, len(self.ATTACK_TYPES))) / len(labels)
            self.model_weights -= learning_rate * np.dot(features.T, grad)
            bias -= learning_rate * np.sum(grad, axis=0)
        
        logger.info("✓ 訓練完成!")
        
        return {
            'final_loss': float(loss),
            'final_accuracy': float(accuracy),
            'epochs': epochs,
            'training_samples': len(labels)
        }
    
    def predict_attack_type(self, features: list[float]) -> tuple[str, float]:
        """預測攻擊類型"""
        if self.model_weights is None:
            raise ValueError("模型尚未訓練!")
        
        features_array = np.array(features).reshape(1, -1)
        logits = np.dot(features_array, self.model_weights)
        predictions = self._softmax(logits)[0]
        
        predicted_id = np.argmax(predictions)
        confidence = predictions[predicted_id]
        
        # 找到對應的攻擊類型
        attack_type = list(self.ATTACK_TYPES.keys())[
            list(self.ATTACK_TYPES.values()).index(predicted_id)
        ]
        
        return attack_type, float(confidence)
    
    def save_model(self, output_file: str = "_out/attack_detection_model.json"):
        """保存訓練好的模型"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'attack_types': self.ATTACK_TYPES,
                'training_samples': len(self.attack_vectors)
            },
            'weights': self.model_weights.tolist() if self.model_weights is not None else None,
            'training_history': self.training_history,
            'attack_vectors': self.attack_vectors,
            'labels': self.labels
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"✓ 模型已保存: {output_file}")
        return str(output_path)
    
    def generate_defense_recommendations(self) -> dict[str, list[str]]:
        """生成防禦建議"""
        recommendations = {}
        
        if not self.training_data:
            return recommendations
        
        for attack_type, pattern_data in self.training_data['attack_patterns'].items():
            count = pattern_data['count']
            recs = self._get_attack_specific_recommendations(attack_type, count)
            if recs:
                recommendations[attack_type] = recs
        
        return recommendations
    
    def _get_attack_specific_recommendations(self, attack_type: str, count: int) -> list[str]:
        """獲取特定攻擊類型的防禦建議"""
        priority = self._get_priority_level(attack_type, count)
        priority_text = f"優先級: {priority} (檢測到 {count} 次)"
        
        base_recommendations = {
            'SQL Injection': [
                "使用參數化查詢 (Prepared Statements)",
                "實施 ORM (Object-Relational Mapping)",
                "啟用 SQL 注入 WAF 規則",
            ],
            'XSS Attack': [
                "實施 Content Security Policy (CSP)",
                "輸出編碼所有用戶輸入",
                "使用 HTTPOnly 和 Secure cookies",
            ],
            'Authentication Bypass': [
                "強制所有端點進行身份驗證",
                "實施 JWT token 驗證",
                "啟用多因素驗證 (MFA)",
                "實施速率限制",
            ],
            'Path Traversal': [
                "驗證和清理所有文件路徑",
                "使用白名單限制可訪問路徑",
                "實施 chroot 環境",
            ],
            'File Upload Attack': [
                "驗證文件類型和擴展名",
                "掃描上傳文件的惡意內容",
                "限制文件大小",
                "隔離上傳文件存儲",
            ],
        }
        
        if attack_type in base_recommendations:
            recommendations = base_recommendations[attack_type].copy()
            recommendations.append(priority_text)
            return recommendations
        
        return []
    
    def _get_priority_level(self, attack_type: str, count: int) -> str:
        """根據攻擊類型和數量決定優先級"""
        thresholds = {
            'SQL Injection': 50,
            'XSS Attack': 30,
            'Authentication Bypass': 100,
            'Path Traversal': 20,
            'File Upload Attack': 10,
        }
        
        threshold = thresholds.get(attack_type, 50)
        if count > threshold:
            return '高'
        elif count > 5:
            return '中'
        else:
            return '低'
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax 激活函數"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
        """One-hot 編碼"""
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[range(len(labels)), labels] = 1
        return one_hot


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🤖 攻擊模式訓練器')
    parser.add_argument('--data', '-d',
                       default='_out/attack_training_data.json',
                       help='訓練數據文件')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='訓練輪數 (預設: 100)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01,
                       help='學習率 (預設: 0.01)')
    parser.add_argument('--output', '-o',
                       default='_out/attack_detection_model.json',
                       help='模型輸出文件')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🤖 攻擊模式訓練器 - Attack Pattern Trainer")
    print("=" * 70)
    print()
    
    # 初始化訓練器
    trainer = AttackPatternTrainer()
    
    # 載入數據
    print("📥 載入訓練數據...\n")
    if not trainer.load_training_data(args.data):
        print("❌ 無法載入訓練數據")
        return
    
    # 準備特徵
    print("\n🔧 準備特徵向量...\n")
    trainer.prepare_features()
    
    # 訓練模型
    print(f"\n🎓 訓練模型 (epochs={args.epochs})...\n")
    results = trainer.train_model(epochs=args.epochs, learning_rate=args.learning_rate)
    
    print("\n✓ 訓練結果:")
    print(f"  - 最終損失: {results['final_loss']:.4f}")
    print(f"  - 最終準確率: {results['final_accuracy']:.2%}")
    print(f"  - 訓練樣本: {results['training_samples']}")
    
    # 保存模型
    print("\n💾 保存模型...\n")
    model_file = trainer.save_model(args.output)
    print(f"✓ 模型已保存至: {model_file}")
    
    # 生成防禦建議
    print("\n💡 生成防禦建議...\n")
    recommendations = trainer.generate_defense_recommendations()
    
    print("=" * 70)
    print("🛡️ 防禦建議")
    print("=" * 70)
    
    for attack_type, recs in recommendations.items():
        print(f"\n【{attack_type}】")
        for rec in recs:
            print(f"  • {rec}")
    
    print("\n" + "=" * 70)
    print("✓ 訓練完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
