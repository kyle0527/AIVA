# 訓練好的模型存放目錄

此目錄用於存放知識蒸餾訓練後的 AIVA 5M 神經網絡模型。

## 文件命名規範

- `best_model.pt` - 驗證集上表現最佳的模型
- `checkpoint_epoch_N.pt` - 第 N 個 epoch 的檢查點
- `final_model.pt` - 最終訓練完成的模型

## 模型文件結構

每個 `.pt` 文件包含：

```python
{
    "epoch": int,                    # 訓練到的 epoch
    "model_state": OrderedDict,      # 模型參數
    "optimizer_state": dict,         # 優化器狀態
    "scheduler_state": dict,         # 學習率調度器狀態
    "best_val_loss": float,          # 最佳驗證損失
    "history": dict,                 # 訓練歷史
    "config": TrainingConfig         # 訓練配置
}
```

## 載入模型

```python
import torch

# 載入檢查點
checkpoint = torch.load("best_model.pt")

# 恢復模型
model = AIVA5MStudentModel(checkpoint["config"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 查看訓練歷史
print(f"訓練 Epoch: {checkpoint['epoch']}")
print(f"最佳驗證損失: {checkpoint['best_val_loss']:.4f}")
print(f"訓練歷史: {checkpoint['history']}")
```

## 模型部署

將訓練好的模型整合到 AIVA 系統：

```bash
# 複製最佳模型到服務目錄
cp best_model.pt ../services/core/aiva_core/cognitive_core/neural/weights/aiva_5m_distilled.pt
```

---

*訓練完成後，此目錄會自動填充模型文件*
