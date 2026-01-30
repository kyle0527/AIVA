# AIVA 訓練模組

> 此模組獨立於 services（運行時服務），專門用於 AIVA 神經網絡的離線訓練

## 📁 目錄結構

```
training/
├── data/                          # 訓練數據
│   ├── security_vocabulary/       # 安全領域詞彙表
│   │   ├── security_vocabulary.json
│   │   ├── security_terms.txt     # 63個安全術語
│   │   ├── security_training_corpus.txt  # 177條訓練語料
│   │   ├── term_contexts.json
│   │   └── vocabulary_report.md
│   │
│   └── distillation_dataset/      # 知識蒸餾數據集
│       ├── distillation_train.json  # 589個訓練樣本
│       ├── distillation_val.json    # 148個驗證樣本
│       ├── distillation_train.csv
│       ├── distillation_val.csv
│       └── dataset_report.md
│
├── scripts/                       # 訓練腳本
│   ├── build_security_vocabulary.py      # 步驟1: 構建詞彙表
│   ├── generate_distillation_dataset.py # 步驟2: 生成訓練數據
│   └── train_student_model.py           # 步驟3: 訓練模型
│
├── models/                        # 訓練好的模型（輸出）
│   └── (訓練後自動生成)
│
└── README.md                      # 本文件
```

## 🎯 訓練流程

### 步驟 1: 構建安全詞彙表

從4份安全知識文檔中提取專業術語，構建領域詞彙表。

```powershell
cd training/scripts
python build_security_vocabulary.py --docs-dir "C:\Users\User\Downloads\新增資料夾" --output-dir "../data/security_vocabulary"
```

**輸出**:
- 63個安全術語（JWT, GraphQL, RCE, XSS...）
- 177條訓練語料（含上下文）
- 詞彙表統計報告

### 步驟 2: 生成蒸餾訓練數據集

使用 Teacher Model（大模型）的知識生成訓練樣本。

```powershell
python generate_distillation_dataset.py --vocab-dir "../data/security_vocabulary" --output-dir "../data/distillation_dataset" --samples-per-type 100
```

**輸出**:
- 737個訓練樣本（7種漏洞類型）
- 訓練集 (80%) / 驗證集 (20%)
- 包含 Teacher 的軟標籤（置信度、嚴重性）

### 步驟 3: 訓練 Student Model

使用知識蒸餾訓練 AIVA 5M 神經網絡。

```powershell
python train_student_model.py
```

**配置**:
- 模型: 5M 參數
- Temperature: 3.0
- 軟標籤權重 (α): 0.7
- 硬標籤權重 (β): 0.3

**輸出**:
- `../models/best_model.pt` - 最佳模型
- `../models/checkpoint_epoch_*.pt` - 檢查點
- 訓練歷史記錄

## 📊 數據集統計

### 詞彙表
- **總術語**: 63個
- **總出現**: 791次
- **來源**: 4份安全知識文檔

**Top 10 術語**:
1. JWT (76次)
2. GraphQL (68次)
3. RCE (59次)
4. XSS (56次)
5. WebSocket (45次)
6. Cloudflare (35次)
7. IDOR (28次)
8. F5 (28次)
9. Imperva (27次)
10. Authorization (25次)

### 蒸餾數據集
- **總樣本**: 737
- **訓練集**: 589 (80%)
- **驗證集**: 148 (20%)

**漏洞類型分布**:
- SQL Injection
- XSS (Cross-Site Scripting)
- SSRF (Server-Side Request Forgery)
- RCE (Remote Code Execution)
- IDOR (Insecure Direct Object Reference)
- JWT Attack
- GraphQL Introspection

**難度分布**:
- Easy: 30%
- Medium: 50%
- Hard: 20%

## 🔬 知識蒸餾原理

### Teacher-Student 架構

```
┌─────────────────────────────────────┐
│ Teacher Model (大語言模型)          │
│ - 專家級安全知識                     │
│ - 生成軟標籤（概率分布）             │
│ - 提供置信度和推理                   │
└────────────┬────────────────────────┘
             │ 知識傳遞
             ▼
┌─────────────────────────────────────┐
│ Student Model (AIVA 5M)             │
│ - 5M 參數（輕量級）                 │
│ - 學習 Teacher 的決策模式            │
│ - 實時推理能力                       │
└─────────────────────────────────────┘
```

### 損失函數

```
L_total = α × L_soft + β × L_hard

L_soft  = KL散度(Student || Teacher)  # 學習不確定性
L_hard  = 交叉熵(Student, 真實標籤)   # 學習正確分類
```

### Temperature Scaling

使用 Temperature=3.0 軟化概率分布，讓 Student 學習到 Teacher 的「不確定性」和「相似類別關係」。

## 🚀 整合到 AIVA 系統

訓練完成後，將模型整合到運行時服務：

```python
# services/core/aiva_core/cognitive_core/neural/real_neural_core.py

# 載入訓練好的 Student Model
student_weights = torch.load("training/models/best_model.pt")
self.ai_core.load_state_dict(student_weights["model_state"])
```

## 📝 依賴項

```bash
pip install torch transformers datasets
```

## ⚙️ 配置修改

修改 `scripts/train_student_model.py` 中的 `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    # 數據路徑
    train_data_path: str = "../data/distillation_dataset/distillation_train.json"
    val_data_path: str = "../data/distillation_dataset/distillation_val.json"
    
    # 蒸餾參數
    temperature: float = 3.0      # 調整軟化程度
    alpha: float = 0.7            # 軟標籤權重
    beta: float = 0.3             # 硬標籤權重
    
    # 訓練參數
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
```

## 📈 訓練監控

查看訓練進度：

```powershell
# 查看數據集報告
cat ../data/distillation_dataset/dataset_report.md

# 查看詞彙表報告
cat ../data/security_vocabulary/vocabulary_report.md

# 訓練時實時監控
# 輸出會顯示：Train Loss, Val Loss, Train Acc, Val Acc
```

## 🎓 知識來源

訓練數據來自以下專業安全文檔：

1. **AI 掃描器漏洞判斷邏輯資料庫.md**
   - SQL注入、XSS、SSRF、IDOR 檢測邏輯

2. **AI 識別高危險 CVE 模組.md**
   - Log4Shell、Spring4Shell、ProxyShell 等

3. **WAF 繞過技術字典生成.md**
   - 編碼混淆、協議攻擊、WAF 繞過

4. **Web 架構安全漏洞檢測指南.md**
   - GraphQL、REST API、WebSocket、JWT

## ⚠️ 注意事項

1. **訓練模組獨立性**: 此目錄與 `services/` 完全分離，不影響運行時系統
2. **數據更新**: 當安全知識文檔更新時，重新執行步驟1-3
3. **模型版本**: 建議保留每次訓練的檢查點，方便回退
4. **資源需求**: 訓練建議使用 GPU（CUDA），CPU 也可但較慢

## 📚 參考資料

- [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [DistilBERT: Distilled version of BERT](https://arxiv.org/abs/1910.01108)
- AIVA 系統架構文檔: `../docs/SERVICES_ARCHITECTURE_ANALYSIS.md`

---

**最後更新**: 2026-01-20  
**AIVA 版本**: 840+  
**訓練框架**: PyTorch 2.0+
