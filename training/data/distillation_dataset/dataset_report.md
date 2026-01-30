# AIVA 知識蒸餾訓練數據集報告

## 數據集概覽

- **總樣本數**: 737
- **訓練集**: 589 (79.9%)
- **驗證集**: 148 (20.1%)

## 漏洞類型分布

- **jwt_attack**: 113 (15.3%)
- **rce**: 110 (14.9%)
- **sql_injection**: 109 (14.8%)
- **ssrf**: 103 (14.0%)
- **idor**: 101 (13.7%)
- **graphql_introspection**: 101 (13.7%)
- **xss**: 100 (13.6%)

## 難度級別分布

- **medium**: 387 (52.5%)
- **easy**: 210 (28.5%)
- **hard**: 140 (19.0%)

## Teacher Model 標籤統計

- **平均嚴重性**: 0.782
- **平均置信度**: 0.857

## 使用方式

```python
# 載入訓練數據
import json

with open("distillation_train.json") as f:
    train_data = json.load(f)

# 訓練 Student Model（5M 參數）
for sample in train_data:
    input_text = sample["scenario_text"]
    
    # Teacher 的軟標籤
    teacher_vuln_type = sample["teacher_vulnerability_type"]
    teacher_severity = sample["teacher_severity"]
    teacher_confidence = sample["teacher_confidence"]
    
    # 使用知識蒸餾損失函數
    # loss = distillation_loss(student_output, teacher_output, temperature=3.0)
```

## 數據質量保證

- ✅ 所有樣本均來自專業安全知識文檔
- ✅ Teacher 標籤基於專家規則和真實上下文
- ✅ 涵蓋 7 種核心漏洞類型
- ✅ 包含 easy/medium/hard 三種難度
