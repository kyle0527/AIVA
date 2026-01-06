#!/usr/bin/env python3
"""
分類六大模組的對內/對外能力
分析 840 個 flows 在對內能力和對外能力的分佈
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

# 數據路徑
data_dir = Path("C:/Users/User/Downloads/data/internal_exploration")
json_file = data_dir / "latest_classification.json"

# 讀取數據
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data.get('flows', [])

# 六大模組的對內/對外分類
MODULE_CLASSIFICATION = {
    'cognitive_core': {
        'type': '對內能力',
        'description': 'AI認知處理、內部決策、智能分析',
        'examples': ['神經網路', 'RAG系統', 'AI模型管理', '認知處理']
    },
    'internal_exploration': {
        'type': '對內能力',
        'description': '自我探索、內部分析、能力發現',
        'examples': ['代碼分析', '能力掃描', '架構探索', '自我認知']
    },
    'task_planning': {
        'type': '混合能力',
        'description': '任務規劃與執行協調（內部規劃 + 外部執行）',
        'examples': ['任務調度', '執行計劃', '資源分配', '工作流管理']
    },
    'external_learning': {
        'type': '對外能力',
        'description': '外部學習、模型訓練、知識獲取',
        'examples': ['模型訓練', '強化學習', '數據學習', '經驗積累']
    },
    'core_capabilities': {
        'type': '對外能力',
        'description': '核心業務能力、攻擊與防禦、安全測試',
        'examples': ['漏洞掃描', '滲透測試', '攻擊執行', 'XSS/SQL注入']
    },
    'service_backbone': {
        'type': '對內能力',
        'description': '基礎設施、性能監控、系統協調',
        'examples': ['性能監控', '健康檢查', '協調器', '基礎服務']
    },
    'unknown': {
        'type': '通用組件',
        'description': '跨模組的共享組件（工具層、UI層）',
        'examples': ['系統工具', 'UI介面', '通用服務']
    }
}

print("\n" + "="*80)
print("🎯 六大模組對內/對外能力分類報告")
print("="*80)

# 統計各模組的 flow 數量
endpoint_modules = Counter()
for flow in flows:
    modules = flow.get('modules', [])
    if modules:
        endpoint_modules[modules[-1]] += 1

print("\n📊 整體分佈:")
print("-" * 80)

internal_total = 0
external_total = 0
hybrid_total = 0
common_total = 0

# 按類型分組
capability_types = {
    '對內能力': [],
    '對外能力': [],
    '混合能力': [],
    '通用組件': []
}

for module, count in sorted(endpoint_modules.items(), key=lambda x: x[1], reverse=True):
    classification = MODULE_CLASSIFICATION.get(module, {})
    cap_type = classification.get('type', '未分類')
    description = classification.get('description', 'N/A')
    
    percentage = count / len(flows) * 100
    
    # 圖示標記
    if cap_type == '對內能力':
        icon = '🔹'
        internal_total += count
        capability_types['對內能力'].append((module, count, percentage))
    elif cap_type == '對外能力':
        icon = '🔸'
        external_total += count
        capability_types['對外能力'].append((module, count, percentage))
    elif cap_type == '混合能力':
        icon = '🔶'
        hybrid_total += count
        capability_types['混合能力'].append((module, count, percentage))
    else:
        icon = '⚪'
        common_total += count
        capability_types['通用組件'].append((module, count, percentage))
    
    print(f"{icon} {module:25} : {count:3} ({percentage:5.1f}%) - {cap_type}")
    print(f"   {description}")
    print()

# 匯總統計
print("="*80)
print("📈 能力類型匯總:")
print("-" * 80)

total_classified = internal_total + external_total + hybrid_total

print(f"\n🔹 對內能力:  {internal_total:3} ({internal_total/len(flows)*100:5.1f}%)")
for module, count, percentage in capability_types['對內能力']:
    print(f"     • {module:23} : {count:3} ({percentage:5.1f}%)")

print(f"\n🔸 對外能力:  {external_total:3} ({external_total/len(flows)*100:5.1f}%)")
for module, count, percentage in capability_types['對外能力']:
    print(f"     • {module:23} : {count:3} ({percentage:5.1f}%)")

print(f"\n🔶 混合能力:  {hybrid_total:3} ({hybrid_total/len(flows)*100:5.1f}%)")
for module, count, percentage in capability_types['混合能力']:
    print(f"     • {module:23} : {count:3} ({percentage:5.1f}%)")

print(f"\n⚪ 通用組件:  {common_total:3} ({common_total/len(flows)*100:5.1f}%)")
for module, count, percentage in capability_types['通用組件']:
    print(f"     • {module:23} : {count:3} ({percentage:5.1f}%)")

print(f"\n{'='*80}")
print(f"總計: {len(flows)} 個 flows")
print(f"  - 已分類 (對內+對外+混合): {total_classified} ({total_classified/len(flows)*100:.1f}%)")
print(f"  - 通用組件: {common_total} ({common_total/len(flows)*100:.1f}%)")

# 詳細說明
print(f"\n{'='*80}")
print("📖 能力類型說明:")
print("-" * 80)

print("""
🔹 對內能力 (Internal Capabilities)
   服務於系統自身的功能，用於維護、優化、監控 AIVA 系統本身
   包括: cognitive_core, internal_exploration, service_backbone
   
   特點:
   • 不直接面向外部用戶
   • 用於系統自我認知、監控和維護
   • 提供基礎設施支持
   • 確保系統穩定運行

🔸 對外能力 (External Capabilities)
   直接服務於用戶/外部任務的功能，面向外部環境的操作能力
   包括: external_learning, core_capabilities
   
   特點:
   • 直接面向用戶需求
   • 執行外部任務（攻擊、掃描、訓練等）
   • 產生可見的外部結果
   • 體現系統的核心價值

🔶 混合能力 (Hybrid Capabilities)
   既服務於內部也服務於外部的功能
   包括: task_planning
   
   特點:
   • 內部規劃 + 外部執行
   • 連接對內和對外能力
   • 協調資源分配
   • 任務調度與管理

⚪ 通用組件 (Common Components)
   跨模組的共享工具和服務
   包括: unknown (tools/, ui/)
   
   特點:
   • 不屬於特定模組
   • 提供通用功能
   • 支持所有模組
""")

# 生成 Markdown 報告
print(f"\n{'='*80}")
print("📝 生成 Markdown 報告...")

report_content = f"""# 六大模組對內/對外能力分類報告

> **生成時間**: 2026-01-01  
> **數據來源**: latest_classification.json  
> **總 Flows**: {len(flows)}

## 📊 整體分佈

| 能力類型 | Flow 數量 | 百分比 | 包含模組 |
|---------|----------|--------|----------|
| 🔹 對內能力 | {internal_total} | {internal_total/len(flows)*100:.1f}% | cognitive_core, internal_exploration, service_backbone |
| 🔸 對外能力 | {external_total} | {external_total/len(flows)*100:.1f}% | external_learning, core_capabilities |
| 🔶 混合能力 | {hybrid_total} | {hybrid_total/len(flows)*100:.1f}% | task_planning |
| ⚪ 通用組件 | {common_total} | {common_total/len(flows)*100:.1f}% | unknown (tools/, ui/) |

## 🔹 對內能力詳細

### 1. cognitive_core (認知核心) - {endpoint_modules['cognitive_core']} flows ({endpoint_modules['cognitive_core']/len(flows)*100:.1f}%)

**功能**: {MODULE_CLASSIFICATION['cognitive_core']['description']}

**典型應用**:
{chr(10).join(f'- {ex}' for ex in MODULE_CLASSIFICATION['cognitive_core']['examples'])}

**代表性 Flows**:
- Flow 5, 13, 18 等

---

### 2. internal_exploration (內部探索) - {endpoint_modules['internal_exploration']} flows ({endpoint_modules['internal_exploration']/len(flows)*100:.1f}%)

**功能**: {MODULE_CLASSIFICATION['internal_exploration']['description']}

**典型應用**:
{chr(10).join(f'- {ex}' for ex in MODULE_CLASSIFICATION['internal_exploration']['examples'])}

**代表性 Flows**:
- Flow 0, 2, 3 等

---

### 3. service_backbone (服務骨幹) - {endpoint_modules['service_backbone']} flows ({endpoint_modules['service_backbone']/len(flows)*100:.1f}%)

**功能**: {MODULE_CLASSIFICATION['service_backbone']['description']}

**典型應用**:
{chr(10).join(f'- {ex}' for ex in MODULE_CLASSIFICATION['service_backbone']['examples'])}

**代表性 Flows**:
- Flow 1, 7 等

---

## 🔸 對外能力詳細

### 1. external_learning (外部學習) - {endpoint_modules['external_learning']} flows ({endpoint_modules['external_learning']/len(flows)*100:.1f}%)

**功能**: {MODULE_CLASSIFICATION['external_learning']['description']}

**典型應用**:
{chr(10).join(f'- {ex}' for ex in MODULE_CLASSIFICATION['external_learning']['examples'])}

**代表性 Flows**:
- Flow 4, 6, 11 等

---

### 2. core_capabilities (核心能力) - {endpoint_modules['core_capabilities']} flows ({endpoint_modules['core_capabilities']/len(flows)*100:.1f}%)

**功能**: {MODULE_CLASSIFICATION['core_capabilities']['description']}

**典型應用**:
{chr(10).join(f'- {ex}' for ex in MODULE_CLASSIFICATION['core_capabilities']['examples'])}

**代表性 Flows**:
- Flow 10, 12 等

---

## 🔶 混合能力詳細

### task_planning (任務規劃) - {endpoint_modules['task_planning']} flows ({endpoint_modules['task_planning']/len(flows)*100:.1f}%)

**功能**: {MODULE_CLASSIFICATION['task_planning']['description']}

**典型應用**:
{chr(10).join(f'- {ex}' for ex in MODULE_CLASSIFICATION['task_planning']['examples'])}

**特殊性**: 
- 內部規劃階段屬於對內能力
- 外部執行階段屬於對外能力
- 是連接內外的橋樑

**代表性 Flows**:
- Flow 59, 163, 205 等

---

## ⚪ 通用組件

### unknown - {endpoint_modules['unknown']} flows ({endpoint_modules['unknown']/len(flows)*100:.1f}%)

**功能**: {MODULE_CLASSIFICATION['unknown']['description']}

**位置**:
- services/core/tools/ (26 flows)
- services/core/ui/ (48 flows)

**典型應用**:
{chr(10).join(f'- {ex}' for ex in MODULE_CLASSIFICATION['unknown']['examples'])}

---

## 💡 戰略意義

### 對內能力的價值 ({internal_total} flows, {internal_total/len(flows)*100:.1f}%)

1. **自我認知與優化**
   - 持續監控系統狀態
   - 發現和改進系統能力
   - 確保系統穩定運行

2. **基礎設施支持**
   - 為對外能力提供堅實基礎
   - 資源管理與協調
   - 性能優化

3. **智能決策**
   - AI 認知處理
   - 內部知識管理
   - 自動化決策支持

### 對外能力的價值 ({external_total} flows, {external_total/len(flows)*100:.1f}%)

1. **直接價值交付**
   - 執行用戶任務
   - 產生可見成果
   - 體現系統核心價值

2. **業務能力**
   - 攻擊與防禦
   - 安全測試
   - 模型訓練

3. **持續進化**
   - 從外部環境學習
   - 累積經驗
   - 能力提升

### 混合能力的價值 ({hybrid_total} flows, {hybrid_total/len(flows)*100:.1f}%)

1. **內外連接**
   - 橋接對內和對外能力
   - 協調資源使用
   - 確保任務完成

2. **智能調度**
   - 任務優先級管理
   - 資源分配優化
   - 執行效率提升

---

## 🎯 架構洞察

### 平衡性分析

```
對內能力: {internal_total} flows ({internal_total/len(flows)*100:.1f}%)
對外能力: {external_total} flows ({external_total/len(flows)*100:.1f}%)
比例: {internal_total/external_total if external_total > 0 else 0:.2f} : 1
```

**解讀**:
- 對內能力占比 {internal_total/len(flows)*100:.1f}%，體現系統重視自我維護和優化
- 對外能力占比 {external_total/len(flows)*100:.1f}%，展現系統的業務價值
- 混合能力 ({hybrid_total} flows) 作為連接器，協調內外能力
- 通用組件 ({common_total} flows) 提供基礎支持

### 設計理念

1. **內外兼修**: 既重視對內優化，也重視對外價值交付
2. **協同作戰**: 混合能力協調對內對外，形成有機整體
3. **模組化架構**: 清晰的能力邊界，便於維護和擴展
4. **持續演進**: 通過內部探索和外部學習不斷進化

---

**生成時間**: 2026-01-01  
**維護狀態**: 🟢 活躍維護
"""

# 保存報告
report_path = Path("C:/D/fold7/AIVA-git/docs/INTERNAL_EXTERNAL_CAPABILITIES_ANALYSIS.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✅ 報告已保存至: {report_path}")

print("\n" + "="*80)
