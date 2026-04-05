# Anti-Hallucination 反幻覺驗證模組

> **路徑**: `cognitive_core/anti_hallucination`  
> **狀態**: ✅ 正常 | **文件數**: 2 | **最後更新**: 2026-04-05

## 概述

基於知識庫驗證 AI 生成的攻擊計畫，移除不合理步驟，防止 AI 產生幻覺輸出。設計原則為「有錯就報錯」，不使用降級機制，確保知識庫可用性。

## 核心組件

### anti_hallucination_module.py

- `KnowledgeBaseUnavailableError` - 知識庫不可用時拋出的異常類
- `AntiHallucinationModule` - 抗幻覺驗證主模組
  - 嚴格模式驗證，知識庫必須可用
  - 多層驗證：規則驗證 + 知識庫驗證 + 統計驗證
  - 基於 MITRE ATT&CK 的技術分類
  - 技術相依性映射和邏輯檢查

### __init__.py

- 模組初始化和導出配置
- 版本: `3.0.0-alpha`

## 依賴關係

- 內部依賴：知識庫實例 (需實現 `search` 方法)
- 外部依賴：`logging`, `pathlib`, `json`

## 主要功能

| 方法 | 說明 |
|------|------|
| `validate_attack_plan()` | 驗證整個攻擊計畫，移除不合理步驟 |
| `_validate_single_step()` | 驗證單個攻擊步驟的合理性 |
| `_validate_with_knowledge_base()` | 使用知識庫驗證步驟 |
| `_is_known_technique()` | 檢查是否為已知攻擊技術 |

## 使用範例

```python
from cognitive_core.anti_hallucination import AntiHallucinationModule

# 初始化模組（需要有效的知識庫）
validator = AntiHallucinationModule(knowledge_base=kb)

# 驗證攻擊計畫
validated_plan = validator.validate_attack_plan(attack_plan)

# 移除的可疑步驟會被記錄在 validation_history 中
print(validator.validation_history[-1])
```
