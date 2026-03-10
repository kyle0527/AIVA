# 模組雙向連結改進方案

## 📑 目錄

- [目標](#目標)
- [現狀問題](#現狀問題)
  - [1. 三個單向模組](#1-三個單向模組)
  - [2. 核心模組間缺少連接](#2-核心模組間缺少連接)
- [解決方案](#解決方案)
  - [方案概述](#方案概述)
  - [技術實現](#技術實現)
    - [1. 反向調用基礎設施](#1-反向調用基礎設施)
    - [2. cognitive_core 反向調用器](#2-cognitive_core-反向調用器)
    - [3. internal_exploration 反向調用器](#3-internal_exploration-反向調用器)
    - [4. task_planning 反向調用器](#4-task_planning-反向調用器)
- [實施計劃](#實施計劃)
  - [階段 1: 基礎設施 (1-2 天)](#階段-1-基礎設施-1-2-天)
  - [階段 2: 核心反向調用器 (2-3 天)](#階段-2-核心反向調用器-2-3-天)
  - [階段 3: Flow 配置更新 (1 天)](#階段-3-flow-配置更新-1-天)
  - [階段 4: 測試驗證 (2-3 天)](#階段-4-測試驗證-2-3-天)
- [預期效果](#預期效果)
- [關鍵優勢](#關鍵優勢)
- [注意事項](#注意事項)
  - [1. 避免循環依賴](#1-避免循環依賴)
  - [2. 性能考慮](#2-性能考慮)
  - [3. 錯誤處理](#3-錯誤處理)
  - [4. 文檔更新](#4-文檔更新)
- [下一步行動](#下一步行動)

---


**方案日期**: 2026-01-01

## 目標

在維持現有 840 flows 架構的前提下，通過添加「反向通道」實現模組間的完整雙向連結。

## 現狀問題

### 1. 三個單向模組

- **cognitive_core**: 124 入站, 0 出站 - 只能被調用
- **internal_exploration**: 201 入站, 0 出站 - 只能被調用
- **task_planning**: 48 入站, 0 出站 - 只能被調用

### 2. 核心模組間缺少連接

- cognitive_core ↔ task_planning
- cognitive_core ↔ core_capabilities
- task_planning ↔ core_capabilities

## 解決方案

### 方案概述

通過「反向通道接口」設計實現雙向連結，核心思想：

1. **不修改現有 flows** - 保持向後兼容
2. **添加反向調用器** - 為單向模組添加主動調用能力
3. **建立直通通道** - 為核心模組建立直接連接
4. **統一接口管理** - 使用統一的跨模組調用基類

### 技術實現

#### 1. 反向調用基礎設施

創建統一的反向調用基類：

```python
# services/core/aiva_core/common/outbound_client_base.py
class OutboundClientBase:
    '''跨模組調用的統一基類'''
    
    def _call_module(self, target_module, target_capability, **kwargs):
        '''統一的跨模組調用接口'''
        # 1. 驗證目標模組和能力
        # 2. 構建調用上下文
        # 3. 執行調用
        # 4. 處理錯誤和超時
        # 5. 記錄調用日誌
        pass
```

#### 2. cognitive_core 反向調用器

為認知核心添加主動調用能力：

**文件結構**:
```
cognitive_core/
├── outbound/
│   ├── __init__.py
│   ├── task_planner_client.py      # → task_planning
│   ├── capability_executor_client.py # → core_capabilities
│   └── learning_requester_client.py  # → external_learning
```

**實現示例**:
```python
# cognitive_core/outbound/task_planner_client.py
from ...common.outbound_client_base import OutboundClientBase

class TaskPlannerClient(OutboundClientBase):
    '''cognitive_core → task_planning 的反向通道'''
    
    def request_plan(self, objective, context):
        '''請求任務規劃'''
        return self._call_module(
            target_module='task_planning',
            target_capability='plan_generator',
            objective=objective,
            context=context
        )
    
    def verify_plan(self, plan):
        '''驗證計劃可行性'''
        return self._call_module(
            target_module='task_planning',
            target_capability='plan_validator',
            plan=plan
        )
```

#### 3. internal_exploration 反向調用器

為內部探索添加主動學習能力：

**文件結構**:
```
internal_exploration/
├── outbound/
│   ├── __init__.py
│   ├── learning_trigger_client.py    # → external_learning
│   ├── decision_validator_client.py  # → cognitive_core
│   └── storage_client.py             # → service_backbone
```

#### 4. task_planning 反向調用器

為任務規劃添加能力驗證：

**文件結構**:
```
task_planning/
├── outbound/
│   ├── __init__.py
│   ├── capability_checker_client.py  # → core_capabilities
│   ├── decision_confirmer_client.py  # → cognitive_core
│   └── resource_query_client.py      # → service_backbone
```

## 實施計劃

### 階段 1: 基礎設施 (1-2 天)

- [ ] 創建 `OutboundClientBase` 基類
- [ ] 為三個單向模組創建 `outbound/` 目錄
- [ ] 實現統一的調用接口和錯誤處理

### 階段 2: 核心反向調用器 (2-3 天)

**高優先級**:
- [ ] cognitive_core → task_planning
- [ ] task_planning → core_capabilities
- [ ] internal_exploration → external_learning

**中優先級**:
- [ ] cognitive_core → core_capabilities
- [ ] cognitive_core → external_learning
- [ ] internal_exploration → cognitive_core

**低優先級**:
- [ ] task_planning → cognitive_core
- [ ] internal_exploration → service_backbone

### 階段 3: Flow 配置更新 (1 天)

- [ ] 為每個新反向調用添加 flow 定義
- [ ] 更新 `latest_classification.json`
- [ ] 驗證新 flows 的正確性

### 階段 4: 測試驗證 (2-3 天)

- [ ] 單元測試: 每個反向調用器
- [ ] 集成測試: 跨模組調用
- [ ] 性能測試: 延遲和吞吐量
- [ ] 回歸測試: 確保不破壞現有功能

## 預期效果

| 指標 | 改進前 | 改進後 | 變化 |
|------|--------|--------|------|
| 單向模組數 | 3 | 0 | ✅ -3 |
| 核心模組連接 | 3/6 對 | 6/6 對 | ✅ +3 |
| 連接密度 | 30.0% | ~45-50% | ✅ +15-20% |
| 孤立模組 | 0 | 0 | ✅ 維持 |
| 預計新增 flows | 0 | 50-80 | 📈 增加 |

## 關鍵優勢

1. **最小侵入性**: 不修改現有 840 flows，完全向後兼容
2. **架構保持**: 模組邊界清晰，依賴關係明確
3. **漸進式實施**: 可分階段實施，每階段獨立驗證
4. **靈活性提升**: 支持更複雜的工作流和自主決策

## 注意事項

### 1. 避免循環依賴
- 使用依賴注入模式
- 明確調用方向和時序
- 添加循環檢測機制

### 2. 性能考慮
- 反向調用可能增加延遲
- 添加調用監控和日誌
- 設置超時和重試機制

### 3. 錯誤處理
- 統一的錯誤處理策略
- 優雅降級機制
- 詳細的錯誤日誌

### 4. 文檔更新
- 更新架構文檔
- 添加反向調用使用示例
- 更新 flow 圖表

## 下一步行動

1. **評審此方案**: 與團隊討論可行性
2. **選擇試點**: 先實施一個高優先級的反向調用器
3. **驗證效果**: 評估試點的效果和影響
4. **全面推廣**: 根據試點結果調整並全面實施
