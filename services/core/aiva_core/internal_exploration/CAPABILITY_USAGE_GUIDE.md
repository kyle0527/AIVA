# AIVA 系統能力使用指南
## 完整的系統能力清單與使用方法

**生成日期**: 2025年12月6日  
**數據來源**: 268個流程分析結果  
**總能力數**: 21個  

---

## 🎯 核心能力分類

### 【AI_CORE】AI 核心系統 (4個能力)
> 提供基礎人工智能功能和神經網絡處理能力

#### 1. **models** - 神經網絡模型管理 ⭐⭐⭐
- **功能**: 管理和維護 AI 模型的核心組件
- **使用頻率**: 155次 (最高)
- **複雜度**: 0.60 (中等)
- **風險等級**: high
- **執行路徑**: `bio → trainer → neural → network → rl → models`
- **依賴**: network, bio, trainer, rl, neural
- **相似能力**: trainers (0.98), core (0.85)

#### 2. **trainers** - 模型訓練管理
- **功能**: 批量管理和協調多個訓練器
- **使用頻率**: 30次
- **執行路徑**: 2種不同實現方式
- **適用場景**: 大規模模型訓練

#### 3. **core** - 核心處理器
- **功能**: 提供系統的核心處理邏輯和基礎功能
- **使用頻率**: 6次
- **執行路徑**: 4種不同實現方式
- **適用場景**: 系統核心功能調用

#### 4. **network** - AI網絡處理
- **功能**: 神經網絡結構管理
- **使用頻率**: 1次
- **適用場景**: 網絡架構設計

### 【ORCHESTRATION】流程編排 (1個能力)
> 任務協調、流程管理和系統編排

#### **orchestrator** - 系統編排器 ⭐⭐
- **功能**: 協調各個系統組件的執行流程和任務分配
- **使用頻率**: 13次 (重要)
- **複雜度**: 0.71 (高)
- **風險等級**: medium
- **執行路徑**: 3種可選方式
  1. `initial → surface → exploit → orchestrator` (最佳路徑，效率1.0)
  2. `bio → trainer → rl → trainers → capability → orchestrator` (最常用)
  3. `bio → trainer → model → trainer → training → orchestrator`
- **依賴**: bio, capability, surface, exploit, trainer
- **使用場景**: 複雜任務協調、多組件編排

### 【STORAGE】數據存儲 (1個能力)
> 數據持久化、存儲管理和緩存機制

#### **store** - 數據存儲中心 ⭐
- **功能**: 管理系統數據的持久化和檢索機制
- **使用頻率**: 25次
- **複雜度**: 0.60 (中等)
- **風險等級**: low
- **執行路徑**: `bio → trainer → rl → trainers → vector → store`
- **依賴**: trainer, trainers, rl, vector, bio
- **使用場景**: 數據持久化、向量存儲

### 【MANAGEMENT】系統管理 (2個能力)
> 資源管理、流程控制和系統監控

#### 1. **manager** - 資源管理器
- **功能**: 負責系統資源的分配和管理操作
- **使用頻率**: 6次
- **執行路徑**: 6種不同實現方式 (最多選擇)
- **使用場景**: 資源分配、系統管理

#### 2. **monitor** - 系統監控
- **功能**: 系統狀態監控和性能追蹤
- **使用頻率**: 4次
- **使用場景**: 系統監控、性能分析

### 【LEARNING】機器學習 (1個能力)
> 實現模型訓練、強化學習和自適應算法

#### **trainer** - 模型訓練器
- **功能**: 實現智能體的強化學習算法和策略優化
- **使用頻率**: 4次
- **執行路徑**: 2種實現方式
- **使用場景**: 單一模型訓練、算法優化

### 【SERVICE】服務接口 (1個能力)
> API 服務、接口提供和外部交互

#### **interface** - 服務提供者
- **功能**: 向外部系統提供 API 服務和功能接口
- **使用頻率**: 1次
- **使用場景**: API 接口、外部服務

---

## 🚀 實際可用腳本

### 內部探索相關 (立即可用)
```bash
✅ aiva_flow_analyzer      # 流程分析器
✅ capability_analyzer     # 能力分析器  
✅ capability_classifier   # 能力分類器
✅ capability_registry     # 能力註冊中心
✅ path_difference_analyzer # 路徑差異分析器
✅ integration_module_sync  # 整合模塊同步
✅ capability_cli          # 統一命令行工具
```

### AI 核心相關 (需定位路徑)
```bash
• models                   # AI模型管理
• model_trainer           # 模型訓練器
• rl_models              # 強化學習模型
• ai_model_manager        # AI模型管理器
• scalable_bio_trainer    # 可擴展生物訓練器
```

### 編排管理相關
```bash
• orchestrator            # 系統編排器
• capability_orchestrator # 能力編排器
• training_orchestrator   # 訓練編排器
• experience_manager      # 經驗管理器
• task_queue_manager      # 任務隊列管理器
```

### 存儲相關
```bash
• postgresql_vector_store # PostgreSQL向量存儲
• unified_vector_store    # 統一向量存儲
• vector_store           # 向量存儲
```

---

## 📋 使用方法指南

### 1. 查看所有可用能力
```bash
cd "c:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration"
python capability_cli.py --flow-dir "c:\D\fold7\AIVA-git\flow_analysis_ai_core" classify
```

### 2. 查看特定能力詳情
```bash
# 查看 AI 模型管理能力
python capability_cli.py --flow-dir "..." classify --capability models

# 查看系統編排能力  
python capability_cli.py --flow-dir "..." classify --capability orchestrator
```

### 3. 分析能力執行路徑
```bash
# 查看有多條路徑的能力
python capability_cli.py --flow-dir "..." compare-paths

# 詳細分析特定能力的路徑差異
python capability_cli.py --flow-dir "..." compare-paths --endpoint orchestrator --detailed
```

### 4. 驗證路徑可執行性
```bash
# 驗證指定路徑
python capability_cli.py validate-paths --nodes aiva_flow_analyzer capability_analyzer

# 驗證所有發現的路徑
python capability_cli.py --flow-dir "..." validate-paths
```

### 5. 構建可執行指令序列
```bash
# 構建內部分析能力 (已驗證可用)
python capability_cli.py build-commands --capability "內部分析" --nodes aiva_flow_analyzer capability_analyzer --detailed

# 構建自定義能力
python capability_cli.py build-commands --capability "數據分析" --nodes <node1> <node2> --detailed
```

### 6. 執行能力 (乾運行模式)
```bash
# 乾運行測試
python capability_cli.py execute-capability --sequence-id "內部分析_0" --dry-run --detailed

# 實際執行 (謹慎使用)
python capability_cli.py execute-capability --sequence-id "內部分析_0" --detailed
```

---

## 🎯 推薦使用場景

### 🔥 高頻核心能力
1. **models** (155次) - 所有AI模型相關操作的核心
2. **trainers** (30次) - 大規模訓練任務
3. **store** (25次) - 數據存儲和檢索
4. **orchestrator** (13次) - 複雜流程編排

### ⚡ 效率最佳路徑
- **orchestrator**: 使用 `initial → surface → exploit → orchestrator` (效率1.0)
- **models**: 標準路徑 `bio → trainer → neural → network → rl → models`

### 🛡️ 風險管理
- **高風險**: models, core (謹慎使用)
- **中風險**: orchestrator, manager
- **低風險**: store, interface, monitor

### 🔧 實際可操作
```bash
# 立即可用的完整工作流
1. 分析系統能力: python capability_cli.py classify
2. 找到最佳路徑: python capability_cli.py compare-paths --endpoint <target>
3. 驗證可執行性: python capability_cli.py validate-paths --nodes <path>
4. 構建指令序列: python capability_cli.py build-commands --capability <name> --nodes <path>
5. 安全執行測試: python capability_cli.py execute-capability --sequence-id <id> --dry-run
```

---

## 💡 重要提醒

### ✅ 確認可用
- **內部探索工具**: 100% 可用，已驗證
- **分析和分類功能**: 完全正常
- **數據同步**: 已成功整合到統一存儲

### ⚠️ 需要注意
- **實際腳本路徑**: 部分能力名稱與實際腳本名稱不完全一致
- **參數配置**: 構建的指令可能需要調整參數
- **依賴關係**: 複雜能力需要確保依賴組件可用

### 🚀 建議工作流
1. 先使用 `classify` 了解全局能力
2. 用 `compare-paths` 找到最優執行方式
3. 用 `validate-paths` 確認可執行性
4. 用 `--dry-run` 模式安全測試
5. 逐步實際執行關鍵能力

**總結**: AIVA 目前擁有 21 個明確的系統能力，覆蓋 AI 核心、編排、存儲、管理等 7 大領域，其中內部探索工具已完全可用，其他能力正在逐步驗證和完善中。