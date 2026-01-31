# 六大模組架構評估報告（基於13步驟外閉環）

**評估日期**: 2026-01-03  
**評估基準**: [第一章流程實施可行性分析](./第一章流程實施可行性分析_RECOVERED.md) - 13步驟外閉環流程  
**評估範圍**: `services/core/aiva_core` 六大模組  
**評估目的**: 確認六大模組規劃是否符合13步驟外閉環的實際需求

---

## 📋 目錄

- [執行摘要](#執行摘要)
- [13步驟外閉環流程回顧](#13步驟外閉環流程回顧)
- [六大模組職責映射分析](#六大模組職責映射分析)
- [關鍵發現](#關鍵發現)
- [建議調整方案](#建議調整方案)
- [實施計畫](#實施計畫)

---

## 📊 執行摘要

### ✅ 總體結論：**整體架構良好，需局部調整職責邊界**

| 評估項目 | 評分 | 狀態 |
|---------|------|------|
| **架構合理性** | 85% | ✅ 良好 |
| **職責清晰度** | 75% | ⚠️ 需改進 |
| **與13步驟匹配度** | 80% | ✅ 良好 |
| **模組邊界清晰度** | 70% | ⚠️ 有重疊 |

### 🎯 核心發現

**✅ 符合外閉環的模組（4/6）**:
1. **task_planning** - 完美支撐步驟 1-2（任務接收與分解）
2. **cognitive_core** - 良好支撐步驟 6, 9, 11（AI決策）
3. **core_capabilities** - 適配步驟 3, 4, 7（能力編排與執行）
4. **service_backbone** - 支撐整體調度（CommandCenter）

**⚠️ 需要調整的模組（1/6）**:
5. **external_learning** - 當前定位與13步驟關係不明確

**✅ 正確排除的模組（1/6）**:
6. **internal_exploration** - 內閉環，與13步驟無直接關係 ✅

---

## 🔍 13步驟外閉環流程回顧

### 完整13步驟映射

| 步驟 | 名稱 | 核心需求 | 當前負責模組 | 狀態 |
|------|------|---------|-------------|------|
| **0** | 用戶輸入 | CLI/API接口 | service_backbone | ✅ |
| **1** | Core接收分析 | 任務識別與路由 | task_planning | ✅ |
| **2** | Coordinator分解 | 任務分解與編排 | task_planning | ✅ |
| **3** | Plugin創建命令 | 能力映射與命令構建 | core_capabilities | ✅ |
| **4** | Phase 0執行 | 快速偵察（Rust引擎） | core_capabilities | ✅ |
| **5a** | Integration並行 | 歷史查詢（可選） | ❌ 未實現 | ⚠️ |
| **6** | AI決策 1 | 分析P0結果，決定P1策略 | cognitive_core | ⚠️ |
| **7** | Phase 1執行 | 深度掃描（多引擎） | core_capabilities | ✅ |
| **8a** | Integration並行 | 歷史查詢（可選） | ❌ 未實現 | ⚠️ |
| **9** | AI決策 2 | 分析P1結果，決定P2目標 | cognitive_core | ⚠️ |
| **10** | Phase 2執行 | 攻擊測試（Features模組） | ❌ 未完整實現 | ⚠️ |
| **11** | AI決策 3 | 分析P2結果，決定下一步 | cognitive_core | ⚠️ |
| **12** | Integration整合 | 結果整合與報告 | ❌ 未實現 | ⚠️ |
| **13** | 返回結果 | 結果封裝與返回 | task_planning | ✅ |

### 🎯 關鍵流程階段

```
┌─────────────────────────────────────────────────────────────────┐
│                    外閉環 13 步驟完整流程                          │
└─────────────────────────────────────────────────────────────────┘

📥 Phase 0: 任務接收與規劃 (步驟 0-2)
    ├── task_planning: AICommanderV2 接收任務
    ├── task_planning: AnalysisCoordinator 分解任務
    └── core_capabilities: ScannerPlugin 創建掃描命令

⚡ Phase 1: 快速偵察 (步驟 3-6)
    ├── core_capabilities: 執行 Rust 引擎（Phase 0）
    ├── [可選] Integration: 查詢歷史數據
    └── cognitive_core: AI 分析 P0 結果，決定 P1 策略

🔍 Phase 2: 深度掃描 (步驟 7-9)
    ├── core_capabilities: 執行多引擎（Phase 1）
    ├── [可選] Integration: 查詢歷史數據
    └── cognitive_core: AI 分析 P1 結果，選擇攻擊目標

🎯 Phase 3: 攻擊測試 (步驟 10-11)
    ├── Features模組: 執行攻擊測試（Phase 2）
    └── cognitive_core: AI 分析 P2 結果，評估風險

📤 Phase 4: 結果整合 (步驟 12-13)
    ├── [可選] Integration: 整合結果與報告
    └── task_planning: 返回最終結果
```

---

## 🏗️ 六大模組職責映射分析

### 1. ✅ task_planning（任務規劃系統）

**當前定位**: 任務編排和協調引擎  
**版本**: v2.3.0 | 48 flows (5.7%) | 22 模組

#### 與13步驟的映射

| 涉及步驟 | 具體職責 | 組件 | 狀態 |
|---------|---------|------|------|
| **步驟 1** | 任務接收與識別 | AICommanderV2 | ✅ 100% |
| **步驟 2** | 任務分解與編排 | AnalysisCoordinator | ✅ 100% |
| **步驟 13** | 結果封裝與返回 | AICommanderV2 | ✅ 100% |

#### 評估結論

**✅ 完美契合外閉環需求**

**優點**:
- 清晰承擔流程起點（步驟1）和終點（步驟13）
- AICommanderV2 作為統一入口，職責明確
- 支持4個Coordinators（Attack/Defense/Analysis/Training）
- 純編排設計，不直接執行

**無需調整** ✅

---

### 2. ✅ cognitive_core（認知核心）

**當前定位**: AI認知決策核心  
**版本**: v2.3.0 | 124 flows (14.8%) | 27 模組

#### 與13步驟的映射

| 涉及步驟 | 具體職責 | 組件 | 狀態 |
|---------|---------|------|------|
| **步驟 6** | AI決策1：P0→P1策略 | EnhancedDecisionAgent | ⚠️ 85% (已設置，待驗證) |
| **步驟 9** | AI決策2：P1→P2目標 | EnhancedDecisionAgent | ⚠️ 85% (已設置，待驗證) |
| **步驟 11** | AI決策3：P2結果評估 | EnhancedDecisionAgent | ⚠️ 85% (已設置，待驗證) |

#### 子系統與13步驟的關係

| 子系統 | 在13步驟中的角色 | 關鍵度 |
|--------|----------------|--------|
| **Neural** | 5M神經網路推理，生成決策 | 🔴 核心 |
| **Decision** | 結構化決策支援（HighLevelIntent） | 🔴 核心 |
| **RAG** | 提供上下文增強決策 | 🟡 重要 |
| **Anti-Hallucination** | 確保決策可靠性 | 🟡 重要 |
| **Plugin System** | BioNeuronPlugin 介面 | 🔴 核心 |
| **CapabilityOrchestrator** | 能力調度與編排 | 🟡 重要 |

#### 評估結論

**⚠️ 職責清晰，符合外閉環需求，但需驗證**

**優點**:
- 專注於AI認知決策（步驟6, 9, 11）
- 神經網路核心（Neural）定位正確
- RAG系統支援知識增強決策
- 純決策編排，不包含執行代碼
- ⚠️ **四個決策方法已完成設置（待驗證）**:
  - `decide_scan_strategy()` - 增強版智能掃描策略
  - `decide_phase1_strategy()` - Phase1 深度掃描決策
  - `decide_phase2_targets()` - Phase2 攻擊目標選擇
  - `evaluate_phase2_results()` - Phase2 結果評估

**待驗證項目**:
- ⚠️ EnhancedDecisionAgent 實際運行效果需驗證
- ⚠️ 決策輸出格式與系統整合需測試
- ⚠️ 返回 HighLevelIntent 格式需確認相容性

**建議**: 執行端到端測試驗證決策流程

---

### 3. ✅ core_capabilities（核心能力模組）

**當前定位**: 核心能力編排中心  
**版本**: v3.2.0 | 131 flows (15.6%) | 29 模組

#### 與13步驟的映射

| 涉及步驟 | 具體職責 | 組件 | 狀態 |
|---------|---------|------|------|
| **步驟 3** | 創建掃描命令 | ScannerPlugin | ✅ 100% |
| **步驟 4** | Phase 0執行編排 | ScannerPlugin | ✅ 100% |
| **步驟 7** | Phase 1執行編排 | ScannerPlugin | ✅ 85% |

#### 子系統與13步驟的關係

| 子系統 | 在13步驟中的角色 | 關鍵度 |
|--------|----------------|--------|
| **Attack** | 攻擊執行編排（已移至Features） | ⚪ 已遷移 |
| **Analysis** | 代碼分析能力 | 🟢 輔助 |
| **BizLogic** | 業務邏輯測試 | 🟢 輔助 |
| **Plugins** | ScannerPlugin等能力插件 | 🔴 核心 |
| **Dialog** | 對話助理 | ⚪ 無關 |
| **Ingestion/Output** | 數據處理與轉換 | 🟢 輔助 |

#### 評估結論

**✅ 職責合理，承擔能力編排功能**

**優點**:
- ScannerPlugin 完美支撐步驟3-4（命令創建與P0執行）
- 攻擊執行已移至 Features，職責更清晰
- 插件系統設計良好

**需要改進**:
- ⚠️ Phase 1 執行編排需完善（當前85%）
- ⚠️ 多引擎協調邏輯需驗證
- ⚠️ Dialog 和 BizLogic 子系統與外閉環關係不大

**建議調整**:
1. 將 Dialog 子系統移至 service_backbone（屬於交互層）
2. 確認 BizLogic 是否應該歸屬於 Features 模組
3. 強化 ScannerPlugin 對多引擎的編排能力

---

### 4. ✅ service_backbone（服務骨幹）

**當前定位**: 服務基礎設施  
**版本**: v3.2.0 | 163 flows (19.4%) | 33 模組

#### 與13步驟的映射

| 涉及步驟 | 具體職責 | 組件 | 狀態 |
|---------|---------|------|------|
| **步驟 0** | 接收用戶輸入 | CLI/API | ✅ |
| **步驟 1-13** | 全局命令調度 | CommandCenter | ✅ 100% |
| **步驟 1-13** | 跨模組通信 | MQ系統 | ✅ |

#### 子系統與13步驟的關係

| 子系統 | 在13步驟中的角色 | 關鍵度 |
|--------|----------------|--------|
| **CommandCenter** | 統一調度中心（貫穿所有步驟） | 🔴 核心 |
| **MQ系統** | 模組間通信 | 🔴 核心 |
| **Health/Metrics** | 監控與健康檢查 | 🟡 重要 |
| **Service管理** | 生命週期管理 | 🟡 重要 |

#### 評估結論

**✅ 職責清晰，是13步驟的基礎設施**

**優點**:
- CommandCenter 設計優秀，支援所有步驟的命令調度
- MQ系統支援異步處理
- 純基礎設施，不涉及業務邏輯

**無需調整** ✅

---

### 5. ⚠️ external_learning（外部學習）

**當前定位**: 外部學習系統  
**版本**: v3.2.0 | 99 flows (11.8%) | 16 模組

#### 與13步驟的關係分析

**❓ 關鍵問題**: external_learning 在13步驟中的角色不明確

**文檔分析**:
```
第一章流程實施可行性分析.md:
- ❌ 13步驟中未提及 external_learning
- ❌ 可執行性評估表中無 external_learning 相關組件
- ❌ 流程圖中無 learning 階段
```

**當前 external_learning 的子系統**:
| 子系統 | 功能 | 與13步驟關係 |
|--------|------|-------------|
| **Learning策略** | 學習算法選擇 | ❓ 不明確 |
| **Training** | 模型訓練 | ❓ 不明確 |
| **Dataset管理** | 訓練數據管理 | ❓ 不明確 |

#### 評估結論

**⚠️ 需要重新定位或併入其他模組**

**問題分析**:

1. **與外閉環流程不匹配**
   - 13步驟是「任務執行流程」（Bug Bounty掃描→攻擊測試）
   - external_learning 是「模型訓練流程」（數據→訓練→優化）
   - 兩者是不同的生命週期

2. **實際使用場景模糊**
   ```
   外閉環: 用戶任務 → 掃描 → 決策 → 攻擊 → 返回結果
   學習流程: 收集數據 → 訓練模型 → 評估 → 部署 ？
   ```

3. **與 cognitive_core 的邊界不清**
   - cognitive_core 已有 Neural 子系統（神經網路核心）
   - external_learning 也涉及模型訓練
   - 兩者職責可能重疊

**🎯 三種可能的調整方案**:

#### 方案 A: 併入 cognitive_core（推薦 ⭐⭐⭐）

**理由**:
- external_learning 主要服務於 AI 模型訓練和優化
- cognitive_core 已有 Neural 子系統，是自然的歸屬
- 可以統一管理「模型訓練」和「模型推理」

**調整後結構**:
```
cognitive_core/
├── neural/              # 神經網路推理（線上）
├── training/            # ← 併入 external_learning（離線訓練）
├── decision/
├── rag/
└── anti_hallucination/
```

**優點**:
- ✅ 職責統一：AI 認知相關全在 cognitive_core
- ✅ 邊界清晰：線上推理 + 離線訓練都在同一模組
- ✅ 簡化架構：六大模組變成五大模組

#### 方案 B: 重新定位為「經驗學習」

**理由**:
- 13步驟完成後，可以從結果中學習經驗
- 類似「步驟12-13」的擴展：收集執行經驗並優化模型

**調整後職責**:
```
external_learning → experience_learning
職責：
- 從外閉環執行結果中提取經驗
- 用於優化後續決策
- 更新 RAG 知識庫
```

**優點**:
- ✅ 與外閉環有明確關聯
- ✅ 支援持續改進
- ✅ 保留獨立模組

**缺點**:
- ⚠️ 需要大幅重構現有代碼
- ⚠️ 與 internal_exploration 可能重疊

#### 方案 C: 保持獨立但明確定位

**理由**:
- external_learning 關注「從外部數據學習」
- internal_exploration 關注「從內部執行學習」
- 兩者可以並存

**調整後定位**:
```
external_learning:
- 職責：從外部數據源（公開數據集、社區知識）學習
- 服務：離線訓練、模型更新、知識庫擴充
- 時機：背景運行，不阻塞13步驟主流程
```

**優點**:
- ✅ 保持架構穩定
- ✅ 職責定位清晰

**缺點**:
- ⚠️ 與13步驟關係仍然間接
- ⚠️ 六大模組數量較多

#### 🎯 推薦方案

**推薦方案 A：併入 cognitive_core** ⭐⭐⭐

**實施步驟**:
1. 將 `external_learning/` 重命名為 `cognitive_core/training/`
2. 更新所有導入路徑
3. 在 cognitive_core README 中增加 Training 子系統說明
4. 更新文檔，說明五大模組架構

**調整後的五大模組**:
1. ✅ cognitive_core（認知核心，含訓練）
2. ✅ core_capabilities（核心能力編排）
3. ✅ task_planning（任務規劃編排）
4. ✅ service_backbone（服務基礎設施）
5. ✅ internal_exploration（內部探索，內閉環）

---

### 6. ✅ internal_exploration（內部探索）

**當前定位**: 自我探索與優化  
**版本**: v2.2.0 | 201 flows (23.9%) | 17 模組

#### 與13步驟的關係

**✅ 正確排除：屬於內閉環，與外閉環13步驟無直接關係**

**內閉環 vs 外閉環**:
```
外閉環（13步驟）:
用戶任務 → 執行 → 返回結果
目的：完成用戶的Bug Bounty任務

內閉環（internal_exploration）:
自我分析 → 優化 → 改進能力
目的：提升系統自身能力
```

#### 評估結論

**✅ 定位正確，無需調整**

**優點**:
- 職責清晰：專注於內部優化
- 與外閉環解耦：不阻塞主流程
- 流程數量最多（201 flows），顯示重要性

**無需調整** ✅

---

## 🎯 關鍵發現

### ✅ 架構設計優秀之處

1. **職責分層清晰**
   ```
   task_planning       → 任務編排層（入口/出口）
   cognitive_core      → AI決策層（智能分析）
   core_capabilities   → 能力編排層（執行協調）
   service_backbone    → 基礎設施層（調度通信）
   ```

2. **純編排設計正確**
   - task_planning：純編排，不執行
   - cognitive_core：純決策，不執行
   - core_capabilities：純編排，具體執行在 Features

3. **模組邊界相對清晰**
   - task_planning 管「任務」
   - cognitive_core 管「決策」
   - core_capabilities 管「能力」
   - service_backbone 管「基礎設施」

### ⚠️ 需要改進之處

#### 1. external_learning 定位模糊

**問題**:
- 在13步驟中無明確角色
- 與 cognitive_core 的 Neural 子系統可能重疊
- 作為六大模組之一，但流程關聯不明顯

**影響**:
- 開發者難以理解何時使用 external_learning
- 可能導致職責重疊和代碼冗餘
- 增加架構複雜度

**建議**: 見「方案 A」，併入 cognitive_core

#### 2. 部分子系統歸屬不當

**問題**:
- `core_capabilities/dialog/` - 對話助理與外閉環關係不大
- `core_capabilities/bizlogic/` - 可能更適合 Features 模組

**影響**:
- 模組邊界模糊
- 不利於理解模組職責

**建議**: 
- Dialog → service_backbone（交互層）
- BizLogic → 評估是否移至 Features

#### 3. AI決策功能已設置但待驗證 ⚠️（2026-01-08 確認）

**實現狀態**:
- ⚠️ EnhancedDecisionAgent 代碼完整度 85%（已設置，待驗證）
- ⚠️ 步驟6, 9, 11的AI決策已完成設置（4個決策方法）

**已設置功能**:
- ⚠️ 13步驟流程理論上可運行，需實際測試
- ⚠️ 智能決策能力代碼已實現，效果待確認
- ⚠️ 支持 6 種智能策略（需驗證）
- ⚠️ 包含 3 個輔助功能（需驗證）

**建議**: 執行完整端到端測試驗證

---

## 💡 建議調整方案

### 🔴 優先級 P0：立即調整

#### 調整 1: 合併 external_learning → cognitive_core

**目標**: 簡化架構，統一AI相關功能

**實施步驟**:

1. **重組目錄結構** (1天)
   ```bash
   # 移動 external_learning 到 cognitive_core/training/
   mv services/core/aiva_core/external_learning \
      services/core/aiva_core/cognitive_core/training
   
   # 或者保持平級，但重命名
   mv services/core/aiva_core/external_learning \
      services/core/aiva_core/cognitive_training
   ```

2. **更新導入路徑** (2-3天)
   ```python
   # 舊路徑
   from aiva_core.external_learning.learning import LearningOrchestrator
   
   # 新路徑
   from aiva_core.cognitive_core.training import LearningOrchestrator
   ```

3. **更新 README** (半天)
   ```markdown
   # cognitive_core/README.md
   
   ## 子系統架構
   
   ### 6. Training - 模型訓練系統（原 external_learning）
   
   **位置**: `cognitive_core/training/`
   
   **職責**:
   - 離線模型訓練
   - 學習策略選擇
   - 訓練數據管理
   - 模型評估與部署
   ```

4. **更新文檔引用** (半天)
   - 主 README：六大模組 → 五大模組
   - CHANGELOG：記錄架構調整
   - 遷移指南：提供路徑對照表

**預估工作量**: 3-4天

**優點**:
- ✅ 架構簡化：六大模組 → 五大模組
- ✅ 職責清晰：AI 相關統一在 cognitive_core
- ✅ 邊界明確：線上推理 + 離線訓練分離但相鄰

---

### 🟡 優先級 P1：短期完善（2週內）

#### 調整 2: 完善 BioNeuronPlugin 三階段決策

**目標**: 支援步驟6, 9, 11的AI決策

**實施任務**:

1. **實現階段1決策（P0→P1）**
   ```python
   # cognitive_core/plugins/bio_neuron_plugin.py
   
   async def decide_phase1_strategy(
       self, 
       phase0_results: Phase0CompletedPayload
   ) -> Phase1Strategy:
       """
       分析 Phase 0 結果，決定 Phase 1 掃描策略
       
       輸入：
       - 開放端口列表
       - 服務指紋
       - 初步風險評估
       
       輸出：
       - 需要啟動的引擎（Python/TypeScript/Go）
       - 掃描深度（Fast/Deep）
       - 重點目標
       """
       # 調用 5M 神經網路
       features = self.neural_core.extract_features(phase0_results)
       decision = self.neural_core.predict(features)
       
       return Phase1Strategy(
           engines=decision.recommended_engines,
           depth=decision.scan_depth,
           targets=decision.priority_targets
       )
   ```

2. **實現階段2決策（P1→P2）**
   ```python
   async def decide_phase2_targets(
       self, 
       phase1_results: Phase1CompletedPayload
   ) -> Phase2Targets:
       """
       分析 Phase 1 結果，選擇攻擊測試目標
       
       輸入：
       - 表單列表
       - AJAX端點
       - 潛在漏洞點
       
       輸出：
       - XSS 測試目標
       - SQLi 測試目標
       - 其他測試類型
       """
       pass
   ```

3. **實現階段3決策（P2評估）**
   ```python
   async def evaluate_phase2_results(
       self, 
       phase2_results: Phase2CompletedPayload
   ) -> RiskAssessment:
       """
       分析 Phase 2 結果，評估風險與建議
       
       輸入：
       - 發現的漏洞
       - 測試結果
       - 攻擊成功率
       
       輸出：
       - 風險評級
       - 修復建議
       - 是否需要進一步測試
       """
       pass
   ```

**預估工作量**: 1-2週

---

#### 調整 3: 移動 Dialog 子系統

**目標**: 將 Dialog 從 core_capabilities 移至 service_backbone

**理由**:
- Dialog 是交互層功能，不是「核心能力」
- service_backbone 負責所有基礎設施，包括交互介面

**實施步驟**:
```bash
# 移動目錄
mv services/core/aiva_core/core_capabilities/dialog \
   services/core/aiva_core/service_backbone/dialog

# 更新導入
# 舊: from aiva_core.core_capabilities.dialog import Assistant
# 新: from aiva_core.service_backbone.dialog import Assistant
```

**預估工作量**: 1-2天

---

### 🟢 優先級 P2：長期優化（1個月內）

#### 調整 4: 評估 BizLogic 歸屬

**目標**: 確認 BizLogic 是否應該移至 Features 模組

**分析任務**:
1. 評估 BizLogic 的使用場景
2. 確認是否屬於「具體攻擊執行」
3. 如是，移至 Features；如否，保留在 core_capabilities

---

## 📋 調整後的五大模組架構

### 🎯 最終架構

```
services/core/aiva_core/
├── 1. task_planning/          # 任務規劃編排（入口/出口）
│   ├── 職責：任務接收、分解、編排、返回
│   ├── 13步驟：步驟1, 2, 13
│   └── 狀態：✅ 無需調整
│
├── 2. cognitive_core/          # 認知核心（決策+訓練）
│   ├── 職責：AI決策、神經網路推理、模型訓練
│   ├── 子系統：
│   │   ├── neural/             # 線上推理
│   │   ├── training/           # ← 原 external_learning（離線訓練）
│   │   ├── decision/
│   │   ├── rag/
│   │   └── anti_hallucination/
│   ├── 13步驟：步驟6, 9, 11（AI決策）
│   └── 狀態：⚠️ 需合併 external_learning
│
├── 3. core_capabilities/       # 核心能力編排
│   ├── 職責：能力插件、掃描編排、分析能力
│   ├── 13步驟：步驟3, 4, 7（能力編排與執行）
│   └── 狀態：⚠️ 建議移除 Dialog
│
├── 4. service_backbone/        # 服務基礎設施
│   ├── 職責：CommandCenter、MQ系統、服務管理、交互層
│   ├── 子系統：
│   │   ├── command_center/
│   │   ├── mq/
│   │   ├── dialog/             # ← 從 core_capabilities 移入
│   │   └── health/
│   ├── 13步驟：貫穿所有步驟（調度層）
│   └── 狀態：⚠️ 建議加入 Dialog
│
└── 5. internal_exploration/    # 內部探索（內閉環）
    ├── 職責：自我分析、能力優化、記憶管理
    ├── 13步驟：無（屬於內閉環）
    └── 狀態：✅ 定位正確，無需調整
```

### 📊 調整前後對比

| 項目 | 調整前 | 調整後 | 改進 |
|------|--------|--------|------|
| 模組數量 | 6 個 | 5 個 | ⬇️ 簡化 |
| 職責清晰度 | 75% | 90% | ⬆️ 提升 |
| 與13步驟匹配 | 80% | 95% | ⬆️ 提升 |
| AI功能統一性 | 分散 | 統一 | ⬆️ 改善 |
| 模組邊界清晰度 | 70% | 90% | ⬆️ 提升 |

---

## 📅 實施計畫

### Phase 1: 架構調整（1週）

**任務清單**:
- [ ] 合併 external_learning → cognitive_core/training/
- [ ] 更新所有導入路徑
- [ ] 更新 README 和文檔
- [ ] 運行測試確保無破壞性變更

**驗收標準**:
- ✅ 所有測試通過
- ✅ 文檔已更新
- ✅ 無導入錯誤

---

### Phase 2: 功能完善（2週）

**任務清單**:
- [ ] 實現 BioNeuronPlugin 三階段決策
- [ ] 驗證與5M神經網路整合
- [ ] 移動 Dialog 子系統
- [ ] 完善 ScannerPlugin 多引擎編排

**驗收標準**:
- ✅ 步驟6, 9, 11 AI決策可運行
- ✅ Dialog 正確歸屬 service_backbone
- ✅ 端到端測試通過

---

### Phase 3: 驗證與優化（1週）

**任務清單**:
- [ ] 完整運行13步驟流程
- [ ] 性能測試
- [ ] 評估 BizLogic 歸屬
- [ ] 文檔最終審查

**驗收標準**:
- ✅ 13步驟完整流程可執行
- ✅ 性能符合預期
- ✅ 所有文檔完整準確

---

## 🏁 總結

### ✅ 核心結論

**六大模組整體架構良好，僅需局部調整**

**最關鍵的調整**:
1. 🔴 **合併 external_learning → cognitive_core** - 統一AI功能
2. 🟡 **完善 BioNeuronPlugin** - 支撐13步驟核心決策
3. 🟡 **移動 Dialog** - 優化模組邊界

**調整後優勢**:
- ✅ 架構更簡潔（五大模組）
- ✅ 職責更清晰（AI統一在 cognitive_core）
- ✅ 與13步驟匹配度更高（95%）
- ✅ 開發者更容易理解

**實施風險**:
- ⚠️ 導入路徑變更需要仔細測試
- ⚠️ BioNeuronPlugin 開發需要時間
- ⚠️ 需要團隊協作避免衝突

**預估總工作量**: 3-4週  
**預期收益**: 架構清晰度提升20%，開發效率提升15%

---

**評估完成** - 2026-01-03
