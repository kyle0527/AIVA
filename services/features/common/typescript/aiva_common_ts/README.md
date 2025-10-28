# AIVA Common TypeScript

[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16.0+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 概述

**AIVA Common TypeScript** 是 AIVA 系統的 TypeScript AI 組件庫，提供與 Python `aiva_common` 對應的 TypeScript 實現。

### 🎯 核心功能

- ✅ **AI 能力評估**: 對應 `aiva_common.ai.capability_evaluator`
- ✅ **經驗管理**: 對應 `aiva_common.ai.experience_manager` 
- ✅ **強化學習支持**: 經驗樣本收集和質量評估
- ✅ **連續監控**: 實時 AI 能力監控和評估
- ✅ **跨語言一致性**: 與 Python 版本保持 API 一致

### 📊 組件統計

- **TypeScript 文件**: 4 個主要文件
- **AI 組件**: 2 個核心組件
- **支持的存儲後端**: Memory, JSON File, SQLite
- **評估維度**: 8 個評估維度
- **證據類型**: 6 種證據類型

---

## 📂 目錄結構

```
services/features/common/typescript/aiva_common_ts/
├── capability-evaluator.ts    # AI 能力評估器實現
├── experience-manager.ts      # 經驗管理器實現
├── index.ts                   # 主入口文件
├── package.json               # NPM 配置
├── tsconfig.json              # TypeScript 配置
└── README.md                  # 本文件
```

---

## 🚀 快速開始

### 安裝

```bash
# 在 TypeScript 項目中安裝
npm install @aiva/common-ts

# 或使用 yarn
yarn add @aiva/common-ts
```

### 基本使用

#### 1. AI 能力評估器

```typescript
import { createCapabilityEvaluator, EvidenceType, EvaluationDimension } from '@aiva/common-ts';

// 創建評估器
const evaluator = createCapabilityEvaluator({
  evaluatorId: 'my_evaluator',
  continuousMonitoring: true
});

// 收集證據
const evidenceId = await evaluator.collectEvidence(
  'capability_001',
  EvidenceType.PERFORMANCE_METRIC,
  {
    executionTime: 1500,
    successRate: 0.85,
    errorCount: 2
  }
);

// 生成評估報告
const assessment = await evaluator.generateAssessment(
  'capability_001',
  EvaluationDimension.PERFORMANCE
);

console.log(`Overall Score: ${assessment.overallScore}/100`);
console.log(`Confidence: ${assessment.confidence}`);
```

#### 2. 經驗管理器

```typescript
import { createExperienceManager, SessionType, StorageBackend } from '@aiva/common-ts';

// 創建經驗管理器
const manager = createExperienceManager({
  managerId: 'exp_manager_001',
  storageBackend: StorageBackend.MEMORY,
  deduplicationEnabled: true
});

// 創建學習會話
const sessionId = await manager.createLearningSession(
  'training_001',
  SessionType.TRAINING
);

// 存儲經驗樣本
const sample = {
  sampleId: 'exp_001',
  sessionId,
  planId: 'plan_001',
  stateBefore: { vulnerability: 'sql_injection', confidence: 0.8 },
  actionTaken: { payload: "' OR '1'='1", method: 'GET' },
  stateAfter: { success: true, response_code: 200 },
  reward: 10.0,
  rewardBreakdown: { success: 8.0, efficiency: 2.0 },
  context: { target: 'login_form' },
  targetInfo: { vulnerabilityType: 'sql_injection' },
  timestamp: new Date(),
  durationMs: 1200,
  isPositive: true,
  confidence: 0.9,
  learningTags: ['sql_injection', 'authentication_bypass'],
  difficultyLevel: 3
};

await manager.storeExperience(sample);

// 獲取統計信息
const stats = await manager.getLearningStatistics();
console.log(`Total Samples: ${stats.totalSamples}`);
console.log(`Success Rate: ${(stats.successRate * 100).toFixed(1)}%`);
```

#### 3. 事件監聽

```typescript
// 監聽評估器事件
evaluator.on('evidence_collected', (evidence) => {
  console.log(`Evidence collected: ${evidence.evidenceId}`);
});

evaluator.on('assessment_generated', (assessment) => {
  console.log(`Assessment generated for ${assessment.capabilityId}`);
});

// 監聽經驗管理器事件
manager.on('experience_stored', (sample) => {
  console.log(`Experience stored: ${sample.sampleId}`);
});

manager.on('session_created', (session) => {
  console.log(`Learning session created: ${session.sessionId}`);
});
```

---

## 🔧 API 參考

### AIVACapabilityEvaluator

#### 主要方法

- `collectEvidence(capabilityId, evidenceType, data, source?)`: 收集能力證據
- `runBenchmarkTests(capabilityId, testSuite?)`: 運行基準測試
- `generateAssessment(capabilityId, dimension?)`: 生成能力評估
- `startContinuousMonitoring()`: 啟動連續監控
- `getAssessment(capabilityId)`: 獲取評估結果
- `getEvaluatorStatus()`: 獲取評估器狀態

#### 事件

- `evidence_collected`: 證據收集完成
- `assessment_generated`: 評估生成完成
- `monitoring_started`: 監控開始
- `periodic_evaluation_completed`: 定期評估完成

### AIVAExperienceManager

#### 主要方法

- `createLearningSession(trainingId?, sessionType?, options?)`: 創建學習會話
- `endLearningSession(sessionId)`: 結束學習會話
- `storeExperience(sample)`: 存儲經驗樣本
- `getExperiences(sessionId?, planId?, qualityThreshold?, limit?)`: 獲取經驗樣本
- `evaluateSampleQuality(sample)`: 評估樣本質量
- `getLearningStatistics(sessionId?)`: 獲取學習統計
- `cleanupOldExperiences(retentionDays?)`: 清理舊經驗

#### 事件

- `session_created`: 會話創建
- `session_ended`: 會話結束
- `experience_stored`: 經驗存儲
- `experiences_cleaned`: 經驗清理

---

## 🏗️ 開發指南

### 構建項目

```bash
# 安裝依賴
npm install

# 構建 TypeScript
npm run build

# 監視模式構建
npm run build:watch

# 運行測試
npm test

# 代碼檢查
npm run lint
```

### 與 Python 版本的對應關係

| TypeScript | Python | 功能 |
|------------|--------|------|
| `AIVACapabilityEvaluator` | `AIVACapabilityEvaluator` | AI 能力評估 |
| `AIVAExperienceManager` | `AIVAExperienceManager` | 經驗管理 |
| `CapabilityEvidence` | `CapabilityEvidence` | 能力證據 |
| `ExperienceSample` | `ExperienceSample` | 經驗樣本 |
| `EvaluationDimension` | `EvaluationDimension` | 評估維度 |
| `SessionType` | `SessionType` | 會話類型 |

### 類型安全

此庫完全使用 TypeScript 編寫，提供完整的類型定義：

```typescript
// 所有接口都有完整的類型定義
interface CapabilityEvidence {
  evidenceId: string;
  capabilityId: string;
  evidenceType: EvidenceType;
  data: Record<string, any>;
  confidence: number;
  timestamp: Date;
  source: string;
  metadata: Record<string, any>;
}

// 枚舉提供類型安全的常量
enum EvidenceType {
  EXECUTION_LOG = "execution_log",
  PERFORMANCE_METRIC = "performance_metric",
  USER_FEEDBACK = "user_feedback",
  // ...
}
```

---

## 🔗 與其他組件的整合

### 與 Python aiva_common 的整合

```typescript
// TypeScript 側發送評估結果到 Python
const assessment = await evaluator.generateAssessment('capability_001');

// 通過 HTTP API 或消息隊列發送到 Python 服務
const response = await fetch('/api/python/assessment', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    assessment_id: assessment.assessmentId,
    capability_id: assessment.capabilityId,
    overall_score: assessment.overallScore,
    confidence: assessment.confidence
  })
});
```

### 與 Go 服務的整合

```typescript
// 通過 gRPC 或 RESTful API 與 Go 服務通信
const experienceData = {
  sample_id: sample.sampleId,
  plan_id: sample.planId,
  reward: sample.reward,
  is_positive: sample.isPositive
};

// 發送到 Go 分析服務
await grpcClient.analyzeExperience(experienceData);
```

---

## 📊 性能特點

### 記憶體使用

- **輕量級設計**: 核心組件僅 ~50KB
- **可配置緩存**: 支持自定義記憶體限制
- **自動清理**: 定期清理過期數據

### 執行性能

- **異步操作**: 所有 I/O 操作都是非阻塞的
- **批處理支持**: 支持批量處理經驗樣本
- **連續監控**: 低開銷的後台監控

### 存儲支持

- **Memory**: 快速原型和測試
- **JSON File**: 簡單持久化
- **SQLite**: 生產環境使用（待實現）

---

## 🚨 已知限制

### 當前版本限制

1. **存儲後端**: 目前僅完整實現 Memory 存儲
2. **測試覆蓋**: 需要更多單元測試和集成測試
3. **文檔**: 需要更多使用範例和最佳實踐

### 計劃改進

- [ ] 實現 JSON File 和 SQLite 存儲後端
- [ ] 添加更多基準測試類型
- [ ] 支持分佈式部署
- [ ] 添加更多評估指標

---

## 📝 版本歷史

### v1.0.0 (2025-10-28)
- ✨ 初始發布
- ✅ AI 能力評估器完整實現
- ✅ 經驗管理器核心功能
- ✅ 與 Python aiva_common v1.0.0 兼容
- ✅ 完整的 TypeScript 類型定義
- ✅ 事件驅動架構
- ✅ 記憶體存儲後端

---

## 📄 授權

本項目採用 MIT 授權 - 詳見 [LICENSE](../../../../../../LICENSE) 文件

---

## 📮 聯絡方式

- **項目維護者**: AIVA 開發團隊
- **問題回報**: 請使用 GitHub Issues
- **功能請求**: 請使用 GitHub Discussions

---

**AIVA Common TypeScript** - 為 TypeScript 項目提供強大的 AI 能力評估和經驗管理功能 🚀