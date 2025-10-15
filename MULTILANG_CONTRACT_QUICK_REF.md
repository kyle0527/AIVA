# AIVA 多語言合約實現對照表

快速參考指南 - 各語言合約實現狀況

---

## 📊 快速概覽

```
語言         檔案數   合約完整度    共用庫     互操作性
=========================================================
Python       263     ████████████  ✅ 完整   ✅ 標準
Go           18      ██░░░░░░░░░░  ⚠️ 部分   ⚠️ 部分可用
TypeScript   984     █░░░░░░░░░░░  ❌ 無     ❌ 不相容
Rust         20      ▌░░░░░░░░░░░  ❌ 無     ❌ 不相容
```

---

## 🔍 詳細對照表

### 核心消息結構

| Schema | Python | Go | TypeScript | Rust | 說明 |
|--------|--------|-----|-----------|------|------|
| MessageHeader | ✅ | ✅ | ❌ | ❌ | 消息頭 |
| AivaMessage | ✅ | ✅ | ❌ | ❌ | 統一包裝 |
| Topic 枚舉 | ✅ 43個 | ❌ | ❌ | ❌ | 消息主題 |

**狀態**: Go 有基礎實現，TS/Rust 完全缺失

---

### 掃描模組合約

| Schema | Python | Go | TypeScript | Rust | 說明 |
|--------|--------|-----|-----------|------|------|
| ScanStartPayload | ✅ | ❌ | ⚠️ 自定義 | ❌ | 掃描啟動 |
| ScanCompletedPayload | ✅ | ❌ | ⚠️ 自定義 | ❌ | 掃描完成 |
| ScanScope | ✅ | ❌ | ❌ | ❌ | 掃描範圍 |
| Authentication | ✅ | ❌ | ❌ | ❌ | 認證信息 |
| RateLimit | ✅ | ❌ | ❌ | ❌ | 速率限制 |
| Asset | ✅ | ❌ | ⚠️ 自定義 | ❌ | 資產信息 |
| Summary | ✅ | ❌ | ⚠️ 自定義 | ❌ | 掃描摘要 |

**狀態**: 僅 Python 完整，TypeScript 有自定義接口但不相容

---

### 功能測試合約

| Schema | Python | Go | TypeScript | Rust | 說明 |
|--------|--------|-----|-----------|------|------|
| FunctionTaskPayload | ✅ | ✅ | ❌ | ⚠️ 簡化 | 任務載荷 |
| FunctionTaskTarget | ✅ | ✅ | ❌ | ⚠️ 簡化 | 任務目標 |
| FunctionTaskContext | ✅ | ✅ | ❌ | ❌ | 任務上下文 |
| FunctionTaskTestConfig | ✅ | ✅ | ❌ | ⚠️ 簡化 | 測試配置 |
| FindingPayload | ✅ | ⚠️ 需更新 | ❌ | ⚠️ 簡化 | 漏洞發現 |
| Vulnerability | ✅ | ⚠️ 基本 | ❌ | ⚠️ 基本 | 漏洞信息 |
| Target/FindingTarget | ✅ | ⚠️ 基本 | ❌ | ⚠️ 基本 | 目標信息 |
| FindingEvidence | ✅ | ⚠️ 欄位不同 | ❌ | ⚠️ 基本 | 證據 |
| FindingImpact | ✅ | ⚠️ 欄位不同 | ❌ | ⚠️ 基本 | 影響 |
| FindingRecommendation | ✅ | ⚠️ 欄位不同 | ❌ | ❌ | 修復建議 |

**狀態**: Go 有基礎但需更新，Rust 極簡版

---

### AI 訓練合約

| Schema | Python | Go | TypeScript | Rust | 說明 |
|--------|--------|-----|-----------|------|------|
| AITrainingStartPayload | ✅ | ❌ | ❌ | ❌ | 訓練啟動 |
| AITrainingProgressPayload | ✅ | ❌ | ❌ | ❌ | 訓練進度 |
| AITrainingCompletedPayload | ✅ | ❌ | ❌ | ❌ | 訓練完成 |
| AIExperienceCreatedEvent | ✅ | ❌ | ❌ | ❌ | 經驗創建 |
| AITraceCompletedEvent | ✅ | ❌ | ❌ | ❌ | 追蹤完成 |
| AIModelUpdatedEvent | ✅ | ❌ | ❌ | ❌ | 模型更新 |
| AIModelDeployCommand | ✅ | ❌ | ❌ | ❌ | 模型部署 |
| AttackPlan | ✅ | ❌ | ❌ | ❌ | 攻擊計畫 |
| AttackResult | ✅ | ❌ | ❌ | ❌ | 攻擊結果 |
| TraceRecord | ✅ | ❌ | ❌ | ❌ | 追蹤記錄 |
| ExperienceSample | ✅ | ❌ | ❌ | ❌ | 經驗樣本 |
| ModelTrainingConfig | ✅ | ❌ | ❌ | ❌ | 訓練配置 |

**狀態**: 僅 Python 實現，其他語言完全缺失

---

### RAG 知識庫合約

| Schema | Python | Go | TypeScript | Rust | 說明 |
|--------|--------|-----|-----------|------|------|
| RAGKnowledgeUpdatePayload | ✅ | ❌ | ❌ | ❌ | 知識更新 |
| RAGQueryPayload | ✅ | ❌ | ❌ | ❌ | 知識查詢 |
| RAGResponsePayload | ✅ | ❌ | ❌ | ❌ | 查詢響應 |

**狀態**: 僅 Python 實現

---

### 統一通訊包裝器

| Schema | Python | Go | TypeScript | Rust | 說明 |
|--------|--------|-----|-----------|------|------|
| AIVARequest | ✅ | ❌ | ❌ | ❌ | 請求包裝 |
| AIVAResponse | ✅ | ❌ | ❌ | ❌ | 響應包裝 |
| AIVAEvent | ✅ | ❌ | ❌ | ❌ | 事件包裝 |
| AIVACommand | ✅ | ❌ | ❌ | ❌ | 命令包裝 |

**狀態**: 僅 Python 實現

---

### 業界標準支持

| Schema | Python | Go | TypeScript | Rust | 說明 |
|--------|--------|-----|-----------|------|------|
| CVSSv3Metrics | ✅ 含計算 | ❌ | ❌ | ❌ | CVSS 評分 |
| CVEReference | ✅ | ❌ | ❌ | ❌ | CVE 參考 |
| CWEReference | ✅ | ❌ | ❌ | ❌ | CWE 參考 |
| MITREAttackTechnique | ✅ | ❌ | ❌ | ❌ | MITRE ATT&CK |
| SARIFResult | ✅ | ❌ | ❌ | ❌ | SARIF 結果 |
| SARIFReport | ✅ | ❌ | ❌ | ❌ | SARIF 報告 |

**狀態**: 僅 Python 實現

---

## 📈 統計圖表

### 合約實現覆蓋率

```
Python:    ████████████████████ 100% (127/127)
Go:        ████░░░░░░░░░░░░░░░░  20% (25/127)
TypeScript: ██░░░░░░░░░░░░░░░░░░  10% (13/127 自定義)
Rust:      █░░░░░░░░░░░░░░░░░░░   5% (6/127 簡化)
```

### 按類別統計

| 類別 | Python | Go | TypeScript | Rust |
|------|--------|-----|-----------|------|
| 核心消息 (3) | 3 ✅ | 2 ✅ | 0 ❌ | 0 ❌ |
| 掃描 (7) | 7 ✅ | 0 ❌ | 3 ⚠️ | 0 ❌ |
| 功能測試 (10) | 10 ✅ | 6 ⚠️ | 0 ❌ | 4 ⚠️ |
| AI 訓練 (12) | 12 ✅ | 0 ❌ | 0 ❌ | 0 ❌ |
| RAG (3) | 3 ✅ | 0 ❌ | 0 ❌ | 0 ❌ |
| 統一包裝器 (4) | 4 ✅ | 0 ❌ | 0 ❌ | 0 ❌ |
| 業界標準 (6) | 6 ✅ | 0 ❌ | 0 ❌ | 0 ❌ |
| 其他 (82) | 82 ✅ | 17 ⚠️ | 10 ⚠️ | 2 ⚠️ |

---

## 🎯 優先級矩陣

### 立即需要 (本週)

| 語言 | 合約 | 優先級 | 理由 |
|------|------|--------|------|
| Go | FindingPayload 更新 | 🔴 最高 | 與 Python 不匹配，影響互操作 |
| Go | 掃描合約 (7個) | 🔴 最高 | 功能模組需要 |

### 短期需要 (2週內)

| 語言 | 合約 | 優先級 | 理由 |
|------|------|--------|------|
| Go | AI 訓練合約 (12個) | 🟡 高 | 支持 AI 功能 |
| Go | RAG 合約 (3個) | 🟡 高 | 支持知識檢索 |
| Go | 統一包裝器 (4個) | 🟡 高 | 標準化通訊 |
| Go | 業界標準 (6個) | 🟡 高 | 安全評估需要 |

### 中期需要 (1個月內)

| 語言 | 合約 | 優先級 | 理由 |
|------|------|--------|------|
| TypeScript | 創建共用庫 | 🟢 中 | 標準化掃描服務 |
| TypeScript | 核心消息 (3個) | 🟢 中 | 基礎通訊 |
| TypeScript | 掃描合約整合 | 🟢 中 | 與現有接口整合 |

### 長期考慮 (2-3個月)

| 語言 | 合約 | 優先級 | 理由 |
|------|------|--------|------|
| Rust | 適配器模式 | 🔵 低 | SAST 相對獨立 |
| 全部 | 自動生成工具 | 🔵 低 | 降低維護成本 |

---

## ⚠️ 互操作性矩陣

### Python ↔ 其他語言

|  | Python → | Go | TypeScript | Rust |
|---|---------|-----|-----------|------|
| **MessageHeader** | ✅ | ✅ 可用 | ❌ 不支持 | ❌ 不支持 |
| **AivaMessage** | ✅ | ✅ 可用 | ❌ 不支持 | ❌ 不支持 |
| **ScanStartPayload** | ✅ | ❌ 不支持 | ⚠️ 格式不同 | ❌ 不支持 |
| **FunctionTaskPayload** | ✅ | ✅ 可用 | ❌ 不支持 | ⚠️ 簡化版 |
| **FindingPayload** | ✅ | ⚠️ 部分可用 | ❌ 不支持 | ⚠️ 簡化版 |
| **AI 訓練相關** | ✅ | ❌ 不支持 | ❌ 不支持 | ❌ 不支持 |

### 跨語言通訊能力

```
Python ←→ Go:        ⚠️  部分可用 (核心功能 OK，Finding 有問題)
Python ←→ TypeScript: ❌  不相容 (需要適配層)
Python ←→ Rust:       ❌  不相容 (需要適配層)
Go ←→ TypeScript:     ❌  不可用
Go ←→ Rust:           ❌  不可用
TypeScript ←→ Rust:   ❌  不可用
```

---

## 🛠️ 快速修復指南

### 對於 Go 開發者

**當前可用**:
```go
// ✅ 可以使用
import "aiva_common_go/schemas"

msg := schemas.AivaMessage{
    Header: schemas.MessageHeader{...},
    Topic: "tasks.function.start",
    Payload: map[string]interface{}{...},
}
```

**需要手動處理**:
```go
// ⚠️ FindingPayload 結構不同，需要轉換
// ❌ 掃描、AI、RAG 相關完全不支持
```

**建議**:
等待 Go 共用庫完善，或使用 Python 服務作為中介

---

### 對於 TypeScript 開發者

**當前狀況**:
```typescript
// ❌ 沒有標準 schema 庫
// ⚠️ 使用自定義接口
interface DynamicScanTask { ... }
```

**臨時方案**:
```typescript
// 手動構造 Python 相容的 JSON
const message = {
  header: {
    message_id: "...",
    trace_id: "...",
    source_module: "SCAN",
    timestamp: new Date().toISOString(),
    version: "1.0"
  },
  topic: "results.scan.completed",
  schema_version: "1.0",
  payload: { ... }
};
```

**建議**:
等待 TypeScript 共用庫創建

---

### 對於 Rust 開發者

**當前可用**:
```rust
// ⚠️ 僅基本結構
#[derive(Serialize, Deserialize)]
pub struct FindingPayload { ... }
```

**建議**:
使用適配器模式，內部使用簡化結構，輸出時轉換為標準 JSON

---

## 📚 相關資源

- [MULTILANG_CONTRACT_STATUS.md](MULTILANG_CONTRACT_STATUS.md) - 詳細狀況報告
- [CONTRACT_VERIFICATION_REPORT.md](CONTRACT_VERIFICATION_REPORT.md) - Python 合約驗證
- [SCHEMA_MAPPING.md](services/function/common/go/aiva_common_go/SCHEMA_MAPPING.md) - Go 映射文檔
- [schemas.py](services/aiva_common/schemas.py) - Python 參考實現

---

**更新時間**: 2025年10月15日  
**維護者**: AIVA Architecture Team
