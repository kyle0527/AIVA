# AIVA 雙閉環系統技術手冊

**版本**: v2.0
**狀態**: Production Ready
**路徑**: 跨越 `services/core/`, `services/integration/`, `cognitive_core/`

---

## 1. 概述

雙閉環（Dual-Loop）是 AIVA 的持續自我優化架構。透過兩個獨立但相互補充的學習環路，系統在每次執行後都能積累知識、優化決策、提升攻擊精準度。

```
┌─────────────────────────────────────────────┐
│                AIVA 雙閉環架構               │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │          內閉環（Internal Loop）      │  │
│  │    Core ↔ Features ↔ Integration     │  │
│  │    目的：即時執行策略優化              │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │          外閉環（External Loop）      │  │
│  │  Integration → 報告 → 回饋 → Core    │  │
│  │  目的：長期知識積累與模型改善          │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## 2. 內閉環（Internal Loop）

### 2.1 目的

在**單次攻擊任務執行期間**，即時監控執行效果並動態調整策略。

### 2.2 資料流

```
Core 發出攻擊指令
  │
  ▼
Features/Scan 執行攻擊
  │ 回傳中間結果
  ▼
Integration (AI Operation Recorder)
  │ 即時分析效果
  ▼
Core 調整後續決策
  │ 更新 Phase 2 目標優先級
  ▼
Features/Scan 繼續執行（使用更新後的策略）
```

### 2.3 內閉環觸發條件

| 觸發事件 | 調整動作 |
|---|---|
| 某模組發現高危漏洞 | 升高對應攻擊面的優先級 |
| 某模組持續失敗 | 切換替代模組或跳過 |
| 回應時間異常 | 降低請求速率 |
| WAF 阻擋增多 | 切換 WAF bypass 策略 |

### 2.4 整合介面

```python
# internal_loop_connector.py
class InternalLoopConnector:
    async def inject_ability_analysis(self, results: dict) -> None:
        """將執行中間結果注入 RAG，供即時決策參考"""

    async def receive_strategy_update(self) -> StrategyUpdate:
        """接收 Core 更新後的策略"""
```

---

## 3. 外閉環（External Loop）

### 3.1 目的

在**任務完成後**，將整個執行過程的知識永久積累，改善未來的決策品質。

### 3.2 資料流

```
攻擊任務完成
  │
  ▼
Integration 生成最終報告
  │
  ├── 客戶/操作者回饋（標記誤報/確認漏洞）
  │
  ▼
external_loop_connector.py
  │ 結構化學習資料
  ▼
learning_system/ (KnowledgeExtractor)
  │
  ▼
sync_experiences.py → RAG vector_store
  │
  ▼
ModelTrainer（定期觸發 5M 神經網路微調）
```

### 3.3 學習資料結構

```json
{
  "session_id": "uuid",
  "target_profile": {
    "tech_stack": ["nginx", "php", "mysql"],
    "waf": "cloudflare"
  },
  "attack_results": [
    {
      "module": "function_sqli",
      "technique": "time-based blind",
      "success": true,
      "bypass_used": "comment_injection"
    }
  ],
  "lessons_learned": [
    "Cloudflare blocks standard UNION-based SQLi; comment_injection effective"
  ],
  "outcome": {
    "vulnerabilities_found": 2,
    "estimated_bounty": 8500
  }
}
```

---

## 4. 學習系統元件（learning_system/）

| 元件 | 功能 |
|---|---|
| `ExperienceReplayMemory` | 儲存近期執行經驗，供訓練採樣 |
| `KnowledgeExtractor` | 從執行結果中提取可學習知識點 |
| `ModelTrainer` | 觸發 5M 神經網路權重更新 |
| `ContinuousLearning` | 管理持續學習排程 |

### 4.1 訓練觸發條件

```python
# 以下條件任一成立時觸發訓練
- 新增經驗 > 1000 筆
- 距上次訓練 > 24 小時
- 發現新型漏洞模式（novelty score > 0.8）
- 手動觸發
```

---

## 5. 資料儲存對應

| 資料類型 | 儲存位置 | 格式 |
|---|---|---|
| 即時執行狀態 | Redis | Key-Value |
| 攻擊路徑圖 | NetworkX (.pkl) | 圖結構 |
| 歷史執行經驗 | SQLite (experience.db) | 關聯式 |
| 向量化知識 | PostgreSQL + pgvector | 512 維向量 |
| 訓練資料集 | JSONL/CSV 檔案 | 批次訓練 |
| 模型權重 | 檔案系統 | PyTorch .pt |

---

## 6. 雙閉環協調協定

### 6.1 內閉環心跳

```
每 30 秒：Integration → Core 傳送執行狀態摘要
每次決策點：Core 查詢 RAG 取得最新上下文
```

### 6.2 外閉環排程

```
任務完成後立即：同步關鍵漏洞知識到 RAG
每日 UTC 02:00：觸發完整模型訓練週期
每週：生成知識庫成長報告
```

---

## 7. 監控指標

追蹤雙閉環健康度的關鍵指標：

| 指標 | 說明 | 健康閾值 |
|---|---|---|
| `rag.hit_rate` | RAG 查詢命中率 | > 70% |
| `learning.new_patterns` | 每次任務新增知識數 | > 5 |
| `model.accuracy_delta` | 每次訓練後精準度提升 | > 0.5% |
| `inner_loop.adjustment_count` | 內閉環動態調整次數 | 每任務 2-10 次 |

---

## 8. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第3冊_執行與適應.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第2冊_AI決策流程.md`
- **技術手冊**：`docs/technical_manuals/06_COGNITIVE_CORE_TECHNICAL_MANUAL.md`
- **技術手冊**：`docs/technical_manuals/07_RAG_SYSTEM_TECHNICAL_MANUAL.md`
- **技術手冊**：`docs/technical_manuals/04_INTEGRATION_MODULE_TECHNICAL_MANUAL.md`
