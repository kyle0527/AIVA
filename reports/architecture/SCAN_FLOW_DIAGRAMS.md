# Scan 模組流程圖完整呈現

## 📑 目錄

- [📋 核心概念](#核心概念)
  - [Scan 模組角色定位](#scan-模組角色定位)
  - [兩階段掃描流程](#兩階段掃描流程)
- [1️⃣ 完整系統流程圖](#1-完整系統流程圖)
- [2️⃣ Scan 模組內部運作詳細流程](#2-scan-模組內部運作詳細流程)
- [3️⃣ 數據流向與存儲位置](#3-數據流向與存儲位置)
- [4️⃣ 時序圖 - Core 與 Scan 完整互動](#4-時序圖-core-與-scan-完整互動)
- [5️⃣ 引擎選擇決策樹](#5-引擎選擇決策樹)
- [6️⃣ 失敗處理與重試機制](#6-失敗處理與重試機制)
- [📊 性能指標](#性能指標)
  - [Phase 0 (Rust 快速偵察)](#phase-0-rust-快速偵察)
  - [Phase 1 (多引擎深度掃描)](#phase-1-多引擎深度掃描)
- [🔍 關鍵技術細節](#關鍵技術細節)
  - [RabbitMQ 消息格式](#rabbitmq-消息格式)
  - [數據 Schema](#數據-schema)
- [📝 總結](#總結)

---


> **文檔目的**: 完整展示 Scan 模組在 AIVA 系統中的運作流程  
> **創建日期**: 2025年11月17日  
> **架構版本**: v6.3

---

## 📋 核心概念

### Scan 模組角色定位

- **指揮官**: Core 模組 (下令執行掃描)
- **執行者**: Scan 模組 (接收命令並執行)
- **通信機制**: RabbitMQ 消息隊列
- **數據流向**: User → Core → Scan → Core

### 兩階段掃描流程

1. **Phase 0**: Core 下令 → Rust 快速偵察 → 回傳初步資產清單
2. **Phase 1**: Core 分析並下令 → 多引擎深度掃描 → 回傳完整資產清單

---

## 1️⃣ 完整系統流程圖

展示從用戶輸入到 Core 後續處理的完整流程

```mermaid
flowchart TB
    USER[用戶輸入目標URL]
    
    subgraph CORE[Core模組指揮中心]
        C1[接收用戶輸入]
        C2[分析目標資訊]
        C3[下令Phase0]
        C4[接收Phase0結果]
        C5[AI分析決策]
        C6[下令Phase1]
        C7[接收完整結果]
        C8[進入7大步驟]
    end
    
    subgraph SCAN[Scan模組執行單元]
        direction TB
        
        subgraph P0[Phase0執行]
            S1[接收Core命令]
            S2[Rust引擎掃描]
            S3[生成初步清單]
            S4[回傳Core]
        end
        
        subgraph P1[Phase1執行]
            S5[接收Core命令]
            S6[Python引擎]
            S7[TypeScript引擎]
            S8[Go引擎]
            S9[Rust引擎]
            S10[整合結果]
            S11[回傳Core]
        end
    end
    
    MQ[RabbitMQ消息隊列]
    
    USER --> C1
    C1 --> C2
    C2 --> C3
    C3 -->|tasks.scan.phase0| MQ
    MQ --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 -->|scan.phase0.completed| MQ
    MQ --> C4
    C4 --> C5
    C5 -->|需要Phase1| C6
    C5 -.已足夠.-> C8
    C6 -->|tasks.scan.phase1| MQ
    MQ --> S5
    S5 --> S6
    S5 --> S7
    S5 --> S8
    S5 --> S9
    S6 --> S10
    S7 --> S10
    S8 --> S10
    S9 --> S10
    S10 --> S11
    S11 -->|scan.completed| MQ
    MQ --> C7
    C7 --> C8
    
    style USER fill:#ffcdd2,stroke:#c62828,stroke-width:3px
    style C1 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C2 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C3 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C4 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C5 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C6 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C7 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C8 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style S1 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style S2 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style S3 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style S4 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style S5 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style S6 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style S7 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style S8 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style S9 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style S10 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style S11 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style MQ fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

**圖表說明**:
- 🟠 橙色: Core 模組節點 (指揮中心)
- 🔵 藍色: Scan 模組節點 (執行單元)
- 🟣 紫色: RabbitMQ 消息隊列
- 🟢 綠色: 結果回傳節點

---

## 2️⃣ Scan 模組內部運作詳細流程

展示 Scan 模組接收命令後的內部執行過程

```mermaid
flowchart TD
    START[接收Core命令]
    JUDGE{判斷階段}
    
    subgraph PHASE0[Phase0執行流程]
        P0_1[初始化Rust引擎]
        P0_2[驗證目標可達性]
        P0_3[敏感資訊掃描]
        P0_4[技術棧指紋識別]
        P0_5[基礎端點發現]
        P0_6[初步攻擊面評估]
        P0_7[聚合結果]
        P0_8[格式化Schema]
        P0_9[暫存內存]
    end
    
    subgraph PHASE1[Phase1執行流程]
        P1_1[解析Core命令]
        P1_2[獲取引擎選擇]
        P1_3[初始化引擎]
        P1_4[分發任務]
        
        P1_5[Python靜態爬取]
        P1_6[Python表單發現]
        P1_7[Python-API分析]
        
        P1_8[TypeScript-JS渲染]
        P1_9[TypeScript-SPA路由]
        P1_10[TypeScript動態內容]
        
        P1_11[Go並發掃描]
        P1_12[Go服務發現]
        P1_13[Go端口掃描]
        
        P1_14[Rust高性能掃描]
        P1_15[Rust大規模處理]
        
        P1_16[收集引擎結果]
        P1_17[整合Phase0和Phase1]
        P1_18[去重關聯分析]
        P1_19[格式化完整清單]
    end
    
    SEND[發送結果回Core]
    LOG[記錄執行日誌]
    END[完成等待下一個命令]
    
    START --> JUDGE
    JUDGE -->|Phase0| PHASE0
    JUDGE -->|Phase1| PHASE1
    
    P0_1 --> P0_2 --> P0_3 --> P0_4 --> P0_5 --> P0_6
    P0_6 --> P0_7 --> P0_8 --> P0_9 --> SEND
    
    P1_1 --> P1_2 --> P1_3 --> P1_4
    P1_4 --> P1_5
    P1_4 --> P1_8
    P1_4 --> P1_11
    P1_4 --> P1_14
    
    P1_5 --> P1_6 --> P1_7 --> P1_16
    P1_8 --> P1_9 --> P1_10 --> P1_16
    P1_11 --> P1_12 --> P1_13 --> P1_16
    P1_14 --> P1_15 --> P1_16
    
    P1_16 --> P1_17 --> P1_18 --> P1_19 --> SEND
    
    SEND --> LOG --> END
    
    style START fill:#ffcdd2,stroke:#c62828,stroke-width:3px
    style JUDGE fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style P0_1 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_2 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_3 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_4 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_5 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_6 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_7 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_8 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P0_9 fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style P1_1 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style P1_2 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style P1_3 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style P1_4 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style P1_5 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_6 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_7 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_8 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_9 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_10 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_11 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_12 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_13 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_14 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_15 fill:#e1f5ff,stroke:#0288d1,stroke-width:1px
    style P1_16 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style P1_17 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style P1_18 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style P1_19 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style SEND fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style LOG fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px
    style END fill:#ffcdd2,stroke:#c62828,stroke-width:3px
```

**圖表說明**:
- 🟠 橙色: 開始/結束/決策節點
- 🔴 紅色: Phase 0 執行步驟 (Rust 快速偵察)
- 🟢 綠色: Phase 1 控制流程
- 🔵 淺藍: 各引擎執行細節
- ⚪ 灰色: 日誌記錄

---

## 3️⃣ 數據流向與存儲位置

展示數據在各模組間的流動和最終存儲位置

```mermaid
flowchart LR
    subgraph CORE[Core模組]
        direction TB
        CC[下達命令]
        CR[接收結果]
        CS[存儲SessionState]
    end
    
    subgraph MQ[RabbitMQ]
        direction TB
        Q1[tasks.scan.phase0]
        Q2[tasks.scan.phase1]
        Q3[scan.phase0.completed]
        Q4[scan.completed]
    end
    
    subgraph SCAN[Scan模組]
        direction TB
        SR[接收命令]
        SE[執行掃描]
        ST[暫存內存]
        SS[發送結果]
        
        subgraph D0[Phase0數據]
            D01[初步資產列表]
            D02[技術棧資訊]
            D03[敏感資訊]
        end
        
        subgraph D1[Phase1數據]
            D11[完整URL清單]
            D12[表單參數]
            D13[API端點]
            D14[入口點]
        end
    end
    
    subgraph DB[數據庫]
        direction TB
        L1[掃描日誌]
        L2[結果歸檔]
    end
    
    CC -->|Phase0命令| Q1
    CC -->|Phase1命令| Q2
    Q1 --> SR
    Q2 --> SR
    SR --> SE
    SE --> ST
    ST --> D0
    ST --> D1
    D0 --> SS
    D1 --> SS
    SS -->|Phase0結果| Q3
    SS -->|最終結果| Q4
    Q3 --> CR
    Q4 --> CR
    CR --> CS
    SE -.日誌.-> L1
    CS -.歸檔.-> L2
    
    style CC fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style CR fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style CS fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Q1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Q2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Q3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Q4 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style SR fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style SE fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style ST fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style SS fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style D01 fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
    style D02 fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
    style D03 fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
    style D11 fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
    style D12 fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
    style D13 fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
    style D14 fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
    style L1 fill:#eceff1,stroke:#607d8b,stroke-width:1px
    style L2 fill:#eceff1,stroke:#607d8b,stroke-width:1px
```

**數據存儲說明**:

| 數據類型 | 存儲位置 | 生命週期 | 用途 |
|---------|---------|---------|------|
| Phase 0 結果 | Scan 內存 (臨時) | 掃描期間 | 傳遞給 Core 分析 |
| Phase 1 結果 | Scan 內存 (臨時) | 掃描期間 | 整合後傳遞給 Core |
| 完整資產清單 | Core SessionState | 會話期間 | Core 7大步驟使用 |
| 掃描日誌 | 數據庫 (可選) | 永久 | 審計和調試 |
| 結果歸檔 | 數據庫 (可選) | 永久 | 歷史查詢和報告 |

---

## 4️⃣ 時序圖 - Core 與 Scan 完整互動

展示完整的時序互動，包含 Phase 0 和 Phase 1

```mermaid
sequenceDiagram
    participant U as 用戶
    participant C as Core模組
    participant M as RabbitMQ
    participant S as Scan模組
    participant R as Rust引擎
    participant P as Python引擎
    participant T as TypeScript引擎
    
    U->>C: 提供目標URL
    activate C
    Note over C: 分析用戶輸入
    
    rect rgb(255, 243, 224)
        Note over C,R: Phase0快速偵察
        C->>M: 發布Phase0命令
        M->>S: 傳遞命令
        activate S
        S->>R: 啟動Rust掃描
        activate R
        R->>R: 敏感資訊掃描
        R->>R: 技術棧識別
        R->>R: 端點發現
        R-->>S: 返回結果
        deactivate R
        S->>S: 格式化Schema
        S->>M: 發送Phase0結果
        deactivate S
        M->>C: 傳遞結果
    end
    
    Note over C: AI分析決策
    
    rect rgb(232, 245, 233)
        Note over C,T: Phase1深度掃描
        alt 需要Phase1
            C->>M: 發布Phase1命令
            Note over M: 指定Python和TypeScript
            M->>S: 傳遞命令
            activate S
            Note over S: 解析命令初始化引擎
            
            par 並行執行
                S->>P: 執行Python掃描
                activate P
                P->>P: 靜態爬取
                P->>P: 表單發現
                P->>P: API分析
                P-->>S: 返回結果
                deactivate P
            and
                S->>T: 執行TypeScript掃描
                activate T
                T->>T: JS渲染
                T->>T: SPA路由
                T->>T: 動態內容
                T-->>S: 返回結果
                deactivate T
            end
            
            S->>S: 整合Phase0和Phase1
            S->>S: 去重關聯分析
            S->>S: 生成完整清單
            S->>M: 發送最終結果
            deactivate S
            M->>C: 傳遞完整結果
        else Phase0已足夠
            Note over C: 跳過Phase1
        end
    end
    
    Note over C: 進入Core七大步驟
    C->>U: 繼續後續流程
    deactivate C
    
    Note over U,T: 總耗時Phase0五到十分鐘加Phase1十到三十分鐘可選
```

**時序說明**:
- 🟨 黃色區塊: Phase 0 執行階段 (5-10 分鐘)
- 🟩 綠色區塊: Phase 1 執行階段 (10-30 分鐘，按需)
- `activate/deactivate`: 顯示組件的活動狀態
- `par`: 表示並行執行

---

## 5️⃣ 引擎選擇決策樹

Core 模組如何決定 Phase 1 使用哪些引擎

```mermaid
flowchart TD
    START[Phase0結果分析]
    
    CHECK1{檢測到JavaScript}
    CHECK2{檢測到表單}
    CHECK3{檢測到API端點}
    CHECK4{大量URL}
    
    USE_TS[選用TypeScript引擎]
    USE_PY[選用Python引擎]
    USE_GO[選用Go引擎]
    USE_RUST[選用Rust引擎]
    
    SKIP[跳過Phase1]
    COMBINE[組合引擎執行]
    
    START --> CHECK1
    
    CHECK1 -->|是| USE_TS
    CHECK1 -->|否| CHECK2
    
    CHECK2 -->|是| USE_PY
    CHECK2 -->|否| CHECK3
    
    CHECK3 -->|是| USE_PY
    CHECK3 -->|否| CHECK4
    
    CHECK4 -->|是| USE_GO
    CHECK4 -->|否| SKIP
    
    USE_TS --> COMBINE
    USE_PY --> COMBINE
    USE_GO --> COMBINE
    USE_RUST --> COMBINE
    
    style START fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style CHECK1 fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style CHECK2 fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style CHECK3 fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style CHECK4 fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style USE_TS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style USE_PY fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style USE_GO fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style USE_RUST fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style SKIP fill:#ffebee,stroke:#c62828,stroke-width:2px
    style COMBINE fill:#c8e6c9,stroke:#1b5e20,stroke-width:3px
```

**決策邏輯**:

| 檢測特徵 | 選擇引擎 | 原因 |
|---------|---------|------|
| 大量 JavaScript | TypeScript | 需要 JS 渲染和 SPA 處理 |
| HTML 表單 | Python | 表單爬取和參數提取 |
| REST API | Python | API 端點深度分析 |
| 大量 URL | Go | 高並發快速掃描 |
| 無特殊需求 | 跳過 Phase 1 | Phase 0 結果已足夠 |

---

## 6️⃣ 失敗處理與重試機制

```mermaid
flowchart TD
    START[執行掃描任務]
    EXEC[執行引擎掃描]
    CHECK{執行成功}
    
    RETRY_CHECK{重試次數<3}
    WAIT[等待退避時間]
    
    SUCCESS[記錄成功]
    PARTIAL[部分失敗處理]
    FAIL[記錄失敗]
    
    RESULT[返回結果給Core]
    
    START --> EXEC
    EXEC --> CHECK
    
    CHECK -->|成功| SUCCESS
    CHECK -->|失敗| RETRY_CHECK
    
    RETRY_CHECK -->|是| WAIT
    RETRY_CHECK -->|否| FAIL
    
    WAIT --> EXEC
    
    SUCCESS --> RESULT
    FAIL --> PARTIAL
    PARTIAL --> RESULT
    
    style START fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style EXEC fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style CHECK fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style RETRY_CHECK fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style WAIT fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style SUCCESS fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style PARTIAL fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style FAIL fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    style RESULT fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
```

**重試策略**:
- 最大重試次數: 3 次
- 退避策略: 指數退避 (1s, 2s, 4s)
- 部分失敗: 返回成功引擎的結果，標記失敗引擎
- 完全失敗: 返回錯誤狀態，Core 決定後續處理

---

## 📊 性能指標

### Phase 0 (Rust 快速偵察)

| 指標 | 目標值 | 說明 |
|-----|--------|------|
| 執行時間 | 5-10 分鐘 | 單目標掃描 |
| 並發連接 | 100+ | Rust 高性能 |
| 發現率 | 80%+ | 基礎資產覆蓋 |
| 內存使用 | < 500MB | 輕量級掃描 |

### Phase 1 (多引擎深度掃描)

| 指標 | 目標值 | 說明 |
|-----|--------|------|
| 執行時間 | 10-30 分鐘 | 依引擎數量 |
| 並發引擎 | 2-4 個 | 按需選擇 |
| 發現率 | 95%+ | 深度覆蓋 |
| 內存使用 | < 2GB | 多引擎並行 |

---

## 🔍 關鍵技術細節

### RabbitMQ 消息格式

**Phase 0 命令**:
```json
{
  "task_type": "phase0",
  "target_url": "https://example.com",
  "scan_id": "uuid-v4",
  "config": {
    "timeout": 600,
    "max_depth": 3
  }
}
```

**Phase 1 命令**:
```json
{
  "task_type": "phase1",
  "target_url": "https://example.com",
  "scan_id": "uuid-v4",
  "engines": ["python", "typescript"],
  "phase0_result": { ... },
  "config": {
    "timeout": 1800,
    "max_depth": 5
  }
}
```

**結果返回**:
```json
{
  "scan_id": "uuid-v4",
  "phase": "phase0|phase1",
  "status": "success|partial|failed",
  "assets": [ ... ],
  "metadata": {
    "execution_time": 450,
    "engines_used": ["rust"],
    "asset_count": 127
  }
}
```

### 數據 Schema

所有數據模型遵循 `aiva_common` 規範:
- 使用 Pydantic v2
- 單一數據來源
- 標準化 AssetSchema
- 禁止重複定義

---

## 📝 總結

Scan 模組作為 AIVA 的執行單元，在 Core 模組的指揮下完成兩階段掃描:

1. **Phase 0**: Rust 引擎快速偵察，提供初步資產清單
2. **Phase 1**: 多引擎深度掃描，生成完整資產清單

通過 RabbitMQ 消息隊列實現模組解耦，確保系統穩定性和可擴展性。所有數據最終存儲在 Core 模組的 SessionStateManager，供後續 7 大步驟使用。
