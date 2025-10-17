# Mermaid 11.11.0+ 語法測試

生成時間: 2025-10-17 09:57:39

## 1. 基本流程圖

```mermaid
%%{init: {'theme':'default'}}%%
flowchart TB
    Start(["開始"]) --> Check{"檢查條件"}
    Check -->|"通過"| Process["處理"]
    Check -->|"失敗"| Error["錯誤處理"]
    Process --> End(["結束"])
    Error --> End
    
    style Start fill:#90EE90,stroke:#2E7D32,stroke-width:2px
    style End fill:#FFB6C1,stroke:#C71585,stroke-width:2px
    style Error fill:#FFCDD2,stroke:#C62828,stroke-width:2px
```

## 2. 使用新形狀語法

```mermaid
flowchart LR
    A@{ shape: rect, label: "矩形" }
    B@{ shape: circle, label: "圓形" }
    C@{ shape: diamond, label: "菱形" }
    D@{ shape: stadium, label: "體育場" }
    
    A --> B --> C --> D
```

## 3. 多語言架構圖

```mermaid
%%{init: {'theme':'default'}}%%
flowchart TB
    subgraph "🐍 Python"
        PY["核心服務"]
    end
    
    subgraph "🦀 Rust"
        RS["SAST 引擎"]
    end
    
    subgraph "🔷 Go"
        GO["SCA 服務"]
    end
    
    subgraph "📘 TypeScript"
        TS["掃描服務"]
    end
    
    PY --> RS
    PY --> GO
    PY --> TS
    
    style PY fill:#3776ab,stroke:#2C5F8D,stroke-width:2px,color:#fff
    style RS fill:#CE422B,stroke:#A33520,stroke-width:2px,color:#fff
    style GO fill:#00ADD8,stroke:#0099BF,stroke-width:2px,color:#fff
    style TS fill:#3178C6,stroke:#2768B3,stroke-width:2px,color:#fff
```

## 4. 時序圖

```mermaid
%%{init: {'theme':'default'}}%%
sequenceDiagram
    autonumber
    participant U as 👤 使用者
    participant A as 🔌 API
    participant D as 💾 資料庫
    
    U->>A: 發送請求
    A->>D: 查詢資料
    D-->>A: 返回結果
    A-->>U: 回應資料
    
    Note over U,D: 完整的請求-回應週期
```

## 5. 類別和樣式

```mermaid
flowchart LR
    A["節點 A"] --> B["節點 B"]
    B --> C["節點 C"]
    
    classDef success fill:#90EE90,stroke:#2E7D32,stroke-width:2px
    classDef warning fill:#FFF59D,stroke:#F57F17,stroke-width:2px
    classDef danger fill:#FFCDD2,stroke:#C62828,stroke-width:2px
    
    class A success
    class B warning
    class C danger
```

---

**測試狀態**: ✓ 所有測試通過
**Mermaid 版本**: 11.11.0+
**生成工具**: AIVA Mermaid Test Suite
