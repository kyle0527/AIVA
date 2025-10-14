# AIVA 多語言架構圖

```mermaid
graph TB
    subgraph "前端層 - TypeScript/Node.js"
        SCAN[aiva_scan_node<br/>Playwright 動態掃描]
        SCAN -->|發現 API| API_DISC[API Discovery]
        SCAN -->|模擬互動| INTERACT[Interaction Simulator]
        SCAN -->|攔截請求| INTERCEPT[Network Interceptor]
    end

    subgraph "核心層 - Python"
        CORE[Core<br/>系統協調]
        INTEG[Integration<br/>任務分發與結果整合]
        AI[AI Engine<br/>BioNeuronRAGAgent]
        LIFECYCLE[Lifecycle Manager<br/>資產與漏洞管理]
        ANALYZER[Correlation Analyzer<br/>根因分析 & SAST-DAST 關聯]
    end

    subgraph "功能層 - Go (高併發 I/O)"
        CSPM[function_cspm_go<br/>雲端安全]
        SCA[function_sca_go<br/>依賴掃描]
        AUTHN[function_authn_go<br/>認證測試]
        SSRF[function_ssrf_go<br/>SSRF 檢測]
    end

    subgraph "功能層 - Rust (CPU 密集)"
        SAST[function_sast_rust<br/>靜態程式碼分析]
        SECRET[info_gatherer_rust<br/>秘密掃描]
    end

    subgraph "共用模組"
        PY_COMMON[aiva_common<br/>Python Schemas & Utils]
        GO_COMMON[aiva_common_go<br/>Go MQ/Logger/Config]
        TS_COMMON[@aiva/common<br/>TypeScript Schemas]
    end

    subgraph "基礎設施"
        MQ[RabbitMQ<br/>訊息佇列]
        DB[(PostgreSQL<br/>資料庫)]
    end

    CORE --> INTEG
    CORE --> AI
    INTEG --> LIFECYCLE
    INTEG --> ANALYZER
    
    INTEG -->|分發任務| MQ
    MQ -->|動態掃描任務| SCAN
    MQ -->|CSPM 任務| CSPM
    MQ -->|SCA 任務| SCA
    MQ -->|SAST 任務| SAST
    MQ -->|秘密掃描任務| SECRET
    
    SCAN -->|結果| MQ
    CSPM -->|結果| MQ
    SCA -->|結果| MQ
    SAST -->|結果| MQ
    SECRET -->|結果| MQ
    AUTHN -->|結果| MQ
    SSRF -->|結果| MQ
    
    MQ -->|收集結果| INTEG
    
    LIFECYCLE --> DB
    ANALYZER --> DB
    
    SCAN -.使用.- TS_COMMON
    CSPM -.使用.- GO_COMMON
    SCA -.使用.- GO_COMMON
    CORE -.使用.- PY_COMMON
    INTEG -.使用.- PY_COMMON

    classDef python fill:#3776ab,stroke:#333,stroke-width:2px,color:#fff
    classDef golang fill:#00add8,stroke:#333,stroke-width:2px,color:#fff
    classDef rust fill:#f74c00,stroke:#333,stroke-width:2px,color:#fff
    classDef typescript fill:#3178c6,stroke:#333,stroke-width:2px,color:#fff
    classDef infra fill:#2ecc71,stroke:#333,stroke-width:2px,color:#fff
    classDef common fill:#f39c12,stroke:#333,stroke-width:2px,color:#fff

    class CORE,INTEG,AI,LIFECYCLE,ANALYZER python
    class CSPM,SCA,AUTHN,SSRF golang
    class SAST,SECRET rust
    class SCAN,API_DISC,INTERACT,INTERCEPT typescript
    class MQ,DB infra
    class PY_COMMON,GO_COMMON,TS_COMMON common
```

## 語言職責分佈

### 🐍 Python (藍色) - 智慧中樞
- 系統協調與任務分發
- AI 引擎與決策邏輯
- 資產與漏洞生命週期管理
- 複雜分析 (根因分析、關聯分析)

### 🔷 Go (青色) - 高效工兵
- 雲端安全組態管理 (CSPM)
- 軟體組成分析 (SCA)
- 認證測試 (暴力破解)
- SSRF 檢測
- 所有高併發 I/O 任務

### 🦀 Rust (橘色) - 效能刺客
- 靜態程式碼分析 (SAST)
- 秘密與敏感資訊掃描
- CPU 密集型正則匹配

### 📘 TypeScript (藍紫色) - 瀏覽器大師
- Playwright 動態掃描
- SPA 渲染與測試
- API 端點自動發現
- 使用者互動模擬

### 🟢 基礎設施 (綠色)
- RabbitMQ: 跨語言訊息通訊
- PostgreSQL: 統一資料存儲

### 🟠 共用模組 (橘色)
- `aiva_common`: Python 共用程式碼
- `aiva_common_go`: Go 共用程式碼
- `@aiva/common`: TypeScript 共用程式碼

## 通訊流程

1. **任務分發**: Core → Integration → RabbitMQ → 各語言 Function
2. **結果收集**: Function → RabbitMQ → Integration → 分析與儲存
3. **資料持久化**: 所有模組 → PostgreSQL
4. **AI 增強**: Analyzer ← AI Engine → 生成修復建議

## 設計原則

✅ **契約先行**: 所有跨語言通訊基於統一 Schema  
✅ **語言專精**: 每種語言做最擅長的事  
✅ **鬆耦合**: 透過訊息佇列解耦  
✅ **可擴展**: 新增功能只需新增對應語言的 Function  
✅ **可維護**: 共用模組消除重複程式碼
