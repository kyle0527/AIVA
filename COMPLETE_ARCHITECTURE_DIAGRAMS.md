# AIVA 完整架構圖集 | Complete Architecture Diagrams

> **生成時間 Generated**: 2025-10-13
> **專案 Project**: AIVA - AI-Powered Intelligent Vulnerability Analysis Platform
> **版本 Version**: v1.0

---

## 目錄 | Table of Contents

- [AIVA 完整架構圖集 | Complete Architecture Diagrams](#aiva-完整架構圖集--complete-architecture-diagrams)
  - [目錄 | Table of Contents](#目錄--table-of-contents)
  - [1. 整體系統架構 | Overall System Architecture](#1-整體系統架構--overall-system-architecture)
  - [2. 四大模組概覽 | Four Core Modules Overview](#2-四大模組概覽--four-core-modules-overview)
  - [3. 核心引擎模組 | Core Engine Module](#3-核心引擎模組--core-engine-module)
  - [4. 掃描引擎模組 | Scan Engine Module](#4-掃描引擎模組--scan-engine-module)
  - [5. 檢測功能模組 | Detection Function Module](#5-檢測功能模組--detection-function-module)
  - [6. 整合服務模組 | Integration Service Module](#6-整合服務模組--integration-service-module)
  - [7. SQL 注入檢測流程 | SQLi Detection Flow](#7-sql-注入檢測流程--sqli-detection-flow)
  - [8. XSS 檢測流程 | XSS Detection Flow](#8-xss-檢測流程--xss-detection-flow)
  - [9. SSRF 檢測流程 | SSRF Detection Flow](#9-ssrf-檢測流程--ssrf-detection-flow)
  - [10. IDOR 檢測流程 | IDOR Detection Flow](#10-idor-檢測流程--idor-detection-flow)
  - [11. 完整掃描工作流程 | Complete Scan Workflow](#11-完整掃描工作流程--complete-scan-workflow)
  - [12. 多語言架構決策 | Multi-Language Architecture Decision](#12-多語言架構決策--multi-language-architecture-decision)
  - [13. 資料流程圖 | Data Flow Diagram](#13-資料流程圖--data-flow-diagram)
  - [14. 部署架構圖 | Deployment Architecture](#14-部署架構圖--deployment-architecture)
  - [圖表說明 | Diagram Descriptions](#圖表說明--diagram-descriptions)
    - [使用方法 | Usage](#使用方法--usage)
    - [圖表類型 | Diagram Types](#圖表類型--diagram-types)
    - [顏色說明 | Color Legend](#顏色說明--color-legend)
  - [生成腳本 | Generation Script](#生成腳本--generation-script)

---

## 1. 整體系統架構 | Overall System Architecture

```mermaid
graph TB
    subgraph "前端層 Frontend Layer"
        UI[Web UI<br/>網頁介面]
        API[REST API<br/>REST 接口]
    end

    subgraph "核心層 Core Layer"
        CORE[AI Core Engine<br/>AI 核心引擎]
        STRATEGY[Strategy Generator<br/>策略生成器]
        TASK[Task Manager<br/>任務管理器]
    end

    subgraph "掃描層 Scan Layer"
        SCAN_PY[Python Scanner<br/>Python 掃描器]
        SCAN_TS[TypeScript Scanner<br/>TypeScript 掃描器]
        SCAN_RS[Rust Info Gatherer<br/>Rust 資訊收集]
    end

    subgraph "檢測層 Detection Layer"
        FUNC_PY[Python Functions<br/>Python 檢測模組]
        FUNC_GO[Go Functions<br/>Go 檢測模組]
        FUNC_RS[Rust Functions<br/>Rust 檢測模組]
    end

    subgraph "整合層 Integration Layer"
        INTG[Integration Service<br/>整合服務]
        REPORT[Report Generator<br/>報告生成器]
        ANALYSIS[Risk Analyzer<br/>風險分析器]
    end

    subgraph "資料層 Data Layer"
        DB[(PostgreSQL<br/>資料庫)]
        MQ[RabbitMQ<br/>訊息佇列]
        REDIS[(Redis<br/>快取)]
    end

    UI --> API
    API --> CORE
    CORE --> STRATEGY
    CORE --> TASK

    TASK --> SCAN_PY
    TASK --> SCAN_TS
    TASK --> SCAN_RS

    SCAN_PY --> MQ
    SCAN_TS --> MQ
    SCAN_RS --> MQ

    MQ --> FUNC_PY
    MQ --> FUNC_GO
    MQ --> FUNC_RS

    FUNC_PY --> MQ
    FUNC_GO --> MQ
    FUNC_RS --> MQ

    MQ --> INTG
    INTG --> REPORT
    INTG --> ANALYSIS

    CORE --> DB
    INTG --> DB
    API --> REDIS

    style UI fill:#E1F5FE
    style CORE fill:#FFE082
    style SCAN_PY fill:#C5E1A5
    style FUNC_PY fill:#CE93D8
    style INTG fill:#FFAB91
    style DB fill:#B0BEC5
```

---

## 2. 四大模組概覽 | Four Core Modules Overview

```mermaid
graph LR
    subgraph "Module 1: Core Engine<br/>模組一：核心引擎"
        CORE_AI[AI Engine<br/>AI 引擎]
        CORE_EXEC[Execution Engine<br/>執行引擎]
        CORE_STATE[State Manager<br/>狀態管理]
    end

    subgraph "Module 2: Scan Engine<br/>模組二：掃描引擎"
        SCAN_STATIC[Static Scanner<br/>靜態掃描]
        SCAN_DYN[Dynamic Scanner<br/>動態掃描]
        SCAN_INFO[Info Collector<br/>資訊收集]
    end

    subgraph "Module 3: Detection Functions<br/>模組三：檢測功能"
        FUNC_WEB[Web Vulnerabilities<br/>Web 漏洞]
        FUNC_CLOUD[Cloud Security<br/>雲端安全]
        FUNC_CODE[Code Analysis<br/>程式碼分析]
    end

    subgraph "Module 4: Integration<br/>模組四：整合服務"
        INTG_API[API Gateway<br/>API 閘道]
        INTG_RPT[Reporting<br/>報告系統]
        INTG_RISK[Risk Assessment<br/>風險評估]
    end

    CORE_AI --> CORE_EXEC
    CORE_EXEC --> CORE_STATE
    CORE_STATE --> SCAN_STATIC

    SCAN_STATIC --> SCAN_DYN
    SCAN_DYN --> SCAN_INFO
    SCAN_INFO --> FUNC_WEB

    FUNC_WEB --> FUNC_CLOUD
    FUNC_CLOUD --> FUNC_CODE
    FUNC_CODE --> INTG_API

    INTG_API --> INTG_RPT
    INTG_RPT --> INTG_RISK

    style CORE_AI fill:#FFD54F
    style SCAN_STATIC fill:#81C784
    style FUNC_WEB fill:#BA68C8
    style INTG_API fill:#FF8A65
```

---

## 3. 核心引擎模組 | Core Engine Module

```mermaid
graph TB
    subgraph "Core Module<br/>核心引擎模組"
        subgraph "AI Engine<br/>AI 引擎"
            BIO[Bio Neuron Core<br/>生物神經網路核心]
            KB[Knowledge Base<br/>知識庫]
            TOOLS[AI Tools<br/>AI 工具]
        end

        subgraph "Analysis<br/>分析模組"
            INIT[Initial Surface<br/>初始攻擊面]
            STRATEGY[Strategy Generator<br/>策略生成器]
            DYNAMIC[Dynamic Adjustment<br/>動態調整]
        end

        subgraph "Execution<br/>執行引擎"
            TASKGEN[Task Generator<br/>任務生成器]
            QUEUE[Task Queue Manager<br/>任務佇列管理]
            MONITOR[Status Monitor<br/>狀態監控]
        end

        subgraph "State Management<br/>狀態管理"
            SESSION[Session State<br/>會話狀態]
            CONTEXT[Scan Context<br/>掃描上下文]
        end

        subgraph "UI Panel<br/>UI 面板"
            DASH[Dashboard<br/>儀表板]
            SERVER[UI Server<br/>UI 服務器]
        end
    end

    BIO --> KB
    KB --> TOOLS
    TOOLS --> INIT

    INIT --> STRATEGY
    STRATEGY --> DYNAMIC
    DYNAMIC --> TASKGEN

    TASKGEN --> QUEUE
    QUEUE --> MONITOR
    MONITOR --> SESSION

    SESSION --> CONTEXT
    CONTEXT --> DASH
    DASH --> SERVER

    style BIO fill:#FFD54F
    style STRATEGY fill:#FFF59D
    style TASKGEN fill:#FFECB3
    style DASH fill:#FFE082
```

---

## 4. 掃描引擎模組 | Scan Engine Module

```mermaid
graph TB
    subgraph "Scan Module<br/>掃描引擎模組"
        subgraph "Python Scanner<br/>Python 掃描器"
            subgraph "Core Crawling<br/>核心爬蟲"
                HTTP[HTTP Client<br/>HTTP 客戶端]
                PARSER[Content Parser<br/>內容解析器]
                URLQ[URL Queue<br/>URL 佇列]
            end

            subgraph "Dynamic Engine<br/>動態引擎"
                BROWSER[Browser Pool<br/>瀏覽器池]
                JS_SIM[JS Simulator<br/>JS 模擬器]
                EXTRACTOR[Content Extractor<br/>內容提取器]
            end

            subgraph "Info Gatherer<br/>資訊收集"
                JS_ANAL[JS Analyzer<br/>JS 分析器]
                FINGER[Fingerprinter<br/>指紋識別]
                SENSITIVE[Sensitive Detector<br/>敏感資訊檢測]
            end
        end

        subgraph "TypeScript Scanner<br/>TypeScript 掃描器"
            TS_SERVICE[Scan Service<br/>掃描服務]
            TS_LOGGER[Logger<br/>日誌]
        end

        subgraph "Rust Info Gatherer<br/>Rust 資訊收集"
            GIT_SCAN[Git History Scanner<br/>Git 歷史掃描]
            SECRET[Secret Detector<br/>秘密檢測器]
        end

        subgraph "Configuration<br/>配置管理"
            AUTH_MGR[Auth Manager<br/>身份驗證管理]
            CONFIG[Config Center<br/>配置中心]
            SCOPE[Scope Manager<br/>範圍管理]
        end
    end

    HTTP --> PARSER
    PARSER --> URLQ
    URLQ --> BROWSER

    BROWSER --> JS_SIM
    JS_SIM --> EXTRACTOR
    EXTRACTOR --> JS_ANAL

    JS_ANAL --> FINGER
    FINGER --> SENSITIVE
    SENSITIVE --> TS_SERVICE

    TS_SERVICE --> TS_LOGGER
    TS_LOGGER --> GIT_SCAN
    GIT_SCAN --> SECRET

    SECRET --> AUTH_MGR
    AUTH_MGR --> CONFIG
    CONFIG --> SCOPE

    style HTTP fill:#81C784
    style BROWSER fill:#AED581
    style TS_SERVICE fill:#4FC3F7
    style GIT_SCAN fill:#E57373
```

---

## 5. 檢測功能模組 | Detection Function Module

```mermaid
graph TB
    subgraph "Function Module<br/>檢測功能模組"
        subgraph "Python Functions<br/>Python 檢測模組"
            SQLI[SQLi Detection<br/>SQL 注入檢測]
            XSS[XSS Detection<br/>XSS 檢測]
            SSRF_PY[SSRF Detection<br/>SSRF 檢測]
            IDOR[IDOR Detection<br/>IDOR 檢測]
        end

        subgraph "Go Functions<br/>Go 檢測模組"
            AUTH[AuthN Detection<br/>身份驗證檢測]
            CSPM[CSPM Scanner<br/>雲端安全掃描]
            SCA[SCA Scanner<br/>軟體組成分析]
            SSRF_GO[SSRF Detection<br/>SSRF 檢測]
        end

        subgraph "Rust Functions<br/>Rust 檢測模組"
            SAST[SAST Analyzer<br/>靜態分析]
        end

        subgraph "Common Components<br/>共用組件"
            CONFIG_DETECT[Detection Config<br/>檢測配置]
            SMART_MGR[Smart Manager<br/>智能管理器]
        end
    end

    CONFIG_DETECT --> SQLI
    CONFIG_DETECT --> XSS
    CONFIG_DETECT --> SSRF_PY
    CONFIG_DETECT --> IDOR

    SMART_MGR --> AUTH
    SMART_MGR --> CSPM
    SMART_MGR --> SCA
    SMART_MGR --> SSRF_GO

    CONFIG_DETECT --> SAST

    SQLI --> SMART_MGR
    XSS --> SMART_MGR
    AUTH --> SAST

    style SQLI fill:#BA68C8
    style XSS fill:#CE93D8
    style AUTH fill:#64B5F6
    style CSPM fill:#4FC3F7
    style SAST fill:#E57373
```

---

## 6. 整合服務模組 | Integration Service Module

```mermaid
graph TB
    subgraph "Integration Module<br/>整合服務模組"
        subgraph "Analysis<br/>分析服務"
            COMPLY[Compliance Checker<br/>合規性檢查]
            RISK[Risk Assessment<br/>風險評估]
            CORREL[Correlation Analyzer<br/>關聯分析]
        end

        subgraph "Attack Path<br/>攻擊路徑"
            ATK_ENGINE[Analysis Engine<br/>分析引擎]
            GRAPH[Graph Builder<br/>圖構建器]
            VISUAL[Visualizer<br/>視覺化]
        end

        subgraph "Reporting<br/>報告系統"
            CONTENT[Content Generator<br/>內容生成器]
            TEMPLATE[Template Selector<br/>模板選擇器]
            EXPORT[Formatter Exporter<br/>格式化匯出]
        end

        subgraph "Performance Feedback<br/>效能回饋"
            META[Metadata Analyzer<br/>元資料分析]
            SUGGEST[Suggestion Generator<br/>建議生成器]
        end

        subgraph "Infrastructure<br/>基礎設施"
            GATEWAY[API Gateway<br/>API 閘道]
            RECEPTION[Data Reception<br/>資料接收]
            DB_INTG[SQL Database<br/>SQL 資料庫]
        end
    end

    COMPLY --> RISK
    RISK --> CORREL
    CORREL --> ATK_ENGINE

    ATK_ENGINE --> GRAPH
    GRAPH --> VISUAL
    VISUAL --> CONTENT

    CONTENT --> TEMPLATE
    TEMPLATE --> EXPORT
    EXPORT --> META

    META --> SUGGEST
    SUGGEST --> GATEWAY
    GATEWAY --> RECEPTION
    RECEPTION --> DB_INTG

    style COMPLY fill:#FF8A65
    style ATK_ENGINE fill:#FFAB91
    style CONTENT fill:#FFCCBC
    style GATEWAY fill:#BCAAA4
```

---

## 7. SQL 注入檢測流程 | SQLi Detection Flow

```mermaid
flowchart TD
    START([Start SQLi Detection<br/>開始 SQL 注入檢測]) --> RECEIVE[Receive Task<br/>接收任務]
    RECEIVE --> FINGERPRINT[Database Fingerprint<br/>資料庫指紋識別]

    FINGERPRINT --> SELECT{Select Engine<br/>選擇引擎}

    SELECT -->|Boolean<br/>布爾盲注| BOOL_ENGINE[Boolean Engine<br/>布爾盲注引擎]
    SELECT -->|Time<br/>時間盲注| TIME_ENGINE[Time Engine<br/>時間盲注引擎]
    SELECT -->|Error<br/>錯誤注入| ERROR_ENGINE[Error Engine<br/>錯誤注入引擎]
    SELECT -->|Union<br/>聯合注入| UNION_ENGINE[Union Engine<br/>聯合注入引擎]
    SELECT -->|OOB<br/>帶外注入| OOB_ENGINE[OOB Engine<br/>帶外注入引擎]

    BOOL_ENGINE --> PAYLOAD[Generate Payload<br/>生成 Payload]
    TIME_ENGINE --> PAYLOAD
    ERROR_ENGINE --> PAYLOAD
    UNION_ENGINE --> PAYLOAD
    OOB_ENGINE --> PAYLOAD

    PAYLOAD --> ENCODE[Encode & Wrap<br/>編碼與包裝]
    ENCODE --> SEND[Send Request<br/>發送請求]
    SEND --> ANALYZE[Analyze Response<br/>分析響應]

    ANALYZE --> DETECT{Vulnerability?<br/>發現漏洞?}
    DETECT -->|Yes<br/>是| CONFIRM[Confirm Vulnerability<br/>確認漏洞]
    DETECT -->|No<br/>否| NEXT{More Tests?<br/>更多測試?}

    NEXT -->|Yes<br/>是| PAYLOAD
    NEXT -->|No<br/>否| REPORT_SAFE[Report Safe<br/>報告安全]

    CONFIRM --> EXTRACT[Extract Data<br/>提取資料]
    EXTRACT --> REPORT_VULN[Report Vulnerability<br/>報告漏洞]

    REPORT_SAFE --> PUBLISH[Publish Result<br/>發布結果]
    REPORT_VULN --> PUBLISH
    PUBLISH --> END([End<br/>結束])

    style START fill:#90EE90
    style BOOL_ENGINE fill:#BA68C8
    style TIME_ENGINE fill:#BA68C8
    style ERROR_ENGINE fill:#BA68C8
    style UNION_ENGINE fill:#BA68C8
    style OOB_ENGINE fill:#BA68C8
    style CONFIRM fill:#FFD54F
    style END fill:#FF6B6B
```

---

## 8. XSS 檢測流程 | XSS Detection Flow

```mermaid
flowchart TD
    START([Start XSS Detection<br/>開始 XSS 檢測]) --> RECEIVE[Receive Target<br/>接收目標]
    RECEIVE --> IDENTIFY{Identify Type<br/>識別類型}

    IDENTIFY -->|Reflected<br/>反射型| REFLECT[Reflected XSS<br/>反射型 XSS]
    IDENTIFY -->|Stored<br/>儲存型| STORED[Stored XSS<br/>儲存型 XSS]
    IDENTIFY -->|DOM<br/>DOM 型| DOM[DOM XSS<br/>DOM XSS]

    REFLECT --> GEN_PAYLOAD[Generate Payloads<br/>生成 Payloads]
    STORED --> GEN_PAYLOAD
    DOM --> ANALYZE_JS[Analyze JavaScript<br/>分析 JavaScript]

    ANALYZE_JS --> FIND_SINK[Find Sinks<br/>查找危險函數]
    FIND_SINK --> GEN_PAYLOAD

    GEN_PAYLOAD --> CONTEXT{Injection Context?<br/>注入上下文?}

    CONTEXT -->|HTML Tag<br/>HTML 標籤| HTML_PAYLOAD[HTML Payload<br/>HTML Payload]
    CONTEXT -->|Attribute<br/>屬性| ATTR_PAYLOAD[Attribute Payload<br/>屬性 Payload]
    CONTEXT -->|JavaScript<br/>JavaScript| JS_PAYLOAD[JavaScript Payload<br/>JS Payload]
    CONTEXT -->|Event<br/>事件| EVENT_PAYLOAD[Event Payload<br/>事件 Payload]

    HTML_PAYLOAD --> INJECT[Inject Payload<br/>注入 Payload]
    ATTR_PAYLOAD --> INJECT
    JS_PAYLOAD --> INJECT
    EVENT_PAYLOAD --> INJECT

    INJECT --> VERIFY{Verify Execution?<br/>驗證執行?}

    VERIFY -->|Executed<br/>已執行| BLIND_CHECK{Blind XSS?<br/>盲 XSS?}
    VERIFY -->|Not Executed<br/>未執行| NEXT{More Payloads?<br/>更多 Payload?}

    BLIND_CHECK -->|Yes<br/>是| LISTENER[Wait for Callback<br/>等待回調]
    BLIND_CHECK -->|No<br/>否| REPORT_VULN[Report Vulnerability<br/>報告漏洞]

    LISTENER --> CALLBACK{Received?<br/>收到回調?}
    CALLBACK -->|Yes<br/>是| REPORT_VULN
    CALLBACK -->|No<br/>否| TIMEOUT[Timeout<br/>超時]

    NEXT -->|Yes<br/>是| GEN_PAYLOAD
    NEXT -->|No<br/>否| REPORT_SAFE[Report Safe<br/>報告安全]

    TIMEOUT --> REPORT_SAFE
    REPORT_VULN --> PUBLISH[Publish Result<br/>發布結果]
    REPORT_SAFE --> PUBLISH
    PUBLISH --> END([End<br/>結束])

    style START fill:#90EE90
    style REFLECT fill:#CE93D8
    style STORED fill:#CE93D8
    style DOM fill:#CE93D8
    style REPORT_VULN fill:#FFD54F
    style END fill:#FF6B6B
```

---

## 9. SSRF 檢測流程 | SSRF Detection Flow

```mermaid
flowchart TD
    START([Start SSRF Detection<br/>開始 SSRF 檢測]) --> RECEIVE[Receive Parameters<br/>接收參數]
    RECEIVE --> SEMANTIC[Semantic Analysis<br/>語義分析]

    SEMANTIC --> IDENTIFY{Identify Target<br/>識別目標類型}

    IDENTIFY -->|URL Parameter<br/>URL 參數| URL_TEST[URL Parameter Test<br/>URL 參數測試]
    IDENTIFY -->|File Path<br/>文件路徑| FILE_TEST[File Path Test<br/>文件路徑測試]
    IDENTIFY -->|API Endpoint<br/>API 端點| API_TEST[API Endpoint Test<br/>API 端點測試]

    URL_TEST --> INTERNAL[Internal Address Detection<br/>內網位址檢測]
    FILE_TEST --> INTERNAL
    API_TEST --> INTERNAL

    INTERNAL --> GEN_PAYLOAD[Generate Payloads<br/>生成 Payloads]

    GEN_PAYLOAD --> PAYLOAD_TYPE{Payload Type<br/>Payload 類型}

    PAYLOAD_TYPE -->|Internal IP<br/>內網 IP| INTERNAL_IP[192.168.x.x<br/>10.x.x.x<br/>172.16.x.x]
    PAYLOAD_TYPE -->|Localhost<br/>本地主機| LOCALHOST[localhost<br/>127.0.0.1<br/>0.0.0.0]
    PAYLOAD_TYPE -->|Cloud Metadata<br/>雲端元資料| METADATA[169.254.169.254<br/>Metadata API]
    PAYLOAD_TYPE -->|DNS Rebinding<br/>DNS 重綁定| DNS_REBIND[DNS Rebinding<br/>DNS 重綁定]

    INTERNAL_IP --> OAST[Setup OAST Platform<br/>設置 OAST 平台]
    LOCALHOST --> OAST
    METADATA --> OAST
    DNS_REBIND --> OAST

    OAST --> SEND[Send Request<br/>發送請求]
    SEND --> WAIT[Wait for Response<br/>等待響應]

    WAIT --> CHECK{Check Response<br/>檢查響應}

    CHECK -->|Success<br/>成功| VERIFY[Verify SSRF<br/>驗證 SSRF]
    CHECK -->|Timeout<br/>超時| OAST_CHECK{OAST Callback?<br/>OAST 回調?}
    CHECK -->|Error<br/>錯誤| NEXT{More Tests?<br/>更多測試?}

    OAST_CHECK -->|Yes<br/>是| VERIFY
    OAST_CHECK -->|No<br/>否| NEXT

    VERIFY --> EXPLOIT[Attempt Exploitation<br/>嘗試利用]
    EXPLOIT --> ASSESS[Assess Impact<br/>評估影響]
    ASSESS --> REPORT_VULN[Report Vulnerability<br/>報告漏洞]

    NEXT -->|Yes<br/>是| GEN_PAYLOAD
    NEXT -->|No<br/>否| REPORT_SAFE[Report Safe<br/>報告安全]

    REPORT_VULN --> PUBLISH[Publish Result<br/>發布結果]
    REPORT_SAFE --> PUBLISH
    PUBLISH --> END([End<br/>結束])

    style START fill:#90EE90
    style INTERNAL fill:#4FC3F7
    style OAST fill:#64B5F6
    style VERIFY fill:#FFD54F
    style END fill:#FF6B6B
```

---

## 10. IDOR 檢測流程 | IDOR Detection Flow

```mermaid
flowchart TD
    START([Start IDOR Detection<br/>開始 IDOR 檢測]) --> RECEIVE[Receive API Endpoints<br/>接收 API 端點]
    RECEIVE --> EXTRACT[Extract Resource IDs<br/>提取資源 ID]

    EXTRACT --> ANALYZE{Analyze ID Pattern<br/>分析 ID 模式}

    ANALYZE -->|Sequential<br/>順序型| SEQ[Sequential IDs<br/>順序 ID]
    ANALYZE -->|UUID<br/>UUID| UUID[UUID Pattern<br/>UUID 模式]
    ANALYZE -->|Hash<br/>雜湊| HASH[Hash Pattern<br/>雜湊模式]
    ANALYZE -->|Custom<br/>自訂| CUSTOM[Custom Pattern<br/>自訂模式]

    SEQ --> TEST_TYPE{Test Type<br/>測試類型}
    UUID --> TEST_TYPE
    HASH --> TEST_TYPE
    CUSTOM --> TEST_TYPE

    TEST_TYPE -->|BFLA<br/>功能級授權| BFLA[BFLA Test<br/>功能級授權測試]
    TEST_TYPE -->|Vertical<br/>垂直提權| VERTICAL[Vertical Escalation<br/>垂直提權]
    TEST_TYPE -->|Horizontal<br/>水平越權| HORIZONTAL[Horizontal Access<br/>水平越權]
    TEST_TYPE -->|Cross-User<br/>跨用戶| CROSS[Cross-User Test<br/>跨用戶測試]

    BFLA --> CREATE_USER[Create Test Users<br/>創建測試用戶]
    VERTICAL --> CREATE_USER
    HORIZONTAL --> CREATE_USER
    CROSS --> CREATE_USER

    CREATE_USER --> USER_ROLE{User Roles<br/>用戶角色}

    USER_ROLE -->|Admin<br/>管理員| ADMIN[Admin User<br/>管理員用戶]
    USER_ROLE -->|Normal<br/>普通用戶| NORMAL[Normal User<br/>普通用戶]
    USER_ROLE -->|Guest<br/>訪客| GUEST[Guest User<br/>訪客用戶]

    ADMIN --> TEST_ACCESS[Test Access Control<br/>測試訪問控制]
    NORMAL --> TEST_ACCESS
    GUEST --> TEST_ACCESS

    TEST_ACCESS --> ATTEMPT[Attempt Unauthorized Access<br/>嘗試未授權訪問]
    ATTEMPT --> VERIFY{Access Granted?<br/>訪問成功?}

    VERIFY -->|Yes<br/>是| MASS_ASSIGN{Mass Assignment?<br/>大量賦值?}
    VERIFY -->|No<br/>否| NEXT{More Tests?<br/>更多測試?}

    MASS_ASSIGN -->|Yes<br/>是| TEST_MASS[Test Mass Assignment<br/>測試大量賦值]
    MASS_ASSIGN -->|No<br/>否| REPORT_VULN[Report Vulnerability<br/>報告漏洞]

    TEST_MASS --> MODIFY[Modify Restricted Fields<br/>修改受限欄位]
    MODIFY --> CHECK_MODIFY{Modification Success?<br/>修改成功?}

    CHECK_MODIFY -->|Yes<br/>是| REPORT_VULN
    CHECK_MODIFY -->|No<br/>否| REPORT_VULN

    NEXT -->|Yes<br/>是| TEST_TYPE
    NEXT -->|No<br/>否| REPORT_SAFE[Report Safe<br/>報告安全]

    REPORT_VULN --> PUBLISH[Publish Result<br/>發布結果]
    REPORT_SAFE --> PUBLISH
    PUBLISH --> END([End<br/>結束])

    style START fill:#90EE90
    style BFLA fill:#BA68C8
    style VERTICAL fill:#BA68C8
    style HORIZONTAL fill:#BA68C8
    style CROSS fill:#BA68C8
    style REPORT_VULN fill:#FFD54F
    style END fill:#FF6B6B
```

---

## 11. 完整掃描工作流程 | Complete Scan Workflow

```mermaid
sequenceDiagram
    autonumber
    participant User as 👤 User<br/>使用者
    participant UI as 🖥️ Web UI<br/>網頁介面
    participant API as 🔌 REST API<br/>REST 接口
    participant Core as 🤖 AI Core<br/>AI 核心
    participant Strategy as 📋 Strategy<br/>策略生成
    participant Queue as 📬 Task Queue<br/>任務佇列
    participant Scan as 🔍 Scanner<br/>掃描器
    participant MQ as 📨 RabbitMQ<br/>訊息佇列
    participant Func as ⚡ Functions<br/>檢測模組
    participant Intg as 🔗 Integration<br/>整合服務
    participant DB as 🗄️ Database<br/>資料庫

    User->>UI: Submit Scan Request<br/>提交掃描請求
    UI->>API: POST /api/scan<br/>發送掃描請求
    API->>Core: Create Scan Task<br/>創建掃描任務
    Core->>DB: Save Task Info<br/>保存任務資訊
    DB-->>Core: Task ID<br/>任務 ID

    Core->>Strategy: Generate Strategy<br/>生成策略
    Strategy-->>Core: Strategy Plan<br/>策略計劃

    Core->>Queue: Dispatch Tasks<br/>分發任務
    Queue->>Scan: Start Scanning<br/>開始掃描

    par Parallel Scanning<br/>並行掃描
        Scan->>Scan: Static Analysis<br/>靜態分析
        Scan->>Scan: Dynamic Analysis<br/>動態分析
        Scan->>Scan: Info Gathering<br/>資訊收集
    end

    Scan->>MQ: Publish Targets<br/>發布目標

    par Parallel Detection<br/>並行檢測
        MQ->>Func: SQLi Detection<br/>SQL 注入檢測
        MQ->>Func: XSS Detection<br/>XSS 檢測
        MQ->>Func: SSRF Detection<br/>SSRF 檢測
        MQ->>Func: IDOR Detection<br/>IDOR 檢測
        MQ->>Func: AuthN Check<br/>身份驗證檢查
        MQ->>Func: Cloud Security<br/>雲端安全
    end

    Func-->>MQ: Detection Results<br/>檢測結果
    MQ->>Intg: Aggregate Results<br/>彙總結果

    Intg->>Intg: Correlation Analysis<br/>關聯分析
    Intg->>Intg: Risk Assessment<br/>風險評估
    Intg->>Intg: Generate Report<br/>生成報告

    Intg->>DB: Save Results<br/>保存結果
    Intg->>Core: Notify Completion<br/>通知完成
    Core->>API: Scan Complete<br/>掃描完成
    API->>UI: Update Status<br/>更新狀態
    UI->>User: Display Report<br/>顯示報告
```

---

## 12. 多語言架構決策 | Multi-Language Architecture Decision

```mermaid
flowchart TD
    START([New Feature Request<br/>新功能需求]) --> ANALYZE[Analyze Requirements<br/>分析需求]

    ANALYZE --> PERF{High Performance<br/>Needed?<br/>需要高效能?}

    PERF -->|Yes 是| CONCUR{High Concurrency?<br/>高併發?}
    PERF -->|No 否| WEB{Web API?<br/>Web 接口?}

    CONCUR -->|Yes 是| MEMORY{Memory Safety<br/>Critical?<br/>記憶體安全<br/>關鍵?}
    CONCUR -->|No 否| LANG_GO[Choose Go<br/>選擇 Go]

    MEMORY -->|Yes 是| LANG_RUST[Choose Rust<br/>選擇 Rust]
    MEMORY -->|No 否| LANG_GO

    WEB -->|Yes 是| LANG_PYTHON[Choose Python<br/>選擇 Python]
    WEB -->|No 否| BROWSER{Browser<br/>Automation?<br/>瀏覽器<br/>自動化?}

    BROWSER -->|Yes 是| LANG_TS[Choose TypeScript<br/>選擇 TypeScript]
    BROWSER -->|No 否| LANG_PYTHON

    LANG_PYTHON --> PY_USE[Use Cases:<br/>使用場景:<br/>• FastAPI<br/>• Core Logic<br/>核心邏輯<br/>• Complex Detection<br/>複雜檢測]

    LANG_GO --> GO_USE[Use Cases:<br/>使用場景:<br/>• AuthN<br/>• CSPM<br/>• SCA<br/>• SSRF]

    LANG_RUST --> RS_USE[Use Cases:<br/>使用場景:<br/>• SAST<br/>• Secret Scanning<br/>秘密掃描<br/>• Performance Critical<br/>效能關鍵]

    LANG_TS --> TS_USE[Use Cases:<br/>使用場景:<br/>• Playwright<br/>• Dynamic Scanning<br/>動態掃描<br/>• Browser Testing<br/>瀏覽器測試]

    PY_USE --> IMPLEMENT[Implement Module<br/>實現模組]
    GO_USE --> IMPLEMENT
    RS_USE --> IMPLEMENT
    TS_USE --> IMPLEMENT

    IMPLEMENT --> INTEGRATE[Integrate via MQ<br/>通過 MQ 整合]
    INTEGRATE --> DEPLOY[Deploy Service<br/>部署服務]
    DEPLOY --> END([Complete<br/>完成])

    style LANG_PYTHON fill:#3776AB
    style LANG_GO fill:#00ADD8
    style LANG_RUST fill:#CE422B
    style LANG_TS fill:#3178C6
    style IMPLEMENT fill:#90EE90
```

---

## 13. 資料流程圖 | Data Flow Diagram

```mermaid
graph TB
    subgraph "Input Layer<br/>輸入層"
        USER_INPUT[User Input<br/>用戶輸入]
        API_REQ[API Request<br/>API 請求]
    end

    subgraph "Validation Layer<br/>驗證層"
        AUTH[Authentication<br/>身份驗證]
        VALIDATE[Input Validation<br/>輸入驗證]
        SANITIZE[Sanitization<br/>清理]
    end

    subgraph "Processing Layer<br/>處理層"
        PARSE[Parse Request<br/>解析請求]
        CREATE_TASK[Create Task<br/>創建任務]
        QUEUE_TASK[Queue Task<br/>任務入隊]
    end

    subgraph "Execution Layer<br/>執行層"
        DISPATCH[Dispatch<br/>分發]
        SCAN_EXEC[Execute Scan<br/>執行掃描]
        DETECT_EXEC[Execute Detection<br/>執行檢測]
    end

    subgraph "Message Layer<br/>訊息層"
        PUBLISH[Publish Message<br/>發布訊息]
        CONSUME[Consume Message<br/>消費訊息]
        MQ_STORE[Message Storage<br/>訊息儲存]
    end

    subgraph "Analysis Layer<br/>分析層"
        COLLECT[Collect Results<br/>收集結果]
        CORRELATE[Correlate Data<br/>關聯資料]
        ASSESS[Assess Risk<br/>評估風險]
    end

    subgraph "Storage Layer<br/>儲存層"
        DB_WRITE[Write to DB<br/>寫入資料庫]
        CACHE_WRITE[Write to Cache<br/>寫入快取]
        FILE_STORE[File Storage<br/>文件儲存]
    end

    subgraph "Output Layer<br/>輸出層"
        FORMAT[Format Output<br/>格式化輸出]
        GENERATE_RPT[Generate Report<br/>生成報告]
        SEND_RESPONSE[Send Response<br/>發送響應]
    end

    USER_INPUT --> API_REQ
    API_REQ --> AUTH
    AUTH --> VALIDATE
    VALIDATE --> SANITIZE

    SANITIZE --> PARSE
    PARSE --> CREATE_TASK
    CREATE_TASK --> QUEUE_TASK

    QUEUE_TASK --> DISPATCH
    DISPATCH --> SCAN_EXEC
    DISPATCH --> DETECT_EXEC

    SCAN_EXEC --> PUBLISH
    DETECT_EXEC --> PUBLISH
    PUBLISH --> MQ_STORE
    MQ_STORE --> CONSUME

    CONSUME --> COLLECT
    COLLECT --> CORRELATE
    CORRELATE --> ASSESS

    ASSESS --> DB_WRITE
    ASSESS --> CACHE_WRITE
    ASSESS --> FILE_STORE

    DB_WRITE --> FORMAT
    CACHE_WRITE --> FORMAT
    FILE_STORE --> FORMAT

    FORMAT --> GENERATE_RPT
    GENERATE_RPT --> SEND_RESPONSE

    style USER_INPUT fill:#E1F5FE
    style AUTH fill:#FFF9C4
    style DISPATCH fill:#C5E1A5
    style COLLECT fill:#CE93D8
    style DB_WRITE fill:#FFAB91
    style SEND_RESPONSE fill:#B0BEC5
```

---

## 14. 部署架構圖 | Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer<br/>負載均衡器"
        LB[Nginx<br/>反向代理]
    end

    subgraph "Container Orchestration<br/>容器編排"
        subgraph "Docker Compose / Kubernetes<br/>Docker Compose / Kubernetes"
            subgraph "Web Layer<br/>Web 層"
                WEB1[FastAPI Instance 1<br/>FastAPI 實例 1]
                WEB2[FastAPI Instance 2<br/>FastAPI 實例 2]
                WEB3[FastAPI Instance 3<br/>FastAPI 實例 3]
            end

            subgraph "Core Services<br/>核心服務"
                CORE1[AI Core Engine<br/>AI 核心引擎]
                CORE2[Task Manager<br/>任務管理器]
            end

            subgraph "Scan Services<br/>掃描服務"
                SCAN_PY[Python Scanner<br/>Python 掃描器]
                SCAN_TS[TypeScript Scanner<br/>TypeScript 掃描器]
                SCAN_RS[Rust Info Gatherer<br/>Rust 資訊收集]
            end

            subgraph "Function Services<br/>檢測服務"
                FUNC_PY[Python Functions<br/>Python 檢測]
                FUNC_GO[Go Functions<br/>Go 檢測]
                FUNC_RS[Rust Functions<br/>Rust 檢測]
            end

            subgraph "Integration Services<br/>整合服務"
                INTG[Integration Service<br/>整合服務]
                REPORT[Report Service<br/>報告服務]
            end
        end
    end

    subgraph "Message Queue<br/>訊息佇列"
        MQ_CLUSTER[RabbitMQ Cluster<br/>RabbitMQ 集群]
    end

    subgraph "Database Layer<br/>資料庫層"
        DB_PRIMARY[(PostgreSQL Primary<br/>PostgreSQL 主庫)]
        DB_REPLICA[(PostgreSQL Replica<br/>PostgreSQL 從庫)]
        REDIS[(Redis Cluster<br/>Redis 集群)]
    end

    subgraph "Storage Layer<br/>儲存層"
        S3[Object Storage<br/>S3/Minio<br/>物件儲存]
    end

    subgraph "Monitoring<br/>監控"
        PROM[Prometheus<br/>指標收集]
        GRAFANA[Grafana<br/>視覺化]
        LOGS[ELK Stack<br/>日誌系統]
    end

    LB --> WEB1
    LB --> WEB2
    LB --> WEB3

    WEB1 --> CORE1
    WEB2 --> CORE1
    WEB3 --> CORE2

    CORE1 --> SCAN_PY
    CORE2 --> SCAN_TS
    CORE2 --> SCAN_RS

    SCAN_PY --> MQ_CLUSTER
    SCAN_TS --> MQ_CLUSTER
    SCAN_RS --> MQ_CLUSTER

    MQ_CLUSTER --> FUNC_PY
    MQ_CLUSTER --> FUNC_GO
    MQ_CLUSTER --> FUNC_RS

    FUNC_PY --> MQ_CLUSTER
    FUNC_GO --> MQ_CLUSTER
    FUNC_RS --> MQ_CLUSTER

    MQ_CLUSTER --> INTG
    INTG --> REPORT

    CORE1 --> DB_PRIMARY
    CORE2 --> DB_PRIMARY
    INTG --> DB_PRIMARY

    DB_PRIMARY -.->|Replication<br/>複製| DB_REPLICA

    WEB1 --> REDIS
    WEB2 --> REDIS
    WEB3 --> REDIS

    REPORT --> S3

    WEB1 -.->|Metrics<br/>指標| PROM
    CORE1 -.->|Metrics<br/>指標| PROM
    FUNC_PY -.->|Metrics<br/>指標| PROM

    PROM --> GRAFANA

    WEB1 -.->|Logs<br/>日誌| LOGS
    CORE1 -.->|Logs<br/>日誌| LOGS
    FUNC_PY -.->|Logs<br/>日誌| LOGS

    style LB fill:#90EE90
    style WEB1 fill:#3776AB
    style CORE1 fill:#FFD54F
    style FUNC_GO fill:#00ADD8
    style FUNC_RS fill:#CE422B
    style MQ_CLUSTER fill:#FFA726
    style DB_PRIMARY fill:#42A5F5
    style GRAFANA fill:#FF6B6B
```

---

## 圖表說明 | Diagram Descriptions

### 使用方法 | Usage

1. **查看圖表 | View Diagrams**
   - 使用支援 Mermaid 的編輯器 (VS Code + Mermaid 擴展)
   - 線上預覽: <https://mermaid.live/>

2. **匯出圖表 | Export Diagrams**

   ```bash
   # 使用 mmdc CLI 匯出為 PNG/SVG
   mmdc -i diagram.mmd -o diagram.png
   ```

3. **整合到文檔 | Integrate to Docs**
   - 直接複製 Mermaid 語法到 Markdown
   - 使用 GitBook / Docusaurus 等支援 Mermaid 的文檔平台

### 圖表類型 | Diagram Types

- **graph TB/LR**: 流程圖 (上下/左右)
- **flowchart TD**: 詳細流程圖
- **sequenceDiagram**: 時序圖
- **pie**: 圓餅圖

### 顏色說明 | Color Legend

| 顏色 Color | 用途 Usage | 十六進位 Hex |
|-----------|-----------|------------|
| 🟦 藍色 | Python 模組 | #3776AB |
| 🟦 淺藍 | Go 模組 | #00ADD8 |
| 🟥 紅色 | Rust 模組 | #CE422B |
| 🟦 藍色 | TypeScript 模組 | #3178C6 |
| 🟨 黃色 | 核心服務 | #FFD54F |
| 🟩 綠色 | 掃描服務 | #81C784 |
| 🟪 紫色 | 檢測服務 | #BA68C8 |
| 🟧 橙色 | 整合服務 | #FF8A65 |

---

## 生成腳本 | Generation Script

使用以下命令重新生成圖表:

```bash
python tools/generate_complete_architecture.py
```

---

**文件版本 Document Version**: v1.0
**最後更新 Last Updated**: 2025-10-13
**維護者 Maintainer**: AIVA Development Team
