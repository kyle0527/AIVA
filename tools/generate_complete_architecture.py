#!/usr/bin/env python3
"""
AIVA 完整架構圖生成工具 | Complete Architecture Diagram Generator
=================================================================

功能 Features:
1. 從模組程式碼自動解析架構 | Auto-parse architecture from module code
2. 生成 Mermaid 語法圖表 | Generate Mermaid syntax diagrams
3. 匯出 PNG/SVG/PDF 格式 | Export to PNG/SVG/PDF formats
4. 中英文雙語標籤 | Bilingual labels (Chinese & English)

使用方法 Usage:
    python tools/generate_complete_architecture.py
    python tools/generate_complete_architecture.py --export png
    python tools/generate_complete_architecture.py --format svg --output docs/diagrams/
"""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess


@dataclass
class DiagramConfig:
    """圖表配置 | Diagram Configuration"""

    name: str  # 圖表名稱 | Diagram name
    name_en: str  # 英文名稱 | English name
    diagram_type: str  # 圖表類型 | Diagram type (graph/flowchart/sequence)
    orientation: str = "TB"  # 方向 | Orientation (TB/LR/TD)
    color_scheme: dict[str, str] = field(
        default_factory=dict
    )  # 顏色方案 | Color scheme
    description: str = ""  # 描述 | Description
    description_en: str = ""  # 英文描述 | English description


class ArchitectureDiagramGenerator:
    """架構圖生成器 | Architecture Diagram Generator"""

    def __init__(self, project_root: str = "/workspaces/AIVA"):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / "_out" / "architecture_diagrams"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 顏色方案 | Color scheme
        self.colors = {
            "python": "#3776AB",
            "go": "#00ADD8",
            "rust": "#CE422B",
            "typescript": "#3178C6",
            "core": "#FFD54F",
            "scan": "#81C784",
            "function": "#BA68C8",
            "integration": "#FF8A65",
            "database": "#42A5F5",
            "queue": "#FFA726",
            "safe": "#90EE90",
            "warning": "#FFD54F",
            "danger": "#FF6B6B",
        }

    def generate_all_diagrams(self) -> list[Path]:
        """生成所有圖表 | Generate all diagrams"""
        print(
            "[START] 開始生成 AIVA 完整架構圖... | Starting AIVA complete architecture diagram generation..."
        )

        diagrams = []

        # 1. 整體系統架構 | Overall System Architecture
        diagrams.append(self._generate_overall_architecture())

        # 2. 四大模組概覽 | Four Core Modules Overview
        diagrams.append(self._generate_modules_overview())

        # 3. 核心引擎模組 | Core Engine Module
        diagrams.append(self._generate_core_module())

        # 4. 掃描引擎模組 | Scan Engine Module
        diagrams.append(self._generate_scan_module())

        # 5. 檢測功能模組 | Detection Function Module
        diagrams.append(self._generate_function_module())

        # 6. 整合服務模組 | Integration Service Module
        diagrams.append(self._generate_integration_module())

        # 7-10. 各功能流程圖 | Function Workflow Diagrams
        diagrams.append(self._generate_sqli_flow())
        diagrams.append(self._generate_xss_flow())
        diagrams.append(self._generate_ssrf_flow())
        diagrams.append(self._generate_idor_flow())

        # 11. 完整掃描工作流程 | Complete Scan Workflow
        diagrams.append(self._generate_complete_workflow())

        # 12. 多語言架構決策 | Multi-Language Architecture Decision
        diagrams.append(self._generate_language_decision())

        # 13. 資料流程圖 | Data Flow Diagram
        diagrams.append(self._generate_data_flow())

        # 14. 部署架構圖 | Deployment Architecture
        diagrams.append(self._generate_deployment_architecture())

        print(
            f"[OK] 完成！生成了 {len(diagrams)} 個圖表 | Completed! Generated {len(diagrams)} diagrams"
        )
        return diagrams

    def _generate_overall_architecture(self) -> Path:
        """生成整體系統架構圖 | Generate overall system architecture"""
        print("  [STATS] 生成整體系統架構... | Generating overall system architecture...")

        mermaid_code = """graph TB
    subgraph "[U+1F3A8] 前端層 Frontend Layer"
        UI["[UI] Web UI<br/>網頁介面<br/><i>FastAPI + React</i>"]
        API["[U+1F50C] REST API<br/>REST 接口<br/><i>OpenAPI 3.0</i>"]
    end

    subgraph "[AI] 核心層 Core Layer"
        CORE["[FAST] AI Core Engine<br/>AI 核心引擎<br/><i>Bio Neuron Network</i>"]
        STRATEGY["[LIST] Strategy Generator<br/>策略生成器<br/><i>Dynamic Planning</i>"]
        TASK["[U+1F4E6] Task Manager<br/>任務管理器<br/><i>Multi-threading</i>"]
    end

    subgraph "[SEARCH] 掃描層 Scan Layer"
        SCAN_PY["[U+1F40D] Python Scanner<br/>Python 掃描器<br/><i>Requests + aiohttp</i>"]
        SCAN_TS["[U+1F4D8] TypeScript Scanner<br/>TypeScript 掃描器<br/><i>Playwright</i>"]
        SCAN_RS["[U+1F980] Rust Info Gatherer<br/>Rust 資訊收集<br/><i>High Performance</i>"]
    end

    subgraph "[FAST] 檢測層 Detection Layer"
        FUNC_PY["[U+1F40D] Python Functions<br/>Python 檢測模組<br/><i>SQLi, XSS, IDOR</i>"]
        FUNC_GO["[U+1F537] Go Functions<br/>Go 檢測模組<br/><i>AuthN, CSPM, SCA</i>"]
        FUNC_RS["[U+1F980] Rust Functions<br/>Rust 檢測模組<br/><i>SAST, Deserialization</i>"]
    end

    subgraph "[U+1F517] 整合層 Integration Layer"
        INTG["[CONFIG] Integration Service<br/>整合服務<br/><i>Result Aggregation</i>"]
        REPORT["[STATS] Report Generator<br/>報告生成器<br/><i>HTML/PDF/JSON</i>"]
        ANALYSIS["[TARGET] Risk Analyzer<br/>風險分析器<br/><i>CVSS Scoring</i>"]
    end

    subgraph "[SAVE] 資料層 Data Layer"
        DB[("[U+1F5C4][U+FE0F] PostgreSQL<br/>資料庫<br/><i>Primary Storage</i>")]
        MQ["[U+1F4E8] RabbitMQ<br/>訊息佇列<br/><i>Task Distribution</i>"]
        REDIS[("[FAST] Redis<br/>快取<br/><i>Session & Cache</i>")]
    end

    UI -->|HTTP Request| API
    API -->|Task Creation| CORE
    CORE -->|Strategy| STRATEGY
    CORE -->|Dispatch| TASK

    TASK -->|Scan Job| SCAN_PY
    TASK -->|Dynamic Scan| SCAN_TS
    TASK -->|Info Collection| SCAN_RS

    SCAN_PY -->|Targets| MQ
    SCAN_TS -->|DOM Data| MQ
    SCAN_RS -->|Secrets| MQ

    MQ -->|Detection Job| FUNC_PY
    MQ -->|Detection Job| FUNC_GO
    MQ -->|Detection Job| FUNC_RS

    FUNC_PY -->|Results| MQ
    FUNC_GO -->|Results| MQ
    FUNC_RS -->|Results| MQ

    MQ -->|Aggregation| INTG
    INTG -->|Generate| REPORT
    INTG -->|Assess| ANALYSIS

    CORE -.->|State| DB
    INTG -.->|Store| DB
    API -.->|Cache| REDIS

    style UI fill:#E3F2FD,stroke:#1976D2,stroke-width:3px
    style API fill:#E3F2FD,stroke:#1976D2,stroke-width:3px
    style CORE fill:#FFF9C4,stroke:#F57F17,stroke-width:3px
    style STRATEGY fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
    style TASK fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
    style SCAN_PY fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style SCAN_TS fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style SCAN_RS fill:#FFCCBC,stroke:#D84315,stroke-width:2px
    style FUNC_PY fill:#E1BEE7,stroke:#7B1FA2,stroke-width:2px
    style FUNC_GO fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style FUNC_RS fill:#FFCCBC,stroke:#D84315,stroke-width:2px
    style INTG fill:#FFE0B2,stroke:#E65100,stroke-width:3px
    style REPORT fill:#FFE0B2,stroke:#E65100,stroke-width:2px
    style ANALYSIS fill:#FFE0B2,stroke:#E65100,stroke-width:2px
    style DB fill:#CFD8DC,stroke:#455A64,stroke-width:3px
    style MQ fill:#FFECB3,stroke:#F57F17,stroke-width:3px
    style REDIS fill:#CFD8DC,stroke:#455A64,stroke-width:3px

    linkStyle default stroke:#666,stroke-width:2px
"""

        output_file = self.output_dir / "01_overall_architecture.mmd"
        self._write_diagram(
            output_file, mermaid_code, "整體系統架構 | Overall System Architecture"
        )
        return output_file

    def _generate_modules_overview(self) -> Path:
        """生成四大模組概覽 | Generate four core modules overview"""
        print("  [STATS] 生成四大模組概覽... | Generating four core modules overview...")

        mermaid_code = """graph LR
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
"""

        output_file = self.output_dir / "02_modules_overview.mmd"
        self._write_diagram(
            output_file, mermaid_code, "四大模組概覽 | Four Core Modules Overview"
        )
        return output_file

    def _generate_core_module(self) -> Path:
        """生成核心引擎模組 | Generate core engine module"""
        print("  [STATS] 生成核心引擎模組... | Generating core engine module...")

        mermaid_code = """graph TB
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
"""

        output_file = self.output_dir / "03_core_module.mmd"
        self._write_diagram(
            output_file, mermaid_code, "核心引擎模組 | Core Engine Module"
        )
        return output_file

    def _generate_scan_module(self) -> Path:
        """生成掃描引擎模組 | Generate scan engine module"""
        print("  [STATS] 生成掃描引擎模組... | Generating scan engine module...")

        # 掃描模組的 Mermaid 代碼 (從之前的檔案複製)
        mermaid_code = """graph TB
    subgraph "Scan Module<br/>掃描引擎模組"
        subgraph "Python Scanner<br/>Python 掃描器"
            HTTP[HTTP Client<br/>HTTP 客戶端]
            PARSER[Content Parser<br/>內容解析器]
            BROWSER[Browser Pool<br/>瀏覽器池]
        end

        subgraph "TypeScript Scanner<br/>TypeScript 掃描器"
            TS_SERVICE[Scan Service<br/>掃描服務]
            TS_LOGGER[Logger<br/>日誌]
        end

        subgraph "Rust Info Gatherer<br/>Rust 資訊收集"
            GIT_SCAN[Git History Scanner<br/>Git 歷史掃描]
            SECRET[Secret Detector<br/>秘密檢測器]
        end
    end

    HTTP --> PARSER
    PARSER --> BROWSER
    BROWSER --> TS_SERVICE
    TS_SERVICE --> TS_LOGGER
    TS_LOGGER --> GIT_SCAN
    GIT_SCAN --> SECRET

    style HTTP fill:#81C784
    style TS_SERVICE fill:#4FC3F7
    style GIT_SCAN fill:#E57373
"""

        output_file = self.output_dir / "04_scan_module.mmd"
        self._write_diagram(
            output_file, mermaid_code, "掃描引擎模組 | Scan Engine Module"
        )
        return output_file

    def _generate_function_module(self) -> Path:
        """生成檢測功能模組 | Generate detection function module"""
        print("  [STATS] 生成檢測功能模組... | Generating detection function module...")

        mermaid_code = """graph TB
    subgraph "Function Module<br/>檢測功能模組"
        subgraph "Python Functions<br/>Python 檢測模組"
            SQLI[SQLi Detection<br/>SQL 注入檢測]
            XSS[XSS Detection<br/>XSS 檢測]
            IDOR[IDOR Detection<br/>IDOR 檢測]
        end

        subgraph "Go Functions<br/>Go 檢測模組"
            AUTH[AuthN Detection<br/>身份驗證檢測]
            CSPM[CSPM Scanner<br/>雲端安全掃描]
            SSRF_GO[SSRF Detection<br/>SSRF 檢測]
        end

        subgraph "Rust Functions<br/>Rust 檢測模組"
            SAST[SAST Analyzer<br/>靜態分析]
        end
    end

    SQLI --> AUTH
    XSS --> CSPM
    IDOR --> SSRF_GO
    AUTH --> SAST

    style SQLI fill:#BA68C8
    style XSS fill:#CE93D8
    style AUTH fill:#64B5F6
    style SAST fill:#E57373
"""

        output_file = self.output_dir / "05_function_module.mmd"
        self._write_diagram(
            output_file, mermaid_code, "檢測功能模組 | Detection Function Module"
        )
        return output_file

    def _generate_integration_module(self) -> Path:
        """生成整合服務模組 | Generate integration service module"""
        print("  [STATS] 生成整合服務模組... | Generating integration service module...")

        mermaid_code = """graph TB
    subgraph "Integration Module<br/>整合服務模組"
        subgraph "Analysis<br/>分析服務"
            COMPLY[Compliance Checker<br/>合規性檢查]
            RISK[Risk Assessment<br/>風險評估]
            CORREL[Correlation Analyzer<br/>關聯分析]
        end

        subgraph "Reporting<br/>報告系統"
            CONTENT[Content Generator<br/>內容生成器]
            TEMPLATE[Template Selector<br/>模板選擇器]
            EXPORT[Formatter Exporter<br/>格式化匯出]
        end

        subgraph "Infrastructure<br/>基礎設施"
            GATEWAY[API Gateway<br/>API 閘道]
            DB_INTG[SQL Database<br/>SQL 資料庫]
        end
    end

    COMPLY --> RISK
    RISK --> CORREL
    CORREL --> CONTENT
    CONTENT --> TEMPLATE
    TEMPLATE --> EXPORT
    EXPORT --> GATEWAY
    GATEWAY --> DB_INTG

    style COMPLY fill:#FF8A65
    style CONTENT fill:#FFCCBC
    style GATEWAY fill:#BCAAA4
"""

        output_file = self.output_dir / "06_integration_module.mmd"
        self._write_diagram(
            output_file, mermaid_code, "整合服務模組 | Integration Service Module"
        )
        return output_file

    def _generate_sqli_flow(self) -> Path:
        """生成 SQL 注入檢測流程 | Generate SQLi detection flow"""
        print("  [STATS] 生成 SQL 注入檢測流程... | Generating SQLi detection flow...")

        mermaid_code = """flowchart TD
    START([Start SQLi Detection<br/>開始 SQL 注入檢測])
    RECEIVE[Receive Task<br/>接收任務]
    SELECT{Select Engine<br/>選擇引擎}
    BOOL[Boolean Engine<br/>布爾盲注引擎]
    TIME[Time Engine<br/>時間盲注引擎]
    PAYLOAD[Generate Payload<br/>生成 Payload]
    DETECT{Vulnerability?<br/>發現漏洞?}
    CONFIRM[Confirm<br/>確認漏洞]
    REPORT[Report<br/>報告]
    END([End<br/>結束])

    START --> RECEIVE --> SELECT
    SELECT -->|Boolean| BOOL --> PAYLOAD
    SELECT -->|Time| TIME --> PAYLOAD
    PAYLOAD --> DETECT
    DETECT -->|Yes| CONFIRM --> REPORT
    DETECT -->|No| REPORT
    REPORT --> END

    style START fill:#90EE90
    style BOOL fill:#BA68C8
    style CONFIRM fill:#FFD54F
    style END fill:#FF6B6B
"""

        output_file = self.output_dir / "07_sqli_flow.mmd"
        self._write_diagram(
            output_file, mermaid_code, "SQL 注入檢測流程 | SQLi Detection Flow"
        )
        return output_file

    def _generate_xss_flow(self) -> Path:
        """生成 XSS 檢測流程 | Generate XSS detection flow"""
        print("  [STATS] 生成 XSS 檢測流程... | Generating XSS detection flow...")

        mermaid_code = """flowchart TD
    START([Start XSS Detection<br/>開始 XSS 檢測])
    IDENTIFY{Identify Type<br/>識別類型}
    REFLECT[Reflected XSS<br/>反射型 XSS]
    STORED[Stored XSS<br/>儲存型 XSS]
    DOM[DOM XSS<br/>DOM XSS]
    PAYLOAD[Generate Payloads<br/>生成 Payloads]
    INJECT[Inject Payload<br/>注入 Payload]
    VERIFY{Verify?<br/>驗證執行?}
    REPORT[Report<br/>報告]
    END([End<br/>結束])

    START --> IDENTIFY
    IDENTIFY -->|Reflected| REFLECT --> PAYLOAD
    IDENTIFY -->|Stored| STORED --> PAYLOAD
    IDENTIFY -->|DOM| DOM --> PAYLOAD
    PAYLOAD --> INJECT --> VERIFY
    VERIFY -->|Yes| REPORT
    VERIFY -->|No| REPORT
    REPORT --> END

    style START fill:#90EE90
    style REFLECT fill:#CE93D8
    style STORED fill:#CE93D8
    style DOM fill:#CE93D8
    style END fill:#FF6B6B
"""

        output_file = self.output_dir / "08_xss_flow.mmd"
        self._write_diagram(
            output_file, mermaid_code, "XSS 檢測流程 | XSS Detection Flow"
        )
        return output_file

    def _generate_ssrf_flow(self) -> Path:
        """生成 SSRF 檢測流程 | Generate SSRF detection flow"""
        print("  [STATS] 生成 SSRF 檢測流程... | Generating SSRF detection flow...")

        mermaid_code = """flowchart TD
    START([Start SSRF Detection<br/>開始 SSRF 檢測])
    SEMANTIC[Semantic Analysis<br/>語義分析]
    INTERNAL[Internal Detection<br/>內網檢測]
    OAST[Setup OAST<br/>設置 OAST]
    SEND[Send Request<br/>發送請求]
    CHECK{Check Response<br/>檢查響應}
    VERIFY[Verify SSRF<br/>驗證 SSRF]
    REPORT[Report<br/>報告]
    END([End<br/>結束])

    START --> SEMANTIC --> INTERNAL
    INTERNAL --> OAST --> SEND --> CHECK
    CHECK -->|Success| VERIFY --> REPORT
    CHECK -->|Fail| REPORT
    REPORT --> END

    style START fill:#90EE90
    style OAST fill:#64B5F6
    style VERIFY fill:#FFD54F
    style END fill:#FF6B6B
"""

        output_file = self.output_dir / "09_ssrf_flow.mmd"
        self._write_diagram(
            output_file, mermaid_code, "SSRF 檢測流程 | SSRF Detection Flow"
        )
        return output_file

    def _generate_idor_flow(self) -> Path:
        """生成 IDOR 檢測流程 | Generate IDOR detection flow"""
        print("  [STATS] 生成 IDOR 檢測流程... | Generating IDOR detection flow...")

        mermaid_code = """flowchart TD
    START([Start IDOR Detection<br/>開始 IDOR 檢測])
    EXTRACT[Extract IDs<br/>提取 ID]
    ANALYZE{Analyze Pattern<br/>分析模式}
    BFLA[BFLA Test<br/>功能級授權測試]
    VERTICAL[Vertical Test<br/>垂直提權測試]
    USERS[Create Test Users<br/>創建測試用戶]
    ACCESS[Test Access<br/>測試訪問]
    VERIFY{Access Granted?<br/>訪問成功?}
    REPORT[Report<br/>報告]
    END([End<br/>結束])

    START --> EXTRACT --> ANALYZE
    ANALYZE -->|BFLA| BFLA --> USERS
    ANALYZE -->|Vertical| VERTICAL --> USERS
    USERS --> ACCESS --> VERIFY
    VERIFY -->|Yes| REPORT
    VERIFY -->|No| REPORT
    REPORT --> END

    style START fill:#90EE90
    style BFLA fill:#BA68C8
    style VERTICAL fill:#BA68C8
    style REPORT fill:#FFD54F
    style END fill:#FF6B6B
"""

        output_file = self.output_dir / "10_idor_flow.mmd"
        self._write_diagram(
            output_file, mermaid_code, "IDOR 檢測流程 | IDOR Detection Flow"
        )
        return output_file

    def _generate_complete_workflow(self) -> Path:
        """生成完整掃描工作流程 | Generate complete scan workflow"""
        print("  [STATS] 生成完整掃描工作流程... | Generating complete scan workflow...")

        mermaid_code = """sequenceDiagram
    participant User as [U+1F464] User<br/>使用者
    participant API as [U+1F50C] API<br/>接口
    participant Core as [AI] Core<br/>核心
    participant Scan as [SEARCH] Scanner<br/>掃描器
    participant MQ as [U+1F4E8] MQ<br/>佇列
    participant Func as [FAST] Functions<br/>檢測
    participant Intg as [U+1F517] Integration<br/>整合

    User->>API: Submit Request<br/>提交請求
    API->>Core: Create Task<br/>創建任務
    Core->>Scan: Start Scan<br/>開始掃描
    Scan->>MQ: Publish Targets<br/>發布目標
    MQ->>Func: Execute Detection<br/>執行檢測
    Func->>MQ: Return Results<br/>返回結果
    MQ->>Intg: Aggregate<br/>彙總
    Intg->>API: Report Ready<br/>報告就緒
    API->>User: Display Report<br/>顯示報告
"""

        output_file = self.output_dir / "11_complete_workflow.mmd"
        self._write_diagram(
            output_file, mermaid_code, "完整掃描工作流程 | Complete Scan Workflow"
        )
        return output_file

    def _generate_language_decision(self) -> Path:
        """生成多語言架構決策 | Generate multi-language architecture decision"""
        print(
            "  [STATS] 生成多語言架構決策... | Generating multi-language architecture decision..."
        )

        mermaid_code = """flowchart TD
    START([New Feature<br/>新功能])
    PERF{High Performance?<br/>高效能?}
    CONCUR{High Concurrency?<br/>高併發?}
    MEMORY{Memory Safety?<br/>記憶體安全?}
    GO[Choose Go<br/>選擇 Go]
    RUST[Choose Rust<br/>選擇 Rust]
    PYTHON[Choose Python<br/>選擇 Python]
    TS[Choose TypeScript<br/>選擇 TypeScript]
    END([Implement<br/>實現])

    START --> PERF
    PERF -->|Yes| CONCUR
    PERF -->|No| PYTHON
    CONCUR -->|Yes| MEMORY
    CONCUR -->|No| GO
    MEMORY -->|Yes| RUST
    MEMORY -->|No| GO
    PYTHON --> END
    GO --> END
    RUST --> END
    TS --> END

    style PYTHON fill:#3776AB
    style GO fill:#00ADD8
    style RUST fill:#CE422B
    style TS fill:#3178C6
"""

        output_file = self.output_dir / "12_language_decision.mmd"
        self._write_diagram(
            output_file,
            mermaid_code,
            "多語言架構決策 | Multi-Language Architecture Decision",
        )
        return output_file

    def _generate_data_flow(self) -> Path:
        """生成資料流程圖 | Generate data flow diagram"""
        print("  [STATS] 生成資料流程圖... | Generating data flow diagram...")

        mermaid_code = """graph TB
    INPUT[User Input<br/>用戶輸入]
    AUTH[Authentication<br/>身份驗證]
    PROCESS[Processing<br/>處理]
    QUEUE[Task Queue<br/>任務佇列]
    EXEC[Execution<br/>執行]
    ANALYSIS[Analysis<br/>分析]
    STORAGE[Storage<br/>儲存]
    OUTPUT[Output<br/>輸出]

    INPUT --> AUTH --> PROCESS
    PROCESS --> QUEUE --> EXEC
    EXEC --> ANALYSIS --> STORAGE
    STORAGE --> OUTPUT

    style INPUT fill:#E1F5FE
    style AUTH fill:#FFF9C4
    style EXEC fill:#C5E1A5
    style STORAGE fill:#FFAB91
"""

        output_file = self.output_dir / "13_data_flow.mmd"
        self._write_diagram(output_file, mermaid_code, "資料流程圖 | Data Flow Diagram")
        return output_file

    def _generate_deployment_architecture(self) -> Path:
        """生成部署架構圖 | Generate deployment architecture"""
        print("  [STATS] 生成部署架構圖... | Generating deployment architecture...")

        mermaid_code = """graph TB
    LB[Load Balancer<br/>負載均衡器]

    subgraph "Web Layer<br/>Web 層"
        WEB1[FastAPI 1]
        WEB2[FastAPI 2]
    end

    subgraph "Services<br/>服務層"
        CORE[Core Service<br/>核心服務]
        SCAN[Scan Service<br/>掃描服務]
        FUNC[Function Service<br/>檢測服務]
    end

    subgraph "Data Layer<br/>資料層"
        DB[(PostgreSQL<br/>資料庫)]
        MQ[RabbitMQ<br/>佇列]
        REDIS[(Redis<br/>快取)]
    end

    LB --> WEB1
    LB --> WEB2
    WEB1 --> CORE
    WEB2 --> CORE
    CORE --> SCAN
    SCAN --> FUNC
    FUNC --> MQ
    CORE --> DB
    WEB1 --> REDIS

    style LB fill:#90EE90
    style WEB1 fill:#3776AB
    style CORE fill:#FFD54F
    style DB fill:#42A5F5
"""

        output_file = self.output_dir / "14_deployment_architecture.mmd"
        self._write_diagram(
            output_file, mermaid_code, "部署架構圖 | Deployment Architecture"
        )
        return output_file

    def _write_diagram(self, output_file: Path, mermaid_code: str, title: str):
        """寫入圖表檔案 | Write diagram file"""
        content = f"""# {title}

```mermaid
{mermaid_code.strip()}
```

---
**Generated by**: AIVA Architecture Diagram Generator
**Timestamp**: {self._get_timestamp()}
"""
        output_file.write_text(content, encoding="utf-8")
        print(f"    [OK] 已生成: {output_file.name}")

    def _get_timestamp(self) -> str:
        """取得時間戳記 | Get timestamp"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def export_diagrams(self, format: str = "png", diagrams: list[Path] | None = None):
        """匯出圖表為圖片格式 | Export diagrams to image format"""
        if not diagrams:
            diagrams = list(self.output_dir.glob("*.mmd"))

        # 檢查是否安裝 mmdc
        if not self._check_mmdc_installed():
            print(
                "[WARN]  未安裝 mermaid-cli，無法匯出圖片 | mermaid-cli not installed, cannot export images"
            )
            print("   安裝方法 Install: npm install -g @mermaid-js/mermaid-cli")
            return

        print(
            f"[U+1F4E4] 開始匯出 {format.upper()} 格式... | Starting {format.upper()} export..."
        )

        for diagram in diagrams:
            output_file = diagram.with_suffix(f".{format}")
            try:
                subprocess.run(
                    [
                        "mmdc",
                        "-i",
                        str(diagram),
                        "-o",
                        str(output_file),
                        "-t",
                        "default",
                        "-b",
                        "transparent",
                    ],
                    check=True,
                    capture_output=True,
                )
                print(f"  [OK] 已匯出: {output_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] 匯出失敗 Failed: {diagram.name} - {e}")

    def _check_mmdc_installed(self) -> bool:
        """檢查是否安裝 mermaid-cli | Check if mermaid-cli is installed"""
        try:
            subprocess.run(["mmdc", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def create_index(self, diagrams: list[Path]):
        """創建索引檔案 | Create index file"""
        print("[LIST] 生成索引檔案... | Generating index file...")

        index_content = f"""# AIVA 架構圖索引 | Architecture Diagrams Index

**生成時間 Generated**: {self._get_timestamp()}
**總計圖表 Total Diagrams**: {len(diagrams)}

---

## 圖表列表 | Diagram List

"""

        for i, diagram in enumerate(sorted(diagrams), 1):
            diagram_name = diagram.stem.replace("_", " ").title()
            index_content += f"{i}. [{diagram_name}]({diagram.name})\n"

        index_content += """
---

## 使用說明 | Usage Guide

### 1. 查看圖表 | View Diagrams

在支援 Mermaid 的環境中查看:
- VS Code + Mermaid 擴展
- GitHub / GitLab
- https://mermaid.live/

### 2. 匯出圖片 | Export Images

```bash
# 匯出 PNG 格式
python tools/generate_complete_architecture.py --export png

# 匯出 SVG 格式
python tools/generate_complete_architecture.py --export svg

# 匯出 PDF 格式
python tools/generate_complete_architecture.py --export pdf
```

### 3. 更新圖表 | Update Diagrams

```bash
# 重新生成所有圖表
python tools/generate_complete_architecture.py

# 生成並匯出
python tools/generate_complete_architecture.py --export png
```

---

**維護者 Maintainer**: AIVA Development Team
"""

        index_file = self.output_dir / "INDEX.md"
        index_file.write_text(index_content, encoding="utf-8")
        print(f"  [OK] 已生成索引: {index_file}")


def main():
    """主程式 | Main program"""
    import argparse

    parser = argparse.ArgumentParser(
        description="AIVA 完整架構圖生成工具 | Complete Architecture Diagram Generator"
    )
    parser.add_argument(
        "--export",
        choices=["png", "svg", "pdf"],
        help="匯出圖表為指定格式 | Export diagrams to specified format",
    )
    parser.add_argument(
        "--output", default="/workspaces/AIVA", help="輸出目錄 | Output directory"
    )

    args = parser.parse_args()

    # 建立生成器 | Create generator
    generator = ArchitectureDiagramGenerator(project_root=args.output)

    # 生成所有圖表 | Generate all diagrams
    diagrams = generator.generate_all_diagrams()

    # 創建索引 | Create index
    generator.create_index(diagrams)

    # 匯出圖片 (如果指定) | Export images (if specified)
    if args.export:
        generator.export_diagrams(format=args.export, diagrams=diagrams)

    print("\n" + "=" * 60)
    print("[SPARKLE] 完成！所有架構圖已生成 | Completed! All diagrams generated")
    print(f"[U+1F4C1] 輸出位置 Output: {generator.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
