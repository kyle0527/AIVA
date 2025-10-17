#!/usr/bin/env python
"""
generate_mermaid_diagrams.py
-----------------------------
為 AIVA 專案生成各種 Mermaid 架構圖
包含多語言架構、模組關係、技術棧選擇等圖表
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def generate_multilang_architecture() -> str:
    """生成多語言架構概覽圖 (Mermaid 11.11.0+)."""
    return """```mermaid
%%{init: {'theme':'default', 'themeVariables': {'fontSize':'14px'}}}%%
flowchart TB
    subgraph "🐍 Python Layer"
        PY_API["FastAPI Web API"]
        PY_CORE["核心引擎"]
        PY_SCAN["掃描服務"]
        PY_INTG["整合層"]
    end

    subgraph "🔷 Go Layer"
        GO_AUTH["身份驗證檢測"]
        GO_CSPM["雲端安全"]
        GO_SCA["軟體組成分析"]
        GO_SSRF["SSRF 檢測"]
    end

    subgraph "🦀 Rust Layer"
        RS_SAST["靜態分析引擎"]
        RS_INFO["資訊收集器"]
    end

    subgraph "📘 TypeScript Layer"
        TS_SCAN["Playwright 掃描"]
    end

    subgraph "🗄️ Data Layer"
        DB[("PostgreSQL")]
        MQ["RabbitMQ"]
    end

    PY_API --> PY_CORE
    PY_CORE --> PY_SCAN
    PY_SCAN --> PY_INTG

    PY_INTG -->|"RPC"| GO_AUTH
    PY_INTG -->|"RPC"| GO_CSPM
    PY_INTG -->|"RPC"| GO_SCA
    PY_INTG -->|"RPC"| GO_SSRF
    PY_INTG -->|"RPC"| RS_SAST
    PY_INTG -->|"RPC"| RS_INFO
    PY_INTG -->|"RPC"| TS_SCAN

    GO_AUTH --> MQ
    GO_CSPM --> MQ
    GO_SCA --> MQ
    GO_SSRF --> MQ
    RS_SAST --> MQ
    RS_INFO --> MQ
    TS_SCAN --> MQ

    MQ --> DB
    PY_CORE --> DB

    style PY_API fill:#3776ab,stroke:#2C5F8D,stroke-width:2px,color:#fff
    style GO_AUTH fill:#00ADD8,stroke:#0099BF,stroke-width:2px,color:#fff
    style RS_SAST fill:#CE422B,stroke:#A33520,stroke-width:2px,color:#fff
    style TS_SCAN fill:#3178C6,stroke:#2768B3,stroke-width:2px,color:#fff
```"""


def generate_code_distribution_pie(stats: dict[str, Any]) -> str:
    """生成程式碼分布圓餅圖."""
    multilang = stats.get("multilang", {})

    python_lines = stats.get("total_lines", 0)
    go_lines = multilang.get("go", {}).get("total_lines", 0)
    rust_lines = multilang.get("rust", {}).get("total_lines", 0)
    ts_lines = multilang.get("typescript", {}).get("total_lines", 0)
    js_lines = multilang.get("javascript", {}).get("total_lines", 0)

    total = python_lines + go_lines + rust_lines + ts_lines + js_lines

    if total == 0:
        return '```mermaid\npie title 程式碼行數分布\n    "無資料" : 1\n```'

    python_pct = (python_lines / total) * 100
    go_pct = (go_lines / total) * 100
    rust_pct = (rust_lines / total) * 100
    ts_js_pct = ((ts_lines + js_lines) / total) * 100

    return f"""```mermaid
pie title 程式碼行數分布
    "Python ({python_pct:.1f}%)" : {python_lines}
    "Go ({go_pct:.1f}%)" : {go_lines}
    "Rust ({rust_pct:.1f}%)" : {rust_lines}
    "TypeScript/JS ({ts_js_pct:.1f}%)" : {ts_lines + js_lines}
```"""


def generate_module_relationships() -> str:
    """生成模組關係圖 (Mermaid 11.11.0+)."""
    return """```mermaid
%%{init: {'theme':'default', 'flowchart': {'curve':'basis'}}}%%
flowchart LR
    subgraph "services"
        aiva_common["aiva_common<br/>共用模組"]
        core["core<br/>核心引擎"]
        function["function<br/>功能模組"]
        integration["integration<br/>整合層"]
        scan["scan<br/>掃描引擎"]
    end

    subgraph "function 子模組"
        func_py["Python 模組"]
        func_go["Go 模組<br/>authn/cspm/sca/ssrf"]
        func_rs["Rust 模組<br/>sast/info_gatherer"]
    end

    subgraph "scan 子模組"
        scan_py["Python 掃描"]
        scan_ts["Node.js 掃描<br/>Playwright"]
    end

    core --> aiva_common
    scan --> aiva_common
    function --> aiva_common
    integration --> aiva_common

    integration --> function
    integration --> scan

    function --> func_py
    function --> func_go
    function --> func_rs

    scan --> scan_py
    scan --> scan_ts

    style aiva_common fill:#90EE90
    style core fill:#FFD700
    style function fill:#87CEEB
    style integration fill:#FFA07A
    style scan fill:#DDA0DD
```"""


def generate_tech_stack_flowchart() -> str:
    """生成技術棧選擇流程圖 (Mermaid 11.11.0+)."""
    return """```mermaid
%%{init: {'theme':'default', 'flowchart': {'htmlLabels': false}}}%%
flowchart TD
    Start(["新功能需求"]) --> Perf{"需要高效能?"}
    Perf -->|"是"| Memory{"需要記憶體安全?"}
    Perf -->|"否"| Web{"是 Web API?"}

    Memory -->|"是"| Rust["使用 Rust<br/>靜態分析/資訊收集"]
    Memory -->|"否"| Go["使用 Go<br/>認證/雲端安全/SCA"]

    Web -->|"是"| Python["使用 Python<br/>FastAPI/核心邏輯"]
    Web -->|"否"| Browser{"需要瀏覽器?"}

    Browser -->|"是"| TS["使用 TypeScript<br/>Playwright 掃描"]
    Browser -->|"否"| Python

    Rust --> MQ["Message Queue"]
    Go --> MQ
    Python --> MQ
    TS --> MQ

    MQ --> Deploy(["部署模組"])

    style Rust fill:#CE422B,stroke:#A33520,stroke-width:2px,color:#fff
    style Go fill:#00ADD8,stroke:#0099BF,stroke-width:2px,color:#fff
    style Python fill:#3776ab,stroke:#2C5F8D,stroke-width:2px,color:#fff
    style TS fill:#3178C6,stroke:#2768B3,stroke-width:2px,color:#fff
```"""


def generate_scanning_workflow() -> str:
    """生成掃描工作流程圖."""
    return """```mermaid
sequenceDiagram
    participant User as 使用者
    participant API as FastAPI
    participant Core as 核心引擎
    participant Intg as 整合層
    participant Go as Go 模組
    participant Rust as Rust 模組
    participant TS as TS 模組
    participant MQ as RabbitMQ
    participant DB as PostgreSQL

    User->>API: 提交掃描請求
    API->>Core: 處理請求
    Core->>DB: 建立掃描任務
    Core->>Intg: 分發任務

    par 平行處理
        Intg->>Go: 認證檢測
        Intg->>Go: 雲端安全檢查
        Intg->>Rust: 靜態分析
        Intg->>TS: 動態掃描
    end

    Go-->>MQ: 發送結果
    Rust-->>MQ: 發送結果
    TS-->>MQ: 發送結果

    MQ->>Core: 彙總結果
    Core->>DB: 儲存報告
    Core->>API: 返回結果
    API->>User: 顯示報告
```"""


def generate_data_flow() -> str:
    """生成資料流程圖 (Mermaid 11.11.0+)."""
    return """```mermaid
%%{init: {'theme':'default'}}}%%
flowchart TD
    A["使用者輸入"] --> B{"驗證參數"}
    B -->|"有效"| C["建立掃描任務"]
    B -->|"無效"| Z["返回錯誤"]

    C --> D["Task Queue"]
    D --> E{"選擇掃描引擎"}

    E -->|"靜態分析"| F["Rust SAST"]
    E -->|"動態掃描"| G["TS Playwright"]
    E -->|"身份驗證"| H["Go Auth"]
    E -->|"雲端安全"| I["Go CSPM"]

    F --> J["RabbitMQ"]
    G --> J
    H --> J
    I --> J

    J --> K["結果處理器"]
    K --> L["儲存至資料庫"]
    L --> M["生成報告"]
    M --> N["返回使用者"]

    style F fill:#CE422B,stroke:#A33520,stroke-width:2px,color:#fff
    style G fill:#3178C6,stroke:#2768B3,stroke-width:2px,color:#fff
    style H fill:#00ADD8,stroke:#0099BF,stroke-width:2px,color:#fff
    style I fill:#00ADD8,stroke:#0099BF,stroke-width:2px,color:#fff
```"""


def generate_deployment_diagram() -> str:
    """生成部署架構圖 (Mermaid 11.11.0+)."""
    return """```mermaid
%%{init: {'theme':'default', 'flowchart': {'rankSpacing': 50}}}%%
flowchart TB
    subgraph "Docker 容器"
        subgraph "Web 層"
            WEB["FastAPI<br/>Python 3.12"]
        end

        subgraph "應用層"
            PY["Python Services"]
            GO["Go Services"]
            RS["Rust Services"]
            TS["Node.js Services"]
        end

        subgraph "訊息層"
            MQ["RabbitMQ"]
        end

        subgraph "資料層"
            DB[("PostgreSQL")]
            CACHE[("Redis")]
        end
    end

    WEB --> PY
    PY --> GO
    PY --> RS
    PY --> TS

    GO --> MQ
    RS --> MQ
    TS --> MQ

    MQ --> PY
    PY --> DB
    PY --> CACHE

    style WEB fill:#3776ab,stroke:#2C5F8D,stroke-width:2px,color:#fff
    style GO fill:#00ADD8,stroke:#0099BF,stroke-width:2px,color:#fff
    style RS fill:#CE422B,stroke:#A33520,stroke-width:2px,color:#fff
    style TS fill:#3178C6,stroke:#2768B3,stroke-width:2px,color:#fff
```"""


def main():
    """主函數."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "_out"

    print("AIVA Mermaid 圖表生成器")
    print("=" * 80)
    print(f"專案根目錄: {project_root}")
    print(f"輸出目錄: {output_dir}")
    print("=" * 80)
    print()

    # 讀取分析報告以取得統計數據
    analysis_dir = output_dir / "analysis"
    latest_report = None

    if analysis_dir.exists():
        json_files = sorted(analysis_dir.glob("analysis_report_*.json"))
        if json_files:
            latest_report = json_files[-1]
            print(f"[STATS] 讀取分析報告: {latest_report.name}")

    stats = {}
    if latest_report:
        try:
            with open(latest_report, encoding="utf-8") as f:
                stats = json.load(f)
        except Exception as e:
            print(f"[WARN]  無法讀取報告: {e}")

    # 生成各種圖表
    output_dir.mkdir(parents=True, exist_ok=True)

    mermaid_content = f"""# AIVA 專案架構圖集

生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 多語言架構概覽

{generate_multilang_architecture()}

## 2. 程式碼分布統計

{generate_code_distribution_pie(stats)}

## 3. 模組關係圖

{generate_module_relationships()}

## 4. 技術棧選擇流程

{generate_tech_stack_flowchart()}

## 5. 掃描工作流程

{generate_scanning_workflow()}

## 6. 資料流程圖

{generate_data_flow()}

## 7. 部署架構圖

{generate_deployment_diagram()}

---

### 圖表說明

- **多語言架構概覽**: 展示各層級間的關係和資料流向
- **程式碼分布統計**: 各語言的程式碼行數佔比
- **模組關係圖**: 服務模組間的依賴關係
- **技術棧選擇流程**: 選擇程式語言的決策流程
- **掃描工作流程**: 漏洞掃描的完整流程
- **資料流程圖**: 資料在系統中的流動
- **部署架構圖**: Docker 容器部署架構

### 如何使用

1. 複製 Mermaid 程式碼到 Markdown 檔案
2. 使用支援 Mermaid 的編輯器預覽 (如 VS Code + Mermaid 外掛)
3. 或使用線上工具: https://mermaid.live/

### 更新圖表

執行以下命令重新生成圖表:

```bash
python tools/generate_mermaid_diagrams.py
```

---

*此檔案由 AIVA 自動生成工具建立*
"""

    # 儲存 Mermaid 圖表
    mermaid_file = output_dir / "ARCHITECTURE_DIAGRAMS.md"
    with open(mermaid_file, "w", encoding="utf-8") as f:
        f.write(mermaid_content)

    print("\n[OK] 圖表已生成:")
    print(f"  [STATS] {mermaid_file}")
    print("\n生成的圖表:")
    print("  1. 多語言架構概覽")
    print("  2. 程式碼分布統計")
    print("  3. 模組關係圖")
    print("  4. 技術棧選擇流程")
    print("  5. 掃描工作流程")
    print("  6. 資料流程圖")
    print("  7. 部署架構圖")
    print("\n" + "=" * 80)
    print("完成！請使用支援 Mermaid 的編輯器查看圖表。")
    print("=" * 80)


if __name__ == "__main__":
    main()
