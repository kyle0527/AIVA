# Attack Path Analyzer

> **✅ 2025-11-16 更新**: 已從 Neo4j 遷移至 NetworkX，零外部依賴，記憶體內高效運算。

攻擊路徑分析引擎，使用 **NetworkX** 圖資料庫建立資產與漏洞的關聯圖，計算攻擊路徑並評估風險。

## 🎯 技術棧變更

| 項目 | 舊版本 (Neo4j) | 新版本 (NetworkX) | 優勢 |
|------|---------------|------------------|------|
| **資料庫** | Neo4j 5.0+ | NetworkX 3.0+ | ✅ 零外部依賴 |
| **查詢語言** | Cypher | Python 原生 | ✅ 更靈活 |
| **持久化** | Neo4j DB | pickle 序列化 | ✅ 更簡單 |
| **適用規模** | 百萬級節點 | 萬級節點 | ✅ AIVA 場景適用 |
| **效能** | 網路通訊 | 記憶體運算 | ✅ 更快速 |

## 功能

- 將 AIVA 發現的資產、漏洞、憑證轉換為圖結構
- 計算從「外部攻擊者」到「核心資產」的攻擊路徑
- 風險評分與路徑排序
- 識別關鍵節點（高中心性節點）
- 視覺化輸出（Mermaid、Cytoscape、HTML）

## 架構

```
attack_path_analyzer/
├── __init__.py
├── engine.py           # 核心引擎 (NetworkX 操作)
├── graph_builder.py    # 圖資料建構器 (從 PostgreSQL 讀取)
├── visualizer.py       # 視覺化工具
└── README.md          # 本文件
```

## 依賴

- **Python 3.11+**
- **NetworkX 3.0+** (圖分析核心)
- **asyncpg** (用於從 PostgreSQL 讀取資料)

> **✅ 已移除依賴**: Neo4j、neo4j-driver

## 使用方式

### 1. 初始化引擎 (使用標準化配置)

```python
from services.integration.aiva_integration.attack_path_analyzer import AttackPathEngine
from services.integration.aiva_integration.config import ATTACK_GRAPH_FILE

# 使用標準化路徑 (自動載入既有圖或建立新圖)
engine = AttackPathEngine(graph_file=ATTACK_GRAPH_FILE)

# 如果是新圖,會自動初始化
# 如果檔案存在,會自動載入
```

### 舊版本 (Neo4j) 遷移說明

```python
# ❌ 舊版本 (已棄用)
engine = AttackPathEngine(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
)

# ✅ 新版本 (推薦)
from services.integration.aiva_integration.config import ATTACK_GRAPH_FILE
engine = AttackPathEngine(graph_file=ATTACK_GRAPH_FILE)
```

### 2. 新增資產與漏洞

```python
from services.aiva_common.schemas import Asset, FindingPayload

# 新增資產
asset = Asset(
    asset_id="asset_123",
    url="https://example.com/api/users",
    type="API_ENDPOINT",
)
engine.add_asset(asset)

# 新增漏洞
finding = FindingPayload(
    finding_id="finding_456",
    task_id="task_789",
    vulnerability=Vulnerability(
        type=VulnerabilityType.SQLI,
        name="SQL Injection in /api/users",
        description="...",
    ),
    severity=Severity.CRITICAL,
)
engine.add_finding(finding)
```

### 3. 尋找攻擊路徑

```python
# 尋找到資料庫的攻擊路徑
paths = engine.find_attack_paths(
    target_node_type="Database",
    max_length=10,
    min_risk_score=5.0,
)

for path in paths:
    print(f"路徑 {path.path_id}:")
    print(f"  風險分數: {path.total_risk_score:.2f}")
    print(f"  路徑長度: {path.length}")
    print(f"  描述: {path.description}")
```

### 4. 從資料庫建立圖

```python
from services.integration.aiva_integration.attack_path_analyzer import GraphBuilder
from services.integration.aiva_integration.config import POSTGRES_DSN

builder = GraphBuilder(
    attack_path_engine=engine,
    postgres_dsn=POSTGRES_DSN,
)

# 建立完整圖
stats = await builder.build_graph_from_database()
print(f"載入 {stats['assets_count']} 個資產, {stats['findings_count']} 個漏洞")

# 增量更新
await builder.incremental_update(since_timestamp="2025-10-13T00:00:00Z")
```

### 5. 視覺化

```python
from services.integration.aiva_integration.attack_path_analyzer import AttackPathVisualizer

# 生成 Mermaid 圖
mermaid_code = AttackPathVisualizer.to_mermaid(paths, title="Critical Attack Paths")
print(mermaid_code)

# 生成互動式 HTML
AttackPathVisualizer.to_html(paths, output_file="attack_paths.html")
# 開啟 attack_paths.html 即可互動式瀏覽
```

## 配置

### 統一配置系統

所有配置統一由 `config.py` 管理，優先級為:
1. 環境變數 (`.env` 檔案)
2. 預設值 (`config.py` 中定義)

### 環境變數配置

在 `.env` 檔案中設定 (已統一定義):

```bash
# ✅ 攻擊路徑圖配置 (NEW - 2025-11-16)
AIVA_ATTACK_GRAPH_FILE=C:/D/fold7/AIVA-git/data/integration/attack_paths/attack_graph.pkl

# ✅ PostgreSQL 配置 (讀取資產/漏洞資料)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=aiva_core
POSTGRES_USER=aiva
POSTGRES_PASSWORD=your_password

# ❌ 已移除配置
# NEO4J_URI - 已遷移至 NetworkX (2025-11-16)
# NEO4J_USER - 已遷移至 NetworkX (2025-11-16)
# NEO4J_PASSWORD - 已遷移至 NetworkX (2025-11-16)
```

### 程式碼中使用配置

```python
from services.integration.aiva_integration.config import (
    ATTACK_GRAPH_FILE,      # 攻擊路徑圖檔案
    POSTGRES_CONFIG,        # PostgreSQL 配置字典
    POSTGRES_DSN,           # PostgreSQL DSN 字串
)

# 初始化引擎
engine = AttackPathEngine(graph_file=ATTACK_GRAPH_FILE)

# 從 PostgreSQL 讀取資料
builder = GraphBuilder(attack_path_engine=engine, postgres_dsn=POSTGRES_DSN)
```

## 圖結構設計

### 節點類型

- **Attacker**: 外部攻擊者（起點）
- **Asset**: 資產（API 端點、網頁等）
- **Vulnerability**: 漏洞
- **Database**: 資料庫（目標）
- **InternalNetwork**: 內部網路
- **Credential**: 憑證
- **APIEndpoint**: API 端點

### 邊類型

- **CAN_ACCESS**: 可訪問（外部攻擊者 → 公開資產）
- **HAS_VULNERABILITY**: 擁有漏洞（資產 → 漏洞）
- **LEADS_TO**: 導致（漏洞 → 內部網路/資料庫）
- **GRANTS_ACCESS**: 授予訪問（漏洞 → API 端點）
- **EXPOSES**: 暴露（XSS → 憑證）

### 漏洞類型與攻擊路徑

| 漏洞類型 | 自動建立的攻擊邊 |
|---------|----------------|
| **SSRF** | Vulnerability → InternalNetwork |
| **SQLi** | Vulnerability → Database |
| **IDOR/BOLA** | Vulnerability → APIEndpoint |
| **XSS** | Vulnerability → Credential |

## 風險評分

```
risk_score = severity_score × confidence_multiplier

severity_score:
  - CRITICAL: 10.0
  - HIGH: 7.5
  - MEDIUM: 5.0
  - LOW: 2.5
  - INFORMATIONAL: 1.0

confidence_multiplier:
  - CERTAIN: 1.0
  - FIRM: 0.8
  - POSSIBLE: 0.5
```

## NetworkX 查詢範例

### 尋找最短攻擊路徑

```python
import networkx as nx

# 使用 NetworkX 內建函式
try:
    shortest_path = nx.shortest_path(
        engine.graph,
        source="external_attacker",
        target="database_node_id"
    )
    print(f"最短路徑: {shortest_path}")
except nx.NetworkXNoPath:
    print("找不到路徑")
```

### 尋找高風險路徑

```python
# 使用自訂權重函式
def edge_weight(u, v, data):
    return data.get('risk', 1.0)

# 尋找所有簡單路徑
paths = nx.all_simple_paths(
    engine.graph,
    source="external_attacker",
    target="database_node_id",
    cutoff=10  # 最大長度
)

# 計算路徑風險並排序
high_risk_paths = []
for path in paths:
    total_risk = sum(
        engine.graph[path[i]][path[i+1]].get('risk', 1.0)
        for i in range(len(path)-1)
    )
    if total_risk >= 20.0:
        high_risk_paths.append((path, total_risk))

# 排序並取前 10
high_risk_paths.sort(key=lambda x: x[1], reverse=True)
for path, risk in high_risk_paths[:10]:
    print(f"路徑: {path}, 總風險: {risk}")
```

### 找出關鍵漏洞節點

```python
# 計算節點中心性
degree_centrality = nx.degree_centrality(engine.graph)

# 篩選漏洞節點
vulnerability_nodes = [
    (node, centrality)
    for node, centrality in degree_centrality.items()
    if engine.graph.nodes[node].get('type') == 'Vulnerability'
    and centrality > 0.1  # 高連接度
]

# 排序並顯示
vulnerability_nodes.sort(key=lambda x: x[1], reverse=True)
for node, centrality in vulnerability_nodes[:10]:
    node_data = engine.graph.nodes[node]
    print(f"{node_data['name']} (嚴重度: {node_data['severity']}, 中心性: {centrality:.3f})")
```

## 整合到 AIVA

### 1. 定期重建圖（每日一次）

```python
import schedule
import asyncio

async def rebuild_graph_job():
    engine = AttackPathEngine(graph_file=ATTACK_GRAPH_FILE)
    builder = GraphBuilder(engine, POSTGRES_DSN)
    await builder.rebuild_graph()
    # NetworkX 會自動保存到 ATTACK_GRAPH_FILE

# 每天 02:00 重建
schedule.every().day.at("02:00").do(lambda: asyncio.create_task(rebuild_graph_job()))
```

### 2. 即時更新（接收到新 Finding 時）

```python
# 在 Integration 模組的 data_reception_layer.py
async def on_new_finding(finding: FindingPayload):
    engine = AttackPathEngine(graph_file=ATTACK_GRAPH_FILE)
    engine.add_finding(finding)
    
    # 檢查是否產生新的高風險路徑
    paths = engine.find_attack_paths(target_node_type="Database", min_risk_score=8.0)
    if paths:
        # 發送告警
        await send_critical_alert(paths)
```

### 3. API 端點（查詢攻擊路徑）

```python
# 在 Integration 的 app.py
from fastapi import APIRouter
from services.integration.aiva_integration.attack_path_analyzer import AttackPathEngine
from services.integration.aiva_integration.config import ATTACK_GRAPH_FILE

router = APIRouter(prefix="/attack-paths", tags=["Attack Paths"])

@router.get("/to-database")
async def get_attack_paths_to_database(
    max_length: int = 10,
    min_risk: float = 5.0,
):
    engine = AttackPathEngine(graph_file=ATTACK_GRAPH_FILE)
    paths = engine.find_attack_paths(
        target_node_type="Database",
        max_length=max_length,
        min_risk_score=min_risk,
    )
    return {"paths": [path.__dict__ for path in paths]}
```

## 效能優化

- ✅ **記憶體內運算**: NetworkX 在記憶體中操作，比 Neo4j 網路通訊快
- ✅ **限制路徑長度**: 使用 `cutoff` 參數避免過深搜尋
- ✅ **增量更新**: 只更新變更的節點和邊
- ✅ **定期備份**: 使用 `backup.py` 自動備份圖檔案

## 測試

```powershell
# 單元測試
pytest services/integration/aiva_integration/attack_path_analyzer/

# 手動測試
python -m services.integration.aiva_integration.attack_path_analyzer.engine
```

## 視覺化範例

生成的 HTML 檔案包含互動式圖表，支援：
- 拖曳節點
- 點擊節點查看詳細資訊
- 縮放與平移
- 自動佈局（breadthfirst, cose, circle 等）

## 🔗 相關文件

### 核心文檔
- 📖 **[整合模組總覽](../../README.md)** - 整合模組主文檔
- 📖 **[資料儲存說明](../../../../data/integration/README.md)** - 完整資料儲存結構
- 📖 **[Integration Core](../README.md)** - 核心模組實現
- 📖 **[Services 總覽](../../../README.md)** - 五大核心服務

### 配置與維護
- 📖 **[config.py 文檔](../config.py)** - 統一配置系統
- 📖 **[維護腳本文檔](../../scripts/README.md)** - 備份與清理工具
- 📖 **[建立報告](../../../../reports/INTEGRATION_DATA_STORAGE_SETUP_REPORT.md)** - 完整建立過程

### 開發指南
- 📖 **[Data Storage Guide](../../../../guides/development/DATA_STORAGE_GUIDE.md)** - 資料儲存總指南
- 📖 **[Reception README](../reception/README.md)** - 經驗資料庫管理

---

**維護**: Integration Team  
**最後更新**: 2025-11-16  
**版本**: v2.0 (NetworkX Migration)  
**下一步**: 整合威脅情資 (ThreatIntel)，動態調整風險分數
