# Attack Path Analyzer

攻擊路徑分析模組 - 使用 Neo4j 圖資料庫建立資產與漏洞的關聯圖，計算從外部攻擊者到核心資產的攻擊路徑。

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
├── engine.py           # 核心引擎（Neo4j 操作）
├── graph_builder.py    # 圖資料建構器（從 PostgreSQL 讀取）
├── visualizer.py       # 視覺化工具
└── README.md
```

## 依賴

- Neo4j 4.4+
- Python 3.11+
- neo4j-driver
- asyncpg (用於從 PostgreSQL 讀取資料)

## 安裝 Neo4j

### Docker 方式

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest
```

### 驗證安裝

訪問 http://localhost:7474，使用 `neo4j/your_password` 登入。

## 使用方式

### 1. 初始化引擎

```python
from services.integration.aiva_integration.attack_path_analyzer import AttackPathEngine

engine = AttackPathEngine(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
)

# 初始化圖結構
engine.initialize_graph()
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
    # ... 其他欄位
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

builder = GraphBuilder(
    attack_path_engine=engine,
    postgres_dsn="postgresql://user:password@localhost:5432/aiva",
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

## Cypher 查詢範例

### 尋找最短攻擊路徑

```cypher
MATCH path = shortestPath(
  (attacker:Attacker {id: 'external_attacker'})
  -[*]->
  (target:Database)
)
RETURN path
```

### 尋找高風險路徑

```cypher
MATCH path = (attacker:Attacker)-[*1..10]->(target:Database)
WITH path, 
     reduce(risk = 0.0, r in relationships(path) | risk + coalesce(r.risk, 1.0)) as total_risk
WHERE total_risk >= 20.0
RETURN path, total_risk
ORDER BY total_risk DESC
LIMIT 10
```

### 找出關鍵漏洞節點

```cypher
MATCH (v:Vulnerability)
WITH v, size((v)--()) as connections
WHERE connections > 3
RETURN v.name, v.severity, connections
ORDER BY connections DESC
LIMIT 10
```

## 整合到 AIVA

### 1. 定期重建圖（每日一次）

```python
import schedule

async def rebuild_graph_job():
    engine = AttackPathEngine()
    builder = GraphBuilder(engine, postgres_dsn)
    await builder.rebuild_graph()
    engine.close()

# 每天 02:00 重建
schedule.every().day.at("02:00").do(lambda: asyncio.create_task(rebuild_graph_job()))
```

### 2. 即時更新（接收到新 Finding 時）

```python
# 在 Integration 模組的 data_reception_layer.py
async def on_new_finding(finding: FindingPayload):
    engine = AttackPathEngine()
    engine.add_finding(finding)
    
    # 檢查是否產生新的高風險路徑
    paths = engine.find_attack_paths(target_node_type="Database", min_risk_score=8.0)
    if paths:
        # 發送告警
        await send_critical_alert(paths)
    
    engine.close()
```

### 3. API 端點（查詢攻擊路徑）

```python
# 在 Integration 的 app.py
from fastapi import APIRouter
from services.integration.aiva_integration.attack_path_analyzer import AttackPathEngine

router = APIRouter(prefix="/attack-paths", tags=["Attack Paths"])

@router.get("/to-database")
async def get_attack_paths_to_database(
    max_length: int = 10,
    min_risk: float = 5.0,
):
    engine = AttackPathEngine()
    try:
        paths = engine.find_attack_paths(
            target_node_type="Database",
            max_length=max_length,
            min_risk_score=min_risk,
        )
        return {"paths": [path.__dict__ for path in paths]}
    finally:
        engine.close()
```

## 效能優化

- 使用 Neo4j 索引加速查詢
- 限制最大路徑長度（避免過深搜尋）
- 增量更新而非全量重建
- 使用連線池管理 Neo4j 連線

## 測試

```bash
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

---

**下一步**: 整合威脅情資 (ThreatIntel)，動態調整風險分數
