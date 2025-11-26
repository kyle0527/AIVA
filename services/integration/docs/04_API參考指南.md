# AIVA 整合模組 - API 參考指南

**版本**: v1.0  
**更新日期**: 2025年11月24日  

---

## 📋 目錄

1. [經驗管理 API](#經驗管理-api)
2. [漏洞管理 API](#漏洞管理-api)
3. [攻擊路徑 API](#攻擊路徑-api)
4. [統計查詢 API](#統計查詢-api)

---

## 經驗管理 API

### `save_attack_experience()`

保存攻擊執行經驗到資料庫。

**簽名**:
```python
def save_attack_experience(
    self,
    plan_id: str,
    attack_type: str,
    ast_graph: Dict[str, Any],
    execution_trace: Dict[str, Any],
    metrics: Dict[str, Any],
    feedback: Dict[str, Any],
    target_info: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> ExperienceRecord
```

**參數**:

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `plan_id` | str | ✓ | 執行計畫 ID |
| `attack_type` | str | ✓ | 攻擊類型 (sqli, xss, idor, etc.) |
| `ast_graph` | Dict | ✓ | AST 圖結構 |
| `execution_trace` | Dict | ✓ | 執行軌跡 |
| `metrics` | Dict | ✓ | 評估指標 |
| `feedback` | Dict | ✓ | 回饋數據（必須包含 "reward" 鍵） |
| `target_info` | Dict | - | 目標資訊 |
| `metadata` | Dict | - | 額外元數據 |

**返回**: `ExperienceRecord` 物件

**範例**:
```python
from services.integration.aiva_integration.unified_data_manager import (
    get_unified_data_manager,
)

manager = get_unified_data_manager()

record = manager.save_attack_experience(
    plan_id="plan_001",
    attack_type="sqli",
    ast_graph={
        "nodes": [
            {"id": "node1", "type": "target"},
            {"id": "node2", "type": "payload"},
        ],
        "edges": [{"from": "node1", "to": "node2"}],
        "strategy": "blind_sqli",
    },
    execution_trace={
        "steps": [
            {"step": 1, "action": "inject_payload", "success": True},
            {"step": 2, "action": "extract_data", "success": True},
        ],
        "success": True,
        "success_rate": 0.95,
        "key_steps": ["inject_payload", "extract_data"],
    },
    metrics={
        "completion_rate": 0.95,
        "overall_score": 0.92,
        "efficiency": 0.88,
    },
    feedback={"reward": 0.9, "penalty": 0.0},
    target_info={"url": "http://example.com", "type": "web_app"},
    metadata={"model_version": "v1.0", "session_id": "sess_001"},
)

print(f"✅ 經驗已保存: {record.experience_id}")
print(f"   分數: {record.overall_score:.2f}")
```

**注意事項**:
- `feedback` 必須包含 `"reward"` 鍵，用於計算 overall_score
- `ast_graph`, `execution_trace`, `metrics` 會以 JSON 格式儲存
- `overall_score` 會自動從 `metrics["overall_score"]` 或 `feedback["reward"]` 計算

---

### `query_high_quality_experiences()`

查詢高質量經驗記錄（按分數過濾）。

**簽名**:
```python
def query_high_quality_experiences(
    self,
    attack_type: str | None = None,
    min_score: float = 0.7,
    limit: int = 100,
) -> List[ExperienceRecord]
```

**參數**:

| 參數 | 類型 | 預設值 | 說明 |
|------|------|-------|------|
| `attack_type` | str \| None | None | 攻擊類型過濾 (None = 所有類型) |
| `min_score` | float | 0.7 | 最低分數閾值 (0.0-1.0) |
| `limit` | int | 100 | 返回數量限制 |

**返回**: `List[ExperienceRecord]` - 按分數降序排序

**範例**:
```python
# 範例 1: 查詢所有高質量經驗
experiences = manager.query_high_quality_experiences(
    min_score=0.8,
    limit=50,
)
print(f"找到 {len(experiences)} 個高質量經驗")

# 範例 2: 查詢特定攻擊類型
sqli_experiences = manager.query_high_quality_experiences(
    attack_type="sqli",
    min_score=0.85,
    limit=100,
)
print(f"找到 {len(sqli_experiences)} 個高質量 SQLi 經驗")

# 範例 3: 分析經驗
for exp in experiences[:5]:
    print(f"經驗ID: {exp.experience_id}")
    print(f"  類型: {exp.attack_type}")
    print(f"  分數: {exp.overall_score:.2f}")
    print(f"  AST: {exp.get_ast_graph()}")
    print(f"  軌跡: {exp.get_execution_trace()}")
```

**ExperienceRecord 屬性**:
- `experience_id`: str - 經驗唯一ID
- `plan_id`: str - 計畫ID
- `attack_type`: str - 攻擊類型
- `overall_score`: float - 綜合分數
- `created_at`: datetime - 創建時間
- `get_ast_graph()`: Dict - 獲取 AST 圖
- `get_execution_trace()`: Dict - 獲取執行軌跡
- `get_metrics()`: Dict - 獲取評估指標

---

### `get_experience_statistics()`

獲取經驗庫統計資訊。

**簽名**:
```python
def get_experience_statistics(self) -> Dict[str, Any]
```

**返回**:
```python
{
    "total_experiences": int,           # 總經驗數
    "average_score": float,             # 平均分數
    "attack_types": [                   # 按攻擊類型統計
        {
            "type": str,                # 攻擊類型
            "count": int,               # 數量
            "avg_score": float,         # 平均分數
        },
        ...
    ]
}
```

**範例**:
```python
stats = manager.get_experience_statistics()

print(f"總經驗數: {stats['total_experiences']}")
print(f"平均分數: {stats['average_score']:.2f}")
print("\n按攻擊類型統計:")
for attack_type_stat in stats['attack_types']:
    print(f"  {attack_type_stat['type']}: "
          f"{attack_type_stat['count']} 個 "
          f"(平均分數: {attack_type_stat['avg_score']:.2f})")
```

---

### `export_training_dataset()`

導出訓練資料集（JSONL 格式）。

**簽名**:
```python
def export_training_dataset(
    self,
    attack_type: str,
    min_score: float = 0.8,
    max_samples: int = 1000,
) -> Path
```

**參數**:

| 參數 | 類型 | 預設值 | 說明 |
|------|------|-------|------|
| `attack_type` | str | - | 攻擊類型 |
| `min_score` | float | 0.8 | 最低分數閾值 |
| `max_samples` | int | 1000 | 最大樣本數 |

**返回**: `Path` - JSONL 文件路徑

**範例**:
```python
# 導出 SQLi 訓練資料集
dataset_path = manager.export_training_dataset(
    attack_type="sqli",
    min_score=0.85,
    max_samples=5000,
)

print(f"✅ 資料集已導出: {dataset_path}")
print(f"   格式: JSONL")

# 讀取資料集
import json
with open(dataset_path, "r") as f:
    for line in f:
        sample = json.loads(line)
        print(f"樣本: {sample['experience_id']}")
        print(f"  分數: {sample['overall_score']}")
```

**JSONL 格式**:
```json
{"experience_id": "exp_001", "attack_type": "sqli", "ast_graph": {...}, "overall_score": 0.92}
{"experience_id": "exp_002", "attack_type": "sqli", "ast_graph": {...}, "overall_score": 0.88}
```

---

## 漏洞管理 API

### `save_finding()`

保存漏洞發現記錄到 PostgreSQL。

**簽名**:
```python
def save_finding(
    self,
    finding: FindingPayload,
    scan_id: str | None = None,
    task_id: str | None = None,
) -> None
```

**參數**:

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `finding` | FindingPayload | ✓ | 漏洞發現物件 (aiva_common.schemas) |
| `scan_id` | str | - | 掃描 ID |
| `task_id` | str | - | 任務 ID |

**範例**:
```python
from services.aiva_common.schemas import FindingPayload
from services.aiva_common.enums import Severity, Confidence

# 創建漏洞發現
finding = FindingPayload(
    finding_id="sqli_001",
    affected_url="http://example.com/api/user?id=1",
    affected_parameter="id",
    severity=Severity.CRITICAL,
    confidence=Confidence.HIGH,
    vulnerability_type="sql_injection",
    description="SQL Injection in user ID parameter",
    evidence={
        "payload": "' OR '1'='1",
        "response": "Welcome, admin!",
        "error_message": "SQL syntax error",
        "detection_method": "error_based",
    },
    cwe="CWE-89",
    cvss_score=9.8,
    remediation="使用參數化查詢或 ORM",
)

# 保存漏洞
manager.save_finding(
    finding=finding,
    scan_id="scan_001",
    task_id="task_001",
)

print("✅ 漏洞已保存到資料庫")
```

**FindingPayload 必填欄位**:
- `finding_id`: str - 漏洞唯一ID
- `affected_url`: str - 受影響的URL
- `severity`: Severity - 嚴重性枚舉
- `confidence`: Confidence - 信心度枚舉
- `vulnerability_type`: str - 漏洞類型
- `description`: str - 描述

---

### `query_findings()`

查詢漏洞發現記錄（支援多種過濾條件）。

**簽名**:
```python
def query_findings(
    self,
    scan_id: str | None = None,
    severity: Severity | None = None,
    confidence: Confidence | None = None,
    vulnerability_type: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]
```

**參數**:

| 參數 | 類型 | 預設值 | 說明 |
|------|------|-------|------|
| `scan_id` | str \| None | None | 掃描 ID 過濾 |
| `severity` | Severity \| None | None | 嚴重性過濾 (CRITICAL/HIGH/MEDIUM/LOW/INFO) |
| `confidence` | Confidence \| None | None | 信心度過濾 |
| `vulnerability_type` | str \| None | None | 漏洞類型過濾 |
| `status` | str \| None | None | 狀態過濾 (active/fixed/false_positive) |
| `limit` | int | 100 | 返回數量限制 (分頁) |
| `offset` | int | 0 | 分頁偏移量 |

**返回**: `List[Dict[str, Any]]` - 漏洞記錄列表

**範例**:
```python
from services.aiva_common.enums import Severity

# 範例 1: 查詢高危漏洞
critical_findings = manager.query_findings(
    severity=Severity.CRITICAL,
    status="active",
    limit=50,
)
print(f"發現 {len(critical_findings)} 個嚴重漏洞")

# 範例 2: 查詢特定掃描的所有漏洞
scan_findings = manager.query_findings(
    scan_id="scan_001",
    limit=1000,
)

# 範例 3: 查詢特定類型漏洞
sqli_findings = manager.query_findings(
    vulnerability_type="sql_injection",
    confidence=Confidence.HIGH,
)

# 範例 4: 分頁查詢
page_size = 20
page1 = manager.query_findings(limit=page_size, offset=0)
page2 = manager.query_findings(limit=page_size, offset=20)

# 範例 5: 分析漏洞
for finding in critical_findings:
    print(f"漏洞ID: {finding['finding_id']}")
    print(f"  類型: {finding['vulnerability_type']}")
    print(f"  嚴重性: {finding['severity']}")
    print(f"  URL: {finding['target_url']}")
    print(f"  證據: {finding['raw_data']['evidence']}")
```

**返回字典格式**:
```python
{
    "id": int,                      # 資料庫主鍵
    "finding_id": str,              # 漏洞ID
    "scan_id": str,                 # 掃描ID
    "vulnerability_type": str,      # 漏洞類型
    "severity": str,                # 嚴重性
    "confidence": str,              # 信心度
    "target_url": str,              # 目標URL
    "status": str,                  # 狀態
    "created_at": datetime,         # 創建時間
    "raw_data": dict,               # 原始資料 (包含 description, evidence 等)
}
```

---

### `update_finding_status()`

更新漏洞狀態。

**簽名**:
```python
def update_finding_status(
    self,
    finding_id: str,
    new_status: str,
) -> bool
```

**參數**:

| 參數 | 類型 | 說明 |
|------|------|------|
| `finding_id` | str | 漏洞 ID |
| `new_status` | str | 新狀態 (active/fixed/false_positive) |

**返回**: `bool` - 是否成功

**範例**:
```python
# 標記漏洞為已修復
success = manager.update_finding_status(
    finding_id="sqli_001",
    new_status="fixed",
)

if success:
    print("✅ 漏洞狀態已更新")
else:
    print("❌ 更新失敗：漏洞不存在")

# 標記為誤報
manager.update_finding_status(
    finding_id="sqli_002",
    new_status="false_positive",
)
```

---

### `get_finding_statistics()`

獲取漏洞統計資訊。

**簽名**:
```python
def get_finding_statistics(
    self,
    scan_id: str | None = None,
) -> Dict[str, Any]
```

**參數**:

| 參數 | 類型 | 預設值 | 說明 |
|------|------|-------|------|
| `scan_id` | str \| None | None | 掃描 ID (None = 所有掃描) |

**返回**:
```python
{
    "total_findings": int,              # 總漏洞數
    "by_severity": {                    # 按嚴重性統計
        "CRITICAL": int,
        "HIGH": int,
        "MEDIUM": int,
        "LOW": int,
        "INFO": int,
    },
    "by_confidence": {                  # 按信心度統計
        "HIGH": int,
        "MEDIUM": int,
        "LOW": int,
    },
    "by_status": {                      # 按狀態統計
        "active": int,
        "fixed": int,
        "false_positive": int,
    },
    "top_vulnerability_types": [        # 最常見漏洞類型
        {"type": str, "count": int},
        ...
    ]
}
```

**範例**:
```python
# 獲取所有掃描的統計
all_stats = manager.get_finding_statistics()

print(f"總漏洞數: {all_stats['total_findings']}")
print(f"嚴重漏洞: {all_stats['by_severity']['CRITICAL']}")
print(f"高危漏洞: {all_stats['by_severity']['HIGH']}")
print(f"活躍漏洞: {all_stats['by_status']['active']}")

print("\n最常見漏洞類型:")
for vuln_type in all_stats['top_vulnerability_types'][:5]:
    print(f"  {vuln_type['type']}: {vuln_type['count']}")

# 獲取特定掃描的統計
scan_stats = manager.get_finding_statistics(scan_id="scan_001")
print(f"\n掃描 scan_001 的統計:")
print(f"  總漏洞: {scan_stats['total_findings']}")
```

---

## 攻擊路徑 API

### `add_asset_to_attack_graph()`

添加資產到攻擊路徑圖。

**簽名**:
```python
def add_asset_to_attack_graph(
    self,
    asset_id: str,
    asset_type: str,
    url: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None
```

**參數**:

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `asset_id` | str | ✓ | 資產 ID (唯一標識) |
| `asset_type` | str | ✓ | 資產類型 (web_application, api, database, server, etc.) |
| `url` | str | - | 資產 URL |
| `metadata` | Dict | - | 額外元數據 |

**範例**:
```python
# 範例 1: 添加 Web 應用
manager.add_asset_to_attack_graph(
    asset_id="web_app_001",
    asset_type="web_application",
    url="https://example.com",
    metadata={
        "framework": "Django",
        "version": "3.2",
        "ip": "192.168.1.100",
        "ports": [80, 443],
    },
)

# 範例 2: 添加 API
manager.add_asset_to_attack_graph(
    asset_id="api_001",
    asset_type="rest_api",
    url="https://api.example.com",
    metadata={
        "authentication": "JWT",
        "version": "v2",
    },
)

# 範例 3: 添加資料庫
manager.add_asset_to_attack_graph(
    asset_id="db_001",
    asset_type="database",
    metadata={
        "db_type": "PostgreSQL",
        "version": "15.0",
        "ip": "192.168.1.101",
        "port": 5432,
    },
)

print("✅ 資產已添加到攻擊路徑圖")
```

---

### `add_vulnerability_to_attack_graph()`

添加漏洞到攻擊路徑圖（自動建立與資產的關聯）。

**簽名**:
```python
def add_vulnerability_to_attack_graph(
    self,
    vuln_id: str,
    vuln_type: str,
    severity: str,
    affected_asset: str,
    metadata: Dict[str, Any] | None = None,
) -> None
```

**參數**:

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `vuln_id` | str | ✓ | 漏洞 ID |
| `vuln_type` | str | ✓ | 漏洞類型 (sql_injection, xss, etc.) |
| `severity` | str | ✓ | 嚴重性 (critical, high, medium, low) |
| `affected_asset` | str | ✓ | 受影響資產 ID |
| `metadata` | Dict | - | 額外元數據 |

**範例**:
```python
# 添加 SQLi 漏洞
manager.add_vulnerability_to_attack_graph(
    vuln_id="sqli_001",
    vuln_type="sql_injection",
    severity="critical",
    affected_asset="web_app_001",  # 自動建立 web_app_001 → sqli_001 的邊
    metadata={
        "cwe": "CWE-89",
        "cvss_score": 9.8,
        "parameter": "id",
    },
)

# 添加 XSS 漏洞
manager.add_vulnerability_to_attack_graph(
    vuln_id="xss_001",
    vuln_type="reflected_xss",
    severity="high",
    affected_asset="web_app_001",
    metadata={
        "cwe": "CWE-79",
        "cvss_score": 7.4,
        "injection_point": "search",
    },
)

print("✅ 漏洞已添加到攻擊路徑圖")
```

---

### `get_attack_paths_for_target()`

獲取目標的所有可能攻擊路徑。

**簽名**:
```python
def get_attack_paths_for_target(
    self,
    target: str,
    max_paths: int = 10,
) -> List[AttackPath]
```

**參數**:

| 參數 | 類型 | 預設值 | 說明 |
|------|------|-------|------|
| `target` | str | - | 目標資產 ID 或 URL |
| `max_paths` | int | 10 | 最大路徑數量 |

**返回**: `List[AttackPath]` - 按風險分數降序排序

**範例**:
```python
# 查詢攻擊路徑
paths = manager.get_attack_paths_for_target(
    target="https://example.com/admin",
    max_paths=5,
)

print(f"發現 {len(paths)} 條攻擊路徑")

for idx, path in enumerate(paths, 1):
    print(f"\n路徑 {idx}:")
    print(f"  描述: {path.description}")
    print(f"  風險分數: {path.total_risk_score:.2f}")
    print(f"  節點: {' → '.join(path.nodes)}")
    print(f"  漏洞: {', '.join(path.vulnerabilities)}")
```

**AttackPath 屬性**:
- `description`: str - 路徑描述
- `total_risk_score`: float - 總風險分數
- `nodes`: List[str] - 路徑節點列表
- `vulnerabilities`: List[str] - 涉及的漏洞列表

---

### `save_attack_graph()`

保存攻擊路徑圖到磁碟（pickle 格式）。

**簽名**:
```python
def save_attack_graph(self) -> None
```

**範例**:
```python
# 添加資產和漏洞後保存
manager.add_asset_to_attack_graph(...)
manager.add_vulnerability_to_attack_graph(...)

# 保存圖
manager.save_attack_graph()
print("✅ 攻擊路徑圖已保存")
```

**注意**: 
- 圖會自動保存到 `data/integration/attack_paths/attack_graph.pkl`
- 建議在批次添加資產/漏洞後統一保存，避免頻繁 I/O

---

### `get_attack_graph_statistics()`

獲取攻擊路徑圖統計資訊。

**簽名**:
```python
def get_attack_graph_statistics(self) -> Dict[str, Any]
```

**返回**:
```python
{
    "total_nodes": int,             # 總節點數
    "total_edges": int,             # 總邊數
    "asset_count": int,             # 資產數量
    "vulnerability_count": int,     # 漏洞數量
    "critical_paths": int,          # 關鍵路徑數 (high risk)
}
```

**範例**:
```python
stats = manager.get_attack_graph_statistics()

print("攻擊路徑圖統計:")
print(f"  總節點: {stats['total_nodes']}")
print(f"  總邊: {stats['total_edges']}")
print(f"  資產數: {stats['asset_count']}")
print(f"  漏洞數: {stats['vulnerability_count']}")
print(f"  關鍵路徑: {stats['critical_paths']}")
```

---

## 統計查詢 API

### `get_unified_statistics()`

獲取統一的綜合統計資訊（整合所有資料源）。

**簽名**:
```python
def get_unified_statistics(self) -> Dict[str, Any]
```

**返回**:
```python
{
    "experiences": {                    # 經驗統計
        "total_experiences": int,
        "average_score": float,
        "attack_types": [...]
    },
    "findings": {                       # 漏洞統計
        "total_findings": int,
        "by_severity": {...},
        "by_confidence": {...},
        "by_status": {...},
        "top_vulnerability_types": [...]
    },
    "attack_paths": {                   # 攻擊路徑統計
        "total_nodes": int,
        "asset_count": int,
        "vulnerability_count": int,
        "critical_paths": int
    },
    "timestamp": str,                   # 統計時間戳
}
```

**範例**:
```python
# 獲取綜合統計
stats = manager.get_unified_statistics()

print("=== AIVA 整合模組統計 ===")
print(f"統計時間: {stats['timestamp']}\n")

print("經驗統計:")
print(f"  總經驗數: {stats['experiences']['total_experiences']}")
print(f"  平均分數: {stats['experiences']['average_score']:.2f}")

print("\n漏洞統計:")
print(f"  總漏洞數: {stats['findings']['total_findings']}")
print(f"  嚴重: {stats['findings']['by_severity']['CRITICAL']}")
print(f"  高危: {stats['findings']['by_severity']['HIGH']}")

print("\n攻擊面統計:")
print(f"  資產數: {stats['attack_paths']['asset_count']}")
print(f"  漏洞數: {stats['attack_paths']['vulnerability_count']}")
print(f"  關鍵路徑: {stats['attack_paths']['critical_paths']}")
```

---

## 下一步

- 🛠️ 查看 [維護運維指南](./05_維護運維指南.md) 學習維護方法
- 📖 查看 [快速開始指南](./01_快速開始指南.md) 回顧基礎用法
- 📚 查看 [模組整合指南](./03_模組整合指南.md) 學習模組整合

---

**維護者**: AIVA Integration Team  
**最後更新**: 2025年11月24日
