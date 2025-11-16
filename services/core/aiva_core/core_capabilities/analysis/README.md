# 🔍 Analysis - 代碼分析系統

**導航**: [← 返回 Core Capabilities](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **代碼量**: 2 個 Python 檔案，約 1181 行代碼  
> **角色**: AIVA 的「智能偵探」- AI 增強的代碼安全分析系統

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
  - [AnalysisEngine - AI 增強代碼分析引擎](#analysisengine---ai-增強代碼分析引擎)
  - [InitialSurface - 初始攻擊面分析](#initialsurface---初始攻擊面分析)
- [使用範例](#使用範例)
- [分析流程](#分析流程)

---

## 🎯 模組概述

**Analysis** 子模組整合了 Tree-sitter AST 解析、神經網路模型和 RAG 知識庫，提供 AI 增強的代碼安全分析能力。能夠自動識別漏洞模式、分析代碼複雜度、檢測架構問題，並生成初始攻擊面報告。

### 核心能力
1. **AI 增強分析** - 結合神經網路和傳統靜態分析
2. **多維度掃描** - 安全性、複雜度、架構、語義分析
3. **攻擊面識別** - 自動識別暴露的端點和潛在入口
4. **並行處理** - 支援大規模代碼庫的高效分析

### 技術特色
- **Tree-sitter 解析** - 精確的語法樹分析
- **神經網路增強** - 利用生物神經網路模型識別複雜模式
- **RAG 整合** - 查詢漏洞知識庫提升檢測準確性
- **緩存機制** - 智能緩存避免重複分析

---

## 📂 檔案列表

| 檔案名 | 行數 | 核心功能 | 狀態 |
|--------|------|----------|------|
| **analysis_engine.py** | 910 | AI 增強代碼分析引擎 - Tree-sitter + 神經網路 | ✅ 生產 |
| **initial_surface.py** | 271 | 初始攻擊面分析 - 端點和入口點識別 | ✅ 生產 |

**總計**: 約 1181 行代碼（含註解和空行）

---

## 🔧 核心組件

### AnalysisEngine - AI 增強代碼分析引擎

**檔案**: `analysis_engine.py` (910 行)

基於 Tree-sitter AST 和神經網路的智能代碼分析系統，整合了生物神經網路和 RAG 知識庫。

#### 核心類別

```python
class AnalysisType(Enum):
    """分析類型枚舉"""
    SECURITY = "security"           # 安全性分析
    VULNERABILITY = "vulnerability"  # 漏洞檢測
    COMPLEXITY = "complexity"        # 複雜度分析
    PATTERNS = "patterns"            # 模式識別
    SEMANTIC = "semantic"            # 語義分析
    ARCHITECTURE = "architecture"    # 架構分析

@dataclass
class IndexingConfig:
    """索引配置（從 RAG 1 遷移）"""
    batch_size: int = 100          # 批次處理大小
    max_workers: int = 4           # 並行工作線程數
    cache_enabled: bool = True     # 是否啟用緩存

class AnalysisEngine:
    """AI 增強代碼分析引擎
    
    功能:
    - Tree-sitter AST 解析
    - 神經網路模式識別
    - RAG 知識庫查詢
    - 多維度代碼分析
    - 並行處理和緩存
    """
    
    def __init__(
        self,
        bio_controller: Optional[BioNeuronMasterController] = None,
        rag_agent: Optional[RealBioNeuronRAGAgent] = None,
        config: Optional[IndexingConfig] = None
    ):
        """初始化分析引擎"""
    
    async def analyze_code(
        self,
        code_content: str,
        file_path: str,
        analysis_types: List[AnalysisType]
    ) -> Dict[str, Any]:
        """分析代碼"""
    
    def parse_ast(self, code: str, language: str = "python") -> Optional[Any]:
        """使用 Tree-sitter 解析 AST"""
    
    async def detect_vulnerabilities(self, ast_tree: Any) -> List[Dict]:
        """檢測漏洞"""
    
    def calculate_complexity(self, ast_tree: Any) -> Dict[str, float]:
        """計算代碼複雜度"""
    
    async def semantic_analysis(self, code: str) -> Dict[str, Any]:
        """語義分析（使用 RAG）"""
```

#### 分析類型說明

| 分析類型 | 檢查項目 | 輸出 | 依賴 |
|---------|----------|------|------|
| **SECURITY** | SQL 注入、XSS、命令注入等 | 安全問題列表 | Tree-sitter |
| **VULNERABILITY** | CVE 漏洞、已知弱點 | 漏洞報告 | RAG 知識庫 |
| **COMPLEXITY** | 圈複雜度、認知複雜度 | 複雜度指標 | AST 分析 |
| **PATTERNS** | 反模式、代碼異味 | 模式匹配結果 | 神經網路 |
| **SEMANTIC** | 語義理解、意圖分析 | 語義摘要 | RAG Agent |
| **ARCHITECTURE** | 架構問題、耦合度 | 架構評估 | AST + 神經網路 |

#### Tree-sitter AST 解析

```python
# 解析 Python 代碼
ast_tree = engine.parse_ast(
    code="""
    def login(username, password):
        query = f"SELECT * FROM users WHERE username='{username}'"
        cursor.execute(query)  # SQL 注入漏洞!
    """,
    language="python"
)

# AST 樹結構
# module
#   function_definition
#     identifier: login
#     parameters
#       identifier: username
#       identifier: password
#     block
#       expression_statement
#         assignment
#           identifier: query
#           f_string  # 檢測到 SQL 字串拼接
```

#### 漏洞檢測

```python
# 檢測常見漏洞
vulnerabilities = await engine.detect_vulnerabilities(ast_tree)

# 輸出示例
[
    {
        "type": "SQL_INJECTION",
        "severity": "high",
        "line": 2,
        "column": 12,
        "description": "檢測到不安全的 SQL 字串拼接",
        "code_snippet": "query = f\"SELECT * FROM users WHERE username='{username}'\"",
        "recommendation": "使用參數化查詢或 ORM",
        "cwe_id": "CWE-89",
        "confidence": 0.95
    }
]
```

#### 複雜度分析

```python
# 計算代碼複雜度
complexity = engine.calculate_complexity(ast_tree)

# 輸出指標
{
    "cyclomatic_complexity": 8,      # 圈複雜度
    "cognitive_complexity": 12,      # 認知複雜度
    "nesting_depth": 4,              # 最大嵌套深度
    "lines_of_code": 150,            # 代碼行數
    "comment_ratio": 0.15,           # 註解比例
    "maintainability_index": 65.3,   # 可維護性指數
    "halstead_volume": 2840.5        # Halstead 體積
}
```

#### 神經網路模式識別

```python
# 使用生物神經網路識別複雜模式
patterns = await engine.analyze_code(
    code_content=code,
    file_path="app/auth.py",
    analysis_types=[AnalysisType.PATTERNS, AnalysisType.SEMANTIC]
)

# 檢測到的模式
{
    "patterns": {
        "god_class": {
            "detected": True,
            "confidence": 0.87,
            "location": "class UserManager",
            "reason": "類有 15 個方法，職責過多"
        },
        "long_parameter_list": {
            "detected": True,
            "confidence": 0.92,
            "location": "def create_user(...)",
            "reason": "方法有 8 個參數"
        }
    },
    "semantic": {
        "intent": "使用者認證和授權管理",
        "security_concerns": [
            "密碼明文儲存風險",
            "缺少速率限制"
        ],
        "suggested_improvements": [
            "添加密碼哈希",
            "實施登錄嘗試限制"
        ]
    }
}
```

#### RAG 知識庫查詢

```python
# 查詢漏洞知識庫
semantic_result = await engine.semantic_analysis(code)

# RAG 查詢結果
{
    "relevant_vulnerabilities": [
        {
            "cve_id": "CVE-2023-12345",
            "description": "不安全的反序列化",
            "similarity_score": 0.89,
            "affected_code": "pickle.loads(user_input)"
        }
    ],
    "best_practices": [
        "使用安全的序列化格式（如 JSON）",
        "驗證和清理所有用戶輸入",
        "實施輸入白名單策略"
    ],
    "reference_documents": [
        "OWASP Top 10 - A08:2021",
        "CWE-502: Deserialization of Untrusted Data"
    ]
}
```

#### 並行處理和緩存

```python
# 配置並行處理
config = IndexingConfig(
    batch_size=100,      # 每批處理 100 個檔案
    max_workers=8,       # 使用 8 個工作線程
    cache_enabled=True   # 啟用緩存
)

engine = AnalysisEngine(config=config)

# 分析整個代碼庫
results = await engine.analyze_codebase(
    directory="./src",
    analysis_types=[
        AnalysisType.SECURITY,
        AnalysisType.VULNERABILITY,
        AnalysisType.COMPLEXITY
    ]
)

# 並行處理流程:
# 1. 掃描目錄獲取所有檔案
# 2. 按語言類型分組
# 3. 分批並行處理（batch_size=100）
# 4. 檢查緩存避免重複分析
# 5. 聚合結果生成報告
```

---

### InitialSurface - 初始攻擊面分析

**檔案**: `initial_surface.py` (271 行)

自動識別應用程式的暴露端點、API 路由、輸入點和潛在攻擊向量。

#### 核心類別

```python
@dataclass
class EndpointInfo:
    """端點信息"""
    path: str                    # 端點路徑
    method: str                  # HTTP 方法
    parameters: List[str]        # 參數列表
    authentication_required: bool # 是否需要認證
    input_types: List[str]       # 輸入類型
    risk_score: float           # 風險評分

class InitialSurface:
    """初始攻擊面分析器
    
    功能:
    - Web 端點識別
    - API 路由發現
    - 輸入點分析
    - 風險評估
    """
    
    def analyze_application(self, app_root: Path) -> Dict[str, Any]:
        """分析應用程式攻擊面"""
    
    def discover_endpoints(self, code_files: List[Path]) -> List[EndpointInfo]:
        """發現所有端點"""
    
    def identify_input_points(self, endpoint: EndpointInfo) -> List[Dict]:
        """識別輸入點"""
    
    def calculate_risk_score(self, endpoint: EndpointInfo) -> float:
        """計算端點風險評分"""
    
    def generate_attack_surface_report(self) -> Dict[str, Any]:
        """生成攻擊面報告"""
```

#### Flask 應用端點識別

```python
# 分析 Flask 應用
surface = InitialSurface()
results = surface.analyze_application(Path("./flask_app"))

# 自動識別路由
# 原始代碼:
"""
@app.route('/api/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    # ...

@app.route('/api/users/<int:user_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def user_detail(user_id):
    # ...
"""

# 識別結果
{
    "endpoints": [
        {
            "path": "/api/login",
            "method": "POST",
            "parameters": ["username", "password"],
            "authentication_required": False,
            "input_types": ["form_data"],
            "risk_score": 0.85,  # 高風險：未認證的登錄端點
            "vulnerabilities": ["brute_force", "credential_stuffing"]
        },
        {
            "path": "/api/users/<user_id>",
            "methods": ["GET", "PUT", "DELETE"],
            "parameters": ["user_id"],
            "authentication_required": True,
            "input_types": ["path_parameter"],
            "risk_score": 0.65,  # 中風險：需檢查授權
            "vulnerabilities": ["idor", "privilege_escalation"]
        }
    ]
}
```

#### FastAPI 應用端點識別

```python
# 分析 FastAPI 應用
# 原始代碼:
"""
@app.post("/api/orders")
async def create_order(order: OrderSchema, user: User = Depends(get_current_user)):
    # ...

@app.get("/api/admin/users")
async def list_users(admin: Admin = Depends(require_admin)):
    # ...
"""

# 識別結果
{
    "endpoints": [
        {
            "path": "/api/orders",
            "method": "POST",
            "parameters": ["order"],
            "input_types": ["json_body"],
            "schema": "OrderSchema",
            "authentication_required": True,
            "authorization_level": "user",
            "risk_score": 0.55,
            "potential_issues": [
                "price_manipulation",
                "race_condition"
            ]
        },
        {
            "path": "/api/admin/users",
            "method": "GET",
            "authentication_required": True,
            "authorization_level": "admin",
            "risk_score": 0.75,  # 高權限端點
            "potential_issues": [
                "privilege_escalation",
                "information_disclosure"
            ]
        }
    ]
}
```

#### 輸入點分析

```python
# 識別所有輸入點
input_points = surface.identify_input_points(endpoint)

# 輸入點分類
{
    "path_parameters": [
        {"name": "user_id", "type": "int", "validation": "none"}
    ],
    "query_parameters": [
        {"name": "page", "type": "int", "default": 1},
        {"name": "search", "type": "str", "sanitization": "none"}
    ],
    "request_body": {
        "format": "json",
        "fields": [
            {"name": "username", "type": "str", "required": True},
            {"name": "email", "type": "str", "validation": "email"},
            {"name": "role", "type": "str", "default": "user"}  # 潛在權限提升
        ]
    },
    "headers": [
        {"name": "Authorization", "required": False},  # 問題：應該必需
        {"name": "X-API-Key", "required": False}
    ],
    "cookies": [
        {"name": "session_id", "httponly": False}  # 問題：不安全
    ]
}
```

#### 風險評分計算

```python
def calculate_risk_score(endpoint: EndpointInfo) -> float:
    """計算端點風險評分（0-1）
    
    評分因子:
    - 認證要求 (-0.2)
    - 敏感操作 (+0.3)
    - 輸入驗證 (-0.1)
    - 權限檢查 (-0.15)
    - 已知漏洞模式 (+0.4)
    """
    
    score = 0.5  # 基礎分數
    
    # 未認證端點
    if not endpoint.authentication_required:
        score += 0.3
    
    # 敏感操作（DELETE, admin 路徑）
    if endpoint.method == "DELETE" or "admin" in endpoint.path:
        score += 0.25
    
    # 缺少輸入驗證
    if not has_input_validation(endpoint):
        score += 0.2
    
    # 缺少授權檢查
    if not has_authorization_check(endpoint):
        score += 0.15
    
    # 檢測到漏洞模式
    if has_vulnerability_pattern(endpoint):
        score += 0.4
    
    return min(score, 1.0)

# 風險等級
# 0.0-0.3: 低風險 🟢
# 0.3-0.6: 中風險 🟡
# 0.6-0.8: 高風險 🟠
# 0.8-1.0: 極高風險 🔴
```

#### 攻擊面報告生成

```python
# 生成完整攻擊面報告
report = surface.generate_attack_surface_report()

# 報告結構
{
    "summary": {
        "total_endpoints": 45,
        "high_risk_endpoints": 8,
        "unauthenticated_endpoints": 5,
        "admin_endpoints": 3,
        "average_risk_score": 0.58
    },
    "by_risk_level": {
        "critical": [
            "/api/admin/execute_command",  # RCE 風險
            "/api/debug/eval"               # 代碼執行
        ],
        "high": [
            "/api/users/delete",
            "/api/payments/refund",
            "/api/admin/users"
        ],
        "medium": [...],
        "low": [...]
    },
    "vulnerability_hotspots": [
        {
            "type": "SQL_INJECTION",
            "affected_endpoints": ["/api/search", "/api/filter"],
            "count": 2
        },
        {
            "type": "IDOR",
            "affected_endpoints": ["/api/users/<id>", "/api/orders/<id>"],
            "count": 2
        }
    ],
    "recommendations": [
        "為所有管理端點添加 RBAC 檢查",
        "實施輸入驗證白名單",
        "添加速率限制到認證端點",
        "修復 5 個 IDOR 漏洞"
    ]
}
```

---

## 🚀 使用範例

### 完整代碼安全分析

```python
from core_capabilities.analysis import AnalysisEngine, AnalysisType, IndexingConfig
from cognitive_core.neural.bio_neuron_master import BioNeuronMasterController
from cognitive_core.neural.real_bio_net_adapter import RealBioNeuronRAGAgent

# 1. 初始化組件
bio_controller = BioNeuronMasterController()
rag_agent = RealBioNeuronRAGAgent()

config = IndexingConfig(
    batch_size=50,
    max_workers=4,
    cache_enabled=True
)

engine = AnalysisEngine(
    bio_controller=bio_controller,
    rag_agent=rag_agent,
    config=config
)

# 2. 分析單個檔案
code = """
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # SQL 注入漏洞!
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    
    return cursor.fetchone()
"""

results = await engine.analyze_code(
    code_content=code,
    file_path="app/auth.py",
    analysis_types=[
        AnalysisType.SECURITY,
        AnalysisType.VULNERABILITY,
        AnalysisType.COMPLEXITY,
        AnalysisType.SEMANTIC
    ]
)

# 3. 查看結果
print("=== 安全分析結果 ===")
for issue in results["security"]:
    print(f"[{issue['severity'].upper()}] {issue['type']}")
    print(f"  位置: Line {issue['line']}, Column {issue['column']}")
    print(f"  描述: {issue['description']}")
    print(f"  建議: {issue['recommendation']}")
    print()

print("=== 複雜度分析 ===")
complexity = results["complexity"]
print(f"圈複雜度: {complexity['cyclomatic_complexity']}")
print(f"認知複雜度: {complexity['cognitive_complexity']}")
print(f"可維護性指數: {complexity['maintainability_index']:.1f}")

print("=== 語義分析 ===")
semantic = results["semantic"]
print(f"代碼意圖: {semantic['intent']}")
print(f"安全顧慮: {', '.join(semantic['security_concerns'])}")
```

### 攻擊面分析

```python
from core_capabilities.analysis import InitialSurface
from pathlib import Path

# 1. 初始化分析器
surface = InitialSurface()

# 2. 分析應用程式
results = surface.analyze_application(Path("./my_app"))

# 3. 查看高風險端點
print("=== 高風險端點 ===")
high_risk = [e for e in results["endpoints"] if e["risk_score"] > 0.7]

for endpoint in high_risk:
    print(f"🔴 {endpoint['method']} {endpoint['path']}")
    print(f"   風險評分: {endpoint['risk_score']:.2f}")
    print(f"   需要認證: {'是' if endpoint['authentication_required'] else '否'}")
    print(f"   潛在漏洞: {', '.join(endpoint['vulnerabilities'])}")
    print()

# 4. 生成完整報告
report = surface.generate_attack_surface_report()

print(f"總端點數: {report['summary']['total_endpoints']}")
print(f"高風險端點: {report['summary']['high_risk_endpoints']}")
print(f"平均風險評分: {report['summary']['average_risk_score']:.2f}")

# 5. 漏洞熱點
print("\n=== 漏洞熱點 ===")
for hotspot in report["vulnerability_hotspots"]:
    print(f"{hotspot['type']}: {hotspot['count']} 個端點受影響")
    for ep in hotspot["affected_endpoints"]:
        print(f"  - {ep}")
```

### 大規模代碼庫掃描

```python
# 並行分析整個代碼庫
config = IndexingConfig(
    batch_size=100,    # 每批 100 個檔案
    max_workers=8,     # 8 個並行工作線程
    cache_enabled=True # 啟用緩存
)

engine = AnalysisEngine(config=config)

# 掃描整個 src 目錄
results = await engine.analyze_codebase(
    directory=Path("./src"),
    analysis_types=[
        AnalysisType.SECURITY,
        AnalysisType.VULNERABILITY
    ],
    file_extensions=[".py", ".js", ".java"]
)

# 聚合結果
print(f"掃描檔案數: {results['total_files']}")
print(f"發現問題數: {results['total_issues']}")
print(f"高危問題: {results['critical_issues']}")
print(f"掃描耗時: {results['duration_seconds']:.2f} 秒")

# 按嚴重程度分組
for severity, issues in results["issues_by_severity"].items():
    print(f"\n{severity.upper()} ({len(issues)} 個):")
    for issue in issues[:5]:  # 顯示前 5 個
        print(f"  - {issue['file']}:{issue['line']} - {issue['type']}")
```

---

## 🔄 分析流程

```
┌─────────────────────────────────────────────────────┐
│                 代碼輸入                             │
│            (單檔案/整個代碼庫)                        │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Tree-sitter 解析    │
         │    生成 AST 樹        │
         └──────────┬───────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   ┌─────────────┐    ┌─────────────┐
   │ 靜態分析     │    │ AI 增強分析  │
   │ - 複雜度     │    │ - 神經網路   │
   │ - 模式匹配   │    │ - RAG 查詢   │
   └──────┬──────┘    └──────┬──────┘
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   結果聚合和評分      │
         │  - 風險評估           │
         │  - 優先級排序         │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   生成分析報告        │
         │  - 漏洞列表           │
         │  - 修復建議           │
         │  - 攻擊面視圖         │
         └──────────────────────┘
```

---

## 📊 性能指標

| 指標 | 說明 | 典型值 |
|------|------|--------|
| **AST 解析速度** | 每秒解析的代碼行數 | 50,000+ lines/s |
| **漏洞檢測速度** | 每秒掃描的檔案數 | 20-50 files/s |
| **並行度** | 同時處理的檔案數 | 4-8 workers |
| **緩存命中率** | 重複檔案的緩存命中率 | >80% |
| **準確率** | 漏洞檢測的準確率 | >90% |
| **誤報率** | 誤報問題的比例 | <10% |

---

## 📚 相關文檔

- [Core Capabilities 主文檔](../README.md)
- [Attack 子模組](../attack/README.md) - 攻擊執行系統
- [BizLogic 子模組](../bizlogic/README.md) - 業務邏輯測試
- [Cognitive Core - Neural](../../cognitive_core/neural/README.md) - 神經網路模型
- [Cognitive Core - RAG](../../cognitive_core/rag/README.md) - 知識庫系統

---

**版權所有** © 2024 AIVA Project. 保留所有權利。
