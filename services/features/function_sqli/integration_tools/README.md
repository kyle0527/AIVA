# 🔗 SQL 注入工具整合層

## 📑 目錄
- [概述](#概述)
- [模組架構](#模組架構)
- [核心組件](#核心組件)
  - [sql_tools.py - SQL 工具管理器](#sql_toolspy---sql-工具管理器)
  - [bounty_hunter.py - Bug Bounty 獵人模式](#bounty_hunterpy---bug-bounty-獵人模式)
- [技術實現](#技術實現)
  - [Sqlmap 整合](#sqlmap-整合)
  - [NoSQLMap 整合](#nosqlmap-整合)
  - [結果標準化](#結果標準化)
- [使用指南](#使用指南)
  - [基礎使用](#基礎使用)
  - [高級配置](#高級配置)
  - [Bug Bounty 模式](#bug-bounty-模式)
- [API 參考](#api-參考)
- [開發指南](#開發指南)

---

## 概述

`integration_tools` 目錄提供與外部 SQL 注入工具的整合接口，將 sqlmap、NoSQLMap 等工具的功能封裝為統一的 Python API，並提供專門為 Bug Bounty 優化的獵人模式。

### 設計目標
- **統一接口**：不同工具使用相同的 API 調用方式
- **結果標準化**：將工具輸出轉換為 AIVA 標準格式
- **自動化管理**：工具安裝、配置、執行全自動化
- **Bug Bounty 優化**：針對賞金獵人場景優化檢測策略

---

## 模組架構

```
integration_tools/
├── __init__.py              # 模組導出
├── sql_tools.py             # SQL 工具整合管理器
└── bounty_hunter.py         # Bug Bounty 獵人模式
```

### 整合流程

```
使用者調用
    ↓
SQLInjectionManager (sql_tools.py)
    ├─→ SqlmapIntegration
    │   ├─ 檢查安裝狀態
    │   ├─ 執行 sqlmap 命令
    │   ├─ 解析 JSON 輸出
    │   └─ 轉換為 SQLInjectionResult
    │
    └─→ BountyHunter (bounty_hunter.py)
        ├─ 高價值目標優先
        ├─ 智能參數選擇
        ├─ 快速初篩 + 深度驗證
        └─ 生成賞金報告
```

---

## 核心組件

### sql_tools.py - SQL 工具管理器

#### 主要類別

##### 1. SQLTarget - 目標定義
```python
@dataclass
class SQLTarget:
    """SQL 注入測試目標"""
    url: str                              # 目標 URL
    method: str = "GET"                   # HTTP 方法
    parameters: Dict[str, str] = None     # 參數字典
    headers: Dict[str, str] = None        # 自定義 HTTP 頭
    cookies: Dict[str, str] = None        # Cookie
    data: Optional[str] = None            # POST 數據
    custom_injection: Optional[str] = None  # 自定義注入點標記
```

**使用示例**：
```python
# GET 請求目標
target = SQLTarget(
    url="http://example.com/search",
    parameters={"q": "test", "page": "1"}
)

# POST 請求目標
target = SQLTarget(
    url="http://example.com/login",
    method="POST",
    data="username=admin&password=test"
)

# 自定義注入點
target = SQLTarget(
    url="http://example.com/api",
    method="POST",
    data='{"user": "*", "pass": "test"}',  # * 標記注入點
    custom_injection="*"
)
```

##### 2. SQLInjectionResult - 檢測結果
```python
@dataclass
class SQLInjectionResult:
    """SQL 注入檢測結果"""
    vulnerable: bool                      # 是否存在漏洞
    injection_type: str                   # 注入類型
    parameter: str                        # 易受攻擊的參數
    payload: str                          # 成功的 payload
    dbms: Optional[str] = None           # 數據庫類型
    dbms_version: Optional[str] = None   # 數據庫版本
    os: Optional[str] = None             # 操作系統
    technique: Optional[str] = None      # 檢測技術
    confidence: int = 0                  # 信心度 (0-100)
    risk: str = "unknown"                # 風險等級
    raw_output: Optional[str] = None     # 原始輸出
    exploitation_data: Dict[str, Any] = None  # 利用數據
```

##### 3. SqlmapIntegration - Sqlmap 整合
```python
class SqlmapIntegration:
    """Sqlmap 工具整合類"""
    
    async def scan(
        self, 
        target: SQLTarget,
        options: Dict[str, Any] = None
    ) -> List[SQLInjectionResult]:
        """
        執行 sqlmap 掃描
        
        Args:
            target: 目標對象
            options: sqlmap 選項
                - level: 測試等級 (1-5)
                - risk: 風險等級 (1-3)
                - threads: 線程數
                - technique: 測試技術 (BEUSTQ)
                
        Returns:
            檢測結果列表
        """
```

**使用示例**：
```python
from integration_tools import SqlmapIntegration, SQLTarget

async def test_sqlmap():
    integration = SqlmapIntegration()
    
    target = SQLTarget(
        url="http://testphp.vulnweb.com/artists.php?artist=1"
    )
    
    results = await integration.scan(
        target,
        options={
            "level": 3,
            "risk": 2,
            "threads": 5,
            "batch": True  # 非交互模式
        }
    )
    
    for result in results:
        print(f"發現注入: {result.parameter}")
        print(f"類型: {result.injection_type}")
        print(f"數據庫: {result.dbms} {result.dbms_version}")
```

##### 4. SQLInjectionManager - 統一管理器
```python
class SQLInjectionManager:
    """SQL 注入工具統一管理器"""
    
    def __init__(self):
        self.sqlmap = SqlmapIntegration()
        self.tools = {
            "sqlmap": self.sqlmap,
            # 未來可添加更多工具
        }
    
    async def quick_scan(
        self,
        url: str,
        tool: str = "sqlmap"
    ) -> List[SQLInjectionResult]:
        """快速掃描模式"""
        
    async def deep_scan(
        self,
        url: str,
        tool: str = "sqlmap"
    ) -> List[SQLInjectionResult]:
        """深度掃描模式"""
    
    async def batch_scan(
        self,
        urls: List[str],
        tool: str = "sqlmap"
    ) -> Dict[str, List[SQLInjectionResult]]:
        """批量掃描"""
```

**使用示例**：
```python
from integration_tools import SQLInjectionManager

async def main():
    manager = SQLInjectionManager()
    
    # 快速掃描（適合初篩）
    results = await manager.quick_scan(
        "http://target.com/page?id=1"
    )
    
    # 深度掃描（確認漏洞）
    if results:
        deep_results = await manager.deep_scan(
            "http://target.com/page?id=1"
        )
    
    # 批量掃描
    urls = [
        "http://site1.com/page?id=1",
        "http://site2.com/search?q=test",
    ]
    batch_results = await manager.batch_scan(urls)
```

---

### bounty_hunter.py - Bug Bounty 獵人模式

#### 設計理念
針對 Bug Bounty 場景優化，關注：
- ✅ **高價值目標優先**：登錄、支付、管理後台
- ✅ **快速驗證**：減少無效測試時間
- ✅ **詳細證據**：生成可提交的報告
- ✅ **隱蔽性**：減少 WAF 觸發

#### 主要類別

##### 1. HighValueTarget - 高價值目標
```python
@dataclass
class HighValueTarget:
    """高價值目標定義"""
    url: str
    method: str = "GET"
    parameters: Dict[str, str] = None
    priority: str = "high"              # high, medium, low
    bounty_potential: int = 0           # 預估獎金（美元）
    confidence_threshold: int = 90      # 信心度門檻
    
    # Bug Bounty 特定屬性
    endpoint_type: str = ""             # login, admin, api, payment
    auth_required: bool = False         # 是否需要認證
    rate_limit: Optional[int] = None    # 請求速率限制
```

##### 2. BountyHunter - 獵人核心類
```python
class BountyHunter:
    """Bug Bounty 獵人模式"""
    
    async def hunt(
        self,
        targets: List[HighValueTarget],
        mode: str = "stealth"
    ) -> Dict[str, Any]:
        """
        執行獵人模式掃描
        
        Args:
            targets: 目標列表
            mode: 掃描模式
                - stealth: 隱蔽模式（低速率、隨機延遲）
                - balanced: 平衡模式（中等速率）
                - aggressive: 激進模式（高速率、深度測試）
                
        Returns:
            包含發現、報告、統計的字典
        """
```

#### 核心功能

##### 1. 智能目標排序
```python
def _prioritize_targets(self, targets: List[HighValueTarget]) -> List[HighValueTarget]:
    """
    基於多個因素排序目標：
    1. 優先級（high > medium > low）
    2. 端點類型（login/admin > api > 其他）
    3. 預估獎金潛力
    """
    def score(target: HighValueTarget) -> int:
        priority_scores = {"high": 100, "medium": 50, "low": 10}
        type_scores = {
            "login": 50, 
            "admin": 50, 
            "payment": 40,
            "api": 30
        }
        
        score = priority_scores.get(target.priority, 0)
        score += type_scores.get(target.endpoint_type, 0)
        score += target.bounty_potential // 10
        
        return score
    
    return sorted(targets, key=score, reverse=True)
```

##### 2. 快速初篩
```python
async def _quick_check(self, target: HighValueTarget) -> bool:
    """
    快速檢查是否可能存在注入（減少深度測試時間）
    
    測試項目：
    1. Error-based（最快）
    2. Boolean-based 簡單 payload
    3. 時間盲注（5 秒延遲）
    
    耗時：10-20 秒
    """
    # 1. 錯誤測試
    error_payloads = ["'", '"', "1' OR '1'='1"]
    for payload in error_payloads:
        if await self._test_error_response(target, payload):
            return True  # 可能存在，進入深度測試
    
    # 2. 布林測試
    if await self._test_boolean_diff(target):
        return True
    
    # 3. 時間測試
    if await self._test_time_delay(target, delay=5):
        return True
    
    return False  # 不太可能存在注入
```

##### 3. 深度驗證
```python
async def _deep_verify(self, target: HighValueTarget) -> Optional[BountyFinding]:
    """
    深度驗證並收集證據
    
    包含：
    1. 多種注入技術測試
    2. 數據庫指紋識別
    3. 數據提取嘗試（證明影響）
    4. 截圖和 HTTP 記錄
    
    耗時：2-10 分鐘
    """
    # 使用 sqlmap 深度掃描
    integration = SqlmapIntegration()
    results = await integration.scan(
        target,
        options={
            "level": 5,
            "risk": 3,
            "threads": 5,
            "technique": "BEUSTQ",  # 全部技術
            "dump": True,           # 嘗試數據提取
        }
    )
    
    if results and results[0].vulnerable:
        # 生成完整證據
        return BountyFinding(
            target=target,
            vulnerability=results[0],
            evidence=self._collect_evidence(results[0]),
            impact_assessment=self._assess_impact(results[0]),
            remediation=self._generate_remediation(results[0])
        )
    
    return None
```

##### 4. 報告生成
```python
@dataclass
class BountyFinding:
    """Bug Bounty 發現"""
    target: HighValueTarget
    vulnerability: SQLInjectionResult
    evidence: Dict[str, Any]
    impact_assessment: str
    remediation: str
    cvss_score: float = 0.0
    estimated_bounty: int = 0

def generate_report(self, findings: List[BountyFinding]) -> str:
    """
    生成專業 Bug Bounty 報告
    
    包含：
    - 執行摘要
    - 漏洞詳情（每個發現）
    - 重現步驟
    - 影響分析
    - 修復建議
    - 附件（截圖、HTTP 記錄）
    """
```

#### 使用示例

```python
from integration_tools import BountyHunter, HighValueTarget

async def bug_bounty_scan():
    hunter = BountyHunter()
    
    # 定義高價值目標
    targets = [
        HighValueTarget(
            url="https://target.com/login",
            method="POST",
            parameters={"username": "test", "password": "test"},
            priority="high",
            endpoint_type="login",
            bounty_potential=500  # 預估 $500
        ),
        HighValueTarget(
            url="https://target.com/admin/users",
            priority="high",
            endpoint_type="admin",
            auth_required=True,
            bounty_potential=1000  # 預估 $1000
        ),
        HighValueTarget(
            url="https://target.com/api/search",
            parameters={"q": "test"},
            priority="medium",
            endpoint_type="api",
            bounty_potential=300
        ),
    ]
    
    # 執行獵人模式（隱蔽）
    results = await hunter.hunt(targets, mode="stealth")
    
    # 輸出結果
    print(f"掃描完成:")
    print(f"  測試目標: {results['targets_scanned']}")
    print(f"  發現漏洞: {len(results['findings'])}")
    print(f"  預估總獎金: ${results['total_potential_bounty']}")
    
    # 生成報告
    if results['findings']:
        report = hunter.generate_report(results['findings'])
        with open("bounty_report.md", "w") as f:
            f.write(report)
```

---

## 技術實現

### Sqlmap 整合

#### 命令構建
```python
def _build_sqlmap_command(self, target: SQLTarget, options: Dict) -> List[str]:
    """構建 sqlmap 命令"""
    cmd = ["sqlmap"]
    
    # 基礎參數
    cmd.extend(["--url", target.url])
    cmd.extend(["--batch"])  # 非交互
    cmd.extend(["--random-agent"])  # 隨機 UA
    
    # 選項參數
    if "level" in options:
        cmd.extend(["--level", str(options["level"])])
    
    if "risk" in options:
        cmd.extend(["--risk", str(options["risk"])])
    
    if "technique" in options:
        cmd.extend(["--technique", options["technique"]])
    
    # POST 數據
    if target.method == "POST" and target.data:
        cmd.extend(["--data", target.data])
    
    # 自定義頭
    if target.headers:
        for key, value in target.headers.items():
            cmd.extend(["--header", f"{key}: {value}"])
    
    # Cookie
    if target.cookies:
        cookie_str = "; ".join(f"{k}={v}" for k, v in target.cookies.items())
        cmd.extend(["--cookie", cookie_str])
    
    # 輸出格式
    cmd.extend(["--output-dir", "/tmp/sqlmap"])
    cmd.extend(["--flush-session"])  # 清除緩存
    
    return cmd
```

#### 結果解析
```python
def _parse_sqlmap_output(self, output_dir: str) -> List[SQLInjectionResult]:
    """解析 sqlmap 輸出目錄"""
    results = []
    
    # 讀取日誌文件
    log_file = Path(output_dir) / "log"
    if not log_file.exists():
        return results
    
    content = log_file.read_text()
    
    # 提取注入點
    injection_pattern = r"Parameter: (.+?) \((.*?)\) is vulnerable"
    matches = re.finditer(injection_pattern, content)
    
    for match in matches:
        parameter = match.group(1)
        location = match.group(2)
        
        # 提取詳細信息
        dbms = self._extract_dbms(content)
        technique = self._extract_technique(content)
        payload = self._extract_payload(content, parameter)
        
        results.append(SQLInjectionResult(
            vulnerable=True,
            injection_type=technique,
            parameter=parameter,
            payload=payload,
            dbms=dbms,
            technique=technique,
            confidence=90
        ))
    
    return results
```

### NoSQLMap 整合

```python
class NoSQLMapIntegration:
    """NoSQL 注入工具整合"""
    
    async def scan(self, target: SQLTarget) -> List[SQLInjectionResult]:
        """掃描 NoSQL 注入"""
        
        # 構建命令
        cmd = [
            "python2",  # NoSQLMap 需要 Python 2
            "nosqlmap.py",
            "-u", target.url
        ]
        
        # 執行
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # 解析結果
        return self._parse_nosqlmap_output(stdout.decode())
```

### 結果標準化

```python
def _standardize_result(
    self,
    tool_name: str,
    raw_result: Any
) -> SQLInjectionResult:
    """
    將不同工具的輸出標準化為 SQLInjectionResult
    
    支持：
    - sqlmap
    - NoSQLMap
    - ghauri
    - 自定義工具
    """
    if tool_name == "sqlmap":
        return self._from_sqlmap(raw_result)
    elif tool_name == "nosqlmap":
        return self._from_nosqlmap(raw_result)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
```

---

## 使用指南

### 基礎使用

#### 1. 快速測試單一 URL
```python
from integration_tools import SQLInjectionManager

async def quick_test():
    manager = SQLInjectionManager()
    results = await manager.quick_scan(
        "http://testphp.vulnweb.com/artists.php?artist=1"
    )
    
    if results:
        print(f"發現 {len(results)} 個注入點")
```

#### 2. POST 請求測試
```python
target = SQLTarget(
    url="http://example.com/login",
    method="POST",
    data="username=admin&password=test",
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

integration = SqlmapIntegration()
results = await integration.scan(target)
```

#### 3. JSON API 測試
```python
target = SQLTarget(
    url="http://api.example.com/search",
    method="POST",
    data='{"query": "*"}',  # * 標記注入點
    headers={"Content-Type": "application/json"},
    custom_injection="*"
)

results = await integration.scan(
    target,
    options={"tamper": "space2comment"}  # 繞過 WAF
)
```

### 高級配置

#### 1. 自定義 Sqlmap 選項
```python
options = {
    "level": 5,              # 測試深度
    "risk": 3,               # 風險等級
    "threads": 10,           # 並發線程
    "technique": "BEUSTQ",   # 測試技術
    "tamper": "space2comment,between",  # WAF 繞過腳本
    "random-agent": True,    # 隨機 User-Agent
    "delay": 2,              # 請求延遲（秒）
    "timeout": 30,           # 超時時間
    "retries": 3,            # 重試次數
}
```

#### 2. 批量掃描
```python
urls = [
    "http://site1.com/page?id=1",
    "http://site2.com/search?q=test",
    "http://site3.com/product?pid=100",
]

manager = SQLInjectionManager()
results = await manager.batch_scan(urls, tool="sqlmap")

for url, findings in results.items():
    print(f"{url}: {len(findings)} vulnerabilities")
```

### Bug Bounty 模式

```python
from integration_tools import BountyHunter, HighValueTarget

# 創建獵人實例
hunter = BountyHunter()

# 定義目標
targets = [
    HighValueTarget(
        url="https://target.com/admin/login",
        priority="high",
        endpoint_type="admin",
        bounty_potential=2000
    ),
]

# 隱蔽模式掃描
results = await hunter.hunt(targets, mode="stealth")

# 生成報告
report = hunter.generate_report(results['findings'])
```

---

## API 參考

### SQLTarget
```python
@dataclass
class SQLTarget:
    url: str                              # 必需
    method: str = "GET"
    parameters: Dict[str, str] = None
    headers: Dict[str, str] = None
    cookies: Dict[str, str] = None
    data: Optional[str] = None
    custom_injection: Optional[str] = None
```

### SQLInjectionResult
```python
@dataclass
class SQLInjectionResult:
    vulnerable: bool
    injection_type: str
    parameter: str
    payload: str
    dbms: Optional[str] = None
    dbms_version: Optional[str] = None
    confidence: int = 0
    risk: str = "unknown"
```

### SqlmapIntegration
```python
class SqlmapIntegration:
    async def scan(target: SQLTarget, options: Dict) -> List[SQLInjectionResult]
    async def check_installation() -> bool
    async def install() -> bool
```

### BountyHunter
```python
class BountyHunter:
    async def hunt(targets: List[HighValueTarget], mode: str) -> Dict
    def generate_report(findings: List[BountyFinding]) -> str
```

---

## 開發指南

### 添加新工具整合

```python
class MyToolIntegration:
    """新工具整合類"""
    
    async def scan(
        self,
        target: SQLTarget,
        options: Dict = None
    ) -> List[SQLInjectionResult]:
        # 1. 構建命令
        cmd = self._build_command(target, options)
        
        # 2. 執行工具
        output = await self._execute(cmd)
        
        # 3. 解析結果
        results = self._parse_output(output)
        
        # 4. 標準化
        return [self._standardize(r) for r in results]
```

### 測試

```python
import pytest
from integration_tools import SqlmapIntegration, SQLTarget

@pytest.mark.asyncio
async def test_sqlmap_integration():
    integration = SqlmapIntegration()
    
    target = SQLTarget(
        url="http://testphp.vulnweb.com/artists.php?artist=1"
    )
    
    results = await integration.scan(target)
    
    assert len(results) > 0
    assert results[0].vulnerable
    assert results[0].dbms == "MySQL"
```

---

## 📚 相關文檔

- [上層文檔：function_sqli README](../README.md)
- [檢測引擎：engines](../engines/README.md)
- [外部工具：external_tools](../external_tools/README.md)

---

**維護者**: AIVA Security Team  
**更新日期**: 2025年12月12日  
**版本**: v1.0.0
