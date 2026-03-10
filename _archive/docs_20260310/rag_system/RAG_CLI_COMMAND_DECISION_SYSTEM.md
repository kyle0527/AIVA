# RAG 真正的應用場景 - CLI 指令決策系統

## 一、RAG 的核心目的

### ❌ 之前的理解（太抽象）
```
RAG → 搜索「知識」→ 生成「建議」
```

### ✅ 實際的需求（具體場景）
```
掃描結果 → RAG 決策 → 選擇 CLI 指令 + 調整參數 → 執行
```

---

## 二、完整流程設計

### 2.1 對內搜索：找合適的 CLI 指令

**場景**：根據當前掃描結果，決定下一步用什麼工具

```
掃描階段 1 (Phase 0: 偵察)
  ↓
  發現: 目標是 PHP 網站，Apache 2.4，開放端口 80, 443
  ↓
RAG 決策:
  問題: "PHP Apache 網站應該用什麼掃描工具？"
  ↓
  搜索 CLI 指令庫:
    1. sqlmap (用於 SQL 注入測試)
    2. XSStrike (用於 XSS 測試)
    3. nikto (用於 Web 漏洞掃描)
  ↓
  決策: 選擇 sqlmap + XSStrike
  ↓
  調整參數:
    - sqlmap: --dbms=MySQL --level=3 --risk=2
    - XSStrike: --fuzzer --crawl
```

### 2.2 CLI 指令知識庫結構

**數據格式**：

```json
{
  "tool_name": "sqlmap",
  "capability": "sqli",
  "command_template": "sqlmap -u {target_url} --batch",
  "parameters": {
    "target_url": {
      "required": true,
      "type": "string",
      "description": "目標 URL"
    },
    "dbms": {
      "required": false,
      "type": "enum",
      "values": ["MySQL", "PostgreSQL", "MSSQL", "Oracle"],
      "description": "資料庫類型",
      "auto_detect": true
    },
    "level": {
      "required": false,
      "type": "integer",
      "range": [1, 5],
      "default": 1,
      "description": "測試等級"
    },
    "risk": {
      "required": false,
      "type": "integer",
      "range": [1, 3],
      "default": 1,
      "description": "風險等級"
    }
  },
  "適用場景": {
    "目標類型": ["web_application"],
    "發現端口": [80, 443, 8080],
    "技術棧": ["PHP", "ASP", "JSP"],
    "前置條件": ["發現參數", "發現表單"]
  },
  "參數調整規則": {
    "如果檢測到 MySQL": {
      "dbms": "MySQL",
      "level": 3
    },
    "如果有 WAF": {
      "tamper": "space2comment,between",
      "random-agent": true
    },
    "如果時間緊急": {
      "level": 1,
      "risk": 1
    }
  }
}
```

### 2.3 參數自動調整邏輯

**根據掃描結果動態調整參數**：

```python
class CLICommandDecisionEngine:
    """CLI 指令決策引擎"""
    
    def decide_command(
        self,
        scan_results: dict,  # 前面階段的掃描結果
        capability: str,     # 需要的能力 (xss, sqli, etc.)
    ) -> dict:
        """
        根據掃描結果決定使用什麼 CLI 指令和參數
        
        Args:
            scan_results: {
                "target": "http://example.com",
                "detected_tech": ["PHP", "Apache", "MySQL"],
                "open_ports": [80, 443],
                "has_waf": true,
                "response_time": 2.5,
                "found_parameters": ["id", "page", "user"]
            }
            capability: "sqli"
        
        Returns:
            {
                "tool": "sqlmap",
                "command": "sqlmap -u http://example.com?id=1 --batch --dbms=MySQL ...",
                "parameters": {...},
                "reasoning": "檢測到 MySQL 和 WAF，建議使用 tamper 腳本"
            }
        """
        
        # 1. 搜索合適的工具
        tools = self._search_tools_for_capability(capability, scan_results)
        
        # 2. 選擇最佳工具
        best_tool = self._select_best_tool(tools, scan_results)
        
        # 3. 調整參數
        parameters = self._adjust_parameters(best_tool, scan_results)
        
        # 4. 生成命令
        command = self._build_command(best_tool, parameters)
        
        return {
            "tool": best_tool["name"],
            "command": command,
            "parameters": parameters,
            "reasoning": self._explain_decision(best_tool, scan_results)
        }
```

---

## 三、對內搜索：CLI 指令庫設計

### 3.1 儲存結構

**選項 1：JSONL 文件（簡單）** ✅ 推薦目前使用

```
services/integration/data/cli_commands/
├── xss_commands.jsonl
├── sqli_commands.jsonl
├── ssrf_commands.jsonl
└── phase0_commands.jsonl
```

**選項 2：SQLite 資料庫（未來）**

```sql
CREATE TABLE cli_commands (
    id INTEGER PRIMARY KEY,
    tool_name TEXT NOT NULL,
    capability TEXT NOT NULL,
    command_template TEXT NOT NULL,
    parameters_json TEXT,  -- JSON 格式參數定義
    适用场景_json TEXT,    -- JSON 格式適用條件
    调整规则_json TEXT,    -- JSON 格式參數調整規則
    priority INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0
);

CREATE INDEX idx_capability ON cli_commands(capability);
```

### 3.2 搜索邏輯

**Step 1: 過濾合適的工具**

```python
def search_cli_commands(
    capability: str,
    scan_results: dict,
) -> list[dict]:
    """搜索合適的 CLI 指令"""
    
    # 讀取該能力的所有指令
    commands = load_commands_for_capability(capability)
    
    # 過濾：檢查適用場景
    matched_commands = []
    for cmd in commands:
        if _is_applicable(cmd, scan_results):
            matched_commands.append(cmd)
    
    # 排序：按優先級和成功率
    matched_commands.sort(
        key=lambda x: (x.get("priority", 0), x.get("success_rate", 0)),
        reverse=True
    )
    
    return matched_commands

def _is_applicable(command: dict, scan_results: dict) -> bool:
    """檢查指令是否適用於當前場景"""
    
    適用場景 = command.get("適用場景", {})
    
    # 檢查技術棧
    if "技術棧" in 適用場景:
        detected_tech = scan_results.get("detected_tech", [])
        required_tech = 適用場景["技術棧"]
        if not any(tech in detected_tech for tech in required_tech):
            return False
    
    # 檢查端口
    if "發現端口" in 適用場景:
        open_ports = scan_results.get("open_ports", [])
        required_ports = 適用場景["發現端口"]
        if not any(port in open_ports for port in required_ports):
            return False
    
    # 檢查前置條件
    if "前置條件" in 適用場景:
        for condition in 適用場景["前置條件"]:
            if not _check_condition(condition, scan_results):
                return False
    
    return True
```

**Step 2: 調整參數**

```python
def adjust_parameters(
    command: dict,
    scan_results: dict,
) -> dict:
    """根據掃描結果調整參數"""
    
    base_params = command.get("parameters", {})
    adjustment_rules = command.get("參數調整規則", {})
    
    final_params = base_params.copy()
    
    # 應用調整規則
    for rule_name, rule_config in adjustment_rules.items():
        if _rule_matches(rule_name, scan_results):
            # 應用這個規則的參數調整
            final_params.update(rule_config)
    
    return final_params

def _rule_matches(rule_name: str, scan_results: dict) -> bool:
    """檢查規則是否匹配當前場景"""
    
    if rule_name == "如果檢測到 MySQL":
        return "MySQL" in scan_results.get("detected_tech", [])
    
    elif rule_name == "如果有 WAF":
        return scan_results.get("has_waf", False)
    
    elif rule_name == "如果時間緊急":
        return scan_results.get("time_limit", float('inf')) < 300  # 5分鐘
    
    # ... 更多規則
    
    return False
```

---

## 四、對外搜索：關鍵字提取

### 4.1 從未知錯誤中提取關鍵字

**場景**：收到從未見過的錯誤訊息

```
執行 sqlmap 後收到響應:
{
    "status_code": 403,
    "error_message": "ModSecurity: Access denied with code 403 (phase 2). 
                      Pattern match \"(?i:(?:\\\\bsys\\\\.user_objects\\\\b|\\\\bsys\\\\.tab(?:s|les)\\\\b))\" 
                      at ARGS:id. [id: 981242]",
    "headers": {
        "Server": "Apache/2.4.41",
        "X-ModSecurity-Version": "2.9.3"
    }
}
```

**關鍵字提取邏輯**：

```python
class KeywordExtractor:
    """從錯誤訊息中提取搜索關鍵字"""
    
    def extract_keywords(self, error_data: dict) -> list[str]:
        """提取關鍵字"""
        
        keywords = []
        
        # 1. 提取技術棧關鍵字
        tech_keywords = self._extract_tech_stack(error_data)
        keywords.extend(tech_keywords)
        # 結果: ["ModSecurity", "Apache", "WAF"]
        
        # 2. 提取錯誤類型
        error_type = self._extract_error_type(error_data)
        keywords.extend(error_type)
        # 結果: ["Access denied", "403 Forbidden", "Pattern match"]
        
        # 3. 提取規則 ID
        rule_id = self._extract_rule_id(error_data)
        if rule_id:
            keywords.append(rule_id)
        # 結果: ["981242", "ModSecurity Rule 981242"]
        
        # 4. 提取攻擊類型（從模式匹配）
        attack_type = self._extract_attack_type(error_data)
        keywords.extend(attack_type)
        # 結果: ["SQL injection", "UNION attack"]
        
        return keywords
    
    def _extract_tech_stack(self, error_data: dict) -> list[str]:
        """提取技術棧"""
        keywords = []
        
        error_msg = error_data.get("error_message", "")
        headers = error_data.get("headers", {})
        
        # 從錯誤訊息提取
        tech_patterns = {
            "ModSecurity": ["ModSecurity", "mod_security"],
            "Cloudflare": ["Cloudflare", "CF-RAY"],
            "WAF": ["Web Application Firewall", "WAF"],
            "Apache": ["Apache"],
            "Nginx": ["nginx"],
        }
        
        for tech, patterns in tech_patterns.items():
            if any(p in error_msg for p in patterns):
                keywords.append(tech)
        
        # 從 Headers 提取
        if "Server" in headers:
            server = headers["Server"]
            keywords.append(server.split("/")[0])  # "Apache/2.4.41" → "Apache"
        
        return keywords
    
    def _extract_rule_id(self, error_data: dict) -> str:
        """提取規則 ID"""
        error_msg = error_data.get("error_message", "")
        
        # ModSecurity 規則 ID
        import re
        match = re.search(r'\[id:\s*(\d+)\]', error_msg)
        if match:
            return f"ModSecurity Rule {match.group(1)}"
        
        return None
    
    def _extract_attack_type(self, error_data: dict) -> list[str]:
        """從模式匹配推測攻擊類型"""
        error_msg = error_data.get("error_message", "")
        
        attack_keywords = []
        
        # 從錯誤訊息中的模式匹配推測
        if "sys.user_objects" in error_msg or "sys.tables" in error_msg:
            attack_keywords.extend(["SQL injection", "Database enumeration"])
        
        if "UNION" in error_msg.upper():
            attack_keywords.append("UNION-based SQL injection")
        
        if "<script" in error_msg or "onerror" in error_msg:
            attack_keywords.append("XSS")
        
        return attack_keywords
```

**提取結果**：

```python
keywords = [
    "ModSecurity",
    "Apache",
    "WAF",
    "Access denied",
    "403 Forbidden",
    "Pattern match",
    "ModSecurity Rule 981242",
    "SQL injection",
    "Database enumeration",
]
```

### 4.2 使用關鍵字對外搜索

```python
async def search_external_with_keywords(keywords: list[str]) -> list[dict]:
    """使用提取的關鍵字對外搜索"""
    
    results = []
    
    # 1. 搜索 ModSecurity 規則
    if any("ModSecurity Rule" in k for k in keywords):
        rule_id = _extract_rule_number(keywords)
        modsec_results = await search_modsecurity_rules(rule_id)
        results.extend(modsec_results)
    
    # 2. 搜索繞過技術
    bypass_query = " ".join(keywords) + " bypass"
    bypass_results = await search_bypass_techniques(bypass_query)
    results.extend(bypass_results)
    
    # 3. 搜索 CVE
    cve_query = " ".join(keywords[:3])  # 前 3 個關鍵字
    cve_results = await search_cve_database(cve_query)
    results.extend(cve_results)
    
    return results
```

**搜索結果示例**：

```json
[
  {
    "type": "modsecurity_rule",
    "source": "ModSecurity GitHub",
    "title": "Rule 981242: SQL Injection Attack - System Table Access",
    "content": "This rule detects attempts to access system tables like sys.user_objects...",
    "bypass_techniques": [
      "使用註釋繞過: sys/**/user_objects",
      "使用十六進制編碼: 0x73797300757365725f6f626a65637473",
      "使用大小寫混淆: SyS.UsEr_ObJeCtS"
    ],
    "url": "https://github.com/SpiderLabs/ModSecurity/blob/master/rules/..."
  },
  {
    "type": "exploit_technique",
    "source": "Exploit-DB",
    "title": "ModSecurity WAF Bypass Techniques for SQL Injection",
    "content": "Various methods to bypass ModSecurity rules...",
    "url": "https://www.exploit-db.com/..."
  }
]
```

---

## 五、完整實現：RAG 決策引擎

```python
class RAGDecisionEngine:
    """RAG 決策引擎 - 統一對內和對外搜索"""
    
    def __init__(self):
        self.cli_commands_db = CLICommandsDatabase()
        self.keyword_extractor = KeywordExtractor()
        self.vector_store = VectorStore()  # 可選，用於語義搜索
    
    async def decide_next_action(
        self,
        current_phase: str,
        scan_results: dict,
        error_data: dict = None,
    ) -> dict:
        """
        決定下一步操作
        
        Args:
            current_phase: 當前階段 (phase0, phase1, xss, sqli, etc.)
            scan_results: 到目前為止的所有掃描結果
            error_data: 如果有錯誤，提供錯誤數據
        
        Returns:
            {
                "action": "execute_command",
                "tool": "sqlmap",
                "command": "sqlmap -u ... --tamper=...",
                "parameters": {...},
                "reasoning": "檢測到 MySQL 和 ModSecurity WAF，建議使用 tamper 腳本繞過"
            }
        """
        
        # 情況 1: 正常流程 - 對內搜索 CLI 指令
        if not error_data:
            return await self._decide_normal_flow(current_phase, scan_results)
        
        # 情況 2: 遇到錯誤 - 提取關鍵字並對外搜索
        else:
            return await self._handle_error_with_rag(error_data, scan_results)
    
    async def _decide_normal_flow(
        self,
        current_phase: str,
        scan_results: dict,
    ) -> dict:
        """正常流程：選擇合適的 CLI 指令"""
        
        # 1. 搜索合適的指令
        commands = self.cli_commands_db.search_commands(
            capability=current_phase,
            scan_results=scan_results
        )
        
        if not commands:
            return {"action": "skip", "reasoning": "沒有找到合適的工具"}
        
        # 2. 選擇最佳指令
        best_command = commands[0]
        
        # 3. 調整參數
        parameters = self.cli_commands_db.adjust_parameters(
            command=best_command,
            scan_results=scan_results
        )
        
        # 4. 構建命令
        command = self._build_command(best_command, parameters, scan_results)
        
        return {
            "action": "execute_command",
            "tool": best_command["tool_name"],
            "command": command,
            "parameters": parameters,
            "reasoning": self._explain_normal_decision(best_command, scan_results)
        }
    
    async def _handle_error_with_rag(
        self,
        error_data: dict,
        scan_results: dict,
    ) -> dict:
        """錯誤處理：提取關鍵字並對外搜索"""
        
        # 1. 提取關鍵字
        keywords = self.keyword_extractor.extract_keywords(error_data)
        
        logger.info(f"提取到關鍵字: {keywords}")
        
        # 2. 先對內搜索（快速）
        internal_results = await self._search_internal_solutions(keywords)
        
        # 3. 如果內部找到解決方案，直接使用
        if internal_results and internal_results[0]["relevance_score"] > 0.8:
            return self._apply_internal_solution(internal_results[0])
        
        # 4. 對外搜索（未知錯誤）
        logger.info("內部未找到高相關解決方案，開始對外搜索...")
        external_results = await self._search_external_resources(keywords)
        
        # 5. 分析外部結果並生成新策略
        new_strategy = self._analyze_external_results(
            external_results,
            error_data,
            scan_results
        )
        
        return new_strategy
    
    async def _search_internal_solutions(
        self,
        keywords: list[str]
    ) -> list[dict]:
        """對內搜索：歷史解決方案"""
        
        # 搜索歷史上遇到相同關鍵字的解決方案
        query = " ".join(keywords)
        
        # 方式 1: 關鍵字匹配
        keyword_results = self.cli_commands_db.search_by_keywords(keywords)
        
        # 方式 2: 向量語義搜索（可選）
        if self.vector_store:
            vector_results = await self.vector_store.search(query, top_k=5)
            keyword_results.extend(vector_results)
        
        return keyword_results
    
    async def _search_external_resources(
        self,
        keywords: list[str]
    ) -> list[dict]:
        """對外搜索：使用關鍵字搜索外部資源"""
        
        results = []
        
        # 1. ModSecurity 規則庫
        if any("ModSecurity" in k for k in keywords):
            modsec_results = await search_modsecurity_rules(keywords)
            results.extend(modsec_results)
        
        # 2. WAF 繞過技術
        if any("WAF" in k or "403" in k for k in keywords):
            bypass_results = await search_waf_bypass_techniques(keywords)
            results.extend(bypass_results)
        
        # 3. CVE 資料庫
        cve_results = await search_cve_with_keywords(keywords)
        results.extend(cve_results)
        
        # 4. Google 技術搜索
        google_query = " ".join(keywords) + " bypass technique"
        google_results = await search_google(google_query)
        results.extend(google_results)
        
        return results
```

---

## 六、資料庫結合方案

### 6.1 CLI 指令庫 + 執行歷史

**表結構**：

```sql
-- CLI 指令定義表
CREATE TABLE cli_commands (
    id INTEGER PRIMARY KEY,
    tool_name TEXT NOT NULL,
    capability TEXT NOT NULL,
    command_template TEXT NOT NULL,
    parameters_json TEXT,
    适用场景_json TEXT,
    调整规则_json TEXT
);

-- 執行歷史表
CREATE TABLE execution_history (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL,
    command_id INTEGER,
    scan_results_json TEXT,  -- 掃描結果（決策依據）
    command_executed TEXT,   -- 實際執行的命令
    parameters_used_json TEXT,  -- 實際使用的參數
    success BOOLEAN,
    error_data_json TEXT,    -- 如果失敗，記錄錯誤
    solution_applied_json TEXT,  -- 應用的解決方案
    timestamp DATETIME,
    FOREIGN KEY (command_id) REFERENCES cli_commands(id)
);

-- 錯誤解決方案表
CREATE TABLE error_solutions (
    id INTEGER PRIMARY KEY,
    error_keywords TEXT,     -- 關鍵字（逗號分隔）
    error_pattern TEXT,      -- 錯誤模式（正則）
    solution_type TEXT,      -- "parameter_adjustment", "command_change", "bypass_technique"
    solution_json TEXT,      -- 解決方案詳情
    success_count INTEGER DEFAULT 0,
    total_attempts INTEGER DEFAULT 0,
    success_rate REAL GENERATED ALWAYS AS (success_count * 1.0 / total_attempts) STORED
);
```

### 6.2 查詢示例

```sql
-- 1. 根據掃描結果查找合適的 CLI 指令
SELECT * FROM cli_commands
WHERE capability = 'sqli'
  AND json_extract(适用场景_json, '$.技術棧') LIKE '%MySQL%'
ORDER BY (
    SELECT AVG(success) FROM execution_history
    WHERE execution_history.command_id = cli_commands.id
) DESC
LIMIT 5;

-- 2. 查找相似錯誤的解決方案
SELECT * FROM error_solutions
WHERE error_keywords LIKE '%ModSecurity%'
  AND error_keywords LIKE '%981242%'
ORDER BY success_rate DESC
LIMIT 3;

-- 3. 統計工具的成功率
SELECT 
    cli_commands.tool_name,
    COUNT(execution_history.id) as total_executions,
    SUM(CASE WHEN execution_history.success THEN 1 ELSE 0 END) as success_count,
    AVG(CASE WHEN execution_history.success THEN 1.0 ELSE 0.0 END) as success_rate
FROM cli_commands
LEFT JOIN execution_history ON cli_commands.id = execution_history.command_id
GROUP BY cli_commands.tool_name
ORDER BY success_rate DESC;
```

---

## 七、總結

### ✅ 對內搜索（CLI 指令決策）

1. **輸入**：當前掃描結果（技術棧、端口、發現等）
2. **搜索**：CLI 指令庫（JSONL 或 SQLite）
3. **匹配**：適用場景（技術棧、端口、前置條件）
4. **調整**：參數自動調整規則
5. **輸出**：完整的 CLI 命令

### ✅ 對外搜索（錯誤解決）

1. **輸入**：未知錯誤/響應
2. **提取**：關鍵字（技術棧、錯誤類型、規則 ID、攻擊類型）
3. **搜索**：外部資源（ModSecurity 規則、WAF 繞過、CVE、Google）
4. **分析**：提取解決方案和繞過技巧
5. **輸出**：新的執行策略

### ✅ 資料庫價值

1. **快速查詢**：根據條件快速過濾指令
2. **統計分析**：工具成功率、參數效果
3. **歷史學習**：從執行歷史中學習最佳實踐
4. **錯誤匹配**：快速找到相似錯誤的解決方案
