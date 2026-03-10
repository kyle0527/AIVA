# AIVA 指令發送與接收架構規劃

## 📑 目錄

- [🎯 核心設計理念](#-核心設計理念)
- [📋 模組通信方式總覽](#-模組通信方式總覽)
- [🔑 核心改動：AI 輸出 CLI 指令](#-核心改動ai-輸出-cli-指令)
  - [AI 決策輸出格式](#ai-決策輸出格式)
  - [CLI 指令規範](#cli-指令規範)
- [🔵 AI 決策引擎（異步）](#-ai-決策引擎異步)
  - [cognitive_core/decision/ai_command_generator.py](#cognitive_coredecisionai_command_generatorpy)
- [🟢 CLI 執行器（同步）](#-cli-執行器同步)
  - [task_planning/executor/cli_executor.py](#task_planningexecutorcli_executorpy)
- [📊 13 步驟與 CLI 指令對應](#-13-步驟與-cli-指令對應)
- [🔄 13 步驟流程（同步/異步標註）](#-13-步驟流程同步異步標註)
- [🏗️ 完整流程：AI 生成 CLI → 執行器執行](#-完整流程ai-生成-cli-執行器執行)
- [📊 完整 CLI 指令規範](#-完整-cli-指令規範)
  - [1. 掃描指令 (aiva scan)](#1-掃描指令-aiva-scan)
  - [2. 功能測試指令 (aiva feature)](#2-功能測試指令-aiva-feature)
  - [3. 偵察指令 (aiva recon)](#3-偵察指令-aiva-recon)
  - [4. 報告指令 (aiva report)](#4-報告指令-aiva-report)
- [📁 需要新增/修改的檔案](#-需要新增修改的檔案)
  - [新增檔案（CLI 薄層入口）](#新增檔案cli-薄層入口)
  - [修改檔案](#修改檔案)
  - [保留不變的檔案 (✅ 100% 保留)](#保留不變的檔案-100-保留)
- [🎯 跨語言支援](#-跨語言支援)
  - [CLI 路由表（語言映射）](#cli-路由表語言映射)
- [🔗 現有模組整合方案](#-現有模組整合方案)
  - [現有架構對照](#現有架構對照)
  - [整合策略](#整合策略)
  - [CLI 入口檔案結構（新增）](#cli-入口檔案結構新增)
  - [CLI 入口範例實現](#cli-入口範例實現)
  - [現有模組調用關係（更新後）](#現有模組調用關係更新後)
  - [attack_coordinator.py 修改](#attack_coordinatorpy-修改)
- [🔧 MQ 使用場景（保留但簡化）](#-mq-使用場景保留但簡化)
  - [保留 MQ 的情況](#保留-mq-的情況)
  - [不使用 MQ 的情況](#不使用-mq-的情況)
- [📊 改善效果](#-改善效果)
- [📝 總結](#-總結)

---


**建立時間**: 2026-01-10  
**更新時間**: 2026-01-10  
**設計原則**: 
1. AI 異步決策，其他模組同步執行
2. **AI 輸出 CLI 指令**，統一跨語言調用

---

## 🎯 核心設計理念

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   🔑 關鍵設計：AI 輸出 CLI 指令（不是 Python 函數調用）                              │
│                                                                                     │
│   好處：                                                                            │
│   ├─ 語言無關：Go, Rust, TypeScript, Python 都透過 CLI 調用                        │
│   ├─ 參數標準化：統一的 --target, --depth, --mode 等參數                           │
│   ├─ 易於擴展：新增引擎只需定義 CLI 介面                                           │
│   └─ 可追溯：CLI 指令可直接記錄和重放                                              │
│                                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                              │  │
│  │   用戶輸入 URL                                                               │  │
│  │        │                                                                     │  │
│  │        ▼                                                                     │  │
│  │   ┌─────────────────────┐                                                    │  │
│  │   │   CLI 入口          │ ← 同步啟動                                         │  │
│  │   └──────────┬──────────┘                                                    │  │
│  │              │                                                               │  │
│  │              ▼                                                               │  │
│  │   ┌─────────────────────┐     ┌─────────────────────────────────────────┐   │  │
│  │   │   task_planning     │ ──► │   cognitive_core (AI)                   │   │  │
│  │   │   (CLI 執行器)      │ ◄── │   輸出: CLI 指令字串                     │   │  │
│  │   └──────────┬──────────┘     │   例: "aiva scan --target xxx --phase 0"│   │  │
│  │              │                └─────────────────────────────────────────┘   │  │
│  │              │                                                               │  │
│  │              │ subprocess.run(cli_command)                                   │  │
│  │              ▼                                                               │  │
│  │   ┌─────────────────────────────────────────────────────────────────────┐   │  │
│  │   │   各語言引擎（透過 CLI 統一介面）                                    │   │  │
│  │   │                                                                      │   │  │
│  │   │   Python:  python -m services.features.xxx --target ...             │   │  │
│  │   │   Rust:    ./scan/rust_engine/scanner --target ...                  │   │  │
│  │   │   Go:      ./scan/go_engine/scanner --target ...                    │   │  │
│  │   │   TS:      node scan/ts_engine/dist/index.js --target ...           │   │  │
│  │   └─────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                              │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 模組通信方式總覽

| 發送方 | 接收方 | 通信方式 | 理由 |
|--------|--------|----------|------|
| CLI | task_planning | 同步函數調用 | 啟動點，需要等待結果 |
| task_planning | cognitive_core (AI 決策) | **異步** | AI 思考需要時間，不阻塞 |
| AI 輸出 | task_planning | **CLI 指令字串** | 語言無關，統一介面 |
| task_planning | 各語言引擎 | **subprocess.run(CLI)** | 跨語言調用 |

---

## 🔑 核心改動：AI 輸出 CLI 指令

### AI 決策輸出格式

```python
@dataclass
class AICommandDecision:
    """AI 決策輸出：CLI 指令"""
    
    # CLI 指令（主要輸出）
    cli_command: str  # 例: "aiva scan --target https://xxx --phase 0 --depth quick"
    
    # 指令元數據
    command_type: str  # "scan" | "feature" | "integration" | "recon"
    priority: int      # 1-10，越高越優先
    timeout: int       # 秒，建議超時時間
    
    # 參數解析（方便日誌記錄）
    parsed_args: dict  # {"target": "https://xxx", "phase": 0, "depth": "quick"}
    
    # 決策理由（供人類審查）
    reasoning: str     # "Phase 0 快速偵察，發現 3 個端點需要深度掃描"
```

### CLI 指令規範

```yaml
# AIVA CLI 指令規範 (AI 必須遵循此格式輸出)

# === 掃描類指令 ===
aiva scan:
  base: "aiva scan --target <URL>"
  parameters:
    --target: "目標 URL（必填）"
    --phase: "0|1|2|3（執行階段）"
    --depth: "quick|normal|deep（掃描深度）"
    --threads: "1-50（並行數）"
    --timeout: "秒數（超時）"
    --output: "json|text|html（輸出格式）"
  
  examples:
    - "aiva scan --target https://example.com --phase 0 --depth quick"
    - "aiva scan --target https://example.com --phase 1 --depth deep --threads 10"

# === 功能測試類指令 ===
aiva feature:
  base: "aiva feature <type> --target <URL>"
  types: [xss, sqli, ssrf, idor, csrf, xxe, ssti, lfi]
  parameters:
    --target: "目標 URL（必填）"
    --payload-level: "basic|advanced|aggressive"
    --bypass: "waf|encoding|none"
    --verify: "true|false（是否驗證）"
    
  examples:
    - "aiva feature xss --target https://example.com/search?q= --payload-level advanced"
    - "aiva feature sqli --target https://example.com/user?id=1 --bypass waf"

# === 偵察類指令 ===
aiva recon:
  base: "aiva recon <type> --target <DOMAIN>"
  types: [subdomain, port, tech, wayback, js-scan]
  parameters:
    --target: "目標域名（必填）"
    --recursive: "true|false"
    --wordlist: "small|medium|large|custom"
    
  examples:
    - "aiva recon subdomain --target example.com --recursive true"
    - "aiva recon tech --target example.com"

# === 報告類指令 ===
aiva report:
  base: "aiva report generate"
  parameters:
    --session: "session ID"
    --format: "json|html|pdf|md"
    --include: "findings|timeline|recommendations"
```

---

## 🔵 AI 決策引擎（異步）

### cognitive_core/decision/ai_command_generator.py

```python
"""
AI 指令生成器 - 將 AI 決策轉換為 CLI 指令
"""
from dataclasses import dataclass
from typing import Optional
import asyncio
from uuid import uuid4

@dataclass
class AICommandDecision:
    """AI 決策輸出"""
    cli_command: str
    command_type: str
    priority: int
    timeout: int
    parsed_args: dict
    reasoning: str


class AICommandGenerator:
    """AI 命令生成器"""
    
    # CLI 指令模板
    COMMAND_TEMPLATES = {
        "scan": "aiva scan --target {target} --phase {phase} --depth {depth}",
        "xss": "aiva feature xss --target {target} --payload-level {level}",
        "sqli": "aiva feature sqli --target {target} --bypass {bypass}",
        "ssrf": "aiva feature ssrf --target {target} --verify {verify}",
        "idor": "aiva feature idor --target {target} --auth {auth}",
        "recon_subdomain": "aiva recon subdomain --target {domain} --recursive {recursive}",
        "recon_port": "aiva recon port --target {domain} --top {top}",
        "recon_tech": "aiva recon tech --target {domain}",
    }
    
    # 參數預設值
    DEFAULT_PARAMS = {
        "phase": 0,
        "depth": "normal",
        "level": "advanced",
        "bypass": "none",
        "verify": "true",
        "recursive": "false",
        "top": 1000,
    }
    
    def __init__(self, decision_agent):
        self.agent = decision_agent  # EnhancedDecisionAgent
        self._request_queue = asyncio.Queue()
    
    async def generate_command(self, context: dict) -> AICommandDecision:
        """
        根據上下文生成 CLI 指令
        
        Args:
            context: {
                "target": "https://example.com",
                "phase": 0,
                "findings_so_far": [...],
                "scan_history": [...],
            }
        
        Returns:
            AICommandDecision with cli_command string
        """
        # 1. AI 決策（異步）
        intent = await self.agent.decide(context)
        
        # 2. 將 AI 意圖轉換為 CLI 指令
        command_type = self._map_intent_to_command_type(intent)
        params = self._extract_params(intent, context)
        
        # 3. 生成 CLI 指令字串
        template = self.COMMAND_TEMPLATES.get(command_type)
        cli_command = self._build_command(template, params)
        
        return AICommandDecision(
            cli_command=cli_command,
            command_type=command_type,
            priority=intent.priority,
            timeout=self._estimate_timeout(command_type, params),
            parsed_args=params,
            reasoning=intent.reasoning,
        )
    
    def _map_intent_to_command_type(self, intent) -> str:
        """將 AI HighLevelIntent 映射到指令類型"""
        intent_mapping = {
            "phase0_recon": "scan",
            "phase1_deep_scan": "scan",
            "test_xss": "xss",
            "test_sqli": "sqli",
            "test_ssrf": "ssrf",
            "test_idor": "idor",
            "enumerate_subdomains": "recon_subdomain",
            "scan_ports": "recon_port",
            "detect_tech": "recon_tech",
        }
        return intent_mapping.get(intent.action, "scan")
    
    def _extract_params(self, intent, context: dict) -> dict:
        """從意圖和上下文提取參數"""
        params = {**self.DEFAULT_PARAMS}
        params["target"] = context.get("target", "")
        params["domain"] = context.get("domain", params["target"])
        
        # 從 AI 意圖覆蓋參數
        if hasattr(intent, "parameters"):
            params.update(intent.parameters)
        
        return params
    
    def _build_command(self, template: str, params: dict) -> str:
        """構建 CLI 指令字串"""
        try:
            return template.format(**params)
        except KeyError as e:
            # 缺少參數，使用預設值
            params[str(e).strip("'")] = self.DEFAULT_PARAMS.get(str(e).strip("'"), "")
            return template.format(**params)
    
    def _estimate_timeout(self, command_type: str, params: dict) -> int:
        """估算指令執行超時時間"""
        timeouts = {
            "scan": {"quick": 300, "normal": 600, "deep": 1800},
            "xss": 120,
            "sqli": 180,
            "ssrf": 120,
            "idor": 90,
            "recon_subdomain": 600,
            "recon_port": 300,
            "recon_tech": 60,
        }
        
        timeout = timeouts.get(command_type, 300)
        if isinstance(timeout, dict):
            depth = params.get("depth", "normal")
            timeout = timeout.get(depth, 600)
        
        return timeout
```

---

## 🟢 CLI 執行器（同步）

### task_planning/executor/cli_executor.py

```python
"""
CLI 執行器 - 執行 AI 生成的 CLI 指令
"""
import subprocess
import json
import shlex
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class CLIExecutionResult:
    """CLI 執行結果"""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    parsed_output: Optional[dict] = None


class CLIExecutor:
    """
    CLI 指令執行器
    
    職責：
    1. 接收 AI 生成的 CLI 指令字串
    2. 同步執行指令（subprocess.run）
    3. 收集並解析輸出
    """
    
    # 允許執行的指令前綴（安全白名單）
    ALLOWED_PREFIXES = [
        "aiva ",           # AIVA 主 CLI
        "python -m ",      # Python 模組
        "./scan/",         # 掃描引擎
        "node ",           # Node.js 引擎
    ]
    
    # 指令路由表（CLI 指令 → 實際執行路徑）
    COMMAND_ROUTER = {
        "aiva scan": "python -m services.scan.cli",
        "aiva feature xss": "python -m services.features.features_ready.function_xss.cli",
        "aiva feature sqli": "python -m services.features.features_ready.function_sqli.cli",
        "aiva feature ssrf": "python -m services.features.features_ready.function_ssrf.cli",
        "aiva feature idor": "python -m services.features.features_ready.function_idor.cli",
        "aiva recon subdomain": "./scan/go_engine/recon --mode subdomain",
        "aiva recon port": "./scan/rust_engine/target/release/port_scanner",
        "aiva recon tech": "python -m services.scan.tech_detector",
    }
    
    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
    
    def execute(self, cli_command: str, timeout: int = 600) -> CLIExecutionResult:
        """
        同步執行 CLI 指令
        
        Args:
            cli_command: AI 生成的 CLI 指令字串
            timeout: 超時秒數
        
        Returns:
            CLIExecutionResult
        """
        # 1. 安全檢查
        if not self._is_allowed_command(cli_command):
            return CLIExecutionResult(
                command=cli_command,
                exit_code=-1,
                stdout="",
                stderr=f"Command not in whitelist: {cli_command}",
                duration_seconds=0,
            )
        
        # 2. 路由轉換（aiva xxx → 實際執行命令）
        actual_command = self._route_command(cli_command)
        
        # 3. 執行
        start_time = datetime.now()
        try:
            result = subprocess.run(
                shlex.split(actual_command),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return CLIExecutionResult(
                command=cli_command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                parsed_output=self._try_parse_json(result.stdout),
            )
            
        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            return CLIExecutionResult(
                command=cli_command,
                exit_code=-2,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                duration_seconds=duration,
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return CLIExecutionResult(
                command=cli_command,
                exit_code=-3,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
            )
    
    def _is_allowed_command(self, command: str) -> bool:
        """檢查指令是否在白名單中"""
        return any(command.startswith(prefix) for prefix in self.ALLOWED_PREFIXES)
    
    def _route_command(self, cli_command: str) -> str:
        """將 aiva CLI 指令路由到實際執行命令"""
        for prefix, actual in self.COMMAND_ROUTER.items():
            if cli_command.startswith(prefix):
                # 提取參數部分
                args_part = cli_command[len(prefix):].strip()
                return f"{actual} {args_part}"
        
        # 未找到路由，直接執行
        return cli_command
    
    def _try_parse_json(self, stdout: str) -> Optional[dict]:
        """嘗試將輸出解析為 JSON"""
        try:
            return json.loads(stdout)
        except:
            return None
```

---

## 📊 13 步驟與 CLI 指令對應

| Step | 階段 | AI 輸出的 CLI 指令範例 | 執行方式 |
|------|------|------------------------|----------|
| 0 | URL 輸入 | N/A (用戶輸入) | CLI 入口 |
| 1 | 目標解析 | `aiva recon tech --target example.com` | subprocess |
| 2 | Phase 0 | `aiva scan --target xxx --phase 0 --depth quick` | subprocess |
| 3 | 結果分析 | N/A (AI 內部) | async |
| 4 | 下一步決策 | AI 決定下一個 CLI 指令 | async |
| 5 | Phase 1 策略 | `aiva scan --target xxx --phase 1 --depth deep` | subprocess |
| 6 | 目標選擇 | AI 選擇重點測試目標 | async |
| 7 | 功能測試 | `aiva feature xss --target xxx --payload-level advanced` | subprocess |
| 8 | 結果收集 | 自動從 stdout 收集 | sync |
| 9 | 評估 | AI 評估是否需要更多測試 | async |
| 10 | Phase 2 | `aiva feature sqli --target xxx --bypass waf` | subprocess |
| 11 | 深度測試 | `aiva feature ssrf --target xxx --verify true` | subprocess |
| 12 | 報告生成 | `aiva report generate --session xxx --format json` | subprocess |
| 13 | 完成 | N/A | 返回結果 |

---

## 🔄 13 步驟流程（同步/異步標註）

```
步驟 0:  CLI 入口                    [同步] 啟動
    │
    ▼
步驟 1:  CommanderCoordinator        [同步] 創建 Session
    │
    ▼
步驟 2:  PlanBuilder                 [同步] 生成 AttackPlan
    │
    ▼
步驟 3:  PlanExecutor                [同步] 開始執行
    │
    ▼
步驟 4:  Phase 0 快速偵察            [同步] 調用 scan/ 引擎
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
步驟 5a: Integration 歷史查詢        [同步]    │
    │                                          │
    ▼                                          │
步驟 6:  🧠 AI 決策 ①                [異步] ◄──┘
         decide_phase1_strategy()
    │
    ▼
步驟 7:  Phase 1 深度掃描            [同步] 調用 features/
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
步驟 8a: Integration 歷史查詢        [同步]    │
    │                                          │
    ▼                                          │
步驟 9:  🧠 AI 決策 ②                [異步] ◄──┘
         decide_phase2_targets()
    │
    ▼
步驟 10: Phase 2 攻擊執行            [同步] 調用 features/
    │
    ▼
步驟 11: 🧠 AI 決策 ③                [異步]
         evaluate_phase2_results()
    │
    ▼
步驟 12: 經驗學習                    [同步] 儲存到 RAG
    │
    ▼
步驟 13: 結果返回                    [同步] 回傳給用戶
```

---

## 🏗️ 完整流程：AI 生成 CLI → 執行器執行

```python
# task_planning/executor/plan_executor.py（重構後）

class PlanExecutor:
    """計畫執行器（CLI 驅動版）"""
    
    def __init__(self):
        # AI 指令生成器（異步）
        self.ai_commander = AICommandGenerator(EnhancedDecisionAgent())
        
        # CLI 執行器（同步）
        self.cli_executor = CLIExecutor(working_dir=PROJECT_ROOT)
        
        # 結果解析器
        self.result_parser = ResultParser()
    
    def execute_plan(self, target_url: str) -> PlanExecutionResult:
        """
        執行攻擊計畫（CLI 驅動）
        
        核心循環：
        1. AI 決策（異步）→ 生成 CLI 指令
        2. 執行器（同步）→ subprocess.run(CLI)
        3. 解析結果 → 回饋給 AI
        4. 重複直到 AI 決定完成
        """
        context = {
            "target": target_url,
            "phase": 0,
            "findings": [],
            "executed_commands": [],
        }
        
        while True:
            # 1. AI 決策：生成下一個 CLI 指令 [異步]
            decision = asyncio.run(self.ai_commander.generate_command(context))
            
            # 檢查是否完成
            if decision.command_type == "complete":
                break
            
            # 記錄指令
            logger.info(f"AI 決策: {decision.cli_command}")
            logger.info(f"理由: {decision.reasoning}")
            
            # 2. 執行 CLI 指令 [同步]
            result = self.cli_executor.execute(
                decision.cli_command,
                timeout=decision.timeout
            )
            
            # 3. 解析結果
            parsed = self.result_parser.parse(result)
            
            # 4. 更新上下文，供下一輪 AI 決策
            context["findings"].extend(parsed.get("findings", []))
            context["executed_commands"].append({
                "command": decision.cli_command,
                "result": parsed,
                "duration": result.duration_seconds,
            })
            context["phase"] = self._determine_phase(context)
        
        # 生成最終報告
        return self._build_final_report(context)
    
    def _determine_phase(self, context: dict) -> int:
        """根據已執行指令判斷當前階段"""
        commands = [c["command"] for c in context["executed_commands"]]
        
        if any("--phase 2" in c or "feature" in c for c in commands):
            return 2
        elif any("--phase 1" in c for c in commands):
            return 1
        return 0
```

---

## 📊 完整 CLI 指令規範

### 1. 掃描指令 (aiva scan)

```bash
# 基本格式
aiva scan --target <URL> [OPTIONS]

# 參數
--target      目標 URL（必填）
--phase       執行階段：0（偵察）、1（深度掃描）、2（驗證）、3（報告）
--depth       掃描深度：quick（快速）、normal（正常）、deep（深度）
--threads     並行線程數：1-50
--timeout     單個請求超時（秒）
--follow-redirects  是否跟隨重定向：true/false
--output      輸出格式：json/text/html

# 範例
aiva scan --target https://example.com --phase 0 --depth quick
aiva scan --target https://example.com --phase 1 --depth deep --threads 10
```

### 2. 功能測試指令 (aiva feature)

```bash
# 基本格式
aiva feature <TYPE> --target <URL> [OPTIONS]

# 類型
xss, sqli, ssrf, idor, csrf, xxe, ssti, lfi, rce

# 通用參數
--target          目標 URL（必填）
--payload-level   有效負載級別：basic/advanced/aggressive
--verify          是否驗證：true/false
--evidence        是否收集證據：true/false

# XSS 特定參數
--context         上下文：html/attr/js/url
--bypass          繞過技術：none/encoding/case/comment

# SQLi 特定參數
--db-type         資料庫類型：mysql/postgres/mssql/oracle/sqlite
--technique       技術：union/blind/error/time
--bypass          繞過技術：none/waf/encoding

# SSRF 特定參數
--protocol        協議：http/https/file/gopher/dict
--internal-scan   是否掃描內網：true/false

# 範例
aiva feature xss --target "https://example.com/search?q=" --payload-level advanced
aiva feature sqli --target "https://example.com/user?id=1" --technique union --bypass waf
aiva feature ssrf --target "https://example.com/fetch?url=" --internal-scan true
```

### 3. 偵察指令 (aiva recon)

```bash
# 基本格式
aiva recon <TYPE> --target <DOMAIN> [OPTIONS]

# 類型
subdomain, port, tech, wayback, js-scan, dns, whois

# 參數
--target      目標域名（必填）
--recursive   遞歸掃描：true/false
--wordlist    字典：small/medium/large/custom
--output      輸出格式：json/text

# 範例
aiva recon subdomain --target example.com --recursive true
aiva recon port --target example.com --top 1000
aiva recon tech --target example.com
aiva recon wayback --target example.com --filter "\.js$"
```

### 4. 報告指令 (aiva report)

```bash
# 基本格式
aiva report generate [OPTIONS]

# 參數
--session     會話 ID（必填）
--format      格式：json/html/pdf/md
--include     包含內容：findings/timeline/recommendations/all
--severity    最低嚴重度：info/low/medium/high/critical

# 範例
aiva report generate --session abc123 --format html --include all
```

---

## 📁 需要新增/修改的檔案

### 新增檔案（CLI 薄層入口）

| 檔案路徑 | 用途 | 代碼量 | 優先級 |
|----------|------|--------|--------|
| `cognitive_core/decision/ai_command_generator.py` | AI CLI 指令生成器 | ~200行 | P0 |
| `task_planning/executor/cli_executor.py` | CLI 指令執行器 | ~150行 | P0 |
| `task_planning/executor/result_parser.py` | CLI 輸出解析器 | ~100行 | P0 |
| `features/features_ready/function_xss/cli.py` | XSS CLI 入口 | ~80行 | P0 |
| `features/features_ready/function_sqli/cli.py` | SQLi CLI 入口 | ~80行 | P0 |
| `features/features_ready/function_ssrf/cli.py` | SSRF CLI 入口 | ~80行 | P0 |
| `features/features_ready/function_idor/cli.py` | IDOR CLI 入口 | ~80行 | P0 |
| `integration/coordinators/cli.py` | 報告/協調 CLI 入口 | ~100行 | P1 |

### 修改檔案

| 檔案路徑 | 修改內容 | 影響範圍 | 優先級 |
|----------|----------|----------|--------|
| `task_planning/commander/attack_coordinator.py` | import → CLI 調用 | ~50行改動 | P0 |
| `task_planning/executor/plan_executor.py` | 改為 CLI 驅動循環 | ~100行改動 | P0 |

### 保留不變的檔案 (✅ 100% 保留)

| 檔案路徑 | 行數 | 說明 |
|----------|------|------|
| `integration/coordinators/base_coordinator.py` | 548行 | 雙閉環基類，作為 CLI 內部實現 |
| `integration/coordinators/xss_coordinator.py` | 439行 | XSS 協調器，CLI 內部調用 |
| `features/function_exploit/executor/attack_executor.py` | 608行 | 攻擊編排，CLI 內部調用 |
| `features/smart_detection_manager.py` | 273行 | 檢測器管理，CLI 內部調用 |
| `features/high_value_manager.py` | 366行 | Bug Bounty 管理，CLI 內部調用 |
| `features/exploit_manager.py` | 968行 | 漏洞利用管理，CLI 內部調用 |
| `scan/go_engine/*` | - | 已有 CLI v3.0 |
| `scan/rust_engine/*` | - | 已有 CLI v3.0 |
| `scan/typescript_engine/*` | - | 已有 CLI v3.0 |

---

## 🎯 跨語言支援

### CLI 路由表（語言映射）

```python
LANGUAGE_ROUTER = {
    # Python 工具
    "aiva scan --phase 0": {
        "engine": "rust",
        "command": "./scan/rust_engine/target/release/fast_scanner",
    },
    "aiva scan --phase 1": {
        "engine": "python",
        "command": "python -m services.scan.deep_scanner",
    },
    "aiva feature xss": {
        "engine": "python",
        "command": "python -m services.features.features_ready.function_xss.cli",
    },
    "aiva feature sqli": {
        "engine": "python", 
        "command": "python -m services.features.features_ready.function_sqli.cli",
    },
    
    # Go 工具
    "aiva recon subdomain": {
        "engine": "go",
        "command": "./scan/go_engine/subdomain_enum",
    },
    "aiva recon dns": {
        "engine": "go",
        "command": "./scan/go_engine/dns_resolver",
    },
    
    # Rust 工具
    "aiva recon port": {
        "engine": "rust",
        "command": "./scan/rust_engine/target/release/port_scanner",
    },
    
    # TypeScript/Node 工具
    "aiva recon js-scan": {
        "engine": "node",
        "command": "node ./scan/ts_engine/dist/js_analyzer.js",
    },
}
```

---

## 🔗 現有模組整合方案

### 現有架構對照

```
現有模組                                    CLI 指令對應
────────────────────────────────────────────────────────────────────────

🎯 integration/coordinators/
├── base_coordinator.py (548行)     ──►  內部使用，不直接暴露 CLI
│   ├── 內循環：實時優化                  由 CLI 執行後內部觸發
│   ├── 外循環：報告生成                  aiva report generate --session xxx
│   └── 標準合約：UnifiedVulnerabilityFinding
│
└── xss_coordinator.py (439行)      ──►  aiva feature xss 內部調用
    └── 作為 XSS CLI 的核心實現

⚡ features/function_exploit/executor/
├── attack_executor.py (608行)      ──►  aiva exploit --target xxx --type <type>
│   ├── 並發執行控制                      --concurrency 參數
│   └── 結果回饋生成                      JSON stdout 輸出
│
└── bizlogic_attack_executor.py     ──►  aiva exploit --type bizlogic

🧠 features/*.py (智能管理器)
├── smart_detection_manager.py      ──►  內部調用，由 CLI 統一入口觸發
│   └── 檢測器註冊/執行
│
├── high_value_manager.py (366行)   ──►  aiva bounty --target xxx --platform hackerone
│   └── HackerOne 報告生成                --generate-report 參數
│
└── exploit_manager.py (968行)      ──►  aiva exploit 內部調用
    └── Payload 選擇/執行

🔧 core/task_planning/commander/
└── attack_coordinator.py (596行)   ──►  AI 決策層直接調用（不走 CLI）
    ├── 調用 Features 功能模組            ⚠️ 這裡改為調用 CLI
    └── Phase 2 攻擊流程

🔄 scan/ (純 CLI v3.0)
├── go_engine/                      ──►  aiva recon subdomain|dns
├── rust_engine/                    ──►  aiva scan --phase 0, aiva recon port
├── typescript_engine/              ──►  aiva recon js-scan
└── python_engine/                  ──►  aiva scan --phase 1|2
```

### 整合策略

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   策略：CLI 作為「外殼」，現有模組作為「核心實現」                            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   AI 輸出: "aiva feature xss --target https://xxx --payload advanced"│   │
│   │                                                                     │   │
│   │                            │                                        │   │
│   │                            ▼                                        │   │
│   │   ┌─────────────────────────────────────────────────────────────┐  │   │
│   │   │  CLI 入口 (新增薄層)                                         │  │   │
│   │   │  services/features/features_ready/function_xss/cli.py       │  │   │
│   │   │                                                             │  │   │
│   │   │  def main():                                                │  │   │
│   │   │      args = parse_args()                                    │  │   │
│   │   │      coordinator = XssCoordinator()  # 現有！               │  │   │
│   │   │      result = coordinator.execute(args.target, ...)        │  │   │
│   │   │      print(json.dumps(result))                             │  │   │
│   │   └─────────────────────────────────────────────────────────────┘  │   │
│   │                            │                                        │   │
│   │                            ▼                                        │   │
│   │   ┌─────────────────────────────────────────────────────────────┐  │   │
│   │   │  現有核心實現 (不變)                                         │  │   │
│   │   │  integration/coordinators/xss_coordinator.py (439行)        │  │   │
│   │   │  integration/coordinators/base_coordinator.py (548行)       │  │   │
│   │   │  features/smart_detection_manager.py (273行)                │  │   │
│   │   └─────────────────────────────────────────────────────────────┘  │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   📌 核心原則：現有模組 100% 保留，只新增 CLI 薄層入口                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### CLI 入口檔案結構（新增）

```
services/
├── features/
│   └── features_ready/
│       ├── function_xss/
│       │   ├── worker.py          # 現有 (保留)
│       │   └── cli.py             # 🆕 新增 CLI 入口
│       ├── function_sqli/
│       │   ├── worker.py          # 現有 (保留)
│       │   └── cli.py             # 🆕 新增 CLI 入口
│       ├── function_ssrf/
│       │   ├── worker.py          # 現有 (保留)
│       │   └── cli.py             # 🆕 新增 CLI 入口
│       └── function_idor/
│           ├── worker.py          # 現有 (保留)
│           └── cli.py             # 🆕 新增 CLI 入口
│
├── integration/
│   └── coordinators/
│       ├── base_coordinator.py    # 現有 (保留)
│       ├── xss_coordinator.py     # 現有 (保留)
│       └── cli.py                 # 🆕 新增 CLI 入口 (報告生成等)
│
└── scan/
    ├── go_engine/                 # 現有 CLI v3.0 (已有 CLI)
    ├── rust_engine/               # 現有 CLI v3.0 (已有 CLI)
    ├── typescript_engine/         # 現有 CLI v3.0 (已有 CLI)
    └── python_engine/
        └── cli.py                 # 🆕 新增 CLI 入口 (若缺少)
```

### CLI 入口範例實現

```python
# services/features/features_ready/function_xss/cli.py
"""
XSS 功能測試 CLI 入口

用法:
    python -m services.features.features_ready.function_xss.cli \
        --target "https://example.com/search?q=" \
        --payload-level advanced \
        --bypass encoding

輸出:
    JSON 格式結果到 stdout
"""
import argparse
import json
import sys
from pathlib import Path

# 確保可以 import 現有模組
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from services.integration.coordinators.xss_coordinator import XssCoordinator
from services.features.smart_detection_manager import SmartDetectionManager


def parse_args():
    parser = argparse.ArgumentParser(description="AIVA XSS 功能測試")
    parser.add_argument("--target", required=True, help="目標 URL")
    parser.add_argument("--payload-level", default="advanced", 
                       choices=["basic", "advanced", "aggressive"])
    parser.add_argument("--bypass", default="none",
                       choices=["none", "encoding", "case", "comment", "waf"])
    parser.add_argument("--context", default="auto",
                       choices=["auto", "html", "attr", "js", "url"])
    parser.add_argument("--verify", type=bool, default=True)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 使用現有的 XssCoordinator (439 行的現有實現)
    coordinator = XssCoordinator()
    
    # 使用現有的 SmartDetectionManager (273 行)
    detection_manager = SmartDetectionManager()
    
    try:
        # 執行 XSS 測試
        result = coordinator.execute_xss_test(
            target=args.target,
            payload_level=args.payload_level,
            bypass_technique=args.bypass,
            context_type=args.context,
            verify=args.verify,
            timeout=args.timeout,
        )
        
        # 輸出 JSON 結果
        output = {
            "success": True,
            "target": args.target,
            "findings": result.findings if hasattr(result, 'findings') else [],
            "metadata": {
                "payload_level": args.payload_level,
                "bypass": args.bypass,
                "duration_seconds": result.duration if hasattr(result, 'duration') else 0,
            }
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        sys.exit(0)
        
    except Exception as e:
        error_output = {
            "success": False,
            "target": args.target,
            "error": str(e),
            "findings": [],
        }
        print(json.dumps(error_output, indent=2, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 現有模組調用關係（更新後）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  AI 決策層                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  cognitive_core/decision/ai_command_generator.py                    │   │
│  │  輸出: "aiva feature xss --target https://xxx --payload advanced"   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  CLI 執行層                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  task_planning/executor/cli_executor.py                             │   │
│  │  subprocess.run("python -m services.features...function_xss.cli")  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  CLI 入口層 (🆕 新增薄層)                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  services/features/features_ready/function_xss/cli.py               │   │
│  │  解析參數 → 調用現有模組 → JSON 輸出                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  現有核心實現 (✅ 100% 保留)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  🎯 integration/coordinators/                                       │   │
│  │  ├── base_coordinator.py (548行) - 雙閉環基類                       │   │
│  │  └── xss_coordinator.py (439行) - XSS 特化                          │   │
│  │                                                                     │   │
│  │  ⚡ features/function_exploit/executor/                             │   │
│  │  ├── attack_executor.py (608行) - 攻擊編排                          │   │
│  │  └── bizlogic_attack_executor.py - 業務邏輯攻擊                     │   │
│  │                                                                     │   │
│  │  🧠 features/*.py                                                   │   │
│  │  ├── smart_detection_manager.py (273行) - 檢測器統一管理            │   │
│  │  ├── high_value_manager.py (366行) - Bug Bounty                     │   │
│  │  └── exploit_manager.py (968行) - 漏洞利用管理                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  掃描引擎層 (✅ 已有 CLI v3.0)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  🔄 scan/                                                           │   │
│  │  ├── go_engine/       - 已有 CLI                                    │   │
│  │  ├── rust_engine/     - 已有 CLI                                    │   │
│  │  ├── typescript_engine/ - 已有 CLI                                  │   │
│  │  └── python_engine/   - 需確認 CLI 入口                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### attack_coordinator.py 修改

```python
# core/task_planning/commander/attack_coordinator.py
# 修改：從直接 import 改為 CLI 調用

# ❌ 舊方式（直接 import）
# from services.integration.coordinators.xss_coordinator import XssCoordinator
# coordinator = XssCoordinator()
# result = await coordinator.execute(target)

# ✅ 新方式（CLI 調用）
from task_planning.executor.cli_executor import CLIExecutor

class AttackCoordinator:
    def __init__(self):
        self.cli_executor = CLIExecutor()
    
    def execute_xss_attack(self, target: str, options: dict) -> dict:
        """執行 XSS 攻擊（透過 CLI）"""
        cli_command = (
            f"aiva feature xss "
            f"--target {target} "
            f"--payload-level {options.get('level', 'advanced')} "
            f"--bypass {options.get('bypass', 'none')}"
        )
        result = self.cli_executor.execute(cli_command)
        return result.parsed_output
```

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│   🔑 AI 只需要學習一套 CLI 語法，不需要知道底層是什麼語言              │
│                                                                        │
│   AI 輸出:                                                             │
│   "aiva scan --target https://xxx --phase 0"                          │
│                                                                        │
│   路由器自動選擇最佳引擎:                                              │
│   ├─ Phase 0 快速掃描 → Rust（高性能）                                │
│   ├─ Phase 1 深度掃描 → Python（靈活性）                              │
│   ├─ 子域名枚舉 → Go（並發好）                                        │
│   └─ JS 分析 → TypeScript（生態系）                                   │
│                                                                        │
│   統一 JSON 輸出格式:                                                  │
│   {                                                                    │
│     "success": true,                                                   │
│     "findings": [...],                                                 │
│     "metadata": {...}                                                  │
│   }                                                                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 MQ 使用場景（保留但簡化）

### 保留 MQ 的情況

| 場景 | 使用 MQ | 理由 |
|------|---------|------|
| 跨服務事件通知 | ✅ | 鬆耦合，非阻塞 |
| 多實例負載均衡 | ✅ | 分散式部署時 |
| 結果日誌廣播 | ✅ | 多個消費者（UI、存儲、分析） |

### 不使用 MQ 的情況

| 場景 | 改用方式 | 理由 |
|------|----------|------|
| Core → 任何引擎 | subprocess.run(CLI) | 統一介面，語言無關 |
| Core → Integration 儲存 | CLI 或直接調用 | 立即確認儲存成功 |

---

## 📊 改善效果

| 指標 | 改動前 | 改動後 |
|------|--------|--------|
| 跨語言支援 | 困難（需要不同綁定） | **容易（統一 CLI）** |
| 代碼複雜度 | 高（MQ + 多語言 SDK） | **低（CLI 字串）** |
| 調試難度 | 高（異步追蹤困難） | **低（可重放 CLI）** |
| 擴展性 | 低（需要寫 MQ 綁定） | **高（只需 CLI 入口）** |
| AI 決策可追溯性 | 低 | **高（CLI 日誌可讀）** |

---

## 📝 總結

**核心變更**：
1. **AI 決策** → 輸出 CLI 指令字串（不是 Python 函數調用）
2. **CLI 執行器** → `subprocess.run()` 執行指令
3. **語言路由** → 根據指令自動選擇 Python/Rust/Go/TS 引擎
4. **統一輸出** → 所有引擎輸出 JSON 格式

**現有模組處理**：
- ✅ **100% 保留**：所有現有核心實現（coordinators, managers, executors）
- 🆕 **新增薄層**：每個功能模組新增 ~80 行 CLI 入口
- 🔄 **修改方式**：`attack_coordinator.py` 從 import 改為 CLI 調用

**好處**：
- ✅ 語言無關：AI 只需學一套 CLI 語法
- ✅ 易於擴展：新引擎只需實現 CLI 入口
- ✅ 可追溯：CLI 指令可直接記錄和重放
- ✅ 調試容易：手動執行 CLI 驗證
- ✅ 現有投資保護：3000+ 行核心代碼完全複用
