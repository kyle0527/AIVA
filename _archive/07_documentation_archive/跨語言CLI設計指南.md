# 跨語言 CLI 統一架構設計方案

> **⚠️ 文檔狀態**: 📚 **參考用 - JSON 合約標準**  
> **兼容性**: ✅ 與「AI 排序器方案」100% 兼容  
> **重要**: 本文件的標準 JSON 合約定義仍然有效，所有 CLI 模組應遵循此標準

**設計日期**: 2026年1月11日  
**目標**: 統一 Features/Scan 模組的 CLI 指令方式，實現 Python 主控、跨語言協同  
**應用場景**: HackerOne 對外黑盒測試

---

## 📊 你的想法分析

### 核心概念

```
┌─────────────────────────────────────────────────┐
│         Python 主控層 (Integration)             │
│  ┌───────────────────────────────────────────┐  │
│  │  CLI 指令發送                             │  │
│  │  ├─ 執行Flow.bat 11 (XSS 掃描)           │  │
│  │  ├─ 執行Flow.bat 50 (SQL 注入)           │  │
│  │  └─ 執行Flow.bat 72 (Rust 快速掃描)      │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                           │
│  ┌───────────────────────────────────────────┐  │
│  │  接收返回資料 (JSON 統一格式)            │  │
│  │  ├─ Python → 直接處理                    │  │
│  │  ├─ Rust   → JSON 返回                   │  │
│  │  ├─ Go     → JSON 返回                   │  │
│  │  └─ TS     → JSON 返回                   │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                           │
│  ┌───────────────────────────────────────────┐  │
│  │  AI 直接處理 (實際架構 ✅)                │  │
│  │  ├─ json.loads(stdout)                   │  │
│  │  ├─ 解析 findings                        │  │
│  │  ├─ 提取 telemetry 學習                  │  │
│  │  └─ 整合結果返回                         │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## ✅ 實際架構驗證（代碼審查完成）

### 1. AI 當前使用的通訊方式

**實際情況**（capability_orchestrator.py 已驗證）：

| 方式 | 使用狀態 | 優點 | 實際用途 |
|------|---------|------|----------|
| **A. CLI + JSON** | ✅ **正在使用** | 簡單、通用 | AI 主要通訊方式 |
| **B. gRPC + Protobuf** | ⚠️ 僅框架存在 | 高效、類型安全 | 未被 AI 實際使用 |
| **C. FFI (Rust)** | ⚠️ 僅框架存在 | 最快 | 未被 AI 實際使用 |

### 2. 當前 Scan 模組的實現

```python
# services/scan/coordinators/multi_engine_coordinator.py
class MultiEngineCoordinator:
    async def run_rust_engine(self, targets):
        # 方式 A: CLI + JSON
        cmd = [
            "cargo", "run", "--release", 
            "--manifest-path", rust_path,
            "--", "scan", "--target", target
        ]
        result = await subprocess_run(cmd)
        return json.loads(result.stdout)  # ← JSON 返回
    
    async def run_python_engine(self, targets):
        # 方式內建: 直接調用 Python 類別
        scanner = PythonScanner()
        return scanner.scan(targets)  # ← Python 對象返回
```

### 3. 當前 Features 模組的實現

```python
# services/features/function_xss/*.py
class XSSScanner:
    def scan(self, target):
        # 直接 Python 實現
        return {
            "findings": [...],
            "status": "completed"
        }
```

**問題**: Features 沒有統一的 CLI 接口！

---

## 🎯 統一架構設計方案

### 方案對比

#### 方案 A: 純 CLI + JSON (推薦用於你的場景)

**優點**:
- ✅ 最簡單，不需要額外服務
- ✅ 所有語言都支援 JSON
- ✅ 容易調試（直接看 stdout）
- ✅ 適合 HackerOne 測試（一次性任務）

**缺點**:
- ⚠️ 每次都要啟動進程（~100ms overhead）
- ⚠️ 不適合高頻調用

**實現**:
```python
# 統一的 CLI 調用接口
class UnifiedCLIExecutor:
    async def execute(self, module, action, params):
        """統一執行 CLI 命令
        
        Args:
            module: "xss" | "sqli" | "rust_scan" | "go_ssrf"
            action: "scan" | "exploit" | "verify"
            params: 參數字典
        
        Returns:
            標準化的 JSON 結果
        """
        cmd = self._build_command(module, action, params)
        result = await subprocess.run(cmd, capture_output=True)
        
        # 所有模組必須返回標準 JSON 格式
        return json.loads(result.stdout)
```

#### 方案 B: gRPC + Protobuf (推薦用於生產環境)

**優點**:
- ✅ 高效能（無進程啟動開銷）
- ✅ 類型安全（Protobuf schema）
- ✅ 支援雙向流
- ✅ 適合微服務架構

**缺點**:
- ⚠️ 需要維護服務常駐
- ⚠️ 需要定義 .proto 文件
- ⚠️ 調試較複雜

**實現**:
```python
# 使用現有的 CrossLanguageService
from aiva_common.cross_language import CrossLanguageService

service = CrossLanguageService()
await service.initialize()

# 調用 Rust 引擎
result = await service.call(
    language="rust",
    service="scan_engine",
    method="fast_scan",
    request={"targets": ["example.com"]}
)
# result 已經是 Python 字典（自動反序列化）
```

---

## 🔧 為你的場景設計的最佳方案

### 推薦: 混合方案（CLI 主導 + JSON 標準化）

**架構**:

```
┌─────────────────────────────────────────────────┐
│    HackerOne 黑盒測試流程                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Python Integration 層 (主控)                   │
│                                                 │
│  class HackerOneTestOrchestrator:               │
│      async def execute_test_suite(target):      │
│          # 1. 發送 CLI 指令                     │
│          xss_result = await cli.execute(        │
│              "xss", "scan", {"url": target}     │
│          )                                      │
│                                                 │
│          # 2. 接收 JSON 結果（統一格式）        │
│          sqli_result = await cli.execute(       │
│              "sqli", "scan", {"url": target}    │
│          )                                      │
│                                                 │
│          # 3. 整合結果                          │
│          return coordinator.integrate_results(   │
│              [xss_result, sqli_result]          │
│          )                                      │
└─────────────────────────────────────────────────┘
                    ↓
┌──────────────┬──────────────┬──────────────┬────┐
│ Python XSS   │ Rust Scanner │ Go SSRF      │ TS │
│              │              │              │    │
│ CLI: xss.py  │ CLI: cargo   │ CLI: go run  │... │
│ 返回: JSON   │ 返回: JSON   │ 返回: JSON   │    │
└──────────────┴──────────────┴──────────────┴────┘
```

### 實現步驟

#### Step 1: 定義標準 JSON 合約

```json
{
  "契約名稱": "AIVA Unified Test Result v1.0",
  "說明": "所有測試模組必須返回此格式",
  
  "標準格式": {
    "status": "completed | failed | timeout",
    "module": "xss | sqli | rust_scan | go_ssrf | ...",
    "target": "https://example.com",
    "execution_time": 1.234,
    "timestamp": "2026-01-11T10:30:00Z",
    
    "findings": [
      {
        "id": "uuid",
        "type": "vulnerability_type",
        "severity": "critical | high | medium | low | info",
        "confidence": 0.95,
        "title": "XSS in search parameter",
        "description": "...",
        "affected_url": "https://example.com/search?q=<script>",
        "evidence": {
          "request": "...",
          "response": "...",
          "payload": "<script>alert(1)</script>"
        },
        "remediation": "...",
        "references": ["CWE-79", "OWASP-A03"]
      }
    ],
    
    "metadata": {
      "scan_id": "uuid",
      "total_requests": 100,
      "error_count": 0
    }
  }
}
```

#### Step 2: 為每個模組創建 CLI 入口

**Python 模組** (已有):
```python
# services/features/function_xss/cli.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["scan", "exploit"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--params", type=json.loads)
    
    args = parser.parse_args()
    
    scanner = XSSScanner()
    result = scanner.scan(args.target, **args.params)
    
    # 輸出標準 JSON 格式
    print(json.dumps(result, ensure_ascii=False))
```

**Rust 模組** (需要添加):
```rust
// services/scan/rust_engine/src/cli.rs
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Serialize)]
struct StandardResult {
    status: String,
    module: String,
    target: String,
    execution_time: f64,
    findings: Vec<Finding>,
    metadata: Metadata,
}

fn main() {
    let args = parse_args();
    let result = run_scan(&args);
    
    // 輸出標準 JSON 格式
    println!("{}", serde_json::to_string(&result).unwrap());
}
```

**Go 模組** (需要添加):
```go
// services/scan/go_engine/cmd/cli/main.go
type StandardResult struct {
    Status        string    `json:"status"`
    Module        string    `json:"module"`
    Target        string    `json:"target"`
    ExecutionTime float64   `json:"execution_time"`
    Findings      []Finding `json:"findings"`
    Metadata      Metadata  `json:"metadata"`
}

func main() {
    args := parseArgs()
    result := runScan(args)
    
    // 輸出標準 JSON 格式
    json, _ := json.Marshal(result)
    fmt.Println(string(json))
}
```

#### Step 3: 創建統一的 Python CLI 執行器

```python
# services/integration/cli_executor.py
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

class UnifiedCLIExecutor:
    """統一 CLI 執行器 - 用於 HackerOne 測試"""
    
    def __init__(self):
        self.module_configs = {
            "xss": {
                "command": ["python", "services/features/function_xss/cli.py"],
                "timeout": 30
            },
            "sqli": {
                "command": ["python", "services/features/function_sqli/cli.py"],
                "timeout": 30
            },
            "rust_scan": {
                "command": ["cargo", "run", "--release", 
                           "--manifest-path", "services/scan/rust_engine/Cargo.toml"],
                "timeout": 10
            },
            "go_ssrf": {
                "command": ["go", "run", "services/scan/go_engine/cmd/cli/main.go"],
                "timeout": 15
            },
            "ts_spider": {
                "command": ["npm", "run", "cli", "--prefix", "services/scan/typescript_engine"],
                "timeout": 60
            }
        }
    
    async def execute(
        self, 
        module: str, 
        action: str, 
        params: Dict[str, Any],
        timeout: int = None
    ) -> Dict[str, Any]:
        """執行 CLI 命令並返回標準化結果
        
        Args:
            module: 模組名稱 (xss, sqli, rust_scan, go_ssrf, ts_spider)
            action: 動作 (scan, exploit, verify)
            params: 參數字典
            timeout: 超時時間（秒）
            
        Returns:
            標準化的 JSON 結果
        """
        config = self.module_configs.get(module)
        if not config:
            raise ValueError(f"Unknown module: {module}")
        
        # 構建命令
        cmd = config["command"] + [
            "--action", action,
            "--params", json.dumps(params)
        ]
        
        # 執行命令
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout or config["timeout"]
            )
        except asyncio.TimeoutError:
            process.kill()
            return {
                "status": "timeout",
                "module": module,
                "error": f"Timeout after {timeout}s"
            }
        
        # 解析返回結果
        if process.returncode == 0:
            try:
                result = json.loads(stdout.decode('utf-8'))
                return self._validate_result(result)
            except json.JSONDecodeError as e:
                return {
                    "status": "failed",
                    "module": module,
                    "error": f"Invalid JSON: {e}"
                }
        else:
            return {
                "status": "failed",
                "module": module,
                "error": stderr.decode('utf-8')
            }
    
    def _validate_result(self, result: Dict) -> Dict:
        """驗證結果格式是否符合標準"""
        required_fields = ["status", "module", "target", "findings"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        return result
    
    async def execute_parallel(
        self, 
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """並行執行多個任務
        
        Args:
            tasks: 任務列表 [{"module": "xss", "action": "scan", "params": {...}}, ...]
            
        Returns:
            結果列表
        """
        coroutines = [
            self.execute(task["module"], task["action"], task["params"])
            for task in tasks
        ]
        return await asyncio.gather(*coroutines, return_exceptions=True)
```

#### Step 4: HackerOne 測試編排器（簡化版，無 Coordinator）

```python
# services/integration/hackerone_orchestrator.py
from typing import Dict, Any, List
from .cli_executor import UnifiedCLIExecutor

class HackerOneTestOrchestrator:
    """HackerOne 黑盒測試編排器（直接處理 JSON）"""
    
    def __init__(self):
        self.cli_executor = UnifiedCLIExecutor()
        # ❌ 不需要 Coordinator - AI 已直接處理 JSON
    
    async def run_comprehensive_test(
        self, 
        target: str,
        test_types: List[str] = None
    ) -> Dict[str, Any]:
        """執行完整的黑盒測試套件
        
        Args:
            target: 目標 URL
            test_types: 測試類型列表，None 表示全部
            
        Returns:
            整合的測試報告
        """
        if test_types is None:
            test_types = ["xss", "sqli", "rust_scan", "go_ssrf", "ts_spider"]
        
        # Step 1: 並行執行所有測試
        tasks = [
            {
                "module": test_type,
                "action": "scan",
                "params": {"target": target}
            }
            for test_type in test_types
        ]
        
        raw_results = await self.cli_executor.execute_parallel(tasks)
        
        # Step 2: 直接處理 JSON 結果（無需 Coordinator）
        processed_results = []
        for raw_result in raw_results:
            if isinstance(raw_result, Exception):
                continue
            
            # ✅ 直接使用 JSON 數據（AI 方式）
            processed_results.append(raw_result)
        
        # Step 3: 生成 HackerOne 格式報告
        return self._generate_hackerone_report(processed_results)
    
    def _generate_hackerone_report(
        self, 
        results: List[Dict]
    ) -> Dict[str, Any]:
        """生成 HackerOne 提交格式的報告"""
        high_value_findings = []
        all_findings = []
        
        for result in results:
            findings = result.get("findings", [])
            for finding in findings:
                all_findings.append(finding)
                if finding.get("severity") in ["critical", "high"]:
                    high_value_findings.append(finding)
        
        return {
            "target": results[0]["target"] if results else "",
            "test_date": datetime.now().isoformat(),
            "total_findings": len(all_findings),
            "critical_findings": len([f for f in all_findings if f["severity"] == "critical"]),
            "high_findings": len([f for f in all_findings if f["severity"] == "high"]),
            
            # HackerOne 格式的發現
            "vulnerabilities": [
                self._format_for_hackerone(f) 
                for f in high_value_findings
            ],
            
            # 完整的技術細節（內部使用）
            "detailed_results": results
        }
    
    def _format_for_hackerone(self, finding: Dict) -> Dict:
        """轉換為 HackerOne 提交格式"""
        return {
            "title": finding["title"],
            "vulnerability_information": finding["description"],
            "steps_to_reproduce": self._extract_steps(finding),
            "impact": self._calculate_impact(finding),
            "proof_of_concept": finding["evidence"],
            "severity_rating": finding["severity"],
            "weakness": finding["references"]
        }
```

---

## 💡 回答你的問題

### Q: 不同語言回傳的也是語言嗎？

**A**: 不是！所有語言都返回 **JSON 字串**。

```python
# Rust 返回 JSON 字串
stdout = b'{"status": "completed", "findings": [...]}'
result = json.loads(stdout)  # → Python 字典

# Go 返回 JSON 字串
stdout = b'{"status": "completed", "findings": [...]}'
result = json.loads(stdout)  # → Python 字典

# TypeScript 返回 JSON 字串
stdout = b'{"status": "completed", "findings": [...]}'
result = json.loads(stdout)  # → Python 字典
```

**關鍵**: JSON 是通用格式，所有語言都支援！

### Q: 有簡單方式可以轉成 Python 嗎？

**A**: 非常簡單！只需要：

```python
import json

# 從任何語言接收 JSON 字串
json_string = subprocess_output.decode('utf-8')

# 轉成 Python 字典
python_dict = json.loads(json_string)

# 現在可以像操作 Python 對象一樣使用
for finding in python_dict["findings"]:
    print(finding["severity"])
```

### Q: 在 HackerOne 對外黑盒測試時？

**A**: 完美適用！建議流程：

```python
# 1. 發送測試任務
orchestrator = HackerOneTestOrchestrator()

# 2. 執行完整測試套件（自動調用所有語言的引擎）
report = await orchestrator.run_comprehensive_test(
    target="https://target.com"
)

# 3. 自動生成 HackerOne 格式報告
hackerone_submission = report["vulnerabilities"]

# 4. 提交到 HackerOne
await submit_to_hackerone(hackerone_submission)
```

---

## 🚀 實施計劃

### Phase 1: 標準化 (1-2 天)

1. 定義標準 JSON 合約文檔
2. 為每個語言創建 JSON schema 驗證
3. 實現 `UnifiedCLIExecutor`

### Phase 2: 模組改造 (3-5 天)

1. 為所有 Python Features 添加 CLI 入口
2. 為 Rust/Go/TS 引擎添加標準 JSON 輸出
3. 測試每個模組的 CLI 接口

### Phase 3: 整合測試 (2-3 天)

1. 實現 `HackerOneTestOrchestrator`
2. 並行執行測試
3. 生成 HackerOne 格式報告

### Phase 4: 優化 (1-2 天)

1. 添加錯誤重試機制
2. 優化並行執行效能
3. 添加進度監控

---

## 📊 預期效果

| 指標 | 當前 | 改進後 |
|------|------|--------|
| 跨語言調用 | ⚠️ 各自實現 | ✅ 統一 CLI |
| 返回格式 | ⚠️ 不一致 | ✅ 標準 JSON |
| Python 整合 | ⚠️ 手動處理 | ✅ 自動化 |
| HackerOne 提交 | ⚠️ 手動整理 | ✅ 一鍵生成 |
| 測試效率 | ⚠️ 串行執行 | ✅ 並行執行 |

---

**總結**: 你的想法非常正確！使用 CLI + JSON 的方式最適合 HackerOne 測試場景。所有語言都返回 JSON 字串，Python 用 `json.loads()` 就能輕鬆轉換，然後通過 Integration Coordinator 處理即可。
