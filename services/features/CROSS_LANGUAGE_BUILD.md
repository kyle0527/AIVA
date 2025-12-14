# 跨語言模組編譯指南

**日期**: 2025-12-11  
**原則**: 有問題就修復，不繞過問題

---

## 🔴 重要聲明

**AIVA 不提供 Python 回退實現！**

- ✅ Go 模組就必須用 Go - 編譯 Go 引擎
- ✅ Rust 模組就必須用 Rust - 編譯 Rust 核心
- ❌ 不會因為編譯問題而降級到 Python
- ✅ 遇到編譯問題請按照本指南修復

**理由**：
1. 保持模組的核心優勢（性能、安全性）
2. 避免功能不完整的"方便"實現誤導使用
3. 明確報錯比默默降級更負責任

---

## 📦 AIVA 跨語言設計：全部使用 CLI 模式

**核心理念**: AI Commander 動態生成 CLI 指令，無跨語言調用問題

```
┌────────────────────────────────┐
│  AI Commander                  │
│  動態生成 CLI 指令              │
│  (internal_exploration)        │
└───────────┬────────────────────┘
            │ 生成指令
            ↓
┌────────────────────────────────┐
│  subprocess.run([              │
│    "crypto-scanner",           │
│    "scan", "--code", "..."     │
│  ])                            │
└───────────┬────────────────────┘
            │ 獨立進程
            ↓
┌────────────────────────────────┐
│  Rust/Go/Python CLI            │
│  輸出標準 JSON                  │
└────────────────────────────────┘
```

### 統一的 CLI 接口設計

**所有 AIVA 模組都提供 CLI 接口**：

| 模組 | 語言 | CLI 命令範例 |
|------|------|-------------|
| **function_crypto** | Rust | `crypto-scanner scan --code "..."` |
| **function_authn_go** | Go | `authn-worker scan --target "..."` |
| **rust_engine** | Rust | `aiva-info-gatherer scan --url "..."` |

**特點**：
- ✅ 統一的 JSON 輸出格式
- ✅ 可獨立測試和運行
- ✅ AI 動態生成指令，無硬編碼
- ✅ 無跨語言調用問題
- ✅ 模組完全獨立

---

## 📦 function_crypto - Rust CLI 編譯

### 前置要求
- ✅ Rust 已安裝（rustc, cargo）

### 編譯步驟

#### 1. 編譯為獨立 CLI 程序
```bash
cd services/features/function_crypto/rust_core

# 開發模式（快速編譯）
cargo build

# 發布模式（優化編譯）
cargo build --release
```

**編譯輸出**：
- 開發: `target/debug/crypto-scanner`
- 發布: `target/release/crypto-scanner`

#### 2. 驗證編譯
```bash
# 檢查版本
./target/release/crypto-scanner --version

# 測試掃描
./target/release/crypto-scanner scan --code "md5(password)"
```

**預期輸出** (JSON):
```json
[
  {
    "issue_type": "WEAK_ALGORITHM",
    "detail": "Detected usage of MD5 algorithm"
  }
]
```

### AI Commander 如何使用

**AI 動態生成指令**（不是寫死的）：

```python
# services/core/aiva_core/internal_exploration/

class CLICommandGenerator:
    def generate_crypto_scan_command(self, code: str) -> list[str]:
        """AI 動態生成指令"""
        return [
            "./services/features/function_crypto/rust_core/target/release/crypto-scanner",
            "scan",
            "--code", code
        ]

# AI 執行生成的指令
async def execute_command(self, cmd: list[str]) -> dict:
    process = await asyncio.create_subprocess_exec(*cmd, ...)
    stdout, _ = await process.communicate()
    return json.loads(stdout.decode())
```

### Python Wrapper 的作用

**只是 subprocess 橋接層**（不是 PyO3 綁定）：

```python
# services/features/function_crypto/python_wrapper/engine_bridge.py

def scan_code(code: str) -> List[Tuple[str, str]]:
    """調用 Rust CLI"""
    result = subprocess.run(
        ["./rust_core/target/release/crypto-scanner", "scan", "--code", code],
        capture_output=True,
        text=True
    )
    findings = json.loads(result.stdout)
    return [(f["issue_type"], f["detail"]) for f in findings]
```

**為何需要這層？**
- ✅ 統一錯誤處理（如果 CLI 未編譯）
- ✅ 數據格式轉換（JSON → Python tuple）
- ✅ 方便 Python 代碼調用（可選，AI 可直接生成 CLI）

---

## 📦 function_authn_go - Go 引擎編譯（CLI 模式）

### 前置要求
- ✅ Go 已安裝（go version）

### 編譯步驟

#### 編譯為可執行文件

**Linux/macOS**:
```bash
cd services/features/function_authn_go
go build -o bin/authn-worker cmd/worker/main.go
chmod +x bin/authn-worker
```

**Windows**:
```bash
cd services/features/function_authn_go
go build -o bin/authn-worker.exe cmd/worker/main.go
```

#### 驗證
```bash
# 測試執行
./bin/authn-worker --version
```

### Python 調用方式（subprocess）

```python
import subprocess
import json

def scan_authentication(target: str, options: dict = None) -> dict:
    """調用 Go 引擎進行認證測試"""
    options = options or {}
    
    # 構建命令
    cmd = ['./bin/authn-worker', 'scan', target]
    if options.get('username'):
        cmd.extend(['--username', options['username']])
    
    # 執行並解析 JSON
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Go 引擎失敗: {result.stderr}")
    
    return json.loads(result.stdout)
```

**為何需要這個調用方式？**
- Go 是獨立程序，不能直接 `import`
- JSON 是語言無關的數據交換格式
- subprocess 是標準跨進程通信方式

---

## 🔧 集成到 AI Commander

### function_crypto 集成

```python
# services/core/aiva_core/task_planning/security_scanner.py

from services.features.function_crypto.detector.crypto_detector import CryptoDetector
import asyncio

class SecurityScanner:
    async def analyze_crypto(self, code: str, task_id: str, scan_id: str):
        """密碼學分析（異步調度）"""
        detector = CryptoDetector()
        
        # 用 asyncio.to_thread() 包裝同步調用
        findings = await asyncio.to_thread(
            detector.detect, code, task_id, scan_id
        )
        
        return findings
```

### function_authn_go 集成

```python
import subprocess
import json
import asyncio

class SecurityScanner:
    async def test_authentication(self, target: str, options: dict):
        """認證測試（異步調度）"""
        def run_go_engine():
            result = subprocess.run(
                ['./bin/authn-worker', 'scan', target],
                capture_output=True,
                text=True,
                timeout=30
            )
            return json.loads(result.stdout)
        
        # 用 asyncio.to_thread() 包裝 subprocess 調用
        findings = await asyncio.to_thread(run_go_engine)
        return findings
```

---

## 📝 檢查清單

### 編譯前檢查
- [ ] Rust 工具鏈已安裝（carg（異步調度）

### function_crypto 集成（直接 import）

```python
# services/core/aiva_core/task_planning/security_scanner.py

from services.features.function_crypto.detector.crypto_detector import CryptoDetector
import asyncio

class SecurityScanner:
    async def analyze_crypto(self, code: str, task_id: str, scan_id: str):
        """密碼學分析（異步調度）"""
        detector = CryptoDetector()
        
        # 用 asyncio.to_thread() 包裝同步調用
        findings = await asyncio.to_thread(
            detector.detect, code, task_id, scan_id
        )
        
        return findings
```

### function_authn_go 集成（subprocess）

```python
import subprocess
import json
import asyncio

class SecurityScanner:
    async def test_authentication(self, target: str, options: dict):
        """認證測試（異步調度）"""
        def run_go_engine():
            result = subprocess.run(
                ['./bin/authn-worker', 'scan', target],
                capture_output=True,
                text=True,
                timeout=30
            )
            return json.loads(result.stdout)
        
        # 用 asyncio.to_thread() 包裝 subprocess 調用
        findings = await asyncio.to_thread(run_go_engine)
        return findings
```

---

## 📝 快速檢查清單

### 編譯完成檢查
```bash
# Rust 模組
python -c "import crypto_engine; print('✅ Rust OK')"

# Go 程序
./bin/authn-worker --version && echo "✅ Go OK"
```

### 生產部署檢查
- [ ] 相同環境編譯相同版本
- [ ] 二進制文件有執行權限（Linux/macOS）