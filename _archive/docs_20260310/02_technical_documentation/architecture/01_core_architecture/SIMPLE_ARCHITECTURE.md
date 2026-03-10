# 🎯 AIVA 功能模組簡化架構設計

**版本**: v5.0 Final  
**日期**: 2025-12-11  
**核心理念**: 簡單直接，各模組按最佳實踐完善

**更新記錄**:
- v5.0 (2025-12-11): ✅ 添加目錄導航，移除所有過時文檔
- v4.0: 移除適配器層和 JSON 契約
- v3.0: 初始版本（已廢棄）

---

## 📚 目錄

1. [📋 核心設計原則](#-核心設計原則)
   - [1. AI Commander 直接調用 CLI 工具](#1-ai-commander-直接調用-cli-工具)
2. [📦 功能模組各自完善](#-功能模組各自完善)
   - [2. 各模組按功能最佳實踐實現](#2-各模組按功能最佳實踐實現)
     - [function_sqli - SQL 注入檢測](#function_sqli---sql-注入檢測按-sql-注入最佳實踐)
     - [function_xss - XSS 檢測](#function_xss---xss-檢測按-xss-檢測最佳實踐)
     - [function_authn_go - 認證檢測](#function_authn_go---認證檢測按認證測試最佳實踐)
     - [function_crypto - 加密檢測](#function_crypto---加密檢測按密碼學最佳實踐)
   - [關鍵理念：功能決定架構](#關鍵理念功能決定架構)
3. [🔧 實際實施方式](#-實際實施方式)
   - [3. 兩種整合方式](#3-兩種整合方式)
     - [方式 1: AI Commander 調用外部 CLI 工具](#方式-1-ai-commander-調用外部-cli-工具)
     - [方式 2: AI Commander 調用 AIVA 功能模組](#方式-2-ai-commander-調用-aiva-功能模組)
4. [📝 完善功能模組的指導原則](#-完善功能模組的指導原則)
   - [4. 各模組完善檢查清單](#4-各模組完善檢查清單)
     - [✅ Python 模組](#-python-模組sqlixssssrfidor)
     - [✅ Go 模組](#-go-模組認證檢測)
     - [✅ Rust 模組](#-rust-模組加密檢測)
5. [🎯 總結](#-總結)
   - [核心理念：功能優先，架構服務於功能](#核心理念功能優先架構服務於功能)
   - [設計哲學](#設計哲學)
6. [📝 實施檢查清單](#-實施檢查清單)
   - [✅ 功能模組開發者](#-功能模組開發者)
   - [✅ AI Commander 開發者](#-ai-commander-開發者)

---

## 📋 核心設計原則

### 1. AI Commander 直接調用 CLI 工具

**不需要適配器層**，AI Commander 直接用 subprocess 調用：

```python
# services/core/aiva_core/task_planning/security_scanner.py

import asyncio
import subprocess
import json

class SecurityScanner:
    """安全掃描器（直接調用 CLI 工具）"""
    
    async def scan_sqli(self, target: str) -> dict:
        """
        SQL 注入掃描
        
        直接調用 sqlmap CLI，解析輸出
        """
        # 構建命令
        cmd = [
            "sqlmap",
            "-u", target,
            "--batch",
            "--output-dir", "/tmp/aiva_scans",
            "--json",  # 如果工具支持 JSON 輸出
        ]
        
        # 直接執行
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # 解析輸出（每個工具自己處理）
        return self._parse_sqlmap_output(stdout.decode())
    
    def _parse_sqlmap_output(self, output: str) -> dict:
        """解析 sqlmap 輸出"""
        # 簡單解析邏輯
        return {
            "tool": "sqlmap",
            "vulnerable": "is vulnerable" in output.lower(),
            "raw_output": output
        }
    
    async def scan_xss(self, target: str) -> dict:
        """XSS 掃描 - 直接調用 XSStrike"""
        cmd = ["python", "XSStrike/xsstrike.py", "-u", target]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        return self._parse_xsstrike_output(stdout.decode())
```

**關鍵點**:
- ✅ **直接調用** - `subprocess.run()` 或 `asyncio.create_subprocess_exec()`
- ✅ **簡單解析** - 每個工具有自己的解析邏輯
- ✅ **不需要抽象** - 不需要統一的適配器模式

---

## 📦 功能模組各自完善

### 2. 各模組按功能最佳實踐實現

**關鍵原則**:
- ✅ **按功能領域的最佳實踐** - 不是按語言
- ✅ **每個子function_sqli - SQL 注入檢測（按 SQL 注入最佳實踐）

**功能特性決定架構**:
- SQL 注入需要多種檢測技術
- 需要 payload 庫和測試引擎
- 可能需要資料庫指紋識別

```
function_sqli/
├── payloads/          # Payload 資料庫（功能需求）
│   ├── mysql.txt
│   ├── postgresql.txt
│   └── mssql.txt
├── engines/           # 多種檢測引擎（功能需求）
│   ├── boolean_blind.py
│   ├── time_based.py
│   ├── error_based.py
│   └── union_based.py
├── detector.py        # 主檢測類（簡單同步）
│   def scan(target):  # 同步函數
│       return result
└── fingerprint.py     # 資料庫指紋（功能需求）

設計原則:function_xss - XSS 檢測（按 XSS 檢測最佳實踐）

**功能特性決定架構**:
- XSS 需要 DOM 分析和 JavaScript 執行
- 可能需要瀏覽器環境
- Context-aware payload 生成

```
function_xss/
├── contexts/          # Context 分析（功能需求）
│   ├── html_context.py
│   ├── js_context.py
│   └── attr_context.py
├── payloads/          # Context-aware payloads
│   └── generator.py
├── dom4 function_crypto - 加密檢測（按密碼學最佳實踐）

**功能特性決定架構**:
- 密碼學需要大量計算
- 需要數學算法和統計分析
- Rust 提供性能和安全性

```
function_crypto/
├── algorithms/        # 密碼學算法（功能需求）
│   ├── entropy.rs
│   ├── frequency.rs
│   └── patterns.rs
├── detectors/         # 各種檢測器（功能需求）
│   ├── weak_random.rs
│   ├── weak_hash.rs
│   └── weak_cipher.rs
├── rust_core/         # Rust 核心
│   pub fn analyze(data: &[u8]) -> Analysis
└── python_binding/    # PyO3 綁定
    pub fn scan(data: bytes) -> dict

設計原則:
- ✅ 按密碼學分析的最佳實踐組織
- ✅ 與其他模組完全不同（因為是數學密集型）
- ✅ Rust 同步實現（專注於計算）
- ❌ 不需要模仿其他模組
```

---

### 關鍵理念：功能決定架構

**不同功能 = 不同架構**:

| 模組 | 功能特性 | 架構特點 | 不同之處 |
|------|---------|---------|---------|
| **SQLi** | Payload 測試 | 多引擎 + Payload 庫 | 需要資料庫指紋 |
| **XSS** | Context 分析 | DOM 分析 + Context 感知 | 可能需要瀏覽器 |
| **認證** | 協議測試 | 會話管理 + 協議解析 | 需要多種認證流程 |
| **加密** | 數學計算 | 算法 + 統計分析 | 密碼學特有邏輯 |

**共同點**（唯一的統一要求）:
- ✅ 簡單的同步介面（`scan()` 函數）
- ✅ 返回結果字典
- ✅ 錯誤處理清晰
- ❌ 架構不需要統一
**功能特性決定架構**:
- 認證測試需要會話管理
- 多種認證協議（OAuth、JWT、SAML）
- Go 適合高並發請求

```
function_authn_go/
├── protocols/         # 認證協議（功能需求）
│   ├── oauth/
│   ├── jwt/
│   └── saml/
├── session/           # 會話管理（功能需求）
│   └── manager.go
├── scanner/           # 主掃描器（同步）
│   └── auth_scanner.go
│       func Scan(target string) Result
└── main.go            # CLI 入口

設計原則:
- ✅ 按認證測試的最佳實踐組織
- ✅ 與 SQLi/XSS 完全不同（因為是不同功能）
- ✅ Go 簡單同步實現
- ❌ 不需要模仿 Python 模組的結構
#### 2.2 Go 模組（function_authn_go）

```
function_authn_go/
├── cmd/worker/        # Go 風格入口
├── internal/          # Go 內部實現
└── go.mod             # Go 模組管理

設計原則:
- ✅ 使用 Cobra CLI 框架
- ✅ Go 語言慣用結構
- ✅ 獨立編譯為二進制
- ❌ 不需要適配器
```

#### 2.3 Rust 模組（function_crypto）

```
function_crypto/
├── rust_core/         # Rust CLI 實現
│   ├── src/
│   │   └── main.rs    # CLI 入口（clap）
│   ├── Cargo.toml     # 依賴: clap, serde_json
│   └── target/release/
│       └── crypto-scanner  # 編譯的二進制
├── python_wrapper/    # subprocess 橋接層（非 PyO3）
│   └── engine_bridge.py
└── detector/          # Python 業務邏輯
    └── crypto_detector.py

設計原則:
- ✅ Rust 編譯成獨立 CLI 程序
- ✅ 接受 CLI 參數，輸出 JSON
- ✅ Python wrapper 只是 subprocess 調用
- ✅ AI Commander 直接生成 CLI 指令
- ❌ 不使用 PyO3（無需 Python 綁定）
```

---

## 🔧 實際實施方式

### 3. AI 動態生成 CLI 指令控制模組

**核心設計**: AI Commander 不直接調用 Python/Rust/Go 代碼，而是**生成 CLI 指令**

```
┌─────────────────────────────────────────┐
│  AI Commander                           │
│  (services/core/aiva_core)              │
│                                         │
│  1. 分析任務需求                        │
│  2. 生成 CLI 指令                       │
│     (internal_exploration)              │
│  3. 執行 subprocess                     │
│  4. 解析 JSON 輸出                      │
└──────────────┬──────────────────────────┘
               │ 生成的 CLI 指令
               ↓
┌─────────────────────────────────────────┐
│  subprocess.run([                       │
│    "crypto-scanner",                    │
│    "scan",                              │
│    "--code", code                       │
│  ])                                     │
└──────────────┬──────────────────────────┘
               │ 獨立進程
               ↓
┌─────────────────────────────────────────┐
│  Rust CLI Binary                        │
│  (完全獨立的可執行文件)                 │
│                                         │
│  輸出 JSON:                             │
│  [{"issue_type":"WEAK_ALGORITHM",       │
│    "detail":"MD5 detected"}]            │
└─────────────────────────────────────────┘
```

#### 方式 1: AI 調用外部 CLI 工具

**適用於**: sqlmap、XSStrike、nuclei 等已存在的工具

```python
# AI Commander 動態生成指令（不是寫死的）
# services/core/aiva_core/internal_exploration/aiva_cli_implementation.py

class CLICommandGenerator:
    def generate_sqlmap_command(self, target: str) -> list[str]:
        """AI 動態生成 sqlmap 指令"""
        return [
            "sqlmap",
            "-u", target,
            "--batch",
            "--output-dir", "/tmp/aiva_scans"
        ]
    
    def generate_xsstrike_command(self, target: str) -> list[str]:
        """AI 動態生成 XSStrike 指令"""
        return [
            "python",
            "XSStrike/xsstrike.py",
            "-u", target
        ]

# AI Commander 執行生成的指令
async def execute_generated_command(self, cmd: list[str]) -> dict:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    return json.loads(stdout.decode())
```

#### 方式 2: AI 調用 AIVA 功能模組（也是 CLI）

**適用於**: AIVA 自己的功能模組（function_crypto、function_authn_go 等）

**關鍵**: **所有 AIVA 模組都提供 CLI 接口**

```python
# AI Commander 生成 AIVA 模組的 CLI 指令
class CLICommandGenerator:
    def generate_crypto_scan_command(self, code: str) -> list[str]:
        """AI 動態生成 crypto-scanner 指令"""
        return [
            "./services/features/function_crypto/rust_core/target/release/crypto-scanner",
            "scan",
            "--code", code
        ]
    
    def generate_authn_scan_command(self, target: str) -> list[str]:
        """AI 動態生成 Go 認證掃描指令"""
        return [
            "./services/features/function_authn_go/bin/authn-worker",
            "scan",
            "--target", target
        ]

# 執行方式完全相同（統一的 CLI 接口）
async def execute_any_module(self, cmd: list[str]) -> dict:
    process = await asyncio.create_subprocess_exec(*cmd, ...)
    stdout, _ = await process.communicate()
    return json.loads(stdout.decode())
```

**設計優勢**:
- ✅ **無跨語言調用** - 一切都是 CLI + JSON
- ✅ **AI 動態生成指令** - 不需要寫死的調用代碼
- ✅ **模組完全獨立** - Rust/Go/Python 各自編譯運行
- ✅ **統一接口** - 都是 `subprocess + JSON 輸出`
- ✅ **易於測試** - 每個模組的 CLI 都可以獨立測試

**工作流程**:
1. AI 分析任務 → 決定使用哪個模組
2. AI 生成 CLI 指令（參考 `internal_exploration/CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md`）
3. AI 執行 subprocess
4. AI 解析 JSON 輸出
5. AI 整合結果

---

## 📝 完善功能模組的指導原則

### 4. 各模組完善檢查清單

#### ✅ Python 模組（SQLi、XSS、SSRF、IDOR）

**當前狀態**: 已有基礎實現，需要完善

**完善方向**:
1. **檢測邏輯優化**
   - 提升檢測準確度
   - 減少誤報率
   - 支持更多攻擊技術

2. **保持簡單同步**
   - ✅ **不需要 asyncio** - 只要簡單的同步函數
   - ✅ AI Commander 負責異步調度
   - ✅ 功能模組專注於檢測邏輯
   
   ```python
   # 功能模組 - 簡單同步實現
   class SQLInjectionDetector:
       def scan(self, target: str) -> dict:
           """同步檢測函數"""
           # 簡單的檢測邏輯
           result = self._test_injection(target)
           return {"vulnerable": result}
   ```

3. **按功能最佳實踐**
   - 每個模組可能完全不同
   - SQLi 可能用 payload 測試
   - XSS 可能用 DOM 分析
   - 不需要統一架構

4. **可選：命令行介面**
   ```python
   # function_sqli/cli.py (可選)
   import click
   
   @click.command()
   @click.option('-u', '--url', required=True)
   def scan(url):
       """SQL 注入掃描"""
       detector = SQLInjectionDetector()
       result = detector.scan(url)  # 同步調用
       print(json.dumps(result, indent=2))
   ```

#### ✅ Go 模組（認證檢測）

**完善方向**:
1. **Go 慣用結構**
   ```go
   cmd/
   ├── worker/
   │   └── main.go      # 入口點
   internal/
   ├── scanner/
   │   └── auth.go      # 認證掃描邏輯
   ├── config/
   │   └── config.go    # 配置管理
   └── models/
       └── types.go     # 數據類型
   ```

2. **高並發設計**
   - 使用 goroutines
   - channel 通信
   - context 管理

3. **獨立二進制**
   - 可以直接執行
   - 也可以被 AI Commander 調用

#### ✅ Rust 模組（加密檢測）

**完善方向**:
1. **Rust 高性能實現**
   ```rust
   // rust_core/src/detector.rs
   pub struct CryptoDetector {
       // 高性能加密分析
   }
   
   impl CryptoDetector {
       pub fn analyze(&self, data: &[u8]) -> Result<Analysis> {
           // Rust 實現的高效算法
       }
   }
   ```

2. **Python 綁定（PyO3）**
   ```rust
   #[pymodule]
   fn crypto_detector(_py: Python, m: &PyModule) -> PyResult<()> {
       m.add_class::<CryptoDetector>()?;
       Ok(())
   }
   ```

3. **零成本抽象**
   - 編譯時優化
   - 記憶體安全
   - 無運行時開銷

---

## 🎯 總結

### 核心理念：功能優先，架構服務於功能

**簡單直接的調用**:

```
┌─────────────────────────────────────────────────┐
│   AI Commander (唯一的異步層)                   │
│   - asyncio 並發調度                             │
│   - 用 asyncio.to_thread() 包裝同步調用         │
└───────────┬─────────────────────────────────────┘
            │
            ├─→ subprocess → 外部 CLI 工具
            │   - sqlmap (SQL 注入)
            │   - XSStrike (XSS)
            │   - nuclei (多功能掃描)
            │
            └─→ import → AIVA 功能模組（全部同步）
                │
                ├─ function_sqli/
                │  └─ def scan(target): ...
                │     架構：多引擎 + Payload 庫
                │
                ├─ function_xss/
                │  └─ def scan(target): ...
                │     架構：DOM 分析 + Context 感知
                │
                ├─ function_authn_go/
                │  └─ func Scan(target): ...
                │     架構：協議解析 + 會話管理
                │
                └─ function_crypto/
                   └─ pub fn scan(data): ...
                      架構：算法 + 統計分析
```

**每個模組完全不同（因為功能不同）**:

| 關注點 | 說明 |
|--------|------|
| **異步** | ❌ 功能模組不需要<br>✅ 只有 AI Commander 需要 |
| **架構** | ❌ 不需要統一<br>✅ 每個功能的最佳實踐 |
| **接口** | ✅ 簡單同步函數 `scan()`<br>✅ 返回字典結果 |
| **實現** | ✅ SQLi 用 Payload 測試<br>✅ XSS 用 DOM 分析<br>✅ 認證用協議解析<br>✅ 加密用數學計算 |

**不需要的東西**:
- ❌ 通用適配器層（過度抽象）
- ❌ JSON 契約定義（不必要）
- ❌ 統一的架構模式（限制創新）
- ❌ 功能模組的異步（複雜化）

**需要做的事情**:
- ✅ **AI Commander**: 異步調度 + subprocess/import
- ✅ **功能模組**: 簡單同步 + 按功能最佳實踐
- ✅ **清晰接口**: `scan()` 函數 + 字典返回
- ✅ **各自完善**: 專注於功能本身的邏輯

**設計哲學**:
> 「各模組按照最能發揮能力的架構完善，只需要符合使用的語言規範及能跟整個程式構聯即可」
> 
> 「功能決定架構，不同功能不同架構」
> 
> 「異步只在 AI Commander，功能模組保持簡單同步」

---

## 📝 實施檢查清單

### ✅ 功能模組開發者

開發新功能模組時，只需考慮：

1. **功能需求**
   - [ ] 這個功能需要什麼樣的檢測技術？
   - [ ] 需要什麼數據結構和算法？
   - [ ] 參考這個功能領域的最佳實踐

2. **簡單接口**
   - [ ] 提供簡單的同步 `scan()` 函數
   - [ ] 返回清晰的字典結果
   - [ ] 不需要考慮異步

3. **專注功能**
   - [ ] 不要模仿其他模組的架構
   - [ ] 按這個功能的最佳方式實現
   - [ ] 不需要適配器、契約等抽象層

### ✅ AI Commander 開發者

整合功能模組時：

1. **調用方式**
   - [ ] 外部工具用 `subprocess`
   - [ ] AIVA 模組用 `import`
   - [ ] 同步函數用 `asyncio.to_thread()` 包裝

2. **異步調度**
   - [ ] 用 `asyncio.gather()` 並行執行
   - [ ] 處理超時和錯誤
   - [ ] 聚合結果

3. **簡單直接**
   - [ ] 不需要適配器層
   - [ ] 直接調用功能模組
   - [ ] 專注於調度邏輯

---

**這才是正確的架構！** 🎉

**簡單、直接、實用、靈活！**
