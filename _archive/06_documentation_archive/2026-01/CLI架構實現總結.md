# CLI 架構實現總結 - 規劃與現實對照


## 📑 目錄

- [🎯 您的 CLI 規劃確認](#-您的-cli-規劃確認)
  - [📚 核心規劃文件](#-核心規劃文件)
- [🔍 您的設計哲學](#-您的設計哲學)
  - [統一的 CLI 接口格式](#統一的-cli-接口格式)
  - [統一的參數規範](#統一的參數規範)
- [💡 CLI 與直接導入的關係 - 您的發現是對的！](#-cli-與直接導入的關係-您的發現是對的)
  - [執行流程對比](#執行流程對比)
  - [本質關係圖](#本質關係圖)
- [🎯 為什麼規劃 CLI 而不是直接導入？](#-為什麼規劃-cli-而不是直接導入)
  - [您的 4 個設計理由](#您的-4-個設計理由)
    - [1️⃣ **統一接口** ✅](#1-統一接口-)
    - [2️⃣ **多語言支持** ✅](#2-多語言支持-)
    - [3️⃣ **進程隔離** ✅](#3-進程隔離-)
    - [4️⃣ **版本管理** ✅](#4-版本管理-)
- [📊 三種執行方式對比表](#-三種執行方式對比表)
- [🎯 實際實現狀態](#-實際實現狀態)
  - [已實現的 CLI 工具](#已實現的-cli-工具)
    - [1. aiva_cli_implementation.py ✅](#1-aiva_cli_implementationpy-)
    - [2. aiva_external_module_cli.py ✅](#2-aiva_external_module_clipy-)
    - [3. aiva_capability_cli.py ✅](#3-aiva_capability_clipy-)
  - [外部模組的 CLI 接口](#外部模組的-cli-接口)
    - [function_xss/__main__.py ✅](#function_xss__main__py-)
    - [function_bizlogic/__main__.py ✅](#function_bizlogic__main__py-)
- [🎯 您規劃的三層架構](#-您規劃的三層架構)
  - [各層職責](#各層職責)
    - [Layer 1: AI 核心](#layer-1-ai-核心)
    - [Layer 2a: CommandHandler（優先）](#layer-2a-commandhandler優先)
    - [Layer 2b: CLI 執行器（跨語言）](#layer-2b-cli-執行器跨語言)
    - [Layer 2c: 直接導入（備用）](#layer-2c-直接導入備用)
- [💡 最終建議：混合架構](#-最終建議混合架構)
  - [根據場景選擇執行方式](#根據場景選擇執行方式)
  - [執行方式決策樹](#執行方式決策樹)
- [📝 總結](#-總結)
  - [您的觀察完全正確！](#您的觀察完全正確)
  - [CLI 的價值在於](#cli-的價值在於)
  - [最佳實踐](#最佳實踐)
  - [您的規劃是對的！](#您的規劃是對的)

---
## 🎯 您的 CLI 規劃確認

根據找到的文件，您確實朝著 **CLI + 參數** 的方向規劃：

### 📚 核心規劃文件

1. **aiva_cli_implementation.py** - 內部模組 CLI 執行器
2. **aiva_external_module_cli.py** - 外部模組 CLI 執行器  
3. **aiva_capability_cli.py** - 能力查詢與執行 CLI
4. **MULTILANG_TOOL_UNIFICATION_PLAN.md** - 多語言工具統一規範

## 🔍 您的設計哲學

### 統一的 CLI 接口格式

```bash
# 內部模組（Python）
python -m aiva_core.internal_exploration.aiva_cli_implementation --flow <id>

# 外部模組（多語言）
python aiva_external_module_cli.py --lang python --flow <id>
python aiva_external_module_cli.py --lang rust --func <function>
python aiva_external_module_cli.py --lang go --func <function>

# 能力查詢
python aiva_capability_cli.py --search <keyword>
python aiva_capability_cli.py --flow <id>
```

### 統一的參數規範

| 參數 | 短參數 | 說明 | 範例 |
|------|--------|------|------|
| `--input` | `-i` | 輸入目錄或文件 | `--input ./src` |
| `--output` | `-o` | 輸出目錄 | `--output ./analysis` |
| `--target` | `-t` | 目標URL或服務 | `--target https://example.com` |
| `--mode` | `-m` | 執行模式 | `--mode pipeline` |
| `--format` | `-f` | 輸出格式 | `--format json` |
| `--verbose` | `-v` | 詳細輸出 | `--verbose` |
| `--flow` | | Flow ID | `--flow 313` |
| `--dry-run` | | 預覽執行計劃 | `--dry-run` |

## 💡 CLI 與直接導入的關係 - 您的發現是對的！

### 執行流程對比

```python
# ==========================================
# 方式 1：CLI 執行（您的規劃）
# ==========================================

# 步驟 1：AI 調用 CLI
subprocess.run([
    "python", 
    "aiva_external_module_cli.py",
    "--lang", "python",
    "--flow", "313"
])

# 步驟 2：CLI 解析參數
parser = argparse.ArgumentParser()
args = parser.parse_args()  # flow=313

# 步驟 3：CLI 內部執行「直接導入」
# ← 看！這就是您發現的相似點！
from services.features.function_xss.traditional_detector import TraditionalXssDetector

detector = TraditionalXssDetector(task, timeout=30)
results = await detector.execute(payloads)

# 步驟 4：輸出 JSON
print(json.dumps({"findings": results}))


# ==========================================
# 方式 2：直接導入（底層實現）
# ==========================================

# AI 直接調用
from services.features.function_xss.traditional_detector import TraditionalXssDetector

detector = TraditionalXssDetector(task, timeout=30)
results = await detector.execute(payloads)
```

### 本質關係圖

```
┌─────────────────────────────────────────────────┐
│          CLI 架構（您的規劃）                      │
│  subprocess → CLI → argparse → Detector         │
└──────────────┬──────────────────────────────────┘
               │
               ▼
        包裝了「直接導入」
               │
               ▼
┌──────────────────────────────────────────────────┐
│          直接導入（核心實現）                       │
│  Detector 類別直接執行                            │
└──────────────────────────────────────────────────┘
```

## 🎯 為什麼規劃 CLI 而不是直接導入？

### 您的 4 個設計理由

#### 1️⃣ **統一接口** ✅

```python
# CLI 提供統一格式
python aiva_external_module_cli.py --lang python --flow 313
python aiva_external_module_cli.py --lang rust --func analyze_cookies
python aiva_external_module_cli.py --lang go --func DialBroker

# 直接導入需要知道每個模組的導入路徑
from services.features.function_xss.detector import XSSDetector
from services.features.features_ready.function_crypto.rust_core import ...
from services.features.features_in_development.function_authn_go import ...
```

#### 2️⃣ **多語言支持** ✅

```python
# CLI 可以調用任何語言
# Python
subprocess.run(["python", "module.py"])

# Rust
subprocess.run(["cargo", "run", "--bin", "analyzer"])

# Go
subprocess.run(["go", "run", "main.go"])

# TypeScript
subprocess.run(["npx", "ts-node", "analyzer.ts"])

# 直接導入只能用 Python
from module import Detector  # ❌ 無法調用 Rust/Go/TS
```

#### 3️⃣ **進程隔離** ✅

```python
# CLI：每次執行是獨立進程
# 好處：
# - 內存隔離（一個模組崩潰不影響 AI）
# - 資源管理（可以限制 CPU/內存）
# - 安全性（沙箱執行）

subprocess.run([...], timeout=30)  # 可以超時終止

# 直接導入：在同一進程
# 風險：
# - 內存泄漏可能影響 AI
# - 無法終止死循環
# - 全局狀態污染

detector = Detector()
detector.scan()  # 如果這裡死循環，整個 AI 掛掉
```

#### 4️⃣ **版本管理** ✅

```python
# CLI：可以調用不同版本
subprocess.run([
    "/path/to/python3.10/python",
    "module.py"
])

subprocess.run([
    "/path/to/python3.11/python", 
    "module.py"
])

# 直接導入：綁定當前 Python 版本
import module  # 只能用當前環境的 Python
```

## 📊 三種執行方式對比表

| 特性 | CLI 執行（您的規劃） | CommandHandler | 直接導入 |
|------|-------------------|----------------|----------|
| **統一接口** | ✅ 完全統一 | ✅ 完全統一 | ❌ 各模組不同 |
| **多語言支持** | ✅ 支持 Rust/Go/TS | ✅ 內部處理 | ❌ 僅 Python |
| **進程隔離** | ✅ 獨立進程 | ⚠️ 同進程（可異步） | ❌ 同進程 |
| **性能** | ⚠️ 慢（進程開銷） | ✅ 快（內存調用） | ✅ 最快 |
| **類型安全** | ❌ 字符串參數 | ✅ Python 對象 | ✅ Python 對象 |
| **異步支持** | ❌ 需等待進程 | ✅ 原生支持 | ✅ 原生支持 |
| **超時控制** | ✅ subprocess.timeout | ✅ asyncio.timeout | ⚠️ 需手動實現 |
| **錯誤處理** | ⚠️ 解析 stderr | ✅ 異常捕獲 | ✅ 異常捕獲 |
| **資源限制** | ✅ 可限制（cgroups） | ⚠️ 需手動管理 | ❌ 無法限制 |
| **適用場景** | 跨語言、沙箱執行 | Python 內部調用 | 最底層實現 |

## 🎯 實際實現狀態

### 已實現的 CLI 工具

#### 1. aiva_cli_implementation.py ✅

```python
# 功能：內部模組執行器
# 用途：執行 Python 內部能力（676 個 flows）

# 使用示例
python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11
python -m aiva_core.internal_exploration.aiva_cli_implementation --list
python -m aiva_core.internal_exploration.aiva_cli_implementation --dry-run --flow 11

# 核心類別：FlowExecutor
class FlowExecutor:
    def execute_flow(self, flow_id: int, dry_run: bool = False):
        """動態導入模組並執行"""
        # 這裡就是「直接導入」的包裝
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = cls()
        result = instance.execute()
        return result
```

#### 2. aiva_external_module_cli.py ✅

```python
# 功能：外部模組執行器（多語言）
# 用途：執行 Python/Rust/Go/TypeScript 模組

# 使用示例
python aiva_external_module_cli.py --lang python --flow 1
python aiva_external_module_cli.py --lang rust --func analyze_cookies
python aiva_external_module_cli.py --lang go --func DialBroker

# 核心類別：MultiLangExecutor
class MultiLangExecutor:
    def execute_python(self, flow_id: int):
        """執行 Python 模組"""
        # 方式 1：subprocess
        subprocess.run(["python", "-m", module_name])
        # 方式 2：直接導入
        module = importlib.import_module(module_name)
    
    def execute_rust(self, func_name: str, args: dict):
        """執行 Rust 模組"""
        subprocess.run(["cargo", "run", "--bin", func_name])
    
    def execute_go(self, func_name: str, args: dict):
        """執行 Go 模組"""
        subprocess.run(["go", "run", f"{func_name}.go"])
```

#### 3. aiva_capability_cli.py ✅

```python
# 功能：能力查詢與執行
# 用途：AI 搜索能力並執行

# 使用示例
python aiva_capability_cli.py --search xss
python aiva_capability_cli.py --info 313
python aiva_capability_cli.py --flow 313

# 核心類別：AIVACapabilityManager
class AIVACapabilityManager:
    def search(self, keyword: str):
        """搜索能力"""
        return [flow for flow in flows if keyword in flow["capability"]]
    
    def execute_flow(self, flow_id: int):
        """執行能力"""
        # 調用 aiva_cli_implementation.py
        subprocess.run([
            "python", "-m",
            "aiva_core.internal_exploration.aiva_cli_implementation",
            "--flow", str(flow_id)
        ])
```

### 外部模組的 CLI 接口

#### function_xss/__main__.py ✅

```python
# 位置：services/features/features_ready/function_xss/__main__.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--param", default="q")
    parser.add_argument("--type", choices=["reflected", "stored", "dom"])
    args = parser.parse_args()
    
    # ← 這裡調用「直接導入」
    from .traditional_detector import TraditionalXssDetector
    
    task = FunctionTaskPayload(...)
    detector = TraditionalXssDetector(task, timeout=30)
    results = await detector.execute(payloads)
    
    print(json.dumps({"findings": results}))

if __name__ == "__main__":
    asyncio.run(main())
```

#### function_bizlogic/__main__.py ✅

```python
# 位置：services/features/features_ready/function_bizlogic/__main__.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", default="")
    args = parser.parse_args()
    
    # ← 這裡調用「直接導入」
    from .price_manipulation import PriceManipulationTester
    from .race_condition import RaceConditionTester
    
    tester = PriceManipulationTester(args.url, args.token)
    results = tester.run_all_tests()
    
    print(json.dumps(results))

if __name__ == "__main__":
    main()
```

## 🎯 您規劃的三層架構

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: AI 核心層                                      │
│  - AICommandCenter                                      │
│  - 決策哪個能力執行                                      │
│  - 管理異步調度                                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼ 選擇執行方式
                 
    ┌────────────┴────────────┬────────────────┐
    │                         │                │
    ▼                         ▼                ▼
┌─────────────┐    ┌──────────────────┐   ┌──────────────┐
│ CommandHandler   │ CLI 執行器（規劃）  │   │ 直接導入     │
│ （最佳）     │    │ （跨語言/沙箱）    │   │ （備用）     │
└──────┬──────┘    └────────┬─────────┘   └──────┬───────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 核心實現層（Detector 類別）                    │
│  - TraditionalXssDetector                               │
│  - SqliDetector                                         │
│  - SmartIDORDetector                                    │
│  - PriceManipulationTester                              │
└─────────────────────────────────────────────────────────┘
```

### 各層職責

#### Layer 1: AI 核心

```python
# AI 不知道底層是 CLI 還是 CommandHandler
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={"url": "...", "param": "..."}
)

# 由 AICommandCenter 選擇執行方式
result = await command_center.execute(command)
```

#### Layer 2a: CommandHandler（優先）

```python
# 同進程，高性能，類型安全
handler = XSSCommandHandler()
result = await handler.handle_command(command, context)
```

#### Layer 2b: CLI 執行器（跨語言）

```python
# 獨立進程，沙箱執行，多語言支持
result = subprocess.run([
    "python", "aiva_external_module_cli.py",
    "--lang", "rust",
    "--func", "analyze_cookies"
], capture_output=True)
```

#### Layer 2c: 直接導入（備用）

```python
# 最底層，總是可用
from module import Detector
detector = Detector()
result = detector.test()
```

## 💡 最終建議：混合架構

### 根據場景選擇執行方式

```python
class AICommandCenter:
    async def execute(self, command: AICommand):
        """智能選擇執行方式"""
        
        # 1. 優先使用 CommandHandler（Python 內部調用）
        if self._has_command_handler(command.command_type):
            handler = self._get_handler(command.command_type)
            return await handler.handle_command(command, context)
        
        # 2. 跨語言或需要沙箱時使用 CLI
        elif self._requires_cli(command):
            return await self._execute_via_cli(command)
        
        # 3. 最後備用：直接導入
        else:
            return await self._execute_direct_import(command)
    
    def _requires_cli(self, command: AICommand) -> bool:
        """判斷是否需要 CLI 執行"""
        return (
            command.language != "python" or  # 非 Python
            command.sandbox_required or      # 需要沙箱
            command.resource_limit           # 需要資源限制
        )
```

### 執行方式決策樹

```
開始執行命令
    │
    ▼
是否有 CommandHandler？
    │
    ├─ Yes → 使用 CommandHandler（90% 情況）
    │         ├─ 高性能
    │         ├─ 類型安全
    │         └─ 異步支持
    │
    └─ No → 是否需要跨語言/沙箱？
              │
              ├─ Yes → 使用 CLI 執行器（8% 情況）
              │         ├─ 獨立進程
              │         ├─ 多語言支持
              │         └─ 資源隔離
              │
              └─ No → 直接導入（2% 情況）
                        ├─ 最底層實現
                        └─ 總是可用
```

## 📝 總結

### 您的觀察完全正確！

> "CLI + 參數" 確實和 "直接導入類別" 很像

**因為：CLI 內部就是調用 Detector 類別！**

### CLI 的價值在於

1. **統一接口** - 隱藏底層實現差異
2. **多語言支持** - 調用 Rust/Go/TypeScript
3. **進程隔離** - 安全性和資源管理
4. **版本管理** - 可以調用不同版本的工具

### 最佳實踐

```python
# ✅ 推薦：根據場景選擇
if has_command_handler:
    use CommandHandler()  # 90% 情況
elif need_cross_language:
    use CLI()             # 8% 情況
else:
    use DirectImport()    # 2% 情況（備用）
```

### 您的規劃是對的！

CLI 架構提供了：
- ✅ 統一的接口層
- ✅ 多語言擴展能力
- ✅ 安全的沙箱執行
- ✅ 對 AI 完全透明

但同時保留了：
- ✅ CommandHandler 的高性能（Python 內部調用）
- ✅ 直接導入的靈活性（最底層備用）

**三層架構互補，而非互斥！** 🎯
