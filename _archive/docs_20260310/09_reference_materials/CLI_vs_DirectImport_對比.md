# 外部功能模組的3種執行方式 - 正確理解


## 📑 目錄

- [🎯 核心架構](#-核心架構)
- [📊 詳細對比](#-詳細對比)
- [🔍 實際代碼對比](#-實際代碼對比)
  - [方式 1：CLI + 參數](#方式-1cli-參數)
  - [方式 2：直接導入類別](#方式-2直接導入類別)
- [💡 為什麼 CLI 模式看起來像第 3 種？](#-為什麼-cli-模式看起來像第-3-種)
- [🎯 關鍵洞察](#-關鍵洞察)
  - [CLI 的三層結構](#cli-的三層結構)
  - [第 3 種方式跳過了前兩層](#第-3-種方式跳過了前兩層)
- [📊 性能對比實測](#-性能對比實測)
- [🎯 什麼時候用哪種？](#-什麼時候用哪種)
  - [使用 CLI + 參數（適合）](#使用-cli-參數適合)
  - [使用直接導入（適合）](#使用直接導入適合)
- [💡 最佳實踐](#-最佳實踐)
  - [AI 核心應該用什麼？](#ai-核心應該用什麼)
  - [結論](#結論)

---
## 🎯 核心架構

**外部功能模組有3種執行方式，第3種直接導入就是CLI實施方式**

```
方式1: CommandHandler（主要執行路徑）
┌─────────────────────────────────────────┐
│  RabbitMQ 或直接調用                     │
│  async def handle_command()             │
└──────────────┬──────────────────────────┘
               │
               ▼
        ┌─────────────┐
        │  Detector    │  ← 核心類別
        │  實際執行    │
        └─────────────┘

方式2: Worker模式（背景服務）
┌─────────────────────────────────────────┐
│  持續運行的背景服務                       │
│  監聽 RabbitMQ 消息隊列                  │
└─────────────────────────────────────────┘

方式3: Direct Import（CLI實施方式）
┌─────────────────────────────────────────┐
│  from module.detector import Detector   │
│  detector = Detector()                  │
│  result = detector.detect()             │
└─────────────────────────────────────────┘
```

**重要說明**: 不需要 __main__.py！第3種方式直接導入就是CLI實施方式。

## 📊 詳細對比

| 特性 | CLI + 參數 | 直接導入類別 |
|------|-----------|------------|
| **本質** | 包裝層 + 核心類別 | 核心類別 |
| **進程** | 新 Python 進程 | 當前進程 |
| **調用方式** | `subprocess.run()` | `import` + 函數調用 |
| **參數傳遞** | 命令行參數（字符串） | Python 對象 |
| **性能** | ❌ 慢（進程創建開銷） | ✅ 快（內存調用） |
| **靈活性** | ⚠️ 受限於 argparse | ✅ 完全控制 |
| **異步支持** | ❌ 需要等待進程結束 | ✅ 原生支持 |
| **適用場景** | 命令行腳本、跨語言 | Python 內部調用 |

## 🔍 實際代碼對比

### 方式 1：CLI + 參數

```python
# AI 核心執行
import subprocess
import json

result = subprocess.run([
    "python", "-m",
    "services.features.features_ready.function_xss",
    "--url", "http://localhost:3000",
    "--param", "q",
    "--type", "reflected",
    "--timeout", "30"
], capture_output=True, text=True)

# 解析 JSON 輸出
data = json.loads(result.stdout)
print(f"發現 {data['findings_count']} 個漏洞")
```

**執行過程：**
```
1. AI 核心調用 subprocess
2. 創建新的 Python 進程
3. 進程執行 __main__.py
4. __main__.py 解析參數
5. __main__.py 創建 Detector
6. Detector 執行測試        ← 這裡才是真正的工作
7. 結果序列化為 JSON
8. 進程結束，返回 stdout
9. AI 核心解析 JSON
```

### 方式 2：直接導入類別

```python
# AI 核心執行
from services.features.function_xss.traditional_detector import TraditionalXssDetector
from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget

task = FunctionTaskPayload(
    task_id="test_001",
    scan_id="scan_001",
    target=FunctionTaskTarget(
        url="http://localhost:3000",
        parameter="q",
        method="GET",
        parameter_location="query"
    )
)

# 直接使用
detector = TraditionalXssDetector(task, timeout=30)
results = await detector.execute(payloads)
print(f"發現 {len(results)} 個漏洞")
```

**執行過程：**
```
1. AI 核心導入 Detector
2. Detector 執行測試        ← 直接開始工作
3. 返回結果對象
```

## 💡 為什麼 CLI 模式看起來像第 3 種？

**因為 CLI 內部就是調用第 3 種！**

```python
# function_xss/__main__.py（CLI 實現）
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--param", default="q")
    args = parser.parse_args()
    
    # ====================================
    # ← 看！這裡就是「直接導入類別」！
    # ====================================
    from .traditional_detector import TraditionalXssDetector
    
    task = FunctionTaskPayload(...)
    detector = TraditionalXssDetector(task, timeout=30)
    results = await detector.execute(payloads)
    # ====================================
    
    # 輸出結果
    print(json.dumps({"findings": results}))

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 關鍵洞察

### CLI 的三層結構

```
┌──────────────────────────────────┐
│  Layer 1: 進程管理層              │
│  subprocess.run()                │
│  - 創建進程                       │
│  - 管理輸入輸出                   │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Layer 2: 參數解析層              │
│  __main__.py + argparse          │
│  - 解析命令行參數                 │
│  - 轉換為 Python 對象            │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Layer 3: 核心實現層              │
│  Detector 類別                    │
│  - 實際執行檢測                   │  ← 第 3 種方式直接用這層！
│  - 返回結果                       │
└──────────────────────────────────┘
```

### 第 3 種方式跳過了前兩層

```python
# ❌ CLI 方式：走完三層
subprocess.run()  → __main__.py → Detector

# ✅ 直接導入：只用第三層
Detector
```

## 📊 性能對比實測

```python
import time

# 測試 CLI 方式
start = time.time()
subprocess.run(["python", "-m", "module", "--url", "..."])
cli_time = time.time() - start

# 測試直接導入
start = time.time()
detector = Detector()
detector.test(url="...")
direct_time = time.time() - start

print(f"CLI 方式: {cli_time:.3f}s")
print(f"直接導入: {direct_time:.3f}s")
print(f"速度提升: {cli_time/direct_time:.1f}x")
```

**典型結果：**
```
CLI 方式: 0.523s
直接導入: 0.045s
速度提升: 11.6x
```

## 🎯 什麼時候用哪種？

### 使用 CLI + 參數（適合）

✅ **跨語言調用**
```bash
# Go/Rust/TypeScript 調用 Python 模組
go run main.go → subprocess → python module
```

✅ **命令行工具**
```bash
# 直接在終端使用
$ python -m module --url xxx --param yyy
```

✅ **獨立腳本**
```bash
# 不需要導入整個項目
./test.sh
```

### 使用直接導入（適合）

✅ **Python 內部調用**
```python
# AI 核心是 Python，直接導入
from module import Detector
```

✅ **高性能需求**
```python
# 需要大量並發調用
await asyncio.gather(*[detector.test(url) for url in urls])
```

✅ **深度定制**
```python
# 需要訪問內部狀態
detector = Detector()
detector.config.timeout = 60
detector.hooks.on_result = custom_handler
```

## 💡 最佳實踐

### AI 核心應該用什麼？

**推薦：CommandHandler（結合了兩者優點）**

```python
# CommandHandler 本質上是「直接導入」的統一接口
command_center = get_command_center()

# 但提供了 CLI 的便利性（統一參數格式）
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={"url": "...", "param": "..."}
)

# 性能接近「直接導入」（同進程）
result = await command_center.execute(command)
```

### 結論

```
CLI + 參數 ≈ 包裝後的「直接導入類別」

CLI        = subprocess + __main__.py + Detector
直接導入    = Detector

效果相同，但：
- CLI 多了 2 層包裝
- 直接導入性能更好
- CLI 更適合跨語言/命令行
- 直接導入更適合 Python 內部
```
