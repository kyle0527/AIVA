# 🔍 AIVA Scan 引擎驗證與修復計劃

## 📑 目錄

- [📊 引擎實際狀態總覽](#引擎實際狀態總覽)
- [🦀 Rust 引擎](#rust-引擎)
  - [實際可用模式](#實際可用模式)
  - [依賴檢查](#依賴檢查)
  - [驗證方法](#驗證方法)
- [🐍 Python 引擎](#python-引擎)
  - [實際可用模式](#實際可用模式-1)
  - [依賴檢查](#依賴檢查-1)
  - [驗證方法](#驗證方法-1)
- [📘 TypeScript 引擎](#typescript-引擎)
  - [實際可用模式](#實際可用模式-2)
  - [依賴檢查](#依賴檢查-2)
  - [驗證方法](#驗證方法-2)
- [🔷 Go 引擎](#go-引擎)
  - [實際可用模式](#實際可用模式-3)
  - [依賴檢查](#依賴檢查-3)
  - [修復建議](#修復建議)
  - [驗證方法](#驗證方法-3)
- [🚀 完整測試腳本](#完整測試腳本)
- [🛠️ 修復優先級](#修復優先級)
  - [高優先級（必須修復）](#高優先級必須修復)
  - [中優先級（建議修復）](#中優先級建議修復)
  - [低優先級（優化）](#低優先級優化)
- [📋 驗證檢查清單](#驗證檢查清單)
  - [環境準備](#環境準備)
  - [引擎編譯](#引擎編譯)
  - [引擎測試](#引擎測試)
  - [清理工作](#清理工作)
- [🎯 預期成果](#預期成果)

---


**創建日期**: 2025年11月21日  
**目標**: 確認各引擎實際可用模式、移除不必要依賴、提供實際驗證方法

---

## 📊 引擎實際狀態總覽

| 引擎 | 核心實現 | 適配器狀態 | RabbitMQ依賴 | 環境變數依賴 | 實際可用性 |
|------|---------|-----------|-------------|-------------|-----------|
| **Rust** | ✅ FFI Bridge | ✅ 完成 | ❌ 無 | ❌ 無 | ✅ 可用 |
| **Python** | ✅ ScanOrchestrator | ✅ 完成 | ❌ 無 | ⚠️ 瀏覽器路徑 | ✅ 可用 |
| **TypeScript** | ✅ Playwright | ✅ 完成 | ❌ 無 | ⚠️ Node.js 路徑 | ⚠️ 需編譯 |
| **Go** | ⚠️ 需編譯 | ✅ 完成 | ⚠️ Legacy檔案 | ❌ 無 | ⚠️ 需編譯 |

---

## 🦀 Rust 引擎

### 實際可用模式

```rust
// 位置: services/scan/engines/rust_engine/
// FFI Bridge: python_bridge/rust_info_gatherer.pyd

// 支援模式
pub enum ScanMode {
    FastDiscovery,    // ✅ 可用 - Phase 0 快速發現
    DeepAnalysis,     // ✅ 可用 - Phase 1 深度掃描
    FocusedVerification  // ✅ 可用 - 聚焦驗證
}
```

### 依賴檢查

**✅ 無 RabbitMQ 依賴**
```bash
# 搜尋結果: 0 matches in rust_engine/*.rs
grep -r "rabbitmq\|amqp" services/scan/engines/rust_engine/src/
```

**✅ 無環境變數依賴**
```rust
// 配置直接通過函數參數傳遞
pub fn scan_target(url: &str, config: ScanConfig) -> ScanResult
```

### 驗證方法

```python
# 測試腳本: test_rust_engine.py
import asyncio
from services.scan.coordinators import MultiEngineCoordinator

async def test_rust():
    coordinator = MultiEngineCoordinator()
    
    result = await coordinator.execute_phase0(
        scan_id="rust_test_001",
        targets=["http://localhost:3000"]  # Juice Shop
    )
    
    print(f"✅ Rust Phase 0:")
    print(f"  - 資產數: {len(result.assets)}")
    print(f"  - 端點: {result.summary.urls_found}")
    print(f"  - 執行時間: {result.execution_time:.2f}s")
    
    # 驗證輸出範例:
    # ✅ Rust Phase 0:
    #   - 資產數: 40-60
    #   - 端點: 40-50
    #   - 執行時間: 0.5-2.0s

asyncio.run(test_rust())
```

**預期結果**:
- ✅ 能成功掃描 Juice Shop
- ✅ 發現 40-60 個資產
- ✅ 執行時間 < 2 秒
- ✅ 無錯誤日誌

**如果靶場無反應**:
1. 確認 Juice Shop 正在運行: `curl http://localhost:3000`
2. 檢查 Rust FFI Bridge: `python -c "from services.scan.engines.rust_engine.python_bridge import rust_info_gatherer; print('OK')"`
3. 查看詳細日誌: 設置 `logging.DEBUG`

---

## 🐍 Python 引擎

### 實際可用模式

```python
# 位置: services/scan/engines/python_engine/scan_orchestrator.py

# 支援策略模式
STRATEGIES = [
    "FAST",          # ✅ 可用 - 快速掃描 (depth=2-3)
    "CONSERVATIVE",  # ✅ 可用 - 保守掃描 (depth=2)
    "BALANCED",      # ✅ 可用 - 平衡掃描 (depth=3-4)
    "DEEP",          # ✅ 可用 - 深度掃描 (depth=5-6)
    "AGGRESSIVE",    # ✅ 可用 - 激進掃描 (depth=7+)
    "STEALTH",       # ✅ 可用 - 隱秘掃描 (低速)
    "TARGETED"       # ✅ 可用 - 目標掃描 (自定義)
]
```

### 依賴檢查

**✅ 無 RabbitMQ 依賴**
```bash
# scan_orchestrator.py 不導入任何 MQ 相關模組
grep -n "from.*mq import\|rabbitmq" services/scan/engines/python_engine/scan_orchestrator.py
# 結果: 無匹配
```

**⚠️ 瀏覽器路徑依賴**
```python
# 問題: Playwright 需要安裝瀏覽器
# 位置: dynamic_engine/headless_browser_pool.py

# 解決方案 1: 全域安裝（推薦）
playwright install chromium

# 解決方案 2: 環境變數指定
export PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers

# 解決方案 3: 跳過動態引擎
# 在 scan_orchestrator.py 中設置
use_dynamic_engine = False  # 只使用靜態爬取
```

### 驗證方法

```python
# 測試腳本: test_python_engine.py
import asyncio
from services.scan.coordinators import MultiEngineCoordinator

async def test_python():
    coordinator = MultiEngineCoordinator()
    
    result = await coordinator.execute_phase1(
        scan_id="python_test_001",
        targets=["http://localhost:3000"],
        selected_engines=["python"],
        max_depth=5,
        max_urls=1000
    )
    
    print(f"✅ Python Phase 1:")
    print(f"  - 資產數: {len(result.assets)}")
    print(f"  - URLs: {result.summary.urls_found}")
    print(f"  - 表單: {result.summary.forms_found}")
    print(f"  - 執行時間: {result.execution_time:.2f}s")
    
    # 驗證輸出範例:
    # ✅ Python Phase 1:
    #   - 資產數: 1400-1500
    #   - URLs: 80-120
    #   - 表單: 20-30
    #   - 執行時間: 30-60s

asyncio.run(test_python())
```

**預期結果**:
- ✅ 能成功爬取 Juice Shop
- ✅ 發現 1400-1500 個資產
- ✅ 發現 80-120 個 URLs
- ✅ 發現 20-30 個表單
- ✅ 執行時間 30-60 秒

**如果靶場無反應（常見問題）**:

1. **BeautifulSoup 錯誤**
   ```bash
   # 錯誤: No module named 'bs4'
   pip install beautifulsoup4 lxml
   ```

2. **Playwright 錯誤**
   ```bash
   # 錯誤: Browser executable not found
   playwright install chromium
   ```

3. **深度不足（只抓到首頁）**
   ```python
   # 問題: max_depth 設置過小
   # 解決: 設置 max_depth=5 或更高
   result = await coordinator.execute_phase1(
       ...,
       max_depth=5,  # ← 增加深度
       max_urls=1000
   )
   ```

4. **速度過慢**
   ```python
   # 問題: 策略設置為 STEALTH 或 CONSERVATIVE
   # 解決: 使用 FAST 或 BALANCED
   request = ScanStartPayload(
       ...,
       strategy="FAST"  # ← 使用快速策略
   )
   ```

---

## 📘 TypeScript 引擎

### 實際可用模式

```typescript
// 位置: services/scan/engines/typescript_engine/src/

// 支援模式
enum ScanMode {
    BASIC_DYNAMIC = "basic_dynamic",        // ✅ 可用 - 基礎動態掃描
    SPA_FRAMEWORK = "spa_framework",        // ✅ 可用 - SPA 框架檢測
    NETWORK_INTERCEPTION = "network",       // ✅ 可用 - 網路攔截
    ADVANCED_INTERACTION = "interaction",   // ✅ 可用 - 進階互動
    FULL_AUTOMATION = "full"                // ✅ 可用 - 完整自動化
}
```

### 依賴檢查

**✅ 無 RabbitMQ 依賴**
```bash
# 搜尋結果: 0 matches in typescript_engine/src/
grep -r "rabbitmq\|amqp" services/scan/engines/typescript_engine/src/
```

**⚠️ Node.js 路徑依賴**
```json
// package.json 依賴
{
  "dependencies": {
    "playwright": "^1.41.0",
    "@types/node": "^20.0.0"
  }
}

// 需要編譯
npm install
npm run build  // 生成 dist/index.js
```

### 驗證方法

**步驟 1: 編譯 TypeScript 引擎**
```powershell
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 安裝依賴
npm install

# 編譯
npm run build

# 驗證編譯產物
Test-Path dist/index.js  # 應該返回 True
```

**步驟 2: 測試引擎**
```python
# 測試腳本: test_typescript_engine.py
import asyncio
from services.scan.coordinators import MultiEngineCoordinator

async def test_typescript():
    coordinator = MultiEngineCoordinator()
    
    result = await coordinator.execute_phase1(
        scan_id="ts_test_001",
        targets=["http://localhost:3000"],
        selected_engines=["typescript"],
        max_depth=3,
        max_urls=200
    )
    
    print(f"✅ TypeScript Phase 1:")
    print(f"  - 資產數: {len(result.assets)}")
    print(f"  - 動態資產: {sum(1 for a in result.assets if 'spa' in a.asset_id.lower())}")
    print(f"  - 執行時間: {result.execution_time:.2f}s")
    
    # 驗證輸出範例:
    # ✅ TypeScript Phase 1:
    #   - 資產數: 50-100
    #   - 動態資產: 30-60
    #   - 執行時間: 20-40s

asyncio.run(test_typescript())
```

**預期結果**:
- ✅ 能成功啟動 Playwright
- ✅ 發現 50-100 個動態資產
- ✅ 執行時間 20-40 秒
- ✅ 無 Node.js 錯誤

**如果靶場無反應**:

1. **編譯產物不存在**
   ```powershell
   # 錯誤: TypeScript 掃描器不存在
   # 解決: 執行編譯
   cd services/scan/engines/typescript_engine
   npm run build
   ```

2. **Node.js 未安裝**
   ```powershell
   # 錯誤: node 不是內部或外部命令
   # 解決: 安裝 Node.js
   winget install OpenJS.NodeJS
   ```

3. **Playwright 瀏覽器未安裝**
   ```bash
   # 錯誤: Browser executable not found
   # 解決: 安裝瀏覽器
   cd services/scan/engines/typescript_engine
   npx playwright install chromium
   ```

4. **JSON 解析錯誤**
   ```python
   # 問題: console.log 污染 JSON 輸出
   # 解決: 適配器已實現 robust_parse_json，應該能處理
   # 如果仍失敗，檢查 TypeScript 代碼是否有多餘的 console.log
   ```

---

## 🔷 Go 引擎

### 實際可用模式

```go
// 位置: services/scan/engines/go_engine/

// 支援掃描器（需編譯）
type Scanner string

const (
    SSRFScanner Scanner = "ssrf-scanner"     // ⚠️ 需編譯
    CSPMScanner Scanner = "cspm-scanner"     // ⚠️ 需編譯
    SCAScanner  Scanner = "sca-scanner"      // ⚠️ 需編譯
)
```

### 依賴檢查

**⚠️ 有 Legacy RabbitMQ 檔案**
```bash
# 發現 2 個檔案包含 RabbitMQ 引用
# 1. go_engine/dispatcher/dispatcher_legacy.py (Legacy，應移除)
# 2. go_engine/dispatcher/worker.py (舊架構，應移除或重構)
```

**✅ 核心掃描器無 RabbitMQ 依賴**
```bash
# Go 掃描器本身不依賴 RabbitMQ
grep -r "rabbitmq\|amqp" services/scan/engines/go_engine/**/*.go
# 結果: 0 matches
```

### 修復建議

**移除或歸檔 Legacy 檔案**:
```powershell
# 移動到 archived_docs
cd C:\D\fold7\AIVA-git\services\scan\engines\go_engine

# 創建歸檔目錄
New-Item -ItemType Directory -Path archived_legacy -Force

# 移動 Legacy 檔案
Move-Item dispatcher/dispatcher_legacy.py archived_legacy/
Move-Item dispatcher/worker.py archived_legacy/

# 或直接刪除（如果確定不需要）
Remove-Item dispatcher/dispatcher_legacy.py
Remove-Item dispatcher/worker.py
```

### 驗證方法

**步驟 1: 編譯 Go 掃描器**
```powershell
cd C:\D\fold7\AIVA-git\services\scan\engines\go_engine

# 編譯 SSRF Scanner
cd ssrf-scanner
go build -o ../bin/ssrf-scanner.exe

# 編譯 CSPM Scanner
cd ../cspm-scanner
go build -o ../bin/cspm-scanner.exe

# 編譯 SCA Scanner
cd ../sca-scanner
go build -o ../bin/sca-scanner.exe

# 驗證編譯產物
Test-Path bin/ssrf-scanner.exe  # 應該返回 True
Test-Path bin/cspm-scanner.exe  # 應該返回 True
Test-Path bin/sca-scanner.exe   # 應該返回 True
```

**步驟 2: 測試引擎**
```python
# 測試腳本: test_go_engine.py
import asyncio
from services.scan.coordinators import MultiEngineCoordinator

async def test_go():
    coordinator = MultiEngineCoordinator()
    
    result = await coordinator.execute_phase1(
        scan_id="go_test_001",
        targets=["http://localhost:3000"],
        selected_engines=["go"],
        max_depth=3,
        max_urls=500
    )
    
    print(f"✅ Go Phase 1:")
    print(f"  - 資產數: {len(result.assets)}")
    print(f"  - 執行時間: {result.execution_time:.2f}s")
    print(f"  - 請求數: {result.metadata.get('requests_made', 0)}")
    
    # 驗證輸出範例:
    # ✅ Go Phase 1:
    #   - 資產數: 30-50
    #   - 執行時間: 5-15s
    #   - 請求數: 100-200

asyncio.run(test_go())
```

**預期結果**:
- ✅ 能成功編譯 3 個掃描器
- ✅ 發現 30-50 個資產
- ✅ 高並發（100-200 請求）
- ✅ 執行時間 5-15 秒

**如果靶場無反應**:

1. **編譯失敗**
   ```bash
   # 錯誤: go: command not found
   # 解決: 安裝 Go
   winget install GoLang.Go
   ```

2. **二進制檔案不存在**
   ```powershell
   # 錯誤: Go 掃描器二進制文件不存在
   # 解決: 執行編譯步驟
   cd services/scan/engines/go_engine
   # 執行上方的編譯命令
   ```

3. **JSON 輸入格式錯誤**
   ```go
   // 問題: Go 掃描器期望特定的 JSON 格式
   // 解決: 檢查 go_adapter.py 的 scan_input 格式
   // 確保與 Go 掃描器的 main.go 期望格式一致
   ```

---

## 🚀 完整測試腳本

**位置**: `services/scan/test_all_engines.py`

```python
"""
完整的四引擎驗證腳本

驗證所有引擎是否能實際對 Juice Shop 產生效果。
"""

import asyncio
import logging
from services.scan.coordinators import MultiEngineCoordinator
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


async def test_single_engine(engine_name: str):
    """測試單一引擎"""
    print(f"\n{'='*60}")
    print(f"🧪 測試 {engine_name.upper()} 引擎")
    print(f"{'='*60}")
    
    coordinator = MultiEngineCoordinator()
    
    try:
        if engine_name == "rust":
            # Rust 測試 Phase 0
            result = await coordinator.execute_phase0(
                scan_id=f"{engine_name}_test",
                targets=["http://localhost:3000"]
            )
        else:
            # 其他引擎測試 Phase 1
            result = await coordinator.execute_phase1(
                scan_id=f"{engine_name}_test",
                targets=["http://localhost:3000"],
                selected_engines=[engine_name],
                max_depth=5,
                max_urls=1000
            )
        
        # 輸出結果
        print(f"\n✅ {engine_name.upper()} 引擎測試成功!")
        print(f"  📦 資產數: {len(result.assets)}")
        print(f"  ⏱️  執行時間: {result.execution_time:.2f}s")
        
        if hasattr(result, 'summary') and result.summary:
            print(f"  🔗 URLs: {result.summary.urls_found}")
            if hasattr(result.summary, 'forms_found'):
                print(f"  📝 表單: {result.summary.forms_found}")
        
        # 顯示前 5 個資產
        if result.assets:
            print(f"\n  📋 前 5 個資產:")
            for i, asset in enumerate(result.assets[:5], 1):
                print(f"    {i}. [{asset.type}] {asset.value[:80]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ {engine_name.upper()} 引擎測試失敗!")
        print(f"  錯誤: {str(e)}")
        import traceback
        print(f"\n  詳細錯誤:")
        print(traceback.format_exc())
        return False


async def test_multi_engine():
    """測試多引擎協同"""
    print(f"\n{'='*60}")
    print(f"🧪 測試多引擎協同 (Python + Rust)")
    print(f"{'='*60}")
    
    coordinator = MultiEngineCoordinator()
    
    try:
        result = await coordinator.execute_phase1(
            scan_id="multi_engine_test",
            targets=["http://localhost:3000"],
            selected_engines=["python", "rust"],
            max_depth=5,
            max_urls=1000
        )
        
        print(f"\n✅ 多引擎測試成功!")
        print(f"  📦 總資產數: {len(result.assets)}")
        print(f"  ⏱️  執行時間: {result.execution_time:.2f}s")
        
        # 分析各引擎貢獻
        if hasattr(result, 'engine_results'):
            print(f"\n  🔧 各引擎貢獻:")
            for engine_name, engine_data in result.engine_results.items():
                print(f"    - {engine_name}: {engine_data.get('asset_count', 0)} 個資產")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 多引擎測試失敗!")
        print(f"  錯誤: {str(e)}")
        return False


async def main():
    """主測試流程"""
    print(f"\n{'#'*60}")
    print(f"# AIVA Scan 引擎完整驗證")
    print(f"# 目標: http://localhost:3000 (Juice Shop)")
    print(f"{'#'*60}")
    
    # 檢查 Juice Shop 是否運行
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3000", timeout=5)
            print(f"\n✅ Juice Shop 運行中 (HTTP {response.status_code})")
    except Exception as e:
        print(f"\n❌ 無法連接 Juice Shop: {e}")
        print(f"   請確認 Juice Shop 正在運行: docker run -p 3000:3000 bkimminich/juice-shop")
        return
    
    # 測試各引擎
    results = {}
    
    # 1. Rust 引擎 (Phase 0)
    results['rust'] = await test_single_engine('rust')
    await asyncio.sleep(2)
    
    # 2. Python 引擎
    results['python'] = await test_single_engine('python')
    await asyncio.sleep(2)
    
    # 3. TypeScript 引擎
    results['typescript'] = await test_single_engine('typescript')
    await asyncio.sleep(2)
    
    # 4. Go 引擎
    results['go'] = await test_single_engine('go')
    await asyncio.sleep(2)
    
    # 5. 多引擎協同
    results['multi'] = await test_multi_engine()
    
    # 總結
    print(f"\n{'='*60}")
    print(f"📊 測試總結")
    print(f"{'='*60}")
    
    for engine_name, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"  {engine_name.upper():<15} {status}")
    
    total_pass = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\n  總計: {total_pass}/{total_tests} 個測試通過")
    
    if total_pass == total_tests:
        print(f"\n🎉 恭喜！所有引擎驗證通過！")
    else:
        print(f"\n⚠️  部分引擎需要修復，請查看上方錯誤信息")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛠️ 修復優先級

### 高優先級（必須修復）

1. **移除 Go Engine Legacy 檔案** ⚠️
   - 檔案: `go_engine/dispatcher/dispatcher_legacy.py`
   - 檔案: `go_engine/dispatcher/worker.py`
   - 原因: 包含 RabbitMQ 依賴，已不使用
   - 操作: 移動到 `archived_docs/` 或直接刪除

2. **編譯 TypeScript 引擎** ⚠️
   - 位置: `typescript_engine/`
   - 原因: 需要 `dist/index.js` 才能運行
   - 操作: `npm install && npm run build`

3. **編譯 Go 掃描器** ⚠️
   - 位置: `go_engine/`
   - 原因: 需要二進制檔案才能運行
   - 操作: 編譯 3 個掃描器

### 中優先級（建議修復）

4. **Python 引擎環境檢查**
   - 檢查: BeautifulSoup、Playwright 是否已安裝
   - 原因: 缺少依賴會導致掃描失敗
   - 操作: 添加依賴檢查和友好錯誤提示

5. **統一錯誤處理**
   - 各適配器的錯誤處理不一致
   - 建議: 統一錯誤格式和日誌級別

### 低優先級（優化）

6. **性能調優**
   - Python 引擎深度掃描較慢
   - TypeScript 引擎啟動時間較長
   - 建議: 添加緩存和並行優化

7. **文檔更新**
   - 更新各引擎 README 的驗證章節
   - 添加常見問題和解決方案

---

## 📋 驗證檢查清單

### 環境準備
- [ ] Juice Shop 運行中: `http://localhost:3000`
- [ ] Python 環境已激活
- [ ] BeautifulSoup 已安裝: `pip install beautifulsoup4`
- [ ] Playwright 已安裝: `pip install playwright && playwright install chromium`
- [ ] Node.js 已安裝: `node --version`
- [ ] Go 已安裝: `go version`

### 引擎編譯
- [ ] Rust FFI Bridge 存在: `services/scan/engines/rust_engine/python_bridge/rust_info_gatherer.pyd`
- [ ] TypeScript 已編譯: `services/scan/engines/typescript_engine/dist/index.js`
- [ ] Go 掃描器已編譯: `services/scan/engines/go_engine/bin/*.exe`

### 引擎測試
- [ ] Rust 引擎測試通過
- [ ] Python 引擎測試通過
- [ ] TypeScript 引擎測試通過
- [ ] Go 引擎測試通過
- [ ] 多引擎協同測試通過

### 清理工作
- [ ] 移除 Go Engine Legacy 檔案
- [ ] 移除所有 RabbitMQ 環境變數引用
- [ ] 更新文檔反映實際狀態

---

## 🎯 預期成果

完成修復後，應該達到以下狀態：

1. **所有引擎無 RabbitMQ 依賴** ✅
2. **所有引擎可直接通過適配器調用** ✅
3. **所有引擎可在 Juice Shop 產生實際效果** ✅
4. **提供完整的驗證腳本和方法** ✅
5. **文檔準確反映實際狀態** ✅

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2025年11月21日
