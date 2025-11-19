# Python Engine 使用指南

> **文檔目的**: 說明如何通過協調器使用 Python Engine 進行掃描  
> **適用角色**: 開發者、測試人員  
> **最後更新**: 2025-11-19  
> **狀態**: ✅ 完全可用並已驗證

---

## 📋 目錄

- [快速開始](#快速開始)
- [基礎概念](#基礎概念)
- [使用方式](#使用方式)
- [參數配置](#參數配置)
- [結果解析](#結果解析)
- [故障排查](#故障排查)
- [性能優化](#性能優化)

---

## 🚀 快速開始

### 最簡單的掃描示例

```python
import asyncio
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
from services.aiva_common.schemas import ScanStartPayload

async def quick_scan():
    """30秒快速測試"""
    coordinator = MultiEngineCoordinator()
    
    request = ScanStartPayload(
        scan_id="quick_test",
        targets=["http://localhost:3000"],
        strategy="quick"
    )
    
    result = await coordinator.execute_coordinated_scan(request)
    print(f"✅ 發現 {result.total_assets} 個資產")
    print(f"⏱️  耗時 {result.total_time:.1f}s")

asyncio.run(quick_scan())
```

**預期輸出**:
```
🎯 開始協調掃描: quick_test
  🐍 Python 引擎: 開始掃描
  🐍 Python 引擎完成: 156 個資產, 8.2s
✅ 發現 156 個資產
⏱️  耗時 8.2s
```

---

## 📚 基礎概念

### 協調器 vs Python Engine

```
┌─────────────────────────────────────────┐
│  MultiEngineCoordinator (協調器)        │
│  - 負責引擎選擇與結果聚合                │
│  - 當前實際只調用 Python Engine          │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐   │
│  │  ScanOrchestrator (Python 引擎)  │   │
│  │  - Phase 1: 靜態內容爬取         │   │
│  │  - Phase 2: 漏洞驗證 (自動觸發)  │   │
│  │  - 返回資產列表                   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**關鍵理解**:
- 協調器是**入口**，但目前只調用一個引擎
- Python Engine 內部有自己的 Phase 1→2 閉環
- 結果會通過協調器統一返回

---

### Phase 1→2 自動閉環

```
Phase 1: 靜態爬取           Phase 2: 漏洞驗證
┌──────────────┐           ┌──────────────┐
│ 發現 URL     │──────────→│ XSS 測試     │
│ 發現 Form    │           │ SQLi 測試    │
│ 發現 API     │           │ CSRF 測試    │
└──────────────┘           └──────────────┘
        ↓                          ↓
   Asset List              Vulnerability List
```

**自動觸發條件**:
- 發現表單 (Forms) → 自動執行 XSS/SQLi 測試
- 發現 API endpoint → 自動執行參數測試
- 無需手動啟動 Phase 2

---

## 🎯 使用方式

### 方式 1: 通過協調器使用 (推薦)

```python
import asyncio
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
from services.aiva_common.schemas import ScanStartPayload

async def scan_via_coordinator():
    """通過協調器調用 Python Engine"""
    coordinator = MultiEngineCoordinator()
    
    request = ScanStartPayload(
        scan_id="scan_001",
        targets=["http://localhost:3000"],
        strategy="normal",
        max_depth=3,
        timeout=300
    )
    
    # 協調器會自動選擇 Python Engine
    result = await coordinator.execute_coordinated_scan(request)
    
    # 提取 Python Engine 的結果
    for engine_result in result.engine_results:
        if engine_result.engine.value == "python":
            print(f"Python 引擎資產: {len(engine_result.assets)}")
            print(f"URLs: {engine_result.metadata.get('urls_found', 0)}")
            print(f"Forms: {engine_result.metadata.get('forms_found', 0)}")

asyncio.run(scan_via_coordinator())
```

**優點**:
- 統一的接口，未來可擴展多引擎
- 結果格式標準化
- 日誌記錄完整

---

### 方式 2: 直接使用 Python Engine

```python
import asyncio
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

async def scan_direct():
    """直接調用 Python Engine (不通過協調器)"""
    orchestrator = ScanOrchestrator()
    
    request = ScanStartPayload(
        scan_id="direct_scan",
        targets=["http://localhost:3000"],
        strategy="quick"
    )
    
    # 直接執行掃描
    scan_result = await orchestrator.execute_scan(request)
    
    print(f"資產數: {len(scan_result.assets)}")
    print(f"URLs: {scan_result.summary.urls_found}")
    print(f"Forms: {scan_result.summary.forms_found}")
    print(f"耗時: {scan_result.summary.scan_duration_seconds:.1f}s")

asyncio.run(scan_direct())
```

**優點**:
- 更直接，減少一層封裝
- 適合只需要 Python Engine 的場景
- 性能稍微好一點點（省略協調器開銷）

**缺點**:
- 無法利用協調器的聚合功能
- 日誌格式可能不同

---

## ⚙️ 參數配置

### ScanStartPayload 參數詳解

```python
from services.aiva_common.schemas import ScanStartPayload

request = ScanStartPayload(
    # 必填參數
    scan_id="unique_scan_id",        # 唯一掃描 ID
    targets=["http://example.com"],  # 目標 URL 列表
    
    # 掃描策略 (影響深度和速度)
    strategy="normal",               # quick | normal | deep | full | custom
    
    # 爬取深度控制
    max_depth=3,                     # 最大爬取層數 (1-10)
    max_pages=100,                   # 最大頁面數
    
    # 超時設置
    timeout=300,                     # 總超時時間 (秒)
    page_timeout=10,                 # 單頁超時 (秒)
    
    # 並發控制
    max_concurrent_requests=5,       # 最大並發請求數
    
    # 可選功能
    enable_javascript=False,         # 是否執行 JS (Python Engine 不支持)
    follow_redirects=True,           # 是否跟隨重定向
    respect_robots_txt=True,         # 是否遵守 robots.txt
)
```

---

### Strategy 參數對照表

| Strategy | max_depth | max_pages | 適用場景 | 預估時間 |
|----------|-----------|-----------|----------|----------|
| `quick` | 1 | 50 | 快速測試、CI/CD | 30s - 2min |
| `normal` | 3 | 100 | 日常掃描 | 2min - 10min |
| `deep` | 5 | 500 | 深度分析 | 10min - 30min |
| `full` | 10 | 無限制 | 完整審計 | 30min+ |
| `custom` | 自定義 | 自定義 | 特殊需求 | 取決於配置 |

**建議**:
- 開發測試: `quick`
- 日常掃描: `normal`
- 安全審計: `deep` 或 `full`

---

### 多目標掃描示例

```python
async def multi_target_scan():
    """同時掃描多個目標"""
    coordinator = MultiEngineCoordinator()
    
    request = ScanStartPayload(
        scan_id="multi_target",
        targets=[
            "http://localhost:3000",  # Juice Shop
            "http://localhost:3001",  # 靶場 2
            "http://localhost:8080",  # 靶場 3
        ],
        strategy="quick",
        max_depth=2
    )
    
    result = await coordinator.execute_coordinated_scan(request)
    
    # 結果會聚合所有目標的資產
    print(f"總資產: {result.total_assets}")
    
    # 可以根據 URL 過濾資產
    assets_by_target = {}
    for engine_result in result.engine_results:
        for asset in engine_result.assets:
            base_url = asset.url.split('/')[2]  # 提取 host:port
            assets_by_target.setdefault(base_url, []).append(asset)
    
    for target, assets in assets_by_target.items():
        print(f"{target}: {len(assets)} 個資產")

asyncio.run(multi_target_scan())
```

**預期輸出**:
```
總資產: 432
localhost:3000: 156 個資產
localhost:3001: 123 個資產
localhost:8080: 153 個資產
```

---

## 📊 結果解析

### 協調器返回結果結構

```python
from services.scan.coordinators.scan_models import CoordinationResult, EngineResult

# 協調器返回的結果
result: CoordinationResult = await coordinator.execute_coordinated_scan(request)

# 頂層信息
result.scan_id              # str: 掃描 ID
result.total_assets         # int: 總資產數
result.total_time           # float: 總耗時 (秒)
result.coordination_strategy # str: 使用的協調策略

# 各引擎結果 (當前只有 Python)
result.engine_results       # List[EngineResult]
```

---

### EngineResult 結構

```python
for engine_result in result.engine_results:
    # 基本信息
    engine_result.engine           # EngineType: PYTHON | TYPESCRIPT | RUST
    engine_result.phase            # ScanPhase: 掃描階段
    engine_result.execution_time   # float: 引擎耗時
    
    # 資產列表
    engine_result.assets           # List[Asset]: 發現的資產
    
    # 元數據
    engine_result.metadata         # Dict[str, Any]: 引擎特定數據
    # Python Engine metadata 包含:
    # - urls_found: int
    # - forms_found: int
    # - scan_duration: float
    
    # 錯誤信息 (如果失敗)
    engine_result.error            # Optional[str]: 錯誤訊息
```

---

### Asset 結構

```python
from services.aiva_common.schemas import Asset, AssetType

for asset in engine_result.assets:
    # 基本信息
    asset.asset_id        # str: 資產唯一 ID
    asset.asset_type      # AssetType: URL | FORM | API | ENDPOINT
    asset.url             # str: 資產 URL
    
    # 發現信息
    asset.method          # str: HTTP 方法 (GET, POST, ...)
    asset.discovered_at   # datetime: 發現時間
    asset.source          # str: 來源 (哪個引擎發現的)
    
    # 詳細數據
    asset.data            # Dict[str, Any]: 資產詳細數據
    # 例如 FORM 類型的 data:
    # {
    #     "action": "/login",
    #     "method": "POST",
    #     "fields": [
    #         {"name": "username", "type": "text"},
    #         {"name": "password", "type": "password"}
    #     ]
    # }
    
    # 漏洞信息 (如果有)
    asset.vulnerabilities # List[Vulnerability]: 關聯的漏洞
```

---

### 完整的結果解析示例

```python
async def analyze_scan_results():
    """完整解析掃描結果"""
    coordinator = MultiEngineCoordinator()
    
    request = ScanStartPayload(
        scan_id="analysis_test",
        targets=["http://localhost:3000"],
        strategy="normal"
    )
    
    result = await coordinator.execute_coordinated_scan(request)
    
    print(f"📊 掃描報告: {result.scan_id}")
    print(f"⏱️  總耗時: {result.total_time:.1f}s")
    print(f"🎯 協調策略: {result.coordination_strategy}\n")
    
    # 分析各引擎結果
    for engine_result in result.engine_results:
        engine_name = engine_result.engine.value
        print(f"--- {engine_name.upper()} 引擎 ---")
        print(f"  資產數: {len(engine_result.assets)}")
        print(f"  耗時: {engine_result.execution_time:.1f}s")
        
        if engine_result.error:
            print(f"  ❌ 錯誤: {engine_result.error}")
            continue
        
        # 統計資產類型
        asset_types = {}
        for asset in engine_result.assets:
            asset_type = asset.asset_type.value
            asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
        
        print(f"  資產類型分佈:")
        for asset_type, count in asset_types.items():
            print(f"    - {asset_type}: {count}")
        
        # 統計漏洞
        total_vulns = sum(
            len(asset.vulnerabilities) 
            for asset in engine_result.assets 
            if asset.vulnerabilities
        )
        print(f"  🔍 發現漏洞: {total_vulns}\n")
    
    # 詳細輸出前 5 個資產
    print("--- 前 5 個資產詳情 ---")
    python_assets = [
        asset 
        for er in result.engine_results 
        if er.engine.value == "python"
        for asset in er.assets
    ][:5]
    
    for i, asset in enumerate(python_assets, 1):
        print(f"{i}. [{asset.asset_type.value}] {asset.url}")
        print(f"   方法: {asset.method}, 來源: {asset.source}")
        if asset.vulnerabilities:
            print(f"   ⚠️  漏洞: {len(asset.vulnerabilities)} 個")

asyncio.run(analyze_scan_results())
```

**輸出示例**:
```
📊 掃描報告: analysis_test
⏱️  總耗時: 8.5s
🎯 協調策略: partial_coordination

--- PYTHON 引擎 ---
  資產數: 156
  耗時: 8.2s
  資產類型分佈:
    - url: 142
    - form: 8
    - api: 6
  🔍 發現漏洞: 3

--- 前 5 個資產詳情 ---
1. [url] http://localhost:3000/
   方法: GET, 來源: python_engine
2. [url] http://localhost:3000/login
   方法: GET, 來源: python_engine
3. [form] http://localhost:3000/login
   方法: POST, 來源: python_engine
   ⚠️  漏洞: 1 個
4. [url] http://localhost:3000/api/products
   方法: GET, 來源: python_engine
5. [api] http://localhost:3000/api/products
   方法: GET, 來源: python_engine
```

---

## 🐛 故障排查

### 問題 1: 返回 0 個資產

**現象**:
```
🐍 Python 引擎完成: 0 個資產
```

**可能原因**:
1. 目標 URL 無法訪問
2. 網絡連接問題
3. 目標網站返回錯誤狀態碼

**排查步驟**:
```python
# 1. 確認目標可訪問
import requests
response = requests.get("http://localhost:3000")
print(response.status_code)  # 應該是 200

# 2. 檢查協調器日誌
# 查看是否有錯誤訊息

# 3. 直接測試 Python Engine
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
orchestrator = ScanOrchestrator()
result = await orchestrator.execute_scan(request)
print(len(result.assets))  # 看是否有資產
```

---

### 問題 2: 掃描超時

**現象**:
```
TimeoutError: Scan exceeded timeout of 300s
```

**解決方案**:
```python
# 增加超時時間
request = ScanStartPayload(
    scan_id="test",
    targets=["http://localhost:3000"],
    strategy="quick",
    timeout=600,        # 增加到 10 分鐘
    page_timeout=20     # 單頁超時也可以增加
)

# 或者降低掃描深度
request = ScanStartPayload(
    scan_id="test",
    targets=["http://localhost:3000"],
    strategy="quick",
    max_depth=2,        # 減少深度
    max_pages=50        # 限制頁面數
)
```

---

### 問題 3: 記憶體使用過高

**現象**:
```
Python 進程記憶體使用超過 2GB
```

**解決方案**:
```python
# 1. 限制並發數
request = ScanStartPayload(
    scan_id="test",
    targets=["http://localhost:3000"],
    strategy="normal",
    max_concurrent_requests=3  # 降低並發 (預設 5)
)

# 2. 限制掃描範圍
request = ScanStartPayload(
    scan_id="test",
    targets=["http://localhost:3000"],
    strategy="quick",
    max_pages=100,            # 限制頁面數
    max_depth=2               # 降低深度
)

# 3. 分批掃描
async def scan_in_batches():
    """分批掃描大型網站"""
    base_url = "http://localhost:3000"
    paths = ["/", "/products", "/admin", "/api"]
    
    all_assets = []
    for path in paths:
        request = ScanStartPayload(
            scan_id=f"batch_{path.replace('/', '_')}",
            targets=[f"{base_url}{path}"],
            strategy="quick"
        )
        result = await coordinator.execute_coordinated_scan(request)
        all_assets.extend(result.engine_results[0].assets)
    
    print(f"總資產: {len(all_assets)}")
```

---

### 問題 4: Phase 2 漏洞驗證未觸發

**現象**:
```
發現了 8 個表單，但沒有漏洞報告
```

**可能原因**:
- Phase 2 驗證被禁用
- 表單字段不符合驗證條件
- 驗證過程中出錯但被捕獲

**排查步驟**:
```python
# 1. 檢查 Python Engine 配置
# 查看 services/scan/engines/python_engine/scan_orchestrator.py
# 確認 Phase 2 是否啟用

# 2. 查看詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 3. 檢查資產的漏洞字段
for asset in result.engine_results[0].assets:
    if asset.asset_type.value == "form":
        print(f"Form: {asset.url}")
        print(f"Vulns: {len(asset.vulnerabilities) if asset.vulnerabilities else 0}")
        if asset.vulnerabilities:
            for vuln in asset.vulnerabilities:
                print(f"  - {vuln.type}: {vuln.severity}")
```

---

## ⚡ 性能優化

### 優化 1: 調整並發數

```python
# 根據目標網站性能調整
request = ScanStartPayload(
    scan_id="optimized",
    targets=["http://localhost:3000"],
    strategy="normal",
    max_concurrent_requests=10  # 目標強: 增加並發
    # max_concurrent_requests=2  # 目標弱: 降低並發
)
```

**基準測試**:
| 並發數 | 耗時 | CPU | 記憶體 |
|--------|------|-----|--------|
| 1 | 45s | 20% | 200MB |
| 3 | 18s | 50% | 400MB |
| 5 | 12s | 70% | 600MB |
| 10 | 10s | 90% | 1GB |

**建議**:
- 本地測試: 3-5
- 生產環境: 5-10
- 弱小目標: 1-2

---

### 優化 2: 使用適當的 Strategy

```python
# 開發階段: 使用 quick 快速迭代
request = ScanStartPayload(
    scan_id="dev_test",
    targets=["http://localhost:3000"],
    strategy="quick",  # 只掃描 1 層，最多 50 頁
)

# CI/CD: 使用 normal 平衡速度和覆蓋率
request = ScanStartPayload(
    scan_id="ci_test",
    targets=["http://localhost:3000"],
    strategy="normal",  # 掃描 3 層，最多 100 頁
)

# 夜間掃描: 使用 deep 獲得完整結果
request = ScanStartPayload(
    scan_id="nightly_scan",
    targets=["http://localhost:3000"],
    strategy="deep",  # 掃描 5 層，最多 500 頁
)
```

---

### 優化 3: 限制掃描範圍

```python
# 只掃描特定路徑
async def scan_specific_paths():
    """針對特定功能模組掃描"""
    coordinator = MultiEngineCoordinator()
    
    # 只掃描登錄相關
    request = ScanStartPayload(
        scan_id="login_only",
        targets=["http://localhost:3000/login"],
        strategy="deep",
        max_depth=2  # 只深入 2 層
    )
    
    result = await coordinator.execute_coordinated_scan(request)
    print(f"登錄模組資產: {result.total_assets}")
```

---

### 優化 4: 使用緩存

```python
# 如果多次掃描同一目標，可以利用結果緩存
from datetime import datetime, timedelta

class ScanCache:
    """簡單的掃描結果緩存"""
    def __init__(self):
        self.cache = {}
    
    async def get_or_scan(self, target: str, max_age: timedelta):
        """獲取緩存或執行新掃描"""
        cache_key = target
        
        if cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < max_age:
                print(f"✅ 使用緩存結果 (age: {datetime.now() - cached_time})")
                return cached_result
        
        # 緩存過期或不存在，執行新掃描
        print(f"🔄 執行新掃描")
        coordinator = MultiEngineCoordinator()
        request = ScanStartPayload(
            scan_id=f"scan_{cache_key}",
            targets=[target],
            strategy="normal"
        )
        result = await coordinator.execute_coordinated_scan(request)
        
        # 更新緩存
        self.cache[cache_key] = (result, datetime.now())
        return result

# 使用示例
cache = ScanCache()
result1 = await cache.get_or_scan("http://localhost:3000", timedelta(hours=1))
result2 = await cache.get_or_scan("http://localhost:3000", timedelta(hours=1))  # 使用緩存
```

---

## 📝 完整測試腳本

```python
import asyncio
import time
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
from services.aiva_common.schemas import ScanStartPayload

async def comprehensive_test():
    """綜合測試腳本"""
    print("========== Python Engine 綜合測試 ==========\n")
    
    coordinator = MultiEngineCoordinator()
    
    # 測試 1: 快速掃描
    print("--- 測試 1: 快速掃描 (strategy=quick) ---")
    start = time.time()
    request = ScanStartPayload(
        scan_id="test_quick",
        targets=["http://localhost:3000"],
        strategy="quick"
    )
    result = await coordinator.execute_coordinated_scan(request)
    print(f"✓ 資產: {result.total_assets}, 耗時: {time.time() - start:.1f}s\n")
    
    # 測試 2: 正常掃描
    print("--- 測試 2: 正常掃描 (strategy=normal) ---")
    start = time.time()
    request = ScanStartPayload(
        scan_id="test_normal",
        targets=["http://localhost:3000"],
        strategy="normal"
    )
    result = await coordinator.execute_coordinated_scan(request)
    print(f"✓ 資產: {result.total_assets}, 耗時: {time.time() - start:.1f}s\n")
    
    # 測試 3: 多目標掃描
    print("--- 測試 3: 多目標掃描 ---")
    start = time.time()
    request = ScanStartPayload(
        scan_id="test_multi",
        targets=[
            "http://localhost:3000",
            "http://localhost:3001"
        ],
        strategy="quick"
    )
    result = await coordinator.execute_coordinated_scan(request)
    print(f"✓ 總資產: {result.total_assets}, 耗時: {time.time() - start:.1f}s\n")
    
    # 測試 4: 結果解析
    print("--- 測試 4: 詳細結果解析 ---")
    for engine_result in result.engine_results:
        if engine_result.engine.value == "python":
            print(f"Python 引擎:")
            print(f"  資產數: {len(engine_result.assets)}")
            print(f"  URLs: {engine_result.metadata.get('urls_found', 0)}")
            print(f"  Forms: {engine_result.metadata.get('forms_found', 0)}")
            
            # 統計資產類型
            asset_types = {}
            for asset in engine_result.assets:
                t = asset.asset_type.value
                asset_types[t] = asset_types.get(t, 0) + 1
            
            print(f"  資產分佈:")
            for asset_type, count in asset_types.items():
                print(f"    {asset_type}: {count}")
            
            # 統計漏洞
            vulns = sum(
                len(a.vulnerabilities) if a.vulnerabilities else 0
                for a in engine_result.assets
            )
            print(f"  漏洞數: {vulns}")
    
    print("\n========== 測試完成 ==========")

# 執行測試
asyncio.run(comprehensive_test())
```

---

## 📚 參考文檔

- **協調器實際狀態**: `COORDINATOR_ACTUAL_STATUS.md`
- **Python Engine 源碼**: `services/scan/engines/python_engine/scan_orchestrator.py`
- **協調器源碼**: `services/scan/coordinators/multi_engine_coordinator.py`
- **數據模型**: `services/aiva_common/schemas.py`

---

**維護者**: AIVA 開發團隊  
**問題反饋**: 如遇到問題請查看故障排查章節或聯繫開發團隊
