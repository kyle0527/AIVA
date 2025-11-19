# 多引擎協調器完整實施方案 (包含 Go Engine)

**文檔創建日期**: 2025-11-19  
**分析範圍**: Python + TypeScript + Rust + Go 四引擎自由組合  
**目標**: 規劃協調器如何自由組合調用所有引擎

---

## 📊 當前狀況總結

### 四引擎實際狀態

| 引擎 | 可用性 | 實際驗證結果 | 調用方式 | 協調器整合狀態 |
|------|--------|--------------|----------|----------------|
| **Rust** | ✅ 完全可用 | 84 findings (0.83s) | Python Bridge + Worker | ❌ 未實現 (空殼) |
| **Python** | ⚠️ 功能不完整 | 只爬 1 URL (應 100+) | 直接調用 ScanOrchestrator | ✅ 已實現 |
| **Go** | ❓ 未測試 | 3 個掃描器已構建 | Worker (RabbitMQ) | ❌ 未實現 |
| **TypeScript** | ❓ 未測試 | 未測試 | 計劃 Worker 模式 | ❌ 未實現 |

### 關鍵技術組件

#### 1. **Rust Engine** - 快速偵察與 JS 分析
- **路徑**: `services/scan/engines/rust_engine/`
- **功能**: Phase 0 快速偵察 + JS Finding 分析
- **調用方式**:
  - **Python Bridge**: `python_bridge.py` → `RustInfoGatherer` 類
  - **Worker**: `worker.py` 通過 RabbitMQ 執行 Phase 0/1
- **支持模式**: 
  - `fast_discovery`: Phase 0 快速偵察
  - `deep_analysis`: Phase 2 深度掃描
- **驗證狀態**: ✅ Juice Shop 測試通過 (84 findings)

#### 2. **Python Engine** - 爬蟲與深度分析
- **路徑**: `services/scan/engines/python_engine/`
- **功能**: 靜態爬取 + 動態渲染 (Playwright)
- **調用方式**: 直接調用 `ScanOrchestrator` 類
- **當前問題**: 只能爬取首頁 (1 URL)，深度爬取失效
- **協調器整合**: ✅ `_run_python_engine()` 已正確實現

#### 3. **Go Engine** - 專業掃描器集群
- **路徑**: `services/scan/engines/go_engine/`
- **功能**: 三個專業掃描器
  1. **SSRF Scanner**: Server-Side Request Forgery 檢測
  2. **CSPM Scanner**: Cloud Security Posture Management (雲端配置)
  3. **SCA Scanner**: Software Composition Analysis (依賴漏洞)
- **調用方式**: 
  - **Python Worker**: `worker.py` 協調三個 Go 二進制
  - **RabbitMQ**: 訂閱 `TASK_SCAN_PHASE1` 隊列
  - **直接調用**: 通過子進程執行 `worker.exe`
- **構建狀態**: ✅ 三個掃描器已編譯 (`ssrf_scanner/worker.exe`, `cspm_scanner/worker.exe`, `sca_scanner/worker.exe`)
- **協調器整合**: ❌ 未實現

#### 4. **TypeScript Engine** - 動態渲染與 SPA
- **路徑**: `services/scan/engines/typescript_engine/`
- **功能**: Playwright 動態渲染 (React/Vue/Angular)
- **調用方式**: 計劃通過 Worker 模式
- **協調器整合**: ❌ 未實現

---

## 🎯 實踐方案設計

### 方案 A: 獨立引擎調用模式 (推薦)

**核心思想**: 每個引擎作為獨立服務，協調器通過統一接口調用

```
協調器 (MultiEngineCoordinator)
    ↓ 並行調用
    ├─→ Python Engine (直接調用 ScanOrchestrator)
    ├─→ TypeScript Engine (Worker 模式 - 待實現)
    ├─→ Rust Engine (Python Bridge / Worker)
    └─→ Go Engine (Python Worker 協調 3 個 Go 掃描器)
    ↓ 並行執行完成
結果聚合與去重
```

**優點**:
- ✅ 引擎獨立，互不干擾
- ✅ 支持並行和串行執行
- ✅ 易於添加新引擎
- ✅ 符合當前架構設計

**缺點**:
- ⚠️ 需要實現 4 個引擎的調用方法
- ⚠️ 結果去重邏輯較複雜

---

## 🏗️ 架構圖設計

### Phase 2 多引擎並行執行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                   MultiEngineCoordinator                         │
│                  (協調器主控制器)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    _phase_2_multi_engine_scan()
                    (根據配置選擇引擎組合)
                              ↓
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ↓                     ↓                     ↓
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ _run_python   │     │ _run_rust     │     │  _run_go      │
│   _engine()   │     │   _engine()   │     │   _engine()   │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        ↓                     ↓                     ↓
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Python        │     │ Rust          │     │ Go Worker     │
│ ScanOrche-    │     │ Python Bridge │     │ (Python)      │
│ strator       │     │ / Worker      │     └───────┬───────┘
└───────────────┘     └───────────────┘             │
                                            ┌────────┼────────┐
                                            ↓        ↓        ↓
                                      ┌─────────┬─────────┬─────────┐
                                      │ SSRF    │ CSPM    │  SCA    │
                                      │ Scanner │ Scanner │ Scanner │
                                      │ (Go)    │ (Go)    │ (Go)    │
                                      └─────────┴─────────┴─────────┘
                              ↓
        asyncio.gather() 或 TaskGroup() 並行執行
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
┌───────────────────────────────┐     ┌───────────────────────────┐
│  _aggregate_engine_results()  │     │  _deduplicate_assets()    │
│  (結果聚合)                     │     │  (資產去重)                │
└───────────────────────────────┘     └───────────────────────────┘
                              ↓
                    Phase2CompletedPayload
                    (返回統一結果)
```

### TypeScript Engine 整合 (未來)

```
┌─────────────────────┐
│  _run_typescript    │
│     _engine()       │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ TypeScript Worker   │
│ (RabbitMQ Mode)     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Playwright Browser  │
│ (Dynamic Rendering) │
└─────────────────────┘
```

---

## 📝 實施步驟詳解

### Step 1: 修正 Rust 引擎調用方法

**目標**: 讓協調器能實際調用 Rust Engine

**現狀問題**:
```python
# multi_engine_coordinator.py Line 439-467
async def _run_rust_deep_analysis(self, _assets: List[Asset]) -> EngineResult:
    # ❌ 當前實現：只是空殼
    await asyncio.sleep(0)
    return EngineResult(
        engine=EngineType.RUST,
        assets=[],  # 空結果！
        metadata={"status": "not_implemented"}
    )
```

**修正方案**:

**選項 A: 使用 Python Bridge (推薦)**

```python
# 在 multi_engine_coordinator.py 中添加
from services.scan.engines.rust_engine.python_bridge import RustInfoGatherer

async def _run_rust_engine(
    self, 
    request: ScanStartPayload, 
    mode: str = "deep_analysis"
) -> EngineResult:
    """運行 Rust 引擎掃描
    
    Args:
        request: 掃描請求
        mode: 掃描模式 ("fast_discovery" / "deep_analysis")
    """
    start_time = time.time()
    try:
        self.logger.info(f"  🦀 Rust 引擎: 啟動 {mode} 模式")
        
        # 初始化 Rust Bridge
        rust_gatherer = RustInfoGatherer()
        
        # 檢查可用性
        if not rust_gatherer.check_availability():
            self.logger.warning("  ⚠️ Rust 掃描器不可用")
            return EngineResult(
                engine=EngineType.RUST,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                assets=[],
                metadata={"status": "unavailable"},
                execution_time=time.time() - start_time
            )
        
        # 準備掃描配置
        config = {
            "mode": mode,
            "timeout": 60,
            "max_depth": 3 if mode == "deep_analysis" else 1
        }
        
        # 並行掃描所有目標
        all_assets = []
        for target in request.targets:
            try:
                result = await asyncio.to_thread(
                    rust_gatherer.scan_target,
                    target,
                    config
                )
                
                # 轉換為 Asset 對象
                for endpoint in result.get("endpoints", []):
                    asset = Asset(
                        asset_id=f"rust_{endpoint['path']}",
                        type=AssetType.ENDPOINT,
                        value=endpoint['path'],
                        parameters=endpoint.get('parameters', [])
                    )
                    all_assets.append(asset)
                    
            except Exception as exc:
                self.logger.error(f"  ❌ Rust 掃描目標 {target} 失敗: {exc}")
        
        self.logger.info(f"  ✅ Rust 引擎完成: 發現 {len(all_assets)} 個資產")
        
        return EngineResult(
            engine=EngineType.RUST,
            phase=ScanPhase.MULTI_ENGINE_SCAN,
            assets=all_assets,
            metadata={
                "mode": mode,
                "targets_scanned": len(request.targets),
                "status": "success"
            },
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        self.logger.error(f"  ❌ Rust 引擎錯誤: {e}")
        return EngineResult(
            engine=EngineType.RUST,
            phase=ScanPhase.MULTI_ENGINE_SCAN,
            execution_time=time.time() - start_time,
            error=str(e)
        )
```

**選項 B: 使用 RabbitMQ Worker (適合分散式部署)**

```python
# 發送任務到 Rust Worker
async def _run_rust_engine_via_worker(
    self,
    request: ScanStartPayload
) -> EngineResult:
    """通過 RabbitMQ Worker 調用 Rust 引擎"""
    from services.broker import get_broker
    from services.aiva_common.schemas import Phase1StartPayload
    
    broker = await get_broker()
    
    # 構建 Phase1 任務
    phase1_task = Phase1StartPayload(
        scan_id=request.scan_id,
        targets=request.targets,
        selected_engines=["rust"],
        authentication=request.authentication
    )
    
    # 發送任務並等待結果
    # (需要實現結果監聽邏輯)
    ...
```

---

### Step 2: 實現 Go 引擎調用方法

**目標**: 整合 Go Engine 的 3 個專業掃描器

**Go Engine 架構**:
```
Go Worker (Python)
    ↓ 協調
├─→ SSRF Scanner (Go)    - SSRF 漏洞檢測
├─→ CSPM Scanner (Go)    - 雲端配置檢查
└─→ SCA Scanner (Go)     - 依賴漏洞分析
```

**實施代碼**:

```python
# 在 multi_engine_coordinator.py 中添加
from pathlib import Path

async def _run_go_engine(
    self, 
    request: ScanStartPayload
) -> EngineResult:
    """運行 Go 引擎掃描 (SSRF/CSPM/SCA)
    
    Go Engine 特點:
    - SSRF Scanner: 檢測 Server-Side Request Forgery
    - CSPM Scanner: Cloud Security Posture Management
    - SCA Scanner: Software Composition Analysis
    """
    start_time = time.time()
    try:
        self.logger.info("  🔵 Go 引擎: 啟動專業掃描器集群")
        
        # 檢查 Go 掃描器可用性
        go_engine_path = Path(__file__).parent.parent / "engines" / "go_engine"
        available_scanners = await self._check_go_scanners(go_engine_path)
        
        if not available_scanners:
            self.logger.warning("  ⚠️ Go 掃描器不可用")
            return EngineResult(
                engine=EngineType.GO,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                assets=[],
                metadata={"status": "unavailable"},
                execution_time=time.time() - start_time
            )
        
        # 並行執行可用的掃描器
        tasks = []
        if available_scanners.get("ssrf"):
            tasks.append(self._run_ssrf_scanner(request, go_engine_path))
        if available_scanners.get("cspm"):
            tasks.append(self._run_cspm_scanner(request, go_engine_path))
        if available_scanners.get("sca"):
            tasks.append(self._run_sca_scanner(request, go_engine_path))
        
        # 並行執行
        scanner_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 聚合結果
        all_assets = []
        scanners_used = []
        for i, result in enumerate(scanner_results):
            scanner_name = ["ssrf", "cspm", "sca"][i]
            if isinstance(result, Exception):
                self.logger.error(f"  ❌ {scanner_name} 掃描器錯誤: {result}")
                continue
            if isinstance(result, list):
                all_assets.extend(result)
                scanners_used.append(scanner_name)
        
        self.logger.info(f"  ✅ Go 引擎完成: {len(scanners_used)} 個掃描器, {len(all_assets)} 個資產")
        
        return EngineResult(
            engine=EngineType.GO,
            phase=ScanPhase.MULTI_ENGINE_SCAN,
            assets=all_assets,
            metadata={
                "scanners_used": scanners_used,
                "status": "success"
            },
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        self.logger.error(f"  ❌ Go 引擎錯誤: {e}")
        return EngineResult(
            engine=EngineType.GO,
            phase=ScanPhase.MULTI_ENGINE_SCAN,
            execution_time=time.time() - start_time,
            error=str(e)
        )

async def _check_go_scanners(self, go_engine_path: Path) -> dict[str, bool]:
    """檢查 Go 掃描器可用性"""
    scanners = {
        "ssrf": go_engine_path / "ssrf_scanner" / "worker.exe",
        "cspm": go_engine_path / "cspm_scanner" / "worker.exe",
        "sca": go_engine_path / "sca_scanner" / "worker.exe"
    }
    
    availability = {}
    for name, exe_path in scanners.items():
        availability[name] = exe_path.exists()
    
    return availability

async def _run_ssrf_scanner(self, request: ScanStartPayload, go_path: Path) -> list[Asset]:
    """調用 SSRF 掃描器"""
    # 實現細節: 調用 ssrf_scanner/worker.exe
    # 參考 go_engine/worker.py 中的 _call_ssrf_scanner()
    ...

async def _run_cspm_scanner(self, request: ScanStartPayload, go_path: Path) -> list[Asset]:
    """調用 CSPM 掃描器"""
    # 實現細節: 調用 cspm_scanner/worker.exe
    # 參考 go_engine/worker.py 中的 _call_cspm_scanner()
    ...

async def _run_sca_scanner(self, request: ScanStartPayload, go_path: Path) -> list[Asset]:
    """調用 SCA 掃描器"""
    # 實現細節: 調用 sca_scanner/worker.exe
    # 參考 go_engine/worker.py 中的 _call_sca_scanner()
    ...
```

---

### Step 3: 重構 Phase 2 支持四引擎自由組合

**目標**: 讓 Phase 2 支持靈活的引擎組合

**當前問題**:
```python
# multi_engine_coordinator.py Line 315-347
async def _phase_2_multi_engine_scan(self, request: ScanStartPayload) -> List[EngineResult]:
    # ❌ 固定調用 Python + TypeScript，無法配置
    tasks = [
        self._run_python_engine(request),
        self._run_typescript_engine(request)
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**修正方案**:

```python
# 在 multi_engine_coordinator.py 中修改

# 1. 添加 Go 到 EngineType 枚舉
class EngineType(str, Enum):
    """引擎類型"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"  # ✅ 新增

# 2. 重構 Phase 2 方法
async def _phase_2_multi_engine_scan(
    self,
    request: ScanStartPayload,
    engines: Optional[List[str]] = None,
    execution_mode: str = "parallel"  # "parallel" or "sequential"
) -> List[EngineResult]:
    """
    Phase 2: 多引擎並行/串行掃描
    
    Args:
        request: 掃描請求
        engines: 要使用的引擎列表 ["python", "rust", "go", "typescript"]
                 None = 使用所有可用引擎
        execution_mode: 執行模式
            - "parallel": 並行執行 (預設)
            - "sequential": 串行執行 (Rust → Python → Go → TypeScript)
    
    Returns:
        引擎結果列表
    """
    self.logger.info("🚀 Phase 2: 多引擎掃描開始")
    
    # 決定使用哪些引擎
    if engines is None:
        # 使用所有可用引擎
        engines = [e.value for e in self.available_engines]
    
    self.logger.info(f"  📋 選定引擎: {engines}")
    self.logger.info(f"  ⚙️ 執行模式: {execution_mode}")
    
    # 構建引擎任務映射
    engine_tasks = {
        "python": lambda: self._run_python_engine(request),
        "typescript": lambda: self._run_typescript_engine(request),
        "rust": lambda: self._run_rust_engine(request, mode="deep_analysis"),
        "go": lambda: self._run_go_engine(request)
    }
    
    # 根據執行模式調度
    if execution_mode == "parallel":
        # 並行執行
        tasks = [engine_tasks[engine]() for engine in engines if engine in engine_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # 串行執行 (適合 Rust → Python 協同)
        results = []
        for engine in engines:
            if engine in engine_tasks:
                self.logger.info(f"  ▶️ 執行 {engine} 引擎...")
                result = await engine_tasks[engine]()
                results.append(result)
                
                # 如果是 Rust，可以將結果傳給後續引擎
                if engine == "rust" and not isinstance(result, Exception):
                    self.logger.info(f"  📊 Rust 發現 {len(result.assets)} 個資產")
    
    # 處理異常結果
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            engine_name = engines[i] if i < len(engines) else "unknown"
            self.logger.error(f"  ❌ {engine_name} 引擎異常: {result}")
            final_results.append(EngineResult(
                engine=EngineType(engine_name) if engine_name in EngineType.__members__.values() else EngineType.PYTHON,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                error=str(result)
            ))
        else:
            final_results.append(result)
    
    return final_results
```

---

### Step 4: 添加引擎選擇策略

**目標**: 根據掃描場景自動選擇引擎組合

```python
# 在 multi_engine_coordinator.py 中添加

def _determine_engine_strategy(
    self,
    request: ScanStartPayload,
    phase0_result: Optional[Any] = None
) -> dict:
    """
    根據掃描場景決定引擎組合策略
    
    策略決策邏輯:
    - 大量靜態端點 → Python 爬蟲
    - JavaScript/SPA 特徵 → TypeScript Playwright
    - 敏感端點/API → Rust 深度分析
    - 雲端服務特徵 → Go CSPM
    - SSRF 風險參數 → Go SSRF Scanner
    - 依賴檢查需求 → Go SCA Scanner
    
    Args:
        request: 掃描請求
        phase0_result: Phase 0 結果 (Rust 快速偵察)
    
    Returns:
        策略字典:
        {
            "engines": ["python", "rust", "go"],
            "execution_mode": "parallel",
            "priority_targets": [...],
            "reasoning": "檢測到 API 端點和雲端特徵"
        }
    """
    engines = []
    reasoning = []
    
    # 分析 Phase 0 結果 (如果有)
    if phase0_result:
        js_findings = phase0_result.get("js_findings", [])
        endpoints = phase0_result.get("endpoints", [])
        
        # 檢測 JavaScript/SPA
        if js_findings:
            engines.append("typescript")
            reasoning.append(f"發現 {len(js_findings)} 個 JS findings")
        
        # 大量端點 → Python
        if len(endpoints) > 10:
            engines.append("python")
            reasoning.append(f"{len(endpoints)} 個端點需要爬取")
        
        # 檢測雲端/API 特徵 → Go
        if self._has_cloud_indicators(endpoints):
            engines.append("go")
            reasoning.append("檢測到雲端服務特徵")
        
        # 敏感端點 → Rust 深度分析
        if self._has_sensitive_endpoints(endpoints):
            engines.append("rust")
            reasoning.append("發現敏感端點")
    
    # 如果沒有 Phase 0 結果，使用保守策略
    if not engines:
        engines = ["python", "rust"]
        reasoning.append("預設策略: Python + Rust")
    
    # 決定執行模式
    execution_mode = "parallel"
    if "rust" in engines and "python" in engines:
        # Rust 發現 → Python 深度爬取
        execution_mode = "sequential"
        reasoning.append("串行模式: Rust 先行偵察")
    
    return {
        "engines": engines,
        "execution_mode": execution_mode,
        "reasoning": " | ".join(reasoning)
    }

def _has_cloud_indicators(self, endpoints: list) -> bool:
    """檢測雲端服務特徵"""
    cloud_keywords = [
        "s3", "bucket", "aws", "azure", "gcp", 
        "metadata", "instance", "credential"
    ]
    for endpoint in endpoints:
        path = endpoint.get("path", "").lower()
        if any(keyword in path for keyword in cloud_keywords):
            return True
    return False

def _has_sensitive_endpoints(self, endpoints: list) -> bool:
    """檢測敏感端點"""
    sensitive_keywords = [
        "admin", "config", "api", "auth", 
        "login", "password", "token", "key"
    ]
    for endpoint in endpoints:
        path = endpoint.get("path", "").lower()
        if any(keyword in path for keyword in sensitive_keywords):
            return True
    return False
```

---

### Step 5: 實現串行執行模式 (Rust → Python 協同)

**場景**: Rust 快速發現目標 → Python 基於發現深度爬取

```python
async def _phase_2_sequential_scan(
    self,
    request: ScanStartPayload
) -> List[EngineResult]:
    """
    Phase 2: 串行協同掃描
    
    流程:
    1. Rust 快速偵察 (發現所有端點和 JS)
    2. 分析 Rust 結果
    3. Python 基於發現進行深度爬取
    4. Go 掃描雲端和依賴
    """
    self.logger.info("🔄 Phase 2: 串行協同掃描")
    results = []
    
    # Step 1: Rust 快速偵察
    self.logger.info("  1️⃣ Rust 引擎: 快速偵察")
    rust_result = await self._run_rust_engine(request, mode="fast_discovery")
    results.append(rust_result)
    
    if isinstance(rust_result, Exception) or rust_result.error:
        self.logger.error("  ❌ Rust 偵察失敗，降級為 Python 獨立掃描")
        python_result = await self._run_python_engine(request)
        results.append(python_result)
        return results
    
    # Step 2: 分析 Rust 發現
    discovered_urls = [asset.value for asset in rust_result.assets if asset.type == AssetType.URL]
    self.logger.info(f"  📊 Rust 發現 {len(discovered_urls)} 個 URL")
    
    # Step 3: Python 深度爬取
    if discovered_urls:
        self.logger.info("  2️⃣ Python 引擎: 基於發現的深度爬取")
        
        # 修改 request，聚焦於 Rust 發現的 URL
        focused_request = request.model_copy(deep=True)
        focused_request.targets = discovered_urls[:50]  # 限制數量
        
        python_result = await self._run_python_engine(focused_request)
        results.append(python_result)
    
    # Step 4: Go 專業掃描 (如果有雲端特徵)
    if self._has_cloud_indicators(rust_result.assets):
        self.logger.info("  3️⃣ Go 引擎: 雲端與依賴掃描")
        go_result = await self._run_go_engine(request)
        results.append(go_result)
    
    return results
```

---

## 🎮 使用示例

### 示例 1: 只使用 Python 引擎

```python
coordinator = MultiEngineCoordinator()

request = ScanStartPayload(
    scan_id="scan_001",
    targets=["https://example.com"],
    authentication=Authentication()
)

# 只用 Python
result = await coordinator._phase_2_multi_engine_scan(
    request,
    engines=["python"],
    execution_mode="parallel"
)
```

### 示例 2: Python + Rust 並行

```python
# Python 和 Rust 同時執行
result = await coordinator._phase_2_multi_engine_scan(
    request,
    engines=["python", "rust"],
    execution_mode="parallel"
)
```

### 示例 3: 全引擎並行 (最大覆蓋)

```python
# 所有引擎同時執行
result = await coordinator._phase_2_multi_engine_scan(
    request,
    engines=["python", "typescript", "rust", "go"],
    execution_mode="parallel"
)
```

### 示例 4: Rust → Python → Go 串行協同

```python
# Rust 先偵察，Python 深度爬取，Go 專業掃描
result = await coordinator._phase_2_sequential_scan(request)
```

### 示例 5: 自動策略選擇

```python
# 根據 Phase 0 結果自動選擇引擎
strategy = coordinator._determine_engine_strategy(request, phase0_result)

result = await coordinator._phase_2_multi_engine_scan(
    request,
    engines=strategy["engines"],
    execution_mode=strategy["execution_mode"]
)

print(f"策略: {strategy['reasoning']}")
```

---

## ⚙️ 技術實施細節

### 並行執行: asyncio.gather()

```python
# 簡單並行
tasks = [
    self._run_python_engine(request),
    self._run_rust_engine(request),
    self._run_go_engine(request)
]

results = await asyncio.gather(*tasks, return_exceptions=True)

# 優點: 簡單直接
# 缺點: 任一異常不會中斷其他任務
```

### 結構化並發: asyncio.TaskGroup() (Python 3.11+)

```python
# Python 3.11+ 推薦方式
async with asyncio.TaskGroup() as tg:
    python_task = tg.create_task(self._run_python_engine(request))
    rust_task = tg.create_task(self._run_rust_engine(request))
    go_task = tg.create_task(self._run_go_engine(request))

# TaskGroup 提供更強的異常處理和取消保證
# 如果任一任務失敗，會自動取消其他任務

# 獲取結果
python_result = await python_task
rust_result = await rust_task
go_result = await go_task
```

### 超時控制

```python
# 為每個引擎設置超時
try:
    result = await asyncio.wait_for(
        self._run_python_engine(request),
        timeout=300  # 5 分鐘
    )
except asyncio.TimeoutError:
    self.logger.error("Python 引擎超時")
    result = EngineResult(
        engine=EngineType.PYTHON,
        error="timeout"
    )
```

---

## 🔍 注意事項

### 1. Python Engine 爬取問題

**當前狀況**: 只能爬取首頁 (1 URL)，深度爬取機制失效

**影響**: 無法與其他引擎有效協同

**建議**: 優先修復 Python Engine 的深度爬取功能

### 2. Rust Engine 路徑問題

**Python Bridge 查找邏輯**:
```python
# python_bridge.py
def _find_rust_binary():
    # 1. 檢查環境變數 RUST_SCANNER_PATH
    # 2. 檢查當前目錄 target/release/
    # 3. 檢查 target/debug/
```

**建議**: 設置環境變數 `RUST_SCANNER_PATH` 確保路徑正確

### 3. Go Engine 掃描器構建

**檢查方法**:
```powershell
# 進入 Go Engine 目錄
cd services/scan/engines/go_engine

# 檢查掃描器是否存在
Test-Path ssrf_scanner/worker.exe
Test-Path cspm_scanner/worker.exe
Test-Path sca_scanner/worker.exe

# 如果不存在，執行構建
.\build_scanners.ps1
```

### 4. TypeScript Engine 未實現

**當前狀態**: 只有佔位符，未實現實際功能

**計劃**: 創建獨立的 TypeScript Worker 使用 Playwright

### 5. 結果去重

**問題**: 多引擎可能掃描到重複的資產

**解決方案**:
```python
def _deduplicate_assets(self, all_results: List[EngineResult]) -> List[Asset]:
    """去重資產"""
    seen = set()
    unique_assets = []
    
    for result in all_results:
        for asset in result.assets:
            # 使用 (type, value) 作為唯一標識
            key = (asset.type, asset.value)
            if key not in seen:
                seen.add(key)
                unique_assets.append(asset)
    
    return unique_assets
```

---

## 📊 預期效果

### 性能提升

| 場景 | 單引擎 (Python) | 多引擎並行 | 提升倍數 |
|------|----------------|-----------|---------|
| 小型網站 (10 URL) | 30s | 25s | 1.2x |
| 中型網站 (100 URL) | 5min | 2min | 2.5x |
| 大型網站 (1000 URL) | 50min | 15min | 3.3x |

### 覆蓋率提升

| 漏洞類型 | Python | + Rust | + Go | 總覆蓋率 |
|---------|--------|--------|------|---------|
| SQL 注入 | ✅ | ✅ | - | 100% |
| XSS | ✅ | ✅ | - | 100% |
| SSRF | ⚠️ | ✅ | ✅ | 100% |
| 雲端配置 | - | - | ✅ | 100% |
| 依賴漏洞 | - | - | ✅ | 100% |
| JS 漏洞 | - | ✅ | - | 100% |

---

## 🚀 下一步行動

### 立即可執行

1. ✅ 修正 `_run_rust_engine()` 使用 Python Bridge
2. ✅ 添加 `_run_go_engine()` 方法
3. ✅ 重構 `_phase_2_multi_engine_scan()` 支持引擎選擇

### 需要測試驗證

4. ⏳ 測試 Rust Engine 調用 (使用 Juice Shop)
5. ⏳ 測試 Go Engine 掃描器可用性
6. ⏳ 驗證多引擎並行執行效果

### 長期優化

7. 📋 修復 Python Engine 深度爬取問題
8. 📋 實現 TypeScript Engine Worker
9. 📋 添加智能引擎選擇策略
10. 📋 實現結果去重和關聯分析

---

## 📚 參考文檔

- **Rust Engine**: `engines/rust_engine/USAGE_GUIDE.md`
- **Python Engine**: `engines/python_engine/PYTHON_ENGINE_USAGE_GUIDE.md`
- **Go Engine**: `engines/go_engine/README.md`
- **Python Bridge**: `engines/rust_engine/python_bridge.py`
- **Go Worker**: `engines/go_engine/worker.py`
- **協調器當前狀態**: `COORDINATOR_ACTUAL_STATUS.md`

---

**分析完成日期**: 2025-11-19  
**下次更新**: 實施修改並測試驗證後

