# 🎯 AIVA Scan - 多語言統一掃描引擎

**導航**: [← 返回 Services 總覽](../README.md) | [📖 文檔中心](../../docs/README.md) | [🔬 引擎驗證指南](./ENGINE_VERIFICATION_AND_FIX_PLAN.md)

> **🎯 設計目標**: 兩階段掃描架構，適配器模式協調四引擎  
> **❌ 實際狀態**: 架構設計完成，但所有引擎均無實際掃描能力  
> **🔄 最後更新**: 2025年11月22日 - 驗證確認：0/4 引擎可用

## 🏗️ 核心架構

### 設計理念

Scan 模組採用**適配器模式**協調四個語言引擎（Python、TypeScript、Rust、Go），實現兩階段掃描流程：

```
┌─────────────────────────────────────────────────────────────┐
│                    AI 命令中心 (Core 模組)                   │
│                                                              │
│  Phase 0 決策 → Rust 快速偵察 → 分析結果 → Phase 1 決策     │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
                    數據合約 (Pydantic)
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│              Scan 模組 - MultiEngineCoordinator              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        適配器層 (coordinators/engines/)              │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  Python Adapter  │  TypeScript Adapter               │  │
│  │  Rust Adapter    │  Go Adapter                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           四引擎並行執行 (asyncio.gather)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**理論優勢** (架構設計):
- ✅ **複雜度降低**: 從 171 降至 17 (-90%)
- ✅ **錯誤隔離**: 單引擎失敗不影響整體
- ✅ **統一接口**: 適配器模式封裝引擎差異
- ✅ **類型安全**: Pydantic 數據合約驗證

**實際驗證結果** (2025年11月22日):
- ❌ **無實際掃描功能**: 所有引擎均未發送真實 HTTP 請求
- ❌ **假陽性結果**: 漏洞掃描器未真實測試目標
- ❌ **引擎不可用**: 4 個引擎中 0 個可正常工作
- ⚠️ **架構與實現脫節**: 代碼結構正確但功能缺失
- 📋 **需要重新實現**: 所有引擎的掃描邏輯需從頭編寫

### 兩階段掃描流程

**Phase 0 - 快速偵察** (5-10 分鐘):
- **執行者**: Rust 引擎 (必須)
- **任務**: 目標驗證、敏感資訊掃描、技術棧識別、基礎端點發現
- **輸出**: 初步資產清單 + AI 決策建議

**Phase 1 - 深度掃描** (10-30 分鐘，按需):
- **選擇引擎**: AI 根據 Phase 0 結果決定組合
- **並行執行**: Python/TypeScript/Rust/Go (1-4 個)
- **任務**: 深度爬取、動態渲染、表單發現、API 分析
- **輸出**: 完整資產清單
```

---

## 📊 系統狀態

### ❌ 驗證失敗 (v2.1 - 2025年11月22日)

| 組件 | 架構狀態 | 功能狀態 | 驗證結果 |
|------|----------|----------|----------|
| **適配器模式** | ✅ 架構完成 | ❌ 未實現 | **無真實請求發送** |
| **四引擎支援** | ⚠️ 代碼存在 | ❌ 不可用 | TypeScript/Go 未安裝，Rust 失敗 |
| **Phase 0 流程** | ✅ 邏輯完成 | ❌ 不可用 | Rust 引擎執行失敗 |
| **Phase 1 流程** | ✅ 邏輯完成 | ❌ 不可用 | Python 無網路活動 |
| **命令處理** | ✅ 接口完成 | ⚠️ 部分可用 | 可接收命令但無實際掃描 |
| **錯誤隔離** | ✅ 完成 | ✅ 正常 | 唯一正常工作的功能 |

**致命問題** (2025年11月22日 驗證):
- ❌ **所有引擎均未發送 HTTP 請求到靶場目標**
- ❌ Python 引擎僅記錄日誌，無實際網路掃描
- ❌ 漏洞掃描器返回假陽性結果（未真實測試）
- ❌ Rust 引擎二進制執行失敗 (exit code 2)
- ❌ TypeScript/Go 引擎未編譯安裝
- 📋 結論: **協調器架構正確，但所有引擎的掃描實現均缺失**

**詳細分析**: 📄 [能力分析報告](_out/SCAN_MODULE_CAPABILITY_ANALYSIS.md)  
**修復計劃**: 📋 [修復計劃文檔](./SCAN_MODULE_RESTORATION_PLAN.md)

---

## ⚠️ 重要聲明

**AIVA Scan 目前狀態** (2025年11月22日驗證):

- ❌ **無實際掃描功能**: 所有引擎均未實現真實的網路請求
- ❌ **架構與實現脫節**: 代碼結構完整但底層功能缺失
- ❌ **需要完整重寫**: 4 個引擎的掃描邏輯需從零實現
- ⚠️ **文檔僅供參考**: 所有功能描述為設計目標，非實際能力

**設計理念**（未實現）:
AIVA Scan 設計為多語言統一掃描引擎，計劃整合 Python、TypeScript、Rust 和 Go 四種技術的優勢，專精於 Bug Bounty 動態檢測和黑盒滲透測試。

---

## 🚀 快速開始

### ⚠️ 重要警告

**目前所有範例代碼僅為架構演示，不會執行實際的網路掃描。**

驗證結果 (2025年11月22日):
- ❌ 代碼可以執行但不會發送 HTTP 請求
- ❌ 返回的結果為空或假陽性數據
- ❌ 所有引擎均無實際掃描能力

### 基本使用（理論架構）

```python
# ⚠️ 警告：此代碼不會執行實際掃描
# 透過 AI 命令中心執行掃描（僅架構演示）
from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas import AICommand, CommandType

# 建立命令
command = AICommand(
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload={
        "scan_id": "scan_001",
        "targets": ["https://example.com"]
    }
)

# ❌ 執行後不會有實際掃描發生
command_center = get_command_center()
result = await command_center.execute(command)  # 返回空結果或假數據
```

### 詳細指南

- 📘 **使用者手冊**: [SCAN_USER_GUIDE.md](./SCAN_USER_GUIDE.md) - 完整操作指南
- 🔧 **API 參考**: [command_handler.py](./command_handler.py) - 命令處理器實現
- 📊 **流程圖解**: [SCAN_FLOW_DIAGRAMS.md](./SCAN_FLOW_DIAGRAMS.md) - 完整流程說明

---

## 📚 文檔導航

### 📖 主要文檔

| 文檔 | 說明 | 適用對象 |
|------|------|----------|
| 📘 [SCAN_USER_GUIDE.md](./SCAN_USER_GUIDE.md) | 使用者手冊 | 所有使用者 |
| 📊 [SCAN_FLOW_DIAGRAMS.md](./SCAN_FLOW_DIAGRAMS.md) | 流程圖解（基準文檔） | 開發者 |
| 📋 [SCAN_MODULE_RESTORATION_PLAN.md](./SCAN_MODULE_RESTORATION_PLAN.md) | 修復計劃與完成狀態 | 開發者 |
| 📄 [SCAN_MODULE_CAPABILITY_ANALYSIS.md](../../_out/SCAN_MODULE_CAPABILITY_ANALYSIS.md) | 能力分析報告 | 開發者 |

### 🔧 引擎文檔（架構設計）

| 引擎 | 架構狀態 | 功能狀態 | 文檔 | 設計目標 |
|------|----------|----------|------|----------|
| 🦀 **Rust** | ⚠️ 代碼存在 | ❌ 執行失敗 | [rust_engine/README.md](./engines/rust_engine/README.md) | Phase 0 快速偵察 |
| 🐍 **Python** | ⚠️ 代碼存在 | ❌ 無網路掃描 | [python_engine/README.md](./engines/python_engine/README.md) | Phase 1 深度爬取 |
| 📘 **TypeScript** | ⚠️ 代碼存在 | ❌ 未編譯 | [typescript_engine/README.md](./engines/typescript_engine/README.md) | Phase 1 動態渲染 |
| 🔷 **Go** | ⚠️ 代碼存在 | ❌ 未編譯 | [go_engine/README.md](./engines/go_engine/README.md) | Phase 1 高並發掃描 |

### 🎯 協調器文檔

| 文檔 | 說明 |
|------|------|
| [coordinators/README.md](./coordinators/README.md) | 協調器總覽和適配器模式 |
| [coordinators/COORDINATOR_ACTUAL_STATUS.md](./coordinators/COORDINATOR_ACTUAL_STATUS.md) | 實際狀態報告 |

---

## 🏗️ 四引擎架構

### 引擎分工

| 引擎 | 主要職責 | 技術優勢 | 支援模式 | 使用階段 |
|------|----------|----------|----------|----------|
| 🦀 **Rust** | 快速偵察、敏感資訊掃描 | 極高性能 (10-100x) | **3種模式**<br/>1. FastDiscovery<br/>2. DeepAnalysis<br/>3. FocusedVerification | Phase 0 必須<br/>Phase 1 可選 |
| 🐍 **Python** | 靜態爬取、表單發現、API 分析 | 生態豐富、開發快速 | **7種策略**<br/>FAST/CONSERVATIVE/BALANCED<br/>DEEP/AGGRESSIVE<br/>STEALTH/TARGETED | Phase 1 主力 |
| 📘 **TypeScript** | JavaScript 渲染、SPA 路由 | Playwright 自動化 | **5種模式**<br/>1. Basic Dynamic<br/>2. SPA Detection<br/>3. Network Interception<br/>4. Content Extraction<br/>5. Interaction Simulation | Phase 1 動態 |
| 🔷 **Go** | 高並發掃描、SSRF/CSPM/SCA | 高並發、低資源 | **3種專業掃描器**<br/>1. SSRF Scanner<br/>2. CSPM Scanner<br/>3. SCA Scanner | Phase 1 可選 |

### 引擎詳細說明

#### 🦀 Rust 引擎
- **位置**: `engines/rust_engine/`
- **狀態**: ✅ 完全可用
- **主要功能**:
  - Phase 0 快速偵察 (必須)
  - 敏感資訊掃描 (API Key, Token, 密鑰)
  - 技術棧指紋識別
- **詳細文檔**: [Rust Engine README](./engines/rust_engine/README.md)

#### 🐍 Python 引擎
- **位置**: `engines/python_engine/`
- **狀態**: ✅ 完全可用
- **主要功能**:
  - 靜態內容爬取
  - 表單與參數發現
  - API 端點分析
- **詳細文檔**: [Python Engine README](./engines/python_engine/README.md)

#### 📘 TypeScript 引擎
- **位置**: `engines/typescript_engine/`
- **狀態**: ✅ 完全可用
- **主要功能**:
  - Playwright 瀏覽器自動化
  - JavaScript 渲染
  - SPA 路由發現
- **詳細文檔**: [TypeScript Engine README](./engines/typescript_engine/README.md)

#### 🔷 Go 引擎
- **位置**: `engines/go_engine/`
- **狀態**: ✅ 完全可用
- **主要功能**:
  - SSRF 漏洞檢測（雲端元數據、內部微服務）
  - 雲端安全態勢管理（AWS/GCP/Azure）
  - 軟體組成分析（依賴包漏洞、許可證合規）
- **詳細文檔**: [Go Engine README](./engines/go_engine/README.md)

---

## 🎯 各引擎掃描模式總覽

### 📊 模式對照表

| 引擎 | 模式名稱 | 特性 | 適用場景 | 切換方式 |
|------|---------|------|---------|----------|
| **Rust** | FastDiscovery | 快速、輕量、無驗證 | Phase 0 必用 | `--mode fast` |
| | DeepAnalysis | 完整密鑰檢測+驗證 | Phase 1 深度 | `--mode deep` |
| | FocusedVerification | 針對性驗證 | AI 決策選擇 | `--mode focused` |
| **Python** | FAST | 深度2-3、極快 | 快速資產發現 | `strategy="FAST"` |
| | CONSERVATIVE | 深度2、低負載 | 避免觸發防護 | `strategy="CONSERVATIVE"` |
| | BALANCED | 深度3-4、平衡 | 日常掃描 | `strategy="BALANCED"` |
| | DEEP | 深度5-6、全面 | 完整覆蓋 | `strategy="DEEP"` |
| | AGGRESSIVE | 深度7+、最慢 | 完整測試 | `strategy="AGGRESSIVE"` |
| | STEALTH | 深度3、隱秘 | 避免檢測 | `strategy="STEALTH"` |
| | TARGETED | 自定義深度 | 特定目標 | `strategy="TARGETED"` |
| **TypeScript** | Basic Dynamic | 基礎渲染 | 傳統網站 | 默認模式 |
| | SPA Detection | 框架識別 | React/Vue/Angular | 自動檢測 |
| | Network Interception | AJAX/Fetch 攔截 | API 端點發現 | 自動啟用 |
| | Content Extraction | 深度 DOM 分析 | 資產發現 | 自動啟用 |
| | Interaction Simulation | 自動點擊/表單 | 互動內容 | 配置啟用 |
| **Go** | SSRF Scanner | 雲端元數據檢測 | 內網探測 | `ssrf-scanner.exe` |
| | CSPM Scanner | AWS 配置審計 | 雲端安全 | `cspm-scanner.exe` |
| | SCA Scanner | 依賴包漏洞 | 供應鏈安全 | `sca-scanner.exe` |

### 🔄 模式切換詳細說明

#### 1. **Rust 引擎模式切換**

**通過 CLI 參數切換**：
```bash
# Mode 1: 快速發現（Phase 0 使用）
./aiva-info-gatherer scan --url http://target.com --mode fast --timeout 10

# Mode 2: 深度分析（Phase 1 使用）
./aiva-info-gatherer scan --url http://target.com --mode deep --timeout 20

# Mode 3: 聚焦驗證（AI 決策使用）
./aiva-info-gatherer scan --url http://target.com --mode focused --verify-keys
```

**通過協調器自動選擇**：
```python
# Phase 0 自動使用 FastDiscovery
phase0_result = await coordinator.execute_phase0(
    scan_id="scan_001",
    targets=["https://example.com"]
)

# Phase 1 可指定使用 DeepAnalysis
phase1_result = await coordinator.execute_phase1(
    scan_id="scan_001",
    targets=["https://example.com"],
    selected_engines=["rust"],  # 自動使用 deep 模式
    max_depth=5
)
```

**配置參數**：
```python
config = {
    "mode": "deep",  # fast/deep/focused
    "timeout": 20,
    "max_depth": 5,
    "verify_keys": True  # 僅 deep/focused 模式
}
```

---

#### 2. **Python 引擎策略切換**

**通過 ScanStartPayload 切換**：
```python
from services.aiva_common.schemas import ScanStartPayload

# 快速掃描
request = ScanStartPayload(
    scan_id="scan_001",
    targets=["http://localhost:3000"],
    strategy="FAST",  # 關鍵參數
    max_depth=2
)

# 深度掃描
request = ScanStartPayload(
    scan_id="scan_002",
    targets=["http://localhost:3000"],
    strategy="DEEP",  # 關鍵參數
    max_depth=5
)

result = await orchestrator.execute_scan(request)
```

**通過協調器預設策略**：
```python
# 方式 1: 使用預設策略方法
result = await coordinator.execute_strategy_fast(scan_id, targets)     # FAST
result = await coordinator.execute_strategy_balanced(scan_id, targets) # BALANCED
result = await coordinator.execute_strategy_comprehensive(scan_id, targets) # DEEP

# 方式 2: 通過 Phase1 配置
result = await coordinator.execute_phase1(
    scan_id=scan_id,
    targets=targets,
    selected_engines=["python"],
    max_depth=5,  # 影響策略選擇
    strategy="AGGRESSIVE"  # 明確指定
)
```

**策略映射關係**：
```python
# aiva_common 標準策略 → Python 引擎策略
"quick" → "FAST"
"normal" → "BALANCED"
"full" → "AGGRESSIVE"
"stealth" → "STEALTH"
```

---

#### 3. **TypeScript 引擎模式切換**

**自動模式檢測**（無需手動切換）：
```typescript
// TypeScript 引擎會自動檢測並啟用相應模式：

// 1. 檢測到 React/Vue/Angular → 啟用 SPA Detection
if (page.url().includes('react') || hasReactRoot) {
    await enableSPADetection();
}

// 2. 監聽所有網路請求 → 自動啟用 Network Interception
page.on('request', captureRequest);
page.on('response', captureResponse);

// 3. 深度 DOM 分析 → Content Extraction 始終啟用
await extractDOMContent(page);

// 4. 互動模擬 → 通過配置控制
if (config.enableInteraction) {
    await simulateInteractions(page);
}
```

**通過協調器配置**：
```python
# TypeScript 引擎配置選項
ts_options = {
    "enable_interaction": True,  # 啟用互動模擬
    "wait_for_network": True,    # 等待網路請求完成
    "capture_websocket": True,   # 捕獲 WebSocket
    "max_wait_time": 5000        # 最大等待時間（ms）
}

result = await coordinator.execute_phase1(
    scan_id=scan_id,
    targets=targets,
    selected_engines=["typescript"],
    options=ts_options
)
```

**模式組合示例**：
```typescript
// 完整模式組合（所有5種模式同時啟用）
const scanConfig = {
    basicDynamic: true,          // 基礎渲染
    spaDetection: true,          // SPA 框架檢測
    networkInterception: true,   // 網路攔截
    contentExtraction: true,     // 內容提取
    interactionSimulation: true  // 互動模擬
};
```

---

#### 4. **Go 引擎掃描器切換**

**通過執行不同二進制**：
```bash
# 掃描器 1: SSRF 檢測
./ssrf-scanner.exe --url https://example.com --param image_url

# 掃描器 2: CSPM 審計
./cspm-scanner.exe --cloud aws --region us-east-1

# 掃描器 3: SCA 分析
./sca-scanner.exe --path ./project --lang nodejs
```

**通過協調器調用**：
```python
# 協調器會根據需求自動選擇掃描器
result = await coordinator.execute_phase1(
    scan_id=scan_id,
    targets=targets,
    selected_engines=["go"],  # 協調器內部會並行執行 3 個掃描器
    options={
        "enable_ssrf": True,   # 啟用 SSRF 掃描
        "enable_cspm": True,   # 啟用 CSPM 掃描
        "enable_sca": True     # 啟用 SCA 掃描
    }
)
```

**Go 引擎內部調度**：
```python
# go_adapter.py 的實現
async def scan(self, targets, options):
    tasks = []
    
    if options.get("enable_ssrf", True):
        tasks.append(self._run_ssrf_scanner(targets))
    
    if options.get("enable_cspm", True):
        tasks.append(self._run_cspm_scanner(targets))
    
    if options.get("enable_sca", True):
        tasks.append(self._run_sca_scanner(targets))
    
    # 並行執行多個掃描器
    results = await asyncio.gather(*tasks)
    return self._merge_results(results)
```

---

### 🎯 協調器如何選擇模式

協調器使用 **5 種預設策略** 自動選擇引擎和模式：

```python
# 1. 快速掃描 - 僅 Python (FAST 策略)
await coordinator.execute_strategy_fast(scan_id, targets)
# 內部映射: Python → FAST 策略，深度=2

# 2. 均衡掃描 - Python (BALANCED) + Rust (FastDiscovery)
await coordinator.execute_strategy_balanced(scan_id, targets)
# 內部映射: 
#   - Python → BALANCED 策略，深度=3
#   - Rust → fast 模式

# 3. 全面掃描 - Python (DEEP) + TypeScript + Rust (DeepAnalysis)
await coordinator.execute_strategy_comprehensive(scan_id, targets)
# 內部映射:
#   - Python → DEEP 策略，深度=5
#   - TypeScript → 所有模式啟用
#   - Rust → deep 模式

# 4. 激進掃描 - 四引擎全開
await coordinator.execute_strategy_aggressive(scan_id, targets)
# 內部映射:
#   - Python → AGGRESSIVE 策略，深度=7
#   - TypeScript → 所有模式啟用
#   - Rust → deep 模式
#   - Go → 3 個掃描器並行

# 5. 智能掃描 - Phase 0 分析後自動選擇
await coordinator.execute_strategy_smart(scan_id, targets)
# 動態決策:
#   1. Phase 0: Rust (fast 模式)
#   2. 分析結果（技術棧、框架）
#   3. 自動選擇最佳引擎組合和模式
```

**智能掃描決策邏輯**：
```python
# 協調器內部的智能決策示例
if "Angular" in phase0_result.tech_stack:
    selected_engines.append("typescript")  # 啟用 TypeScript (SPA 模式)

if phase0_result.has_api_endpoints:
    python_strategy = "DEEP"  # 提升 Python 掃描深度

if phase0_result.cloud_metadata_detected:
    selected_engines.append("go")  # 啟用 Go (SSRF 掃描器)
```

---

### 📋 模式選擇決策樹

```
掃描需求分析
├─ 需要快速驗證？
│  └─ 使用 execute_strategy_fast
│     └─ Python (FAST) 單引擎
│
├─ 一般 Web 應用？
│  └─ 使用 execute_strategy_balanced
│     └─ Python (BALANCED) + Rust (fast)
│
├─ SPA 應用（React/Vue/Angular）？
│  └─ 使用 execute_strategy_comprehensive
│     └─ Python (DEEP) + TypeScript (全模式) + Rust (deep)
│
├─ 大型應用或完整評估？
│  └─ 使用 execute_strategy_aggressive
│     └─ 四引擎全開（最大覆蓋）
│
└─ 不確定目標類型？
   └─ 使用 execute_strategy_smart
      └─ Phase 0 分析 → 自動決策
```

---

### 💡 最佳實踐建議

1. **開發測試階段**
   - 使用 `execute_strategy_fast` 快速驗證
   - 單引擎（Python FAST）即可

2. **日常掃描**
   - 使用 `execute_strategy_balanced` 平衡速度與覆蓋
   - Python + Rust 雙引擎

3. **完整安全評估**
   - 使用 `execute_strategy_aggressive` 最大覆蓋
   - 四引擎全開，所有模式啟用

4. **未知目標**
   - 使用 `execute_strategy_smart` 智能決策
   - 根據 Phase 0 結果自動選擇最佳組合

5. **自定義需求**
   - 直接調用 `execute_phase1`
   - 明確指定引擎列表和配置參數

---

## 📁 目錄結構

```
services/scan/
├── coordinators/                    # 協調器模組
│   ├── engines/                     # 適配器層 (888 lines)
│   │   ├── base.py                 # 基礎適配器（實際檔名）
│   │   ├── python_adapter.py       # Python 引擎適配器
│   │   ├── typescript_adapter.py   # TypeScript 引擎適配器
│   │   ├── rust_adapter.py         # Rust 引擎適配器
│   │   ├── go_adapter.py           # Go 引擎適配器
│   │   └── __init__.py             # 模組初始化
│   ├── multi_engine_coordinator.py # 多引擎協調器 (647 lines, 複雜度 17)
│   ├── scan_models.py              # 數據模型定義
│   ├── unified_scan_engine.py      # 統一掃描引擎
│   ├── target_generators/          # 測試目標生成器
│   ├── README.md                    # 協調器文檔
│   ├── COORDINATOR_ACTUAL_STATUS.md # 狀態報告
│   └── COORDINATOR_ENGINE_INTEGRATION_DESIGN.md # 引擎整合設計
├── engines/                         # 四個掃描引擎
│   ├── rust_engine/                # Rust 引擎（Phase 0 必須）
│   ├── python_engine/              # Python 引擎（Phase 1 主力）
│   ├── typescript_engine/          # TypeScript 引擎（Phase 1 動態）
│   ├── go_engine/                  # Go 引擎（Phase 1 可選）
│   └── ENGINES_DOCUMENTATION_INDEX.md # 引擎文檔索引
├── archived_docs/                   # 歷史文檔歸檔
├── command_handler.py              # AI 命令處理器 (461 lines)
├── README.md                        # 本文檔（架構總覽）
├── SCAN_USER_GUIDE.md              # 使用者手冊
├── SCAN_FLOW_DIAGRAMS.md           # 流程圖解（基準文檔，不可修改）
├── SCAN_MODULE_RESTORATION_PLAN.md # 修復計劃與完成狀態
└── __init__.py                      # 模組初始化
```

---

## 📄 重要檔案說明

### 核心檔案

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `command_handler.py` | AI 命令處理器，處理 SCAN_PHASE0 和 SCAN_PHASE1 命令 | ✅ 完成 (461 lines) |
| `__init__.py` | Scan 模組初始化，定義公開接口 | ✅ 完成 |

### 文檔檔案

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `README.md` | 架構總覽（本文檔） | ✅ 已更新 |
| `SCAN_USER_GUIDE.md` | 使用者手冊，包含 AI 命令接口使用範例 | ✅ 已更新 |
| `SCAN_FLOW_DIAGRAMS.md` | 流程圖解（**基準文檔，不可修改**） | 📌 基準 |
| `SCAN_MODULE_RESTORATION_PLAN.md` | 修復計劃與重構完成狀態 | ✅ 已更新 |

### 協調器檔案 (coordinators/)

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `multi_engine_coordinator.py` | 多引擎協調器，複雜度從 171 降至 17 | ✅ 完成 (647 lines) |
| `scan_models.py` | 數據模型定義（遵循 aiva_common 規範） | ✅ 完成 |
| `unified_scan_engine.py` | 統一掃描引擎 | ✅ 完成 |
| `engines/base.py` | 基礎適配器（統一引擎接口） | ✅ 完成 |
| `engines/*_adapter.py` | 各引擎適配器（Python/TypeScript/Rust/Go） | ✅ 完成 (888 lines) |
| `README.md` | 協調器總覽文檔 | ✅ 已更新 |
| `COORDINATOR_USAGE_GUIDE.md` | 協調器使用指南 | ✅ 新增 |
| `COORDINATOR_ACTUAL_STATUS.md` | 實際狀態報告 | ✅ 已更新 |

### 引擎檔案 (engines/)

| 引擎 | README 位置 | 狀態 |
|------|-------------|------|
| Rust | `engines/rust_engine/README.md` | ✅ Phase 0 可用 |
| Python | `engines/python_engine/README.md` | ✅ Phase 1 可用 |
| TypeScript | `engines/typescript_engine/README.md` | ✅ Phase 1 可用 |
| Go | `engines/go_engine/README.md` | ✅ Phase 1 可用 |

### 歸檔目錄

- `archived_docs/` - 歷史文檔歸檔（舊版 RabbitMQ 相關文檔）

---

## 📊 系統統計

- **總檔案數**: 139 個檔案 (Python: 39, TypeScript: 17, Rust: 9, Go: 30)
- **程式碼規模**: 22,000+ 行代碼
- **核心組件**:
  - MultiEngineCoordinator: 647 lines (複雜度 17，從 171 降低 90%)
  - 適配器層: 888 lines (統一四引擎接口)
  - 命令處理器: 461 lines (AI 命令接口)
- **支援協議**: HTTP/HTTPS、WebSocket、GraphQL、gRPC
- **輸出格式**: SARIF 2.1.0、JSON、XML、CSV

---

## 🛠️ 開發指南

### 開發環境

詳細的開發環境設置請參考：
- 📘 [使用者手冊](./SCAN_USER_GUIDE.md)
- 🔧 [命令處理器](./command_handler.py)
- 📊 [流程圖解](./SCAN_FLOW_DIAGRAMS.md)

### 新增引擎

1. 在 `coordinators/engines/` 建立新的適配器
2. 繼承 `BaseEngineAdapter`
3. 實現 `scan_async()` 和 `validate_config()` 方法
4. 在 `MultiEngineCoordinator` 註冊新引擎

### 測試

```bash
# 測試 Phase 0
python test_ai_command_scan.py

# 測試多引擎協調
python test_two_phase_scan.py

# 測試命令處理器
python test_command_handler_quick.py
```

---

## 🔧 修復規範

**保留未使用函數原則**: 在程式碼修復過程中，若發現有定義但尚未使用的函數或方法，只要不影響程式正常運作，建議予以保留。這些函數可能是：
- 預留的 API 端點或介面
- 未來功能的基礎架構
- 測試或除錯用途的輔助函數
- 向下相容性考量的舊版介面

---

## 📋 更新記錄

- **2025年11月21日**: 完成適配器模式重構 (v2.1)
- **2025年11月17日**: 第二次完整修復
- **2025年10月**: 初始版本 (v2.0)

---
