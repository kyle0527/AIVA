# AIVA 掃描引擎統一文檔索引

**更新日期**: 2025-11-22  
**版本**: 1.1.0  
**狀態**: 各引擎文檔完整，模式切換指南已添加

---

## 📚 文檔導航

### Rust Engine (Phase0 核心引擎)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `rust_engine/README.md` | 引擎概述、架構、**3種掃描模式** |
| **使用指南** | `rust_engine/USAGE_GUIDE.md` | 詳細使用說明、模式切換 |
| **驗證狀態** | `rust_engine/WORKING_STATUS_2025-11-19.md` | 實際驗證結果 |
| **優化路線** | `rust_engine/OPTIMIZATION_ROADMAP.md` | 3階段優化計劃 |
| **Phase0 計劃** | `rust_engine/PHASE0_IMPLEMENTATION_PLAN.md` | 實施細節 |

**支持的 3 種掃描模式**:
1. **FastDiscovery** - Phase 0 專用（10分鐘內）
2. **DeepAnalysis** - Phase 1 高效能（10-100x 快於 Python）
3. **FocusedVerification** - AI 決策專用（針對性驗證）

**快速開始**:
```bash
cd services/scan/engines/rust_engine
cargo build --release

# Mode 1: 快速發現
./target/release/aiva-info-gatherer scan --url http://target.com --mode fast

# Mode 2: 深度分析
./target/release/aiva-info-gatherer scan --url http://target.com --mode deep

# Mode 3: 聚焦驗證
./target/release/aiva-info-gatherer scan --url http://target.com --mode focused
```

---

### Python Engine (Phase1 主力引擎)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `python_engine/README.md` | 引擎概述、**7種掃描策略** |
| **驗證計劃** | `python_engine/VALIDATION_TEST_PLAN.md` | 驗證測試計劃 |
| **Worker** | `python_engine/worker.py` | Phase1 Worker 實現 |
| **Orchestrator** | `python_engine/scan_orchestrator.py` | 掃描編排器 |

**支持的 7 種掃描策略**:
1. **FAST** - 快速掃描（深度2-3，極快）
2. **CONSERVATIVE** - 保守掃描（深度2，低負載）
3. **BALANCED** - 平衡掃描（深度3-4，推薦）
4. **DEEP** - 深度掃描（深度5-6，全面）
5. **AGGRESSIVE** - 激進掃描（深度7+，最慢）
6. **STEALTH** - 隱秘掃描（深度3，避免檢測）
7. **TARGETED** - 目標掃描（自定義深度）

**特色功能**:
- ✅ 靜態爬取 (90% 完成)
- ✅ 表單發現 (完整實現)
- ✅ API 分析 (完整實現)
- ✅ 動態渲染 (Playwright 整合)
- ✅ JavaScript 分析 (敏感資訊檢測)

**策略切換示例**:
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
```

---

### TypeScript Engine (Phase1 SPA 專家)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `typescript_engine/README.md` | 引擎概述、**5種掃描模式** |
| **驗證狀態** | `typescript_engine/VALIDATION_STATUS.md` | 代碼完整度評估 |
| **Worker** | `typescript_engine/worker.py` | Phase1 Worker 實現 |
| **Scan Service** | `typescript_engine/src/services/scan-service.ts` | 核心掃描邏輯 (440行) |
| **網路攔截器** | `typescript_engine/src/services/network-interceptor.service.ts` | AJAX 攔截 |

**支持的 5 種掃描模式**（自動檢測）:
1. **Basic Dynamic** - 真實瀏覽器渲染 + JavaScript 執行
2. **SPA Detection** - React/Vue/Angular 框架識別 + 路由提取
3. **Network Interception** - AJAX/Fetch/WebSocket 完整攔截
4. **Content Extraction** - 深度 DOM 分析 + JS 變數提取
5. **Interaction Simulation** - 自動點擊、表單填寫、滾動觸發

**特色功能** (獨有):
- ✅ SPA 路由發現 (React/Vue/Angular)
- ✅ 動態 AJAX 攔截
- ✅ WebSocket 檢測
- ✅ 事件處理器提取
- ✅ 動態表單識別

**模式自動啟用**（無需手動切換）:
```typescript
// TypeScript 引擎會自動檢測並啟用相應模式：

// 檢測到 SPA 框架 → 啟用 SPA Detection
if (hasReactRoot || hasVueApp) {
    await enableSPADetection();
}

// 監聽網路請求 → 自動啟用 Network Interception
page.on('request', captureRequest);
page.on('response', captureResponse);

// 深度 DOM 分析 → Content Extraction 始終啟用
await extractDOMContent(page);
```

**快速開始**:
```bash
cd services/scan/engines/typescript_engine
npm install
npm run install:browsers
npm run build
node dist/index.js
```

---

### Go Engine (Phase1 高並發掃描器)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `go_engine/README.md` | 引擎概述、**3種專業掃描器** |
| **使用指南** | `go_engine/USAGE_GUIDE.md` | 詳細使用說明 |
| **Worker** | `go_engine/dispatcher/worker.py` | Phase1 Worker 實現 |

**支持的 3 種專業掃描器**:
1. **SSRF Scanner** - 服務端請求偽造檢測（雲端元數據、內部微服務）
2. **CSPM Scanner** - 雲端安全態勢管理（AWS/GCP/Azure 配置審計）
3. **SCA Scanner** - 軟體組成分析（依賴包漏洞、許可證合規）

**特色功能**:
- ✅ SSRF 檢測 (30-60秒/目標)
- ✅ CSPM 雲端配置檢測 (1-2分鐘/目標)
- ✅ SCA 依賴漏洞掃描 (2-3分鐘/目標)
- ✅ 高並發處理（10+ 並發）

**掃描器切換**:
```bash
# 掃描器 1: SSRF 檢測
./bin/ssrf-scanner.exe --url https://example.com --param image_url

# 掃描器 2: CSPM 審計
./bin/cspm-scanner.exe --cloud aws --region us-east-1

# 掃描器 3: SCA 分析
./bin/sca-scanner.exe --path ./project --lang nodejs
```

**通過協調器調用**（並行執行多個掃描器）:
```python
result = await coordinator.execute_phase1(
    scan_id=scan_id,
    targets=targets,
    selected_engines=["go"],
    options={
        "enable_ssrf": True,   # 啟用 SSRF 掃描
        "enable_cspm": True,   # 啟用 CSPM 掃描
        "enable_sca": True     # 啟用 SCA 掃描
    }
)
```

**構建指令**:
```bash
# Windows
cd services/scan/engines/go_engine
make build

# Linux/macOS
make build
```

---

## 📊 引擎對比速查表

### 掃描模式總覽

| 引擎 | 模式數量 | 模式類型 | 切換方式 | 適用場景 |
|------|---------|---------|---------|---------|
| **Rust** | **3種** | 命令行參數 | `--mode fast/deep/focused` | Phase 0 必用 + Phase 1 可選 |
| **Python** | **7種** | 策略參數 | `strategy="FAST/BALANCED/DEEP"` 等 | Phase 1 主力（靜態+動態） |
| **TypeScript** | **5種** | 自動檢測 | 無需手動切換（智能啟用） | Phase 1 SPA 專用 |
| **Go** | **3種** | 獨立二進制 | 執行不同掃描器 exe | Phase 1 專業掃描 |
| **總計** | **18種** | - | - | 全方位覆蓋 |

### 詳細模式對照

#### Rust Engine - 3 種模式

| 模式 | CLI 參數 | 特性 | 執行時間 | 使用階段 |
|------|---------|------|---------|---------|
| FastDiscovery | `--mode fast` | 輕量、無驗證、並行最大 | 10分鐘內 | Phase 0 必用 |
| DeepAnalysis | `--mode deep` | 完整密鑰檢測+驗證 | 10-30分鐘 | Phase 1 可選 |
| FocusedVerification | `--mode focused` | 針對性驗證、AI 決策 | 5-15分鐘 | AI 動態選擇 |

#### Python Engine - 7 種策略

| 策略 | 深度 | 速度 | 負載 | 使用場景 |
|------|------|------|------|---------|
| FAST | 2-3 | 極快 | 低 | 快速驗證、開發測試 |
| CONSERVATIVE | 2 | 快 | 低 | 避免觸發防護、謹慎掃描 |
| BALANCED | 3-4 | 中 | 中 | **日常掃描（推薦）** |
| DEEP | 5-6 | 慢 | 高 | 全面覆蓋、完整分析 |
| AGGRESSIVE | 7+ | 最慢 | 極高 | 完整測試、最大覆蓋 |
| STEALTH | 3 | 極慢 | 低 | 避免檢測、隱秘掃描 |
| TARGETED | 自定義 | 自定義 | 中 | 特定目標、自定義配置 |

#### TypeScript Engine - 5 種模式（自動啟用）

| 模式 | 觸發條件 | 功能 | 適用場景 |
|------|---------|------|---------|
| Basic Dynamic | 始終啟用 | 瀏覽器渲染 + JS 執行 | 所有網站 |
| SPA Detection | 檢測到 React/Vue/Angular | 框架識別 + 路由提取 | SPA 應用 |
| Network Interception | 始終啟用 | AJAX/Fetch/WebSocket 攔截 | API 端點發現 |
| Content Extraction | 始終啟用 | 深度 DOM 分析 + JS 變數 | 資產發現 |
| Interaction Simulation | 配置啟用 | 自動點擊、表單填寫 | 需互動的內容 |

#### Go Engine - 3 種掃描器

| 掃描器 | 二進制文件 | 功能 | 執行時間 | 適用場景 |
|--------|-----------|------|---------|---------|
| SSRF Scanner | `ssrf-scanner.exe` | 雲端元數據、內部探測 | 30-60秒/目標 | SSRF 漏洞檢測 |
| CSPM Scanner | `cspm-scanner.exe` | AWS/GCP/Azure 配置審計 | 1-2分鐘/目標 | 雲端安全評估 |
| SCA Scanner | `sca-scanner.exe` | 依賴漏洞、許可證分析 | 2-3分鐘/目標 | 供應鏈安全 |

---

### 功能矩陣

| 功能 | Rust | Python | TypeScript | Go |
|------|------|--------|-----------|-----|
| **Phase0 必執行** | ✅ | ❌ | ❌ | ❌ |
| **靜態爬取** | ⚠️ 字典 | ✅ 完整 | ✅ 完整 | ✅ 高並發 |
| **動態渲染** | ❌ | ✅ Playwright | ✅ Playwright | ❌ |
| **SPA 路由** | ❌ | ❌ | ✅ **獨有** | ❌ |
| **AJAX 攔截** | ❌ | ⚠️ 有限 | ✅ **最優** | ❌ |
| **表單發現** | ❌ | ✅ 靜態/動態 | ✅ 動態 | ❌ |
| **JS 分析** | ✅ 靜態 | ✅ 完整 | ✅ 動態 | ❌ |
| **端點發現** | ✅ 字典 | ✅ 爬取 | ✅ 爬取 | ✅ 爆破 |
| **SSRF 檢測** | ❌ | ❌ | ❌ | ✅ **獨有** |
| **CSPM** | ❌ | ❌ | ❌ | ✅ **獨有** |
| **SCA** | ❌ | ❌ | ❌ | ✅ **獨有** |
| **WebSocket** | ❌ | ❌ | ✅ **獨有** | ❌ |

### 性能對比

| 指標 | Rust | Python | TypeScript | Go |
|------|------|--------|-----------|-----|
| **掃描速度** | ⭐⭐⭐⭐⭐ (178ms) | ⭐⭐⭐ (~10-30s) | ⭐⭐⭐⭐ (~30-60s) | ⭐⭐⭐⭐⭐ (並發) |
| **內存使用** | ⭐⭐⭐⭐⭐ (~5MB) | ⭐⭐⭐ (~50-100MB) | ⭐⭐⭐ (~300-500MB) | ⭐⭐⭐⭐ (~20-50MB) |
| **並發能力** | ⭐⭐⭐⭐ (4+) | ⭐⭐⭐ (2-4) | ⭐⭐⭐ (2-3) | ⭐⭐⭐⭐⭐ (10+) |
| **資源消耗** | 極低 | 中等 | 較高 | 低 |

### 最佳使用場景

#### Rust Engine
- ✅ **必用場景**: Phase0 快速偵察 (每次掃描必執行)
- ✅ **技術棧識別**: 基礎指紋識別
- ✅ **敏感資訊掃描**: 配置洩漏、備份文件
- ✅ **多目標並行**: 4+ 目標同時掃描
- ⚠️ **限制**: 不支援 SPA、動態渲染

#### Python Engine
- ✅ **靜態網站爬取**: HTML 解析、表單提取
- ✅ **API 端點發現**: RESTful API 分析
- ✅ **表單參數挖掘**: 完整的表單處理
- ✅ **JavaScript 分析**: 敏感資訊檢測
- ✅ **動態內容**: Playwright 支援
- ⚠️ **限制**: 性能不如 Rust、SPA 支援有限

#### TypeScript Engine
- ✅ **現代 SPA 應用**: React、Vue、Angular
- ✅ **SPA 路由發現**: 動態路由提取 (獨有)
- ✅ **AJAX 端點攔截**: 實時 API 監控 (最優)
- ✅ **WebSocket 應用**: 實時通訊檢測 (獨有)
- ✅ **複雜互動流程**: 需要點擊/輸入觸發的內容
- ⚠️ **限制**: 資源消耗高、執行時間長

#### Go Engine
- ✅ **SSRF 檢測**: 專業 SSRF 漏洞掃描 (獨有)
- ✅ **雲端安全**: CSPM 配置檢測 (獨有)
- ✅ **依賴分析**: SCA 漏洞掃描 (獨有)
- ✅ **高並發掃描**: 大量 URL 同時處理
- ✅ **端口掃描**: 服務發現
- ⚠️ **限制**: 不支援動態渲染、前端分析

---

## 🎯 引擎選擇決策樹

```
開始掃描
    |
    ├─> Phase0 (必執行) → Rust Engine
    |                    ├─ 端點發現 (字典)
    |                    ├─ JS 分析 (靜態)
    |                    ├─ 技術棧識別
    |                    └─ 敏感資訊掃描
    |
    └─> Phase1 (根據目標類型選擇)
        |
        ├─> 目標是 SPA? (React/Vue/Angular)
        |   └─ YES → TypeScript Engine (必選)
        |             ├─ SPA 路由發現
        |             ├─ AJAX 攔截
        |             └─ WebSocket 檢測
        |
        ├─> 需要完整表單分析?
        |   └─ YES → Python Engine
        |             ├─ 靜態爬取
        |             ├─ 表單提取
        |             └─ API 分析
        |
        ├─> 需要 SSRF/CSPM/SCA?
        |   └─ YES → Go Engine
        |             ├─ SSRF 檢測
        |             ├─ CSPM 檢測
        |             └─ SCA 檢測
        |
        └─> 大量 URL 需要爆破?
            └─ YES → Go Engine (高並發)
```

---

## 📋 引擎狀態總覽

| 引擎 | 完成度 | 驗證狀態 | 優先級 | 下一步 |
|------|--------|---------|-------|--------|
| **Rust** | 80% | ✅ 已驗證 | 🔴 最高 | 優化 (A1, A2) |
| **Python** | 90% | ⚠️ 待驗證 | 🟡 中 | 實際測試 |
| **TypeScript** | 80% (代碼) | ⚠️ 待驗證 | 🔴 最高 | 編譯測試 |
| **Go** | 70% | ⚠️ 待驗證 | 🟢 低 | 功能增強 |

### Rust Engine ✅
- **狀態**: 生產可用
- **驗證結果**: ✅ 4 靶場並行成功
- **性能**: ✅ 713ms / 4 目標
- **待改進**: A1 (代碼去重), A2 (Regex 優化)
- **文檔**: ✅ 完整

### Python Engine ⚠️
- **狀態**: 代碼完整,待實測
- **已實現**: ✅ 90% 功能
- **待驗證**: 實際靶場測試、Phase0 整合
- **文檔**: ✅ 驗證計劃已建立

### TypeScript Engine ⚠️
- **狀態**: 代碼 80% 完整,未驗證
- **已實現**: ✅ SPA 路由、AJAX 攔截、WebSocket
- **待驗證**: 編譯、實測、Worker 整合
- **文檔**: ✅ 驗證狀態報告已建立

### Go Engine ⚠️
- **狀態**: 基礎可用
- **已實現**: ✅ SSRF、CSPM、SCA 掃描器
- **待改進**: 規則庫擴充、性能優化
- **文檔**: ✅ README 完整

---

## 🚀 後續行動計劃

### 短期 (1-2 週)

#### 優先級 1: Python Engine 驗證
```bash
# 目標: 確認 Python 引擎實際可用性
cd services/scan/engines/python_engine

# 1. 環境準備
playwright install

# 2. 執行驗證測試 (參考 VALIDATION_TEST_PLAN.md)
pytest test_validation.py

# 3. 記錄結果,更新文檔
```

#### 優先級 2: TypeScript Engine 驗證
```bash
# 目標: 確認 TypeScript 引擎實際可用性
cd services/scan/engines/typescript_engine

# 1. 編譯測試
npm install
npm run build

# 2. 單元測試
npm test

# 3. 實際靶場測試 (Juice Shop)
# 參考 VALIDATION_STATUS.md

# 4. 記錄結果,更新文檔
```

#### 優先級 3: Go Engine 驗證
```bash
# 目標: 驗證 Go 掃描器功能
cd services/scan/engines/go_engine

# 1. 構建所有掃描器
./build_scanners.sh  # Linux/macOS
.\build_scanners.ps1  # Windows

# 2. 單獨測試每個掃描器
./ssrf_scanner/worker.exe --task-file test_ssrf.json
./cspm_scanner/worker.exe --task-file test_cspm.json
./sca_scanner/worker.exe --task-file test_sca.json

# 3. 記錄結果
```

### 中期 (2-4 週)

#### Rust Engine 優化
- A1: 消除重複代碼 (60行)
- A2: Regex 編譯優化 (15-20% 性能提升)

#### Python Engine 改進
- Phase0 結果整合優化
- 去重邏輯增強
- 性能調優

#### TypeScript Engine 完善
- 錯誤處理增強
- Asset 去重優化
- 內存洩漏檢查

#### Go Engine 增強
- SSRF payload 變種
- CSPM 規則庫擴充 (50+ 規則)
- SCA 漏洞庫更新

### 長期 (1-2 月)

#### 多引擎協調優化
- 結果去重和聚合
- 動態引擎選擇策略
- 性能基準測試

#### 文檔和示例
- 完整使用示例
- 最佳實踐指南
- 故障排除手冊

---

## 📞 技術支持

### 各引擎聯絡點

- **Rust Engine**: 參考 `rust_engine/USAGE_GUIDE.md`
- **Python Engine**: 參考 `python_engine/VALIDATION_TEST_PLAN.md`
- **TypeScript Engine**: 參考 `typescript_engine/VALIDATION_STATUS.md`
- **Go Engine**: 參考 `go_engine/README.md`

### 通用資源

- **架構文檔**: `ENGINE_COMPLETION_ANALYSIS.md`
- **掃描流程**: `SCAN_FLOW_DIAGRAMS.md`
- **Schema 規範**: `services/aiva_common/schemas/`

---

## 📝 版本歷史

### v1.1.0 (2025-11-22) - 模式完整化 ✨ 最新
- ✅ **添加各引擎模式詳細說明**
  - Rust: 3種模式（FastDiscovery/DeepAnalysis/FocusedVerification）
  - Python: 7種策略（FAST/CONSERVATIVE/BALANCED/DEEP/AGGRESSIVE/STEALTH/TARGETED）
  - TypeScript: 5種模式（自動檢測啟用）
  - Go: 3種專業掃描器（SSRF/CSPM/SCA）
- ✅ **添加模式切換指南**
  - CLI 參數切換（Rust/Go）
  - 策略參數切換（Python）
  - 自動檢測說明（TypeScript）
  - 協調器集成示例
- ✅ **更新掃描模式總覽表格**
  - 18種模式完整對照
  - 使用場景說明
  - 切換方式詳解
- ✅ **協調器如何選擇引擎模式**
  - 5種預設策略映射關係
  - 智能決策邏輯
  - 手動配置示例
- ✅ **更新主 README 和協調器文檔**
  - 完整的模式切換教學
  - 實際使用場景示例
  - 協調器操作指南

### v1.0.0 (2025-11-19)
- ✅ Rust Engine 驗證完成,生產可用
- ✅ Rust USAGE_GUIDE.md 建立
- ✅ Python VALIDATION_TEST_PLAN.md 建立
- ✅ TypeScript VALIDATION_STATUS.md 建立
- ✅ Go Engine 基礎文檔完整
- ✅ 統一索引文檔建立

### 預定 v1.2.0
- Python Engine 驗證完成
- TypeScript Engine 驗證完成
- Go Engine 驗證完成
- 各引擎 USAGE_GUIDE.md 完整

### 預定 v2.0.0
- 所有引擎優化完成
- 多引擎協調優化
- 完整最佳實踐指南
