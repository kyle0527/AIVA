# AIVA 掃描引擎統一文檔索引

**更新日期**: 2025-11-19  
**版本**: 1.0.0  
**狀態**: 各引擎獨立文檔已建立

---

## 📚 文檔導航

### Rust Engine (Phase0 核心引擎)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `rust_engine/README.md` | 引擎概述、架構、性能數據 |
| **使用指南** | `rust_engine/USAGE_GUIDE.md` | ✨ **新增** - 詳細使用說明 |
| **驗證狀態** | `rust_engine/WORKING_STATUS_2025-11-19.md` | 實際驗證結果 |
| **優化路線** | `rust_engine/OPTIMIZATION_ROADMAP.md` | 3階段優化計劃 |
| **Phase0 計劃** | `rust_engine/PHASE0_IMPLEMENTATION_PLAN.md` | 實施細節 |

**快速開始**:
```bash
cd services/scan/engines/rust_engine
cargo build --release
./target/release/aiva-info-gatherer scan \
  --url http://target.com --mode fast --timeout 10
```

---

### Python Engine (Phase1 主力引擎)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `python_engine/README.md` | 引擎概述 |
| **驗證計劃** | `python_engine/VALIDATION_TEST_PLAN.md` | ✨ **新增** - 驗證測試計劃 |
| **Worker** | `python_engine/worker.py` | Phase1 Worker 實現 |
| **Orchestrator** | `python_engine/scan_orchestrator.py` | 掃描編排器 |

**特色功能**:
- ✅ 靜態爬取 (90% 完成)
- ✅ 表單發現 (完整實現)
- ✅ API 分析 (完整實現)
- ✅ 動態渲染 (Playwright 整合)
- ✅ JavaScript 分析 (敏感資訊檢測)

**待驗證項目**:
- Phase0 結果整合
- 實際靶場測試
- 去重邏輯驗證

---

### TypeScript Engine (Phase1 SPA 專家)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `typescript_engine/README.md` | 引擎概述 |
| **驗證狀態** | `typescript_engine/VALIDATION_STATUS.md` | ✨ **新增** - 代碼完整度評估 |
| **Worker** | `typescript_engine/worker.py` | Phase1 Worker 實現 |
| **Scan Service** | `typescript_engine/src/services/scan-service.ts` | 核心掃描邏輯 (440行) |
| **網路攔截器** | `typescript_engine/src/services/network-interceptor.service.ts` | AJAX 攔截 |

**特色功能** (獨有):
- ✅ SPA 路由發現 (React/Vue/Angular)
- ✅ 動態 AJAX 攔截
- ✅ WebSocket 檢測
- ✅ 事件處理器提取
- ✅ 動態表單識別

**待驗證項目**:
- 編譯和構建測試
- Juice Shop SPA 測試
- Worker 整合測試
- 性能和內存測試

**快速開始**:
```bash
cd services/scan/engines/typescript_engine
npm install
npm run install:browsers
npm run build
```

---

### Go Engine (Phase1 高並發掃描器)

| 文檔 | 路徑 | 說明 |
|------|------|------|
| **README** | `go_engine/README.md` | 引擎概述 |
| **Worker** | `go_engine/worker.py` | Phase1 Worker 實現 |
| **SSRF Scanner** | `go_engine/ssrf_scanner/` | SSRF 漏洞檢測 |
| **CSPM Scanner** | `go_engine/cspm_scanner/` | 雲端安全檢測 |
| **SCA Scanner** | `go_engine/sca_scanner/` | 依賴漏洞分析 |

**特色功能**:
- ✅ SSRF 檢測 (30-60秒/目標)
- ✅ CSPM 雲端配置檢測 (1-2分鐘/目標)
- ✅ SCA 依賴漏洞掃描 (2-3分鐘/目標)
- ✅ 高並發處理

**構建指令**:
```bash
# Windows
cd services/scan/engines/go_engine
.\build_scanners.ps1

# Linux/macOS
chmod +x build_scanners.sh
./build_scanners.sh
```

---

## 📊 引擎對比速查表

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

### v1.0.0 (2025-11-19)
- ✅ Rust Engine 驗證完成,生產可用
- ✅ Rust USAGE_GUIDE.md 建立
- ✅ Python VALIDATION_TEST_PLAN.md 建立
- ✅ TypeScript VALIDATION_STATUS.md 建立
- ✅ Go Engine 基礎文檔完整
- ✅ 統一索引文檔建立

### 預定 v1.1.0
- Python Engine 驗證完成
- TypeScript Engine 驗證完成
- Go Engine 驗證完成
- 各引擎 USAGE_GUIDE.md 完整

### 預定 v2.0.0
- 所有引擎優化完成
- 多引擎協調優化
- 完整最佳實踐指南
