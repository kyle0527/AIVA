# AIVA 外部模組數據流分析完整報告

**生成日期**: 2026-01-13  
**分析範圍**: 外部功能模組 + 掃描引擎  
**分析工具**: Python Flow Analyzer + Rust/Go/TypeScript Tools

---

## 📊 總覽統計

### 全系統數據流統計

| 類別 | 模組數 | 總 Flows | 相同起終點不同路徑 | 平均 Flow 長度 |
|------|--------|----------|-------------------|---------------|
| **功能模組 (Python)** | 5 | 203 | 13 | 2-4 層 |
| **功能模組 (Go)** | 1 | 4 | 0 | 2 層 |
| **功能模組 (Rust)** | 1 | 4 | 0 | 2 層 |
| **掃描引擎 (Rust)** | 1 | 4 | 0 | 2 層 |
| **掃描引擎 (TypeScript)** | 1 | 3 | 3 | 2-3 層 |
| **總計** | **10** | **222** | **16** | **2-4 層** |

### 關鍵發現

✅ **總流程數**: 222 個數據流路徑  
✅ **多路徑設計**: 16 組相同起終點但路徑不同的流程 (7.2%)  
✅ **模組化設計**: 平均每個模組 22.2 個流程  
⚠️ **複雜度集中**: XSS 模組佔 49% 的流程 (109/222)

---

## 🔍 功能模組詳細分析

### 1. Python 功能模組

#### 1.1 function_xss - XSS 漏洞檢測
**最複雜的外部模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 109 | 最多的數據流路徑 |
| **唯一起終點組合** | 90 | 90 種不同的調用路徑 |
| **多路徑組合** | 13 | 13 組相同目標的不同實現方式 |
| **架構類型** | 完整架構 | 支援多種 XSS 檢測場景 |

**數據流特徵**:
- ✅ **多層次架構設計** - 支援 DOM XSS、Reflected XSS、Stored XSS
- ✅ **豐富的測試路徑** - 13 組不同路徑到達相同檢測器
- ✅ **Payload 多樣化** - 包含繞過技術和編碼變換
- ⚠️ **高複雜度** - 需要仔細維護流程關係

**多路徑範例**:

**範例 1: XssPayloadGenerator (2 條路徑)**
```
路徑 A: main → run_reflected_test → XssPayloadGenerator
路徑 B: main → run_stored_test → XssPayloadGenerator
```
→ **設計意圖**: 反射型 vs 儲存型 XSS 測試走不同路徑，但使用同一個 payload 生成器

**範例 2: converter (2 條路徑)**
```
路徑 A: retireJs → requester → converter (長度 3)
路徑 B: retireJs → main_scanner → scan_uri → requester → converter (長度 6)
```
→ **設計意圖**: 直接調用 vs 透過掃描器調用，支援不同使用場景

**分析結果位置**: `module_analysis/function_xss/`

---

#### 1.2 function_ssrf - SSRF 漏洞檢測
**第二複雜的功能模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 35 | 豐富的檢測路徑 |
| **唯一起終點組合** | 32 | 32 種不同的調用方式 |
| **多路徑組合** | 3 | 3 組同步/異步實現 |
| **架構類型** | 完整架構 | 支援內網探測、雲端檢測 |

**數據流特徵**:
- ✅ **內網探測能力** - 支援私有 IP 段掃描
- ✅ **雲端元數據檢測** - AWS/Azure/GCP 元數據服務
- ✅ **雙重調用路徑** - 直接調用 vs 任務系統調用
- ✅ **協議走私支援** - HTTP 協議繞過檢測

**多路徑範例**:

**範例 1: ParamSemanticsAnalyzer (2 條路徑)**
```
路徑 A: run → ParamSemanticsAnalyzer (長度 2) [直接調用]
路徑 B: run → _execute_task → process_task → ParamSemanticsAnalyzer (長度 4) [任務系統]
```
→ **設計意圖**: CLI 直接執行 vs Worker 異步執行

**範例 2: InternalAddressDetector (2 條路徑)**
```
路徑 A: run → InternalAddressDetector (長度 2) [直接調用]
路徑 B: run → _execute_task → process_task → InternalAddressDetector (長度 4) [任務系統]
```
→ **設計意圖**: 同步檢測 vs 背景任務檢測

**分析結果位置**: `module_analysis/function_ssrf/`

---

#### 1.3 function_sqli - SQL 注入檢測

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 32 | 完整的 SQL 注入檢測 |
| **唯一起終點組合** | 32 | 每條流程都是獨特路徑 |
| **多路徑組合** | 0 | 單一路徑設計 |
| **架構類型** | 完整架構 | 支援多種資料庫類型 |

**數據流特徵**:
- ✅ **多資料庫支援** - MySQL, PostgreSQL, MSSQL, Oracle
- ✅ **NoSQL 整合** - 包含 NoSQLMap 外部工具
- ✅ **多種注入技術** - 時間盲注、布林盲注、聯合查詢
- ✅ **單一路徑設計** - 每個檢測器有明確的調用路徑

**流程特徵**:
- 無多路徑設計，每個功能有專屬入口
- 流程清晰，易於追蹤和維護
- 適合需要精確控制的注入測試場景

**分析結果位置**: `module_analysis/function_sqli/`

---

#### 1.4 function_idor - IDOR 漏洞檢測

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 19 | 訪問控制檢測 |
| **唯一起終點組合** | 19 | 每條流程都是獨特路徑 |
| **多路徑組合** | 0 | 單一路徑設計 |
| **架構類型** | 簡單架構 | IDOR 和權限提升檢測 |

**數據流特徵**:
- ✅ **對象引用檢測** - 直接對象引用漏洞
- ✅ **權限提升測試** - 水平/垂直權限提升
- ✅ **參數遍歷** - ID 參數模糊測試
- ✅ **清晰設計** - 單一路徑，易於理解

**流程特徵**:
- 簡單直接的調用關係
- 專注於訪問控制問題
- 適合快速檢測和驗證

**分析結果位置**: `module_analysis/function_idor/`

---

#### 1.5 function_bizlogic - 業務邏輯漏洞

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 8 | 業務邏輯檢測 |
| **唯一起終點組合** | 8 | 每條流程都是獨特路徑 |
| **多路徑組合** | 0 | 單一路徑設計 |
| **架構類型** | 簡單架構 | 業務流程安全檢測 |

**數據流特徵**:
- ✅ **支付流程檢測** - 價格篡改、支付繞過
- ✅ **工作流程檢測** - 步驟跳過、狀態篡改
- ✅ **限制繞過** - 速率限制、數量限制繞過
- ✅ **最小化設計** - 8 個核心流程

**流程特徵**:
- 最簡潔的模組設計
- 專注於高價值業務邏輯漏洞
- 流程數少但影響大

**分析結果位置**: `module_analysis/function_bizlogic/`

---

### 2. Go 功能模組

#### 2.1 function_authn_go - 身份驗證

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 4 | 身份驗證檢測 |
| **唯一起終點組合** | 4 | 每條流程都是獨特路徑 |
| **多路徑組合** | 0 | 單一路徑設計 |
| **語言** | Go | 高性能身份驗證檢測 |

**數據流特徵**:
- ✅ **Go 實現** - 高性能並發檢測
- ✅ **JWT 分析** - Token 驗證和繞過
- ✅ **Session 檢測** - 會話管理問題
- ✅ **暴力破解** - 認證爆破測試

**流程特徵**:
- Go 語言的並發優勢
- 適合大規模認證測試
- 與 Python 模組互補

**分析結果位置**: `module_analysis/function_authn_go/`

---

### 3. Rust 功能模組

#### 3.1 function_crypto - 加密相關漏洞

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 4 | 加密問題檢測 |
| **唯一起終點組合** | 4 | 每條流程都是獨特路徑 |
| **多路徑組合** | 0 | 單一路徑設計 |
| **語言** | Rust | 安全的加密分析 |

**數據流特徵**:
- ✅ **Cookie 安全分析** - Secure/HttpOnly 標記檢查
- ✅ **Header 安全檢查** - HSTS, CSP 策略分析
- ✅ **JavaScript 加密問題** - 硬編碼密鑰、弱算法檢測
- ✅ **TLS/SSL 分析** - 協議版本、密碼套件檢查

**數據流路徑**:

**Flow 1**: `main` → `cookie_analyzer::analyze_cookies`
- 功能：Cookie 安全檢查
- 檢測：缺失 Secure/HttpOnly、敏感 Cookie、SameSite

**Flow 2**: `main` → `header_analyzer::analyze_headers`
- 功能：HTTP Header 安全
- 檢測：HSTS、CSP、X-Frame-Options、X-Content-Type-Options

**Flow 3**: `main` → `js_crypto_analyzer::scan_javascript`
- 功能：JavaScript 加密問題
- 檢測：硬編碼密鑰、弱加密算法、JWT 問題、不安全存儲

**Flow 4**: `main` → `tls_analyzer::analyze_tls`
- 功能：TLS/SSL 分析
- 檢測：協議版本、密碼套件強度（異步實現）

**架構特徵**:
- ✅ **中央調度器模式** - main 根據命令分派到對應 analyzer
- ✅ **模組化設計** - 4 個 analyzer 完全獨立
- ✅ **清晰的數據流** - 單向調用，無循環依賴
- ✅ **Rust 安全性** - 內存安全，無數據競爭

**分析結果位置**: `services/features/function_crypto/`

---

### 4. 掃描引擎

#### 4.1 rust_engine - Rust 掃描引擎

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 4 | 核心掃描流程 |
| **唯一起終點組合** | 4 | 每條流程都是獨特路徑 |
| **多路徑組合** | 0 | 單一路徑設計 |
| **語言** | Rust | 高性能掃描引擎 |

**數據流路徑**:

**Flow 1**: `scan_single_target` → `EndpointDiscoverer::new`
- 功能：初始化端點發現器
- 用途：Web 端點探測和識別

**Flow 2**: `scan_single_target` → `JsAnalyzer::new`
- 功能：初始化 JavaScript 分析器
- 用途：前端代碼安全分析

**Flow 3**: `scan_single_target` → `SensitiveInfoScanner::with_mode`
- 功能：初始化敏感信息掃描器
- 用途：數據洩漏和敏感信息檢測

**Flow 4**: `scan_single_target` → `AttackSurfaceAssessor::new`
- 功能：初始化攻擊面評估器
- 用途：系統風險評估和漏洞優先級排序

**引擎特色**:
- ✅ **並行處理** - 使用 rayon 實現並發掃描
- ✅ **異步支持** - tokio 異步運行時
- ✅ **正則優化** - aho-corasick 高效模式匹配
- ✅ **SPA/Cloud 識別** - 現代應用架構支援

**分析結果位置**: `services/scan/rust_engine/`

---

#### 4.2 typescript_engine - TypeScript 掃描引擎

| 指標 | 數值 | 說明 |
|------|------|------|
| **總流程數** | 3 | 核心掃描流程 |
| **唯一起終點組合** | 2 | 有共享終點 |
| **多路徑組合** | 3 | 3 組不同實現路徑 |
| **語言** | TypeScript | 前端專用掃描引擎 |

**數據流特徵**:
- ✅ **TypeScript Compiler API** - 深度語法分析
- ✅ **ES6+ 模組分析** - 現代 JavaScript 支援
- ✅ **多路徑設計** - 支援不同調用場景
- ✅ **前端專精** - 專注於前端安全問題

**多路徑範例**:
- 3 條流程展示了不同的掃描入口點
- 支援 CLI、API、整合三種調用方式
- 靈活的架構適應不同使用場景

**分析結果位置**: `services/scan/typescript_engine/`

---

## 📊 多路徑設計分析

### 多路徑統計

| 模組名稱 | 多路徑組數 | 佔總流程比例 | 設計意圖 |
|---------|-----------|-------------|---------|
| function_xss | 13 | 11.9% | 支援多種測試場景 |
| function_ssrf | 3 | 8.6% | 同步/異步雙重路徑 |
| typescript_engine | 3 | 100% | 多入口調用支援 |
| **總計** | **16** | **7.2%** | **靈活性與可維護性平衡** |

### 多路徑設計模式

#### 模式 1: 測試場景分離 (function_xss)
```
同一個檢測器，根據不同測試類型走不同路徑：
- 反射型 XSS 測試路徑
- 儲存型 XSS 測試路徑
- DOM XSS 測試路徑
```
**優點**: 清晰的場景分離，易於針對性測試

#### 模式 2: 執行方式分離 (function_ssrf)
```
同一個分析器，根據調用方式走不同路徑：
- 直接調用路徑 (CLI/API)
- 任務系統路徑 (Worker/Background)
```
**優點**: 支援同步和異步執行，靈活性高

#### 模式 3: 入口點多樣化 (typescript_engine)
```
同一個引擎，支援多種入口：
- CLI 命令行入口
- API 程式化入口
- 整合系統入口
```
**優點**: 多種整合方式，適應不同使用場景

---

## 🎯 架構設計評估

### 優秀設計實踐

✅ **模組化設計**
- 每個功能模組職責明確
- 低耦合，高內聚
- 易於獨立測試和維護

✅ **多語言整合**
- Python: 靈活的檢測邏輯
- Rust: 高性能掃描引擎
- Go: 並發認證測試
- TypeScript: 前端專用分析

✅ **多路徑支援**
- 7.2% 的流程支援多路徑
- 平衡了靈活性和複雜度
- 主要用於關鍵功能（XSS、SSRF）

✅ **清晰的流程設計**
- 92.8% 的流程是單一路徑
- 降低維護成本
- 易於理解和除錯

### 改進建議

⚠️ **XSS 模組複雜度**
- 109 個流程佔總數 49%
- 建議：拆分為子模組（DOM XSS、Reflected XSS、Stored XSS）
- 好處：降低單一模組複雜度，提高可維護性

⚠️ **統一多路徑模式**
- 目前三種不同的多路徑設計
- 建議：制定統一的多路徑設計規範
- 好處：統一開發模式，降低學習成本

⚠️ **語言檢測準確性**
- 部分 Python 模組被標記為 "Unknown"
- 建議：改進語言檢測邏輯
- 好處：更準確的統計和分類

---

## 📈 流程長度分析

### 流程長度分布

| 長度 | 流程數 | 百分比 | 說明 |
|------|--------|--------|------|
| 2 層 | ~150 | 67.6% | 直接調用，最常見 |
| 3 層 | ~50 | 22.5% | 中間層調用 |
| 4 層 | ~15 | 6.8% | 任務系統調用 |
| 5+ 層 | ~7 | 3.1% | 複雜調用鏈 |

**分析**:
- 大部分流程保持 2-3 層，符合良好的架構設計
- 4+ 層的流程主要來自任務系統和工具鏈整合
- 避免過深的調用鏈，保持程式碼可讀性

---

## 🔧 工具鏈整合

### 分析工具

1. **Python Flow Analyzer** (`aiva_flow_analyzer.py`)
   - Python AST 分析
   - 函數調用追蹤
   - 數據流路徑生成

2. **Rust Analyzer** (`rust_analyzer`)
   - Cargo.toml 解析
   - Rust 函數圖生成
   - CLI 指令提取

3. **Go Analyzer** (`go2mermaid`)
   - Go AST 分析
   - 模組依賴追蹤

4. **TypeScript Analyzer** (`ts2mermaid`)
   - TypeScript Compiler API
   - ES6+ 模組分析

### 批次處理工具

**aiva_external_module_batch_classifier.py**
- 掃描所有外部模組
- 統一格式轉換
- 生成整合報告
- 支援多語言模組

---

## 📝 總結

### 關鍵指標

| 指標 | 數值 | 評價 |
|------|------|------|
| **總模組數** | 10 | 適中的模組規模 |
| **總流程數** | 222 | 豐富的檢測能力 |
| **多路徑率** | 7.2% | 良好的平衡 |
| **平均流程長度** | 2-3 層 | 優秀的架構設計 |
| **語言多樣性** | 4 種 | 充分利用語言優勢 |

### 整體評估

🎯 **成熟度**: ⭐⭐⭐⭐☆ (4/5)
- 完整的功能覆蓋
- 清晰的模組劃分
- 良好的多語言整合

🎯 **可維護性**: ⭐⭐⭐⭐☆ (4/5)
- 大部分流程清晰簡單
- XSS 模組需要優化
- 統一的分析工具鏈

🎯 **擴展性**: ⭐⭐⭐⭐⭐ (5/5)
- 模組化設計易於擴展
- 支援多語言整合
- 靈活的多路徑設計

---

## 📋 附錄

### A. 完整模組清單

#### 功能模組 (Features)
1. function_xss - XSS 漏洞檢測 (Python, 109 flows)
2. function_sqli - SQL 注入檢測 (Python, 32 flows)
3. function_ssrf - SSRF 漏洞檢測 (Python, 35 flows)
4. function_idor - IDOR 漏洞檢測 (Python, 19 flows)
5. function_bizlogic - 業務邏輯漏洞 (Python, 8 flows)
6. function_authn_go - 身份驗證 (Go, 4 flows)
7. function_crypto - 加密相關漏洞 (Rust, 4 flows)

#### 掃描引擎 (Scan)
8. rust_engine - Rust 掃描引擎 (Rust, 4 flows)
9. typescript_engine - TypeScript 掃描引擎 (TypeScript, 3 flows)
10. scan_engine - 通用掃描引擎 (0 flows)

### B. 分析數據位置

- **模組分析結果**: `module_analysis/*/analysis_results.json`
- **整合報告**: `external_module_reports/`
- **內部文檔**: `services/core/aiva_core/internal_exploration/`

### C. 相關工具

- **流程分析器**: `services/core/aiva_core/internal_exploration/aiva_flow_analyzer.py`
- **批次分類器**: `services/core/aiva_core/internal_exploration/python_tools/aiva_external_module_batch_classifier.py`
- **流程分析腳本**: `analyze_external_flows.py`

---

**報告結束**
