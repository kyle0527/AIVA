# AIVA 數據流分析完整報告

**生成日期**: 2026-01-13  
**最後更新**: 2026-01-18  
**分析範圍**: 多語言工具 + 功能模組  
**分析工具**: Python/Go/Rust/TypeScript AST 分析工具  
**架構狀態**: ✅ 已完成架構重構（語言層與業務邏輯層分離）

---

## 🎯 架構演進說明

### 當前架構 (v3.0)

```
語言層 (Language Layer) - 只負責 AST 解析
  ├── python_tools/aiva_flow_analyzer.py
  ├── go_tools/go2mermaid.go  
  ├── rust_tools/src/main.rs
  └── typescript_tools/ts2mermaid.ts
         ↓ (輸出 JSON Schema v3.3)
         
業務邏輯層 (Business Logic Layer)
  ├── aiva_internal_classifier.py (AIVAFlowClassifier)
  ├── aiva_internal_executor.py (FlowExecutor)
  ├── aiva_external_classifier.py (MultiLanguageClassifier)  
  └── aiva_external_executor.py (MultiLangExecutor)
```

### 本報告的定位

本報告記錄了使用多語言 AST 工具進行的**完整數據流分析結果**，包括：
- ✅ Python 功能模組分析（features/）
- ✅ Go 認證模組分析（function_authn_go）
- ✅ Rust 加密模組分析（function_crypto）
- ✅ TypeScript 掃描引擎分析（typescript_engine）

**注意**: 此報告為歷史分析記錄，當前架構已將分類/執行邏輯移至獨立腳本。

---

## 📊 總覽統計

### 全系統數據流統計

| 類別 | 模組數 | 總檔案 | 總 Flows | 總連接 | 總函數 |
|------|--------|--------|----------|--------|--------|
| **功能模組 (Python)** | 6 | 118 | 157 | 85 | - |
| **功能模組 (Rust)** | 1 | 5 | 0 | 0 | 16 |
| **掃描引擎 (Rust)** | 1 | 11 | 4 | 4 | 255 |
| **認證模組 (Go)** | 1 | 4 | 4 | 4 | - |
| **掃描引擎 (TypeScript)** | 1 | 11 | 3 | 3 | 75 |
| **總計** | **10** | **149** | **168** | **96** | **346+** |

---

## 🔍 功能模組詳細分析

### 1. Python 功能模組 (features_ready)

#### 1.1 function_xss - XSS 漏洞檢測
**最複雜的功能模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 47 | 包含多種 XSS 檢測引擎 |
| **數據流** | 83 | 最多的內部數據流 |
| **真實連接** | 35 | 複雜的模組間調用 |
| **數據源頭** | 6 | 6個入口點 |
| **架構類型** | 完整架構 | 多層次調用結構 |

**數據流特徵**:
- ✅ 多層次架構設計
- ✅ 包含 DOM XSS、Reflected XSS、Stored XSS 檢測
- ✅ 豐富的 payload 庫和繞過技術
- ⚠️ 存在 invalid escape sequence 警告（非阻斷性）

**分析結果位置**: `module_analysis/function_xss/`

---

#### 1.2 function_sqli - SQL 注入檢測
**第二複雜的功能模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 34 | 包含 NoSQL 和 SQL 注入檢測 |
| **數據流** | 48 | 豐富的數據流路徑 |
| **真實連接** | 30 | 良好的模組化設計 |
| **數據源頭** | 4 | 4個主要入口 |
| **架構類型** | 完整架構 | 多類型 SQL 注入支援 |

**數據流特徵**:
- ✅ 支援多種資料庫類型
- ✅ 包含 NoSQLMap 外部工具整合
- ✅ 時間盲注、布林盲注、聯合查詢等技術
- ⚠️ 包含 Python 2 代碼（NoSQLMap），有語法警告

**外部工具**:
- NoSQLMap (external_tools/NoSQLMap/) - 5個檔案有 Python 2/3 語法問題

**分析結果位置**: `module_analysis/function_sqli/`

---

#### 1.3 function_ssrf - SSRF 漏洞檢測
**中等複雜度模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 14 | SSRF 檢測與繞過 |
| **數據流** | 19 | 適中的數據流複雜度 |
| **真實連接** | 13 | 清晰的調用關係 |
| **數據源頭** | 5 | 5個入口點 |
| **架構類型** | 完整架構 | 涵蓋多種 SSRF 場景 |

**數據流特徵**:
- ✅ 內網探測能力
- ✅ 雲端元數據服務檢測 (AWS, Azure, GCP)
- ✅ URL 繞過技術
- ✅ 協議走私檢測

**分析結果位置**: `module_analysis/function_ssrf/`

---

#### 1.4 function_idor - IDOR 漏洞檢測
**簡單架構模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 11 | IDOR 和訪問控制檢測 |
| **數據流** | 5 | 簡單清晰的流程 |
| **真實連接** | 5 | 線性調用關係 |
| **數據源頭** | 3 | 3個入口點 |
| **架構類型** | 簡單架構 | 專注核心功能 |

**數據流特徵**:
- ✅ 簡潔的設計
- ✅ 專注於 ID 枚舉和權限提升
- ✅ 參數污染技術
- ✅ UUID/GUID 預測

**分析結果位置**: `module_analysis/function_idor/`

---

#### 1.5 function_bizlogic - 業務邏輯漏洞
**簡單架構模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 10 | 業務邏輯漏洞檢測 |
| **數據流** | 2 | 最簡單的流程 |
| **真實連接** | 2 | 最少的連接數 |
| **數據源頭** | 2 | 2個入口點 |
| **架構類型** | 簡單架構 | 精簡設計 |

**數據流特徵**:
- ✅ 聚焦核心業務邏輯問題
- ✅ 競態條件檢測
- ✅ 支付邏輯漏洞
- ✅ 工作流繞過

**分析結果位置**: `module_analysis/function_bizlogic/`

---

#### 1.6 function_info_leak - 信息洩漏檢測
**✅ 已修復並增強 (2026-01-28)**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 2 | 敏感信息檢測器 |
| **數據流** | 0 | 獨立工具設計 |
| **真實連接** | 0 | 無跨檔案調用 |
| **數據源頭** | 0 | N/A |
| **架構類型** | 工具集 | 獨立檢測器 |
| **代碼行數** | 1307 | 從 547 行擴展 |

**數據流特徵**:
- ✅ **0 flows 為正常** - 獨立檢測器設計
- ✅ 編碼問題已修復（完全重建）
- 功能：50+ 敏感信息檢測模式（AWS, GCP, Azure, GitHub, JWT 等）
- 新增：Shannon 熵值分析、SARIF v2.1.0 輸出、風險評分

**檔案**:
- `sensitive_info_detector.py` - ✅ 已修復並增強至 1307 行
- `__init__.py` - 模組入口

**分析結果位置**: `module_analysis/function_info_leak/`

---

### 2. Rust 功能模組

#### 2.1 function_crypto - 密碼學安全檢測
**✅ 已完成且有完整數據流的模組**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 5 | 4個analyzer + 1個main |
| **數據流** | **4** | ✅ main調度4個analyzer |
| **真實連接** | **4** | ✅ 跨檔案調用 |
| **函數數** | 16 | CLI 調度函數 |
| **架構類型** | 調度器架構 | main作為中央調度器 |
| **完成度** | **100%** | **功能完整，數據流正常** |

**修復說明**：
- 🔧 **分析工具已修復** - 之前的 Rust 工具無法識別模組路徑調用 (`mod::func`)
- ✅ **現在正確檢測** - 4條數據流全部識別成功

**數據流分析**:

**Flow 1**: `main` → `scan_javascript`
- 從: main.rs
- 到: js_crypto_analyzer.rs
- 調用: `js_crypto_analyzer::scan_javascript(&content)`
- 用途: JavaScript 加密問題掃描

**Flow 2**: `main` → `analyze_tls`
- 從: main.rs
- 到: tls_analyzer.rs
- 調用: `tls_analyzer::analyze_tls(&target, port).await`
- 用途: TLS/SSL 配置分析

**Flow 3**: `main` → `analyze_cookies`
- 從: main.rs
- 到: cookie_analyzer.rs
- 調用: `cookie_analyzer::analyze_cookies(&cookies_json, &url)`
- 用途: Cookie 安全性分析

**Flow 4**: `main` → `analyze_headers`
- 從: main.rs
- 到: header_analyzer.rs
- 調用: `header_analyzer::analyze_headers(&headers_json, &url)`
- 用途: HTTP Header 安全檢查

**架構特徵**:
- ✅ **中央調度器模式** - main 根據命令分派到對應 analyzer
- ✅ **模組化設計** - 4個analyzer完全獨立
- ✅ **清晰的數據流** - 單向調用，無循環依賴

**分析器組成** (4個完整的獨立模組):

1. **cookie_analyzer.rs** (87行) - Cookie 安全分析
   - ✅ 檢測缺失 Secure/HttpOnly 標記
   - ✅ 敏感 Cookie 識別
   - ✅ SameSite 屬性檢查
   - **獨立函數**: `analyze_cookies()`, `extract_cookie_name()`, `is_sensitive_cookie()`
   
2. **header_analyzer.rs** - HTTP Header 安全
   - ✅ HSTS 檢查（max-age 驗證）
   - ✅ CSP 策略分析
   - ✅ 安全 Header 缺失檢測 (X-Frame-Options, X-Content-Type-Options)
   - **獨立函數**: `analyze_headers()`, `extract_max_age()`

3. **js_crypto_analyzer.rs** - JavaScript 加密問題
   - ✅ 硬編碼金鑰檢測（API keys, AWS credentials, JWT secrets）
   - ✅ 弱加密算法識別（MD5, DES, RC4）
   - ✅ JWT 問題檢測（alg: none, weak signatures）
   - ✅ 不安全存儲（localStorage 存敏感數據）
   - **5個獨立檢測函數**: `scan_javascript()`, `detect_hardcoded_keys()`, `detect_weak_crypto_usage()`, `detect_weak_random()`, `detect_jwt_issues()`, `detect_insecure_storage()`

4. **tls_analyzer.rs** - TLS/SSL 分析
   - ✅ 協議版本檢查
   - ✅ 密碼套件強度評估
   - **異步實現**: `analyze_tls()` (async fn)

**main.rs 的角色**:
- 🎯 **CLI 入口** - 使用 clap 解析命令
- 🎯 **路由調度** - 根據子命令調用對應 analyzer
- 🎯 **結果格式化** - 統一輸出 JSON 格式
- ⚠️ **不含業務邏輯** - 只做調度，不做檢測

**CLI 指令**: 16個 cargo 命令，分3類（ANALYSIS, OTHER, RECONNAISSANCE）

**為什麼這是好的設計？**
- ✅ **單一職責**: 每個 analyzer 專注一個領域
- ✅ **易於測試**: 可單獨測試每個 analyzer
- ✅ **易於維護**: 修改一個不影響其他
- ✅ **易於擴展**: 新增 analyzer 不需修改現有代碼

**分析結果位置**: `services/integration/data/internal_exploration/analysis_results/rust/`

---

#### 2.2 rust_engine - 掃描引擎
**完整架構引擎**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 11 | 完整的掃描引擎 |
| **程式碼量** | 201 KB | 大型專案 |
| **數據流** | 4 | 核心模組初始化流程 |
| **真實連接** | 4 | 主控流程調用 |
| **函數數** | 255 | 複雜的功能實現 |
| **架構類型** | 引擎架構 | 模組化掃描引擎 |

**數據流分析**:

**Flow 1**: `scan_single_target` → `EndpointDiscoverer::new`
- 功能：初始化端點發現器
- 用途：Web 端點探測

**Flow 2**: `scan_single_target` → `JsAnalyzer::new`
- 功能：初始化 JavaScript 分析器
- 用途：前端代碼分析

**Flow 3**: `scan_single_target` → `SensitiveInfoScanner::with_mode`
- 功能：初始化敏感信息掃描器
- 用途：數據洩漏檢測

**Flow 4**: `scan_single_target` → `AttackSurfaceAssessor::new`
- 功能：初始化攻擊面評估器
- 用途：風險評估

**模組組成**:
- `main.rs` - 主程序入口
- `scanner.rs` - 掃描協調器
- `endpoint_discovery.rs` - 端點發現
- `js_analyzer.rs` - JS 分析
- `secret_detector.rs` - 密碼檢測
- `attack_surface.rs` - 攻擊面評估
- `auth_brute.rs` - 認證爆破
- `smuggling_detector.rs` - 走私檢測
- `verifier.rs` - 結果驗證
- `schemas/` - 數據結構

**特色**:
- ✅ 並行處理 (rayon)
- ✅ 異步支持 (tokio)
- ✅ 正則優化 (aho-corasick)
- ✅ SPA/Cloud 平台識別

**分析結果位置**: `services/scan/rust_engine/rust_analysis_output/`

---

### 3. Go 功能模組

#### 3.1 function_authn_go - 認證模組
**Go 實現的認證功能**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 4 | 認證相關功能 |
| **數據流** | 4 | 清晰的調用鏈 |
| **真實連接** | 4 | 模組間調用 |
| **架構類型** | 完整架構 | Go 認證實現 |

**數據流特徵**:
- ✅ Go 語言實現
- ✅ 認證流程完整
- ✅ 模組化設計

**分析結果位置**: `services/features/features_in_development/function_authn_go/analysis_output/`

---

### 4. TypeScript 掃描引擎

#### 4.1 typescript_engine - TypeScript 掃描引擎
**TypeScript 實現的掃描工具**

| 指標 | 數值 | 說明 |
|------|------|------|
| **檔案數** | 11 | TypeScript 掃描引擎 |
| **程式碼量** | 94 KB | 中型專案 |
| **數據流** | 3 | 核心調用流程 |
| **真實連接** | 3 | 模組調用 |
| **函數數** | 75 | 功能豐富 |
| **架構類型** | 引擎架構 | TS 掃描實現 |

**數據流特徵**:
- ✅ TypeScript 實現
- ✅ 現代化架構
- ✅ 掃描引擎完整

**分析結果位置**: `services/scan/typescript_engine/analysis_output/`

---

## 📈 數據流複雜度排名

### 按 Flows 數量排名

| 排名 | 模組 | 語言 | Flows | 檔案 | 複雜度評級 |
|------|------|------|-------|------|-----------|
| 🥇 1 | function_xss | Python | 83 | 47 | ⭐⭐⭐⭐⭐ 極高 |
| 🥈 2 | function_sqli | Python | 48 | 34 | ⭐⭐⭐⭐ 高 |
| 🥉 3 | function_ssrf | Python | 19 | 14 | ⭐⭐⭐ 中 |
| 4 | function_idor | Python | 5 | 11 | ⭐⭐ 低 |
| 5 | function_authn_go | Go | 4 | 4 | ⭐⭐ 低 |
| 5 | rust_engine | Rust | 4 | 11 | ⭐⭐ 低* |
| 7 | typescript_engine | TypeScript | 3 | 11 | ⭐ 極低 |
| 8 | function_bizlogic | Python | 2 | 10 | ⭐ 極低 |
| 9 | function_crypto | Rust | 0 | 5 | - 工具集 |
| 9 | function_info_leak | Python | 0 | 2 | - 工具集 |

*rust_engine 雖然只有4個flows，但有255個函數，實際複雜度極高

---

## 🏗️ 架構模式分析

### 架構類型分布

| 架構類型 | 模組數 | 特徵 | 範例 |
|----------|--------|------|------|
| **完整架構** | 6 | 多層次調用，豐富數據流 | XSS, SQLi, SSRF |
| **簡單架構** | 2 | 線性流程，少量連接 | IDOR, BizLogic |
| **引擎架構** | 2 | 複雜邏輯，模組初始化 | rust_engine, typescript_engine |
| **工具集** | 2 | 獨立工具，無跨檔案流 | crypto, info_leak |

### 數據流密度分析

**數據流密度 = Flows / 檔案數**

| 模組 | 密度值 | 評級 | 說明 |
|------|--------|------|------|
| function_sqli | 1.41 | 🔥 極高 | 平均每個檔案參與1.4個flow |
| function_xss | 1.77 | 🔥 極高 | 最高的數據流密度 |
| function_ssrf | 1.36 | 🔥 高 | 良好的模組化 |
| function_authn_go | 1.00 | ✅ 適中 | Go模組設計合理 |
| function_idor | 0.45 | ✅ 適中 | 簡潔設計 |
| rust_engine | 0.36 | ✅ 低 | 大量輔助函數 |
| typescript_engine | 0.27 | ✅ 低 | 工具類較多 |
| function_bizlogic | 0.20 | ✅ 極低 | 精簡設計 |

---

## 🎯 數據流健康度評估

### 各模組健康度

| 模組 | 數據流 | 連接數 | 語法問題 | 健康度 | 建議 |
|------|--------|--------|----------|--------|------|
| function_xss | ✅ 83 | ✅ 35 | ⚠️ 輕微 | 🟢 90% | 修復 escape sequence 警告 |
| function_sqli | ✅ 48 | ✅ 30 | ⚠️ 中等 | 🟡 85% | 升級 NoSQLMap 到 Python 3 |
| function_ssrf | ✅ 19 | ✅ 13 | ✅ 無 | 🟢 100% | 無需改進 |
| function_idor | ✅ 5 | ✅ 5 | ✅ 無 | 🟢 100% | 無需改進 |
| function_bizlogic | ✅ 2 | ✅ 2 | ✅ 無 | 🟢 100% | 無需改進 |
| function_info_leak | ✅ 0* | ✅ 0* | ✅ 無 | 🟢 100% | ✅ 已修復並增強 (2026-01-28) |
| function_crypto | ✅ 0* | ✅ 0* | ✅ 無 | 🟢 100% | 正常（工具集） |
| rust_engine | ✅ 4 | ✅ 4 | ✅ 無 | 🟢 100% | 無需改進 |
| function_authn_go | ✅ 4 | ✅ 4 | ✅ 無 | 🟢 100% | 無需改進 |
| typescript_engine | ✅ 3 | ✅ 3 | ✅ 無 | 🟢 100% | 無需改進 |

*0 flows 為工具集設計，屬於正常情況

---

## 🔧 發現的問題與建議

### 🔴 高優先級

~~1. **function_info_leak 編碼問題**~~ ✅ 已完成 (2026-01-28)
   - ✅ 已修復：完全重建 sensitive_info_detector.py（547 → 1307 行）
   - ✅ 新增：50+ 檢測模式、熵值分析、SARIF 輸出
   - ✅ 功能測試通過

### 🟡 中優先級

2. **function_sqli NoSQLMap Python 2 代碼**
   - **問題**: 5個檔案有 `Missing parentheses in call to 'print'`
   - **影響**: 語法警告，可能影響執行
   - **建議**: 升級 NoSQLMap 到 Python 3 或使用 2to3 工具

### 🟢 低優先級

3. **function_xss Escape Sequence 警告**
   - **問題**: `invalid escape sequence '\-'`
   - **影響**: 非阻斷性警告
   - **建議**: 使用 raw string (r"...") 或正確轉義

---

## 📁 分析結果檔案位置

### Python 模組分析

```
module_analysis/
├── function_xss/           # XSS 模組分析結果
│   ├── analysis_report.txt
│   ├── flow_chains.json
│   └── *.mmd
├── function_sqli/          # SQLi 模組分析結果
├── function_ssrf/          # SSRF 模組分析結果
├── function_idor/          # IDOR 模組分析結果
├── function_bizlogic/      # BizLogic 模組分析結果
└── function_info_leak/     # InfoLeak 模組分析結果
```

### Rust 模組分析

```
services/integration/data/internal_exploration/analysis_results/rust/
├── analysis_results.json   # function_crypto 完整分析
├── cli_commands.sh         # 16個 CLI 指令
├── system_flow.mmd         # 系統架構圖
└── *.mmd                   # 49個函數流程圖

services/scan/rust_engine/rust_analysis_output/
├── analysis_results.json   # rust_engine 完整分析
├── cli_commands.sh         # CLI 指令
└── *.mmd                   # 流程圖
```

### Go/TypeScript 模組分析

```
services/features/features_in_development/function_authn_go/analysis_output/
└── analysis_results.json

services/scan/typescript_engine/analysis_output/
└── analysis_results.json
```

---

## 🎓 分析方法論

### 使用的工具

1. **Python Flow Analyzer** (`aiva_flow_analyzer.py`)
   - AST 分析
   - 跨檔案數據流追蹤
   - 深度3層分析

2. **Rust Analyzer** (`rs2mermaid`)
   - Cargo 工具鏈
   - 跨檔案數據流串接
   - CLI 指令生成

3. **Go Analyzer** (`go2mermaid`)
   - Go AST 分析
   - 模組依賴追蹤

4. **TypeScript Analyzer** (`ts2mermaid`)
   - TypeScript Compiler API
   - ES6+ 模組分析

### 分析參數

- **追蹤深度**: 3層
- **最大路徑**: 10條
- **輸出格式**: JSON Schema v3.3
- **視覺化**: Mermaid 流程圖

---

## 📊 統一 JSON Schema v3.3

所有分析結果遵循統一格式：

```json
{
  "metadata": {
    "tool": "工具名稱",
    "version": "2.0",
    "language": "語言",
    "generated_at": "ISO8601時間戳",
    "total_flows": 數量,
    "total_files": 檔案數,
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "id": 1,
      "path": ["從", "到"],
      "full_path": ["完整路徑1", "完整路徑2"],
      "length": 2,
      "start": "起點",
      "end": "終點",
      "classifications": [...],
      "language": "語言",
      "cli_command": "執行命令"
    }
  ],
  "functions": [...],
  "summary": {...}
}
```

---

## 🎯 下一步計劃

### Phase 2: 跨模組數據流分析

1. **模組間調用關係**
   - 分析功能模組之間的依賴
   - 繪製完整的系統調用圖

2. **多語言整合**
   - Python → Rust FFI 調用
   - Go/TypeScript 服務整合

3. **FlowExecutor 擴展**
   - 支援 Rust/Go/TypeScript CLI flows
   - 統一執行介面

### Phase 3: 智能優化建議

1. **性能瓶頸識別**
   - 分析高頻調用路徑
   - 識別潛在性能問題

2. **架構優化建議**
   - 過度耦合檢測
   - 模組重構建議

3. **測試覆蓋分析**
   - 未測試的數據流路徑
   - 測試用例生成建議

---

## 📝 附錄

### A. 術語表

- **Flow**: 數據流路徑，從一個函數到另一個函數的調用鏈
- **Connection**: 真實連接，兩個腳本/函數之間的直接調用
- **Data Source**: 數據源頭節點，流程的起點
- **Schema v3.3**: AIVA 統一的 JSON 輸出格式版本 3.3

### B. 分析命令參考

**Python 模組分析**:
```bash
python aiva_flow_analyzer.py \
  --target "模組路徑" \
  --output "輸出目錄" \
  --depth 3
```

**Rust 模組分析**:
```bash
cargo run --manifest-path rust_tools/Cargo.toml -- 模組路徑
```

**Go 模組分析**:
```bash
go run go2mermaid.go 模組路徑
```

**TypeScript 模組分析**:
```bash
npx ts-node ts2mermaid.ts 模組路徑
```

### C. 參考文檔

- [Phase 1 完成報告](PHASE1_COMPLETION_REPORT.md)
- [多語言分析架構](Multi-Language_Analysis_Execution_Architecture.md)
- [Internal Exploration README](README.md)
- [JSON 格式統一計劃](MULTILANG_JSON_FORMAT_UNIFICATION.md)

---

**報告生成**: AIVA Internal Exploration System  
**版本**: 1.0  
**更新頻率**: 每次重大分析後更新
