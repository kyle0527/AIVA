# AIVA AI 組件探索與 CLI 轉化完整指南

> 🎯 **目標**: 使用 AIVA 自身的 AI 組件來發現程式實際功能，並將其轉化為可用的 CLI 指令
> 
> 📅 **更新日期**: 2025-10-28
> 
> 🧠 **適用場景**: 系統能力發現、功能驗證、CLI 自動生成

---

## 📋 目錄

- [🔍 AI 組件探索方法](#-ai-組件探索方法)
- [🤖 可用的 AI 探索工具](#-可用的-ai-探索工具)
- [⚡ CLI 轉化流程](#-cli-轉化流程)
- [📊 實際操作示例](#-實際操作示例)
- [🎯 最佳實踐](#-最佳實踐)
- [⚠️ 常見問題與解決方案](#️-常見問題與解決方案)

---

## 🔍 AI 組件探索方法

### 🎯 基本原理

AIVA 系統具備自我探索能力，通過 AI 組件可以：
1. **自動發現**系統中所有可用的掃描器和功能
2. **智能分析**代碼結構和功能特點
3. **生成 CLI 指令**供直接使用
4. **驗證功能可用性**並提供狀態報告

### 🧠 探索架構

```
🔍 AI 探索系統
├── ai_component_explorer.py      # 主要探索工具
├── ai_functionality_validator.py # 功能驗證器
├── ai_system_explorer_v3.py      # 深度系統分析器
├── scanner_statistics.py         # 掃描器統計分析器
└── health_check.py               # 系統健康檢查
```

---

## 🤖 可用的 AI 探索工具

### 1. 🔌 AI 組件探索器 (`ai_component_explorer.py`)

**用途**: 全面探索 AIVA 的 AI 組件和五大模組架構

```bash
python ai_component_explorer.py
```

**探索範圍**:
- ✅ 五大模組架構分析 (aiva_common, core, scan, integration, features)  
- ✅ AI 組件統計 (22 個 AI 組件)
- ✅ 可插拔組件識別 (15 個)
- ✅ CLI 指令自動生成 (11 個命令)

**輸出示例**:
```
🤖 AI組件統計:
   總AI組件數: 22
   可插拔組件: 15
   生成CLI命令: 11

🧠 發現的AI組件:
   🔌 ai_models                 | AI組件         | 函數:  0 | 類別: 27
   🔌 ai_commander              | AI指揮官        | 函數:  5 | 類別:  4
   🔌 learning_engine           | AI引擎         | 函數:  5 | 類別: 10
```

### 2. 🔍 掃描器統計分析器 (`scanner_statistics.py`)

**用途**: 精確統計系統中所有可用的掃描器

```bash
python scanner_statistics.py
```

**分析維度**:
- ✅ Python 功能掃描器統計
- ✅ 核心掃描引擎發現
- ✅ AI 智能檢測器識別
- ✅ 跨語言模組狀態檢查

**輸出結果**:
```
📊 掃描器統計總覽:
   Go 掃描器:     0 個
   Rust 掃描器:   0 個
   Python 掃描器: 15 個
   AI 智能檢測器: 4 個
   總計:          19 個掃描器
```

### 3. 🧠 AI 功能驗證器 (`ai_functionality_validator.py`)

**用途**: 驗證腳本功能並生成可用的 CLI 指令

```bash
python ai_functionality_validator.py --help
```

**驗證能力**:
- ✅ 腳本功能理解和描述
- ✅ CLI 指令自動生成
- ✅ 可執行性驗證
- ✅ 語法正確性檢查

**分析結果**:
```
🧠 AI理解能力展示:
   ✅ ai_system_explorer_v3.py  | AIVA AI 系統探索器 v3.0
   ✅ comprehensive_pentest_runner.py | AIVA 綜合實戰滲透測試執行器

🎯 生成的可用CLI指令:
   ✅ python ai_system_explorer_v3.py --detailed --output=json
   ✅ python comprehensive_pentest_runner.py --comprehensive
```

### 4. 🏥 系統健康檢查器 (`health_check.py`)

**用途**: 檢查系統組件可用性和配置狀態

```bash
python health_check.py
```

**檢查項目**:
- ✅ Schema 可用性狀態
- ✅ 專業工具安裝狀態 (Go, Rust, Node.js)
- ✅ AI 探索器存在性
- ✅ 關鍵目錄結構

**健康報告**:
```
🛠️ 專業工具狀態:
   Go: ✅ go1.25.0
   Rust: ✅ 1.90.0
   Node.js: ✅ v22.19.0

🎉 系統健康狀態: 優秀 (所有組件正常)
```

---

## ⚡ CLI 轉化流程

### 🎯 自動 CLI 生成流程

```mermaid
flowchart TD
    A[AI 組件探索] --> B[功能分析]
    B --> C[CLI 指令生成]
    C --> D[可執行性驗證]
    D --> E[指令優化]
    E --> F[最終 CLI 輸出]
    
    A --> A1[ai_component_explorer.py]
    B --> B1[ai_functionality_validator.py]
    C --> C1[自動生成語法正確指令]
    D --> D1[--help 可用性測試]
    E --> E1[參數最佳化]
    F --> F1[可用 CLI 指令集]
```

### 📝 CLI 轉化步驟

#### 步驟 1: 探索 AI 組件
```bash
python ai_component_explorer.py
```
- 🔍 發現系統中的所有 AI 組件
- 📊 統計組件數量和類型
- ⚡ 初步生成 CLI 指令

#### 步驟 2: 驗證功能性
```bash
python ai_functionality_validator.py
```
- 🧠 AI 理解腳本功能
- ✅ 驗證 CLI 指令可執行性
- 📈 計算成功率統計

#### 步驟 3: 統計掃描器
```bash
python scanner_statistics.py
```
- 🔢 精確統計掃描器數量
- 📂 分析掃描器分布
- 💾 生成詳細 JSON 報告

#### 步驟 4: 健康檢查
```bash
python health_check.py
```
- 🏥 驗證系統健康狀態
- 🛠️ 檢查依賴工具
- ⚠️ 識別配置問題

---

## 📊 實際操作示例

### 🎯 完整探索流程示例

```bash
# 1. 探索所有 AI 組件和模組架構
python ai_component_explorer.py

# 2. 驗證腳本功能並生成 CLI
python ai_functionality_validator.py

# 3. 統計掃描器詳細資訊
python scanner_statistics.py

# 4. 檢查系統健康狀態
python health_check.py
```

### 📈 探索結果分析

#### AI 組件發現結果:
```
📋 五大模組架構:
   aiva_common     | AI組件:  0 | 傳統組件: 10 | 通用基礎模組
   core            | AI組件: 14 | 傳統組件: 10 | 核心業務模組 (AI組件密集)       
   scan            | AI組件:  0 | 傳統組件: 10 | 掃描發現模組
   integration     | AI組件:  3 | 傳統組件: 10 | 整合服務模組
   features        | AI組件:  5 | 傳統組件: 10 | 功能檢測模組
```

#### 自動生成的 CLI 指令:
```bash
# AI 控制指令
python -m services.core.aiva_core.ai_commander --mode=interactive
python -m services.core.aiva_core.learning_engine --auto-train

# 掃描指令  
python -m services.scan.aiva_scan.vulnerability_scanner --target=localhost:3000
python -m services.scan.aiva_scan.network_scanner --range=192.168.1.0/24

# 功能檢測指令
python -m services.features.function_sqli --payload-file=payloads.txt
python -m services.features.function_xss --target=http://localhost:3000

# 系統測試指令
python ai_security_test.py --comprehensive
python ai_autonomous_testing_loop.py --max-iterations=5
```

### 🔍 掃描器詳細統計

#### Python 功能掃描器 (11個):
```
✅ sqli_scanner        - SQL 注入檢測
✅ xss_scanner         - XSS 漏洞掃描
✅ ssrf_scanner        - SSRF 漏洞檢測
✅ idor_scanner        - 權限繞過檢測
✅ crypto_scanner      - 加密漏洞分析
✅ authn_go_scanner    - 認證機制測試
✅ cspm_go_scanner     - 雲安全配置檢查
✅ postex_scanner      - 後滲透分析
✅ sast_rust_scanner   - 靜態代碼分析
✅ sca_go_scanner      - 軟體組成分析
✅ ssrf_go_scanner     - Go 版 SSRF 檢測
```

#### 核心掃描引擎 (4個):
```
✅ core_crawling_engine - 網頁爬取引擎
✅ dynamic_engine       - 動態分析引擎
✅ info_gatherer        - 資訊收集器
✅ examples             - 範例引擎
```

#### AI 智能檢測器 (4個):
```
✅ smart_detection_manager         - 智能檢測管理器
✅ unified_smart_detection_manager - 統一智能檢測管理器
✅ smart_idor_detector             - 智能 IDOR 檢測器
✅ smart_ssrf_detector             - 智能 SSRF 檢測器
```

---

## 🎯 最佳實踐

### 1. 🔄 定期更新探索

```bash
# 建議的定期探索流程
#!/bin/bash
echo "🔍 開始 AIVA 系統探索..."

# 探索 AI 組件
python ai_component_explorer.py > reports/ai_exploration_$(date +%Y%m%d).log

# 統計掃描器
python scanner_statistics.py

# 健康檢查
python health_check.py

echo "✅ 探索完成，報告已生成"
```

### 2. 📊 結果驗證

```bash
# 驗證生成的 CLI 指令
python ai_functionality_validator.py --help

# 測試關鍵功能
python -m services.features.function_sqli --help
python -m services.scan.aiva_scan.vulnerability_scanner --help
```

### 3. 🎨 報告生成

所有探索工具都會自動生成詳細報告：
- `reports/ai_diagnostics/ai_components_exploration_*.json`
- `reports/scanner_statistics/scanner_statistics_*.json`
- `reports/ai_diagnostics/ai_functionality_analysis_*.json`

---

## ⚠️ 常見問題與解決方案 (2025-10-28 更新)

### ✅ 問題 1: RabbitMQ 配置錯誤 (已解決)

**之前的錯誤訊息**:
```
ValueError: AIVA_RABBITMQ_URL or AIVA_RABBITMQ_USER/AIVA_RABBITMQ_PASSWORD must be set
```

**新的解決方案 (推薦)**:
```bash
# 方式一: 使用離線模式 (一鍵解決)
python launch_offline_mode.py        # 自動配置所有環境變數
python health_check.py               # 驗證環境 - 現在顯示 ✅ Schemas OK

# 方式二: 自動修復環境
python fix_offline_dependencies.py   # 自動修補配置檔案
```

**傳統解決方案 (仍可用)**:
```bash
# 手動設置環境變數
export AIVA_RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
export AIVA_RABBITMQ_USER="guest"
export AIVA_RABBITMQ_PASSWORD="guest"
```

### ✅ 問題 2: AI 組件完全可用 (已解決)

**現在的狀態**:
- ✅ 所有 22 個 AI 組件正常運行
- ✅ 離線模式支援完整功能
- ✅ 實戰測試驗證通過

**推薦使用方式**:
```bash
# AI 安全測試 (完全可用)
python ai_security_test.py --target http://localhost:3000

# AI 自主學習 (完全可用)  
python ai_autonomous_testing_loop.py --target http://localhost:3000

# AI 組件探索 (完全可用)
python ai_component_explorer.py
```

### 問題 3: Go/Rust 掃描器未激活

**原因**: Go 和 Rust 掃描器需要額外配置

**解決方案**:
1. 確認 Go 和 Rust 工具鏈已安裝
2. 配置相應的環境變數
3. 編譯 Rust 掃描器：
   ```bash
   cd services/scan/aiva_scan/advanced_scanner_rust
   cargo build --release
   ```

### 問題 4: CLI 指令生成失敗

**診斷步驟**:
```bash
# 1. 檢查腳本語法
python ai_functionality_validator.py

# 2. 驗證系統健康
python health_check.py

# 3. 查看詳細錯誤
python ai_component_explorer.py --verbose
```

---

## 🎯 結論

AIVA 的 AI 組件探索系統提供了完整的自我發現能力：

### ✅ 關鍵優勢
1. **自動化發現**: 無需手動檢查，AI 自動探索系統能力
2. **準確統計**: 精確統計 19 個掃描器，而非文檔中錯誤的 10 個
3. **CLI 生成**: 自動生成可用的命令列指令
4. **持續驗證**: 定期檢查系統健康和功能可用性

### 🚀 最終建議
1. **定期執行**探索腳本以保持文檔更新
2. **配置 RabbitMQ**以啟用完整 AI 功能
3. **使用生成的 CLI 指令**進行日常操作
4. **監控系統健康**狀態以確保最佳性能

通過這套完整的 AI 探索流程，AIVA 系統能夠準確地自我認知和持續優化。

---

**📅 文檔版本**: v1.0  
**🔄 最後更新**: 2025-10-28 17:30:00  
**✅ 驗證狀態**: 已通過完整測試  
**🎯 適用版本**: AIVA v4.0+