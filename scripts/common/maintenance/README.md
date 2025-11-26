# AIVA 維護工具集 (Maintenance Tools)

本目錄包含 AIVA 專案的維護、診斷和優化工具，用於系統健康檢查、問題修復和專案管理。

## 📁 目錄結構

```
maintenance/
├── README.md                                    # 本說明文件
├── check_status.ps1                             # 系統狀態檢查腳本
├── diagnose_system.ps1                          # 系統診斷工具
├── fix_import_paths.py                          # 導入路徑修復工具
├── generate_project_report.ps1                  # 專案完整報告生成器
├── generate_stats.ps1                           # 專案統計生成腳本
├── generate_tree_ultimate_chinese.ps1           # 程式碼樹狀圖生成器（終極整合版）
├── generate_tree_ultimate_chinese_backup.ps1    # 備份版本
├── health_check_multilang.ps1                   # 多語言系統健康檢查
├── optimize_core_modules.ps1                    # 核心模組優化腳本
└── system_repair_tool.py                        # 系統自動修復工具
```

---

## 🛠️ 工具說明

### 1. **check_status.ps1** - 系統狀態檢查

**功能：** 快速檢查 AIVA 系統所有服務的運行狀態

**使用方式：**
```powershell
.\check_status.ps1
```

**檢查項目：**
- ✅ Docker 容器狀態（aiva, rabbitmq, postgres）
- ✅ 服務埠號檢查（8001, 8003, 5672, 15672, 5432）
- ✅ Python 進程狀態（uvicorn, worker）
- ✅ API 端點健康檢查（Core API, Integration API, RabbitMQ）

**輸出示例：**
```
========================================
📊 AIVA 系統狀態檢查
========================================

🐳 Docker 容器狀態:
   ✅ aiva-core       Up 2 hours
   ✅ rabbitmq        Up 2 hours

🔌 服務埠號檢查:
   ✅ Port 8001 - Core API
   ✅ Port 8003 - Integration API

🐍 Python 進程:
   ✅ 運行中: 3 個 Python 進程
```

---

### 2. **diagnose_system.ps1** - 系統診斷工具

**功能：** 全面診斷系統問題並提供修復建議

**使用方式：**
```powershell
.\diagnose_system.ps1
```

**診斷項目：**
1. **基本環境檢查**
   - Python、Node.js、Go、Rust、Docker 安裝狀態
   - Docker 服務運行狀態

2. **項目結構檢查**
   - 必要目錄存在性
   - Python 環境配置

3. **配置文件檢查**
   - pyproject.toml、requirements.txt、docker-compose.yml

4. **端口占用檢查**
   - 8001, 8003, 5432, 5672, 15672, 6379, 7474, 7687

**輸出示例：**
```
🔍 檢查基本環境...
   ✅ Python: Python 3.11.5
   ✅ Node.js: v18.17.0
   ✅ Docker: Docker version 24.0.6

⚠️  診斷完成 - 發現 1 個問題

🔧 發現的問題:
   • Docker 服務未運行

💡 修復建議:
   • 啟動 Docker Desktop
```

---

### 3. **fix_import_paths.py** - 導入路徑修復工具

**功能：** 自動修復 Python 腳本的導入路徑問題

**使用方式：**
```bash
python fix_import_paths.py
```

**修復項目：**
- 修正相對路徑導入為絕對路徑
- 統一 `sys.path` 設定
- 驗證關鍵模組導入

**修復模式：**
```python
# 修復前
sys.path.insert(0, str(Path(__file__).parent))

# 修復後
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

**輸出示例：**
```
🔧 AIVA 導入路徑修復工具
==================================================
📁 項目根目錄: C:\D\fold7\AIVA-git
📄 找到 45 個Python文件需要檢查

✅ 修復: scripts\testing\test_script.py
⚪ 無需修復: scripts\common\utils.py

🎯 修復完成! 共修復 12 個文件

🔍 檢查導入問題...
✅ services.aiva_common.enums.modules
✅ services.scan.aiva_scan
```

---

### 4. **generate_project_report.ps1** - 專案完整報告生成器

**功能：** 生成整合了樹狀圖、統計數據和架構圖的完整專案報告

**使用方式：**
```powershell
.\generate_project_report.ps1
```

**參數：**
```powershell
.\generate_project_report.ps1 -ProjectRoot "C:\D\fold7\AIVA-git" -OutputDir "C:\D\fold7\AIVA-git\_out"
```

**生成內容：**
1. **統計數據**
   - 檔案類型統計（Top 10）
   - 程式碼行數統計（依副檔名）
   - 多語言架構分析（Python、Go、Rust、TypeScript）

2. **目錄結構**
   - 完整樹狀圖（帶圖示）
   - 最多 10 層深度
   - 自動排除快取和虛擬環境

3. **Mermaid 架構圖**
   - 多語言架構概覽
   - 程式碼分布統計（餅圖）
   - 模組關係圖
   - 技術棧選擇流程圖

**輸出文件：**
- `PROJECT_REPORT.txt` - 完整文字報告
- `tree.mmd` - Mermaid 架構圖

**報告示例：**
```
═══════════════════════════════════════════════════════════════
📊 專案統計摘要
═══════════════════════════════════════════════════════════════

總文件數量: 1,234
總程式碼行數: 45,678
程式碼檔案數: 456

💻 程式碼行數統計 (依副檔名)
───────────────────────────────────────────────────────────────
  .py          25,123 行  (234 個檔案, 平均 107.4 行/檔案)
  .go           8,456 行  ( 89 個檔案, 平均  95.0 行/檔案)
  .rs           5,234 行  ( 45 個檔案, 平均 116.3 行/檔案)
  .ts           3,456 行  ( 67 個檔案, 平均  51.6 行/檔案)

📈 專案規模分析
───────────────────────────────────────────────────────────────
🐍 Python 程式碼: 25,123 行 (234 個檔案, 55.0% 佔比)
🔷 Go 程式碼: 8,456 行 (89 個檔案, 18.5% 佔比)
🦀 Rust 程式碼: 5,234 行 (45 個檔案, 11.5% 佔比)
📘 TypeScript/JavaScript: 3,456 行 (67 個檔案, 7.6% 佔比)
```

---

### 5. **generate_stats.ps1** - 專案統計生成腳本

**功能：** 生成專案文件統計和程式碼行數統計

**使用方式：**
```powershell
.\generate_stats.ps1
```

**生成文件：**
1. `ext_counts.csv` - 副檔名統計
2. `loc_by_ext.csv` - 程式碼行數統計（依副檔名）
3. `tree_ascii.txt` - ASCII 樹狀圖
4. `tree_unicode.txt` - Unicode 樹狀圖
5. `tree.mmd` - Mermaid 格式樹狀圖
6. `tree.html` - HTML 可視化版本
7. `tree.md` - Markdown 樹狀圖（帶圖示）

**輸出示例：**
```
📊 專案統計摘要:
  總文件數: 1,234
  總程式碼行數: 45,678

🎯 Top 5 副檔名 (依文件數):
  .py: 234 個文件
  .md: 123 個文件
  .json: 89 個文件

💻 Top 5 副檔名 (依程式碼行數):
  .py: 25,123 行 (234 個文件, 平均 107.4 行/文件)
  .go: 8,456 行 (89 個文件, 平均 95.0 行/文件)
```

---

### 6. **generate_tree_ultimate_chinese.ps1** - 程式碼樹狀圖生成器（終極整合版）

**功能：** 生成帶中文註解和差異標記的程式碼樹狀圖

**使用方式：**
```powershell
# 首次執行
.\generate_tree_ultimate_chinese.ps1 -ShowColorInTerminal -AddChineseComments

# 與上一版比對
.\generate_tree_ultimate_chinese.ps1 -PreviousTreeFile "tree_ultimate_chinese_20251025_120000.txt" -ShowColorInTerminal -AddChineseComments
```

**參數：**
- `-ProjectRoot` - 專案根目錄（預設：當前目錄）
- `-OutputDir` - 輸出目錄（預設：`_out`）
- `-PreviousTreeFile` - 上一版樹狀圖檔案（用於差異對比）
- `-ShowColorInTerminal` - 在終端顯示彩色輸出
- `-AddChineseComments` - 添加中文檔名說明

**特色功能：**
1. **差異標記**
   - `[+]` = 🟢 新增的檔案或目錄
   - `[-]` = 🔴 已刪除的檔案或目錄
   - `    ` = ⚪ 保持不變

2. **中文檔名說明**
   - 自動識別 Python、Go、Rust、TypeScript 檔案
   - 智慧推測功能用途
   - 覆蓋 AIVA 專案特定模組

3. **語言分布統計**
   - Python、Go、Rust、TypeScript 佔比
   - 檔案數量和行數統計

**輸出示例：**
```
AIVA                                              # AIVA 安全檢測平台
├─services/                                       # 服務模組
│   ├─aiva_common/                               # AIVA 共用模組
│   │   ├─__init__.py                            # 模組初始化
│   │   ├─config.py                              # 配置管理
[+] │   │   ├─new_module.py                      # 新增模組
│   │   └─enums/                                 # 列舉定義
│   │       ├─__init__.py                        # 模組初始化
│   │       └─modules.py                         # 模組列舉
│   ├─core/                                       # 核心模組
│   │   └─aiva_core/                             # AIVA 核心模組
│   │       ├─app.py                             # 應用程式入口
│   │       ├─ai_engine/                         # AI 引擎
│   │       │   ├─bio_neuron_core.py             # 生物神經元核心
│   │       │   └─ai_commander.py                # AI 指揮官

────────────────────────────────────────────────────────────────
🔴 已刪除的項目 (共 3 個):
────────────────────────────────────────────────────────────────
[-] old_module.py                                # 舊模組
[-] deprecated_service/                          # 已廢棄服務
```

---

### 7. **health_check_multilang.ps1** - 多語言系統健康檢查

**功能：** 檢查多語言微服務架構的健康狀態

**使用方式：**
```powershell
.\health_check_multilang.ps1
```

**檢查項目：**
1. **基礎設施狀態**
   - Docker 容器運行狀態

2. **Web 服務端點**
   - Core API (8001)
   - Integration API (8003)
   - RabbitMQ (15672)
   - PostgreSQL (5432)
   - Redis (6379)
   - Neo4j (7474)

3. **運行進程**
   - Python 進程
   - Node.js 進程
   - Go 進程
   - Rust 進程

4. **系統資源使用**
   - CPU 使用率
   - 記憶體使用
   - 磁碟使用

**輸出示例：**
```
🔍 AIVA 多語言系統 - 健康檢查

🏗️  基礎設施狀態
   ✅ Docker 容器狀態

🌐 Web 服務端點檢查
   ✅ Core API: 正常
   ✅ Integration API: 正常
   ❌ Redis: 無法連接 localhost:6379

⚡ 運行進程檢查
   ✅ Python 進程: 5 個運行中
   ✅ Node.js 進程: 2 個運行中

📊 系統資源使用
   💻 CPU 使用率: 45.2%
   🧠 記憶體使用: 8.5 GB / 16 GB (53.1%)
   💽 磁碟使用 (C:): 120.3 GB / 512 GB (23.5%)
```

---

### 8. **optimize_core_modules.ps1** - 核心模組優化腳本

**功能：** 執行核心模組的優化任務

**使用方式：**
```powershell
# 顯示幫助
.\optimize_core_modules.ps1 help

# 執行所有優化（推薦）
.\optimize_core_modules.ps1 all

# 預覽模式
.\optimize_core_modules.ps1 all -DryRun

# 單獨執行優化
.\optimize_core_modules.ps1 unify-ai         # 統一AI引擎
.\optimize_core_modules.ps1 refactor-app     # 重構app.py
.\optimize_core_modules.ps1 split-optimized  # 拆分optimized_core.py
.\optimize_core_modules.ps1 monitor          # 設置效能監控
```

**優化項目：**

1. **unify-ai** - 統一 AI 引擎版本
   - 合併 `bio_neuron_core.py` 和 `bio_neuron_core_v2.py`
   - 備份舊版本到 `legacy/`
   - 生成差異分析報告

2. **refactor-app** - 重構 app.py 依賴注入
   - 創建依賴注入容器
   - 創建組件工廠
   - 分離初始化邏輯

3. **split-optimized** - 拆分 optimized_core.py
   - 創建專業化模組：
     - `parallel_processing.py` - 並行處理
     - `neural_optimization.py` - 神經網路優化
     - `memory_management.py` - 記憶體管理
     - `metrics_collection.py` - 指標收集
     - `component_pooling.py` - 元件池

4. **monitor** - 設置效能監控系統
   - 創建效能監控器
   - 設置指標收集
   - 配置閾值告警

**輸出示例：**
```
🚀 開始執行所有核心模組優化...

✅ 前置條件檢查完成

ℹ️  步驟 1/4: 統一AI引擎
✅ 備份文件: bio_neuron_core.py.backup -> legacy\
✅ 差異分析完成，結果保存至 ai_engine_diff.txt
✅ AI引擎統一完成

ℹ️  步驟 2/4: 重構app.py
✅ 依賴注入容器創建完成
✅ 組件工廠創建完成

✅ 所有核心模組優化完成！

後續步驟：
1. 手動完成 optimized_core.py 的程式碼遷移
2. 更新 app.py 以使用新的依賴注入系統
3. 運行測試確保功能正常
```

---

### 9. **system_repair_tool.py** - 系統自動修復工具

**功能：** 自動診斷和修復系統問題

**使用方式：**
```bash
python system_repair_tool.py
```

**修復項目：**

1. **Python 導入問題**
   - 創建缺失的 `__init__.py`
   - 執行 schema 導入修復腳本
   - 驗證模組導入

2. **Go 模組依賴**
   - 清理 `go.sum`
   - 執行 `go mod tidy`
   - 下載依賴
   - 編譯測試
   - 修復特定錯誤（如 SSRF 未使用變數）

3. **Rust 編譯問題**
   - 清理 `target/` 目錄
   - 執行 `cargo update`
   - 運行 `cargo check`

4. **系統通連性檢查**
   - 測試模組導入
   - 驗證模組間通信

5. **靶場連接驗證**
   - 測試 localhost:3000 連接

**輸出示例：**
```
🚀 AIVA 系統修復開始
======================================================================

🐍 修復 Python 導入...
✅ [Python Import] Create __init__.py: services/__init__.py
✅ [Python Import] Run schema fix: success

🔧 修復 Go 模組依賴...
✅ [function_authn_go] Clean go.sum: success
✅ [function_authn_go] Go mod tidy: success
✅ [function_ssrf_go] Fix unused variable: 已修復 awsV2Token 未使用問題

🦀 修復 Rust 模組...
✅ [SAST Analyzer] Clean target: success
✅ [SAST Analyzer] Cargo check: 編譯成功，3 個警告

🔍 檢查系統通連性...
✅ [Connectivity] Import Common Module: success
✅ [Connectivity] Import Core Module: success

📊 生成修復報告...
✅ [System] Save report: aiva_system_repair_report_20251025_120000.json

======================================================================
🔧 AIVA 系統修復報告
======================================================================
🕐 修復時間: 2025-10-25T12:00:00
📊 修復動作: 32 個
✅ 成功: 28 個
❌ 失敗: 2 個
⚠️ 警告: 2 個
📈 成功率: 87.5%

📋 模組修復狀態:
   ✅ Python Import: 5/5 成功
   ✅ function_authn_go: 4/4 成功
   ✅ SAST Analyzer: 3/3 成功
   ⚠️  function_cspm_go: 2/4 成功

✅ 系統修復完成！
```

---

## 🚀 快速開始

### 基本健康檢查
```powershell
# 1. 檢查系統狀態
.\check_status.ps1

# 2. 診斷問題
.\diagnose_system.ps1

# 3. 自動修復
python system_repair_tool.py
```

### 專案報告生成
```powershell
# 生成完整報告
.\generate_project_report.ps1

# 生成統計數據
.\generate_stats.ps1

# 生成程式碼樹狀圖（帶中文註解）
.\generate_tree_ultimate_chinese.ps1 -ShowColorInTerminal -AddChineseComments
```

### 系統優化
```powershell
# 優化核心模組
.\optimize_core_modules.ps1 all

# 修復導入路徑
python fix_import_paths.py
```

---

## 📊 工具分類

### 狀態檢查類
- `check_status.ps1` - 快速狀態檢查
- `health_check_multilang.ps1` - 全面健康檢查
- `diagnose_system.ps1` - 問題診斷

### 修復類
- `system_repair_tool.py` - 自動修復工具
- `fix_import_paths.py` - 導入路徑修復

### 報告生成類
- `generate_project_report.ps1` - 完整專案報告
- `generate_stats.ps1` - 統計數據
- `generate_tree_ultimate_chinese.ps1` - 程式碼樹狀圖

### 優化類
- `optimize_core_modules.ps1` - 核心模組優化

---

## 🔧 常見使用場景

### 場景 1：每日系統檢查
```powershell
# 早上啟動系統時
.\check_status.ps1
.\health_check_multilang.ps1
```

### 場景 2：發現問題後
```powershell
# 診斷問題
.\diagnose_system.ps1

# 自動修復
python system_repair_tool.py

# 手動修復導入問題
python fix_import_paths.py
```

### 場景 3：專案報告
```powershell
# 生成完整報告（週報、月報）
.\generate_project_report.ps1

# 生成程式碼樹狀圖（追蹤變更）
.\generate_tree_ultimate_chinese.ps1 -PreviousTreeFile "上一版.txt" -AddChineseComments
```

### 場景 4：系統優化
```powershell
# 定期優化核心模組
.\optimize_core_modules.ps1 all -DryRun  # 先預覽
.\optimize_core_modules.ps1 all          # 確認後執行
```

---

## 📝 輸出文件位置

所有生成的報告和文件預設儲存在：
```
C:\D\fold7\AIVA-git\_out\
├── PROJECT_REPORT.txt                    # 專案完整報告
├── tree.mmd                              # Mermaid 架構圖
├── ext_counts.csv                        # 副檔名統計
├── loc_by_ext.csv                        # 程式碼行數統計
├── tree_ascii.txt                        # ASCII 樹狀圖
├── tree_unicode.txt                      # Unicode 樹狀圖
├── tree.html                             # HTML 可視化
├── tree.md                               # Markdown 樹狀圖
├── tree_ultimate_chinese_*.txt           # 中文樹狀圖（帶時間戳）
└── aiva_system_repair_report_*.json      # 修復報告（帶時間戳）
```

---

## ⚙️ 環境需求

### PowerShell 腳本
- PowerShell 5.1 或更高版本
- Windows 作業系統

### Python 腳本
- Python 3.8 或更高版本
- 已安裝必要的 Python 套件

### 系統工具
- Docker Desktop（用於容器檢查）
- Git（用於版本控制檢查）
- Go 工具鏈（用於 Go 模組修復）
- Rust 工具鏈（用於 Rust 模組修復）

---

## 🔍 故障排除

### 問題：PowerShell 腳本無法執行
**解決方案：**
```powershell
# 設置執行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 問題：Python 腳本找不到模組
**解決方案：**
```bash
# 確保在專案根目錄執行
cd C:\D\fold7\AIVA-git

# 或使用修復工具
python scripts\common\maintenance\fix_import_paths.py
```

### 問題：Docker 檢查失敗
**解決方案：**
```powershell
# 確保 Docker Desktop 正在運行
# 手動啟動 Docker Desktop 或執行：
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

---

## 📚 相關文件

- [AIVA 開發者指南](../../../DEVELOPER_GUIDE.md)
- [專案結構說明](../../../REPOSITORY_STRUCTURE.md)
- [快速參考](../../../QUICK_REFERENCE.md)

---

## 💡 最佳實踐

1. **定期執行健康檢查**
   - 每日啟動系統時執行 `check_status.ps1`
   - 每週執行 `health_check_multilang.ps1`

2. **問題修復流程**
   - 先診斷：`diagnose_system.ps1`
   - 再修復：`system_repair_tool.py`
   - 最後驗證：`check_status.ps1`

3. **專案報告生成**
   - 每週生成一次專案報告
   - 使用差異比對追蹤變更
   - 保留歷史版本用於趨勢分析

4. **系統優化**
   - 先使用 `-DryRun` 預覽變更
   - 在測試環境驗證
   - 確認無誤後在生產環境執行

---

## 🤝 貢獻

如需新增或改進維護工具，請遵循以下步驟：

1. 在本目錄創建新腳本
2. 更新本 README 文件
3. 添加使用範例和說明
4. 進行充分測試
5. 提交 Pull Request

---

## 📞 支援

如遇到問題或需要協助：

1. 查看本說明文件
2. 執行診斷工具：`diagnose_system.ps1`
3. 檢查系統日誌
4. 聯繫開發團隊

---

**最後更新：** 2025-10-25  
**版本：** 1.0.0  
**維護者：** AIVA 開發團隊
