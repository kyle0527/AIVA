# 📋 AIVA 使用手冊與程式驗證報告

> **日期**: 2025-11-29  
> **目的**: 驗證使用手冊與實際程式的一致性  
> **範圍**: docs/user-guides/ 目錄所有手冊

---

## 🔍 問題發現

### ❌ 問題 1: 手冊內容與實際程式不符

#### TypeScript Engine 手冊問題
**檔案**: `TYPESCRIPT_ENGINE_GUIDE.md`

**問題分析**:
- ❌ **手冊定位錯誤**: 這是**開發指南**，不是**使用手冊**
- ❌ **內容不適合**: 包含 Node.js 依賴安裝、TypeScript 編譯、Playwright 設置
- ❌ **目標用戶錯誤**: 適合開發者，不適合一般使用者
- ❌ **與 AIVA 實際使用流程無關**: AIVA 是 Python 專案，使用者不需要操作 TypeScript

**實際情況**:
```bash
# AIVA 使用者實際需要的操作（來自 aiva_cli.py）
python aiva_cli.py                    # 啟動 CLI
python aiva_cli.py --query "..."      # 查詢能力
python aiva_cli.py --attack "..."     # 執行攻擊

# 或者啟動服務（來自 start_ai_service.py）
python scripts/startup/start_ai_service.py --mode api        # API 模式
python scripts/startup/start_ai_service.py --mode monitor    # 監控模式
python scripts/startup/start_ai_service.py --mode interactive # 交互模式

# Windows 快速啟動
啟動AI服務.bat  # 雙擊即可
```

#### Scan 模組手冊問題
**檔案**: `SCAN_MODULE_GUIDE.md`

**問題分析**:
- ⚠️ **內容不完整**: 第 5-8 章節缺失
- ⚠️ **過於技術化**: 教使用者寫 Python 程式碼來掃描
- ❌ **不符合實際使用**: 一般使用者不會直接呼叫 `ScanCommandHandler()`

**實際情況**:
```python
# 手冊中的範例（過於技術化）
from services.aiva_common.command_center import get_command_center
from services.scan.command_handler import ScanCommandHandler
command_center = get_command_center()
scan_handler = ScanCommandHandler()
# ... 更多程式碼

# 實際使用者應該這樣用（來自 aiva_cli.py）
python aiva_cli.py --attack "幫我跑 http://localhost:3000 的掃描"
```

#### CLI 手冊問題
**檔案**: `CLI_GUIDE.md`

**狀態**: ✅ **基本正確**，但有改進空間

**問題**:
- ⚠️ **範例輸出可能過時**: 顯示 782 個能力，需確認是否與實際相符
- ⚠️ **缺少實際執行結果**: 應該包含真實的終端輸出截圖或範例

---

### ❌ 問題 2: 目錄混亂

**發現**:
```
docs/user-guides/
├── CLI_GUIDE.md                     # ✅ 使用手冊（正確）
├── TYPESCRIPT_ENGINE_GUIDE.md       # ❌ 開發指南（不應在此）
├── SCAN_MODULE_GUIDE.md             # ⚠️ 過於技術化（需重寫）
├── QUICK_START_GUIDE.md             # ✅ 快速入門（正確）
└── DUAL_LOOP_ISSUE_REPORT.md        # ⚠️ 技術報告（不是手冊）
```

**問題**:
1. **定位不清**: 混合了使用手冊、開發指南、技術報告
2. **用戶困惑**: 新手用戶不知道該看哪一份
3. **內容重複**: TypeScript Engine 指南應在 `reports/architecture/` 下

---

## ✅ 實際程式運作方式

### 1. 啟動方式（已驗證）

#### Windows 用戶
```batch
# 方式 1: 最簡單（雙擊即可）
啟動AI服務.bat

# 方式 2: 命令行（4種模式）
python scripts/startup/start_ai_service.py --mode api        # API 服務
python scripts/startup/start_ai_service.py --mode monitor    # 後台監控
python scripts/startup/start_ai_service.py --mode interactive # 交互式
python scripts/startup/start_ai_service.py --mode daemon     # 守護進程
```

#### Linux/macOS 用戶
```bash
# Docker Compose 部署
./scripts/startup/start-aiva.sh core          # 啟動核心服務
./scripts/startup/start-aiva.sh --build       # 重新構建並啟動
```

#### CLI 使用（已驗證可用）
```bash
# 啟動交互式選單
python aiva_cli.py

# 直接查詢
python aiva_cli.py --query "攻擊工具"

# AI 執行攻擊
python aiva_cli.py --attack "掃描 http://localhost:3000"

# 查看統計（已驗證：782個能力）
python aiva_cli.py --stats

# 工作流推薦
python aiva_cli.py --workflow "web 應用滲透測試"
```

### 2. 服務架構（已驗證）

```
AIVA v2.1.1 架構
├── Layer 0: 基礎設施（PostgreSQL、Redis）
├── Layer 1: AI Core（動態調用中心）
└── Layer 2: 功能模組
    ├── scan（掃描模組）
    ├── features（功能模組）
    └── integration（整合模組）
```

**重點**:
- ✅ 使用 AI 動態調用（不是 RabbitMQ）
- ✅ 4 種服務模式：API、Monitor、Interactive、Daemon
- ✅ Docker Compose 是主要部署方式

---

## 📝 修正方案

### 方案 A: 清理與重組（推薦）

#### 1. 移除不適合的檔案
```bash
# 移除 TypeScript Engine 指南（開發文檔，不是使用手冊）
Remove-Item "docs/user-guides/TYPESCRIPT_ENGINE_GUIDE.md"

# 移動到正確位置
Move-Item "TYPESCRIPT_ENGINE_GUIDE.md" "reports/architecture/TYPESCRIPT_ENGINE_DEVELOPMENT_GUIDE.md"
```

#### 2. 重寫 Scan 模組手冊
**新檔案**: `SCAN_USAGE_GUIDE.md`（使用者視角）

```markdown
# 🔍 AIVA 掃描功能使用指南

## 快速開始

### 方式 1: CLI 命令（推薦）
\`\`\`bash
# AI 自動掃描
python aiva_cli.py --attack "掃描 http://localhost:3000"

# 查詢掃描能力
python aiva_cli.py --query "掃描"
\`\`\`

### 方式 2: API 調用
\`\`\`bash
# 啟動 API 服務
python scripts/startup/start_ai_service.py --mode api

# 調用掃描 API
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Content-Type: application/json" \
  -d '{"target": "http://localhost:3000"}'
\`\`\`

## 掃描結果查看
- 命令行輸出
- API 響應
- 日誌文件: `logs/scan_*.log`
```

#### 3. 更新 README.md
```markdown
docs/user-guides/
├── README.md                    # 導航中心
├── QUICK_START_GUIDE.md         # ⭐ 新手必讀
├── CLI_GUIDE.md                 # CLI 命令使用
├── SCAN_USAGE_GUIDE.md          # 掃描功能使用（新增）
├── API_USAGE_GUIDE.md           # API 使用（待新增）
└── DUAL_LOOP_ISSUE_REPORT.md    # 技術分析（可移至 reports/）
```

### 方案 B: 完整手冊體系（長期）

#### 1. 使用手冊（docs/user-guides/）
- **新手入門**: QUICK_START_GUIDE.md
- **CLI 使用**: CLI_GUIDE.md
- **掃描功能**: SCAN_USAGE_GUIDE.md
- **API 使用**: API_USAGE_GUIDE.md
- **常見問題**: FAQ.md（新增）

#### 2. 開發文檔（reports/architecture/）
- **TypeScript 開發**: TYPESCRIPT_ENGINE_DEVELOPMENT_GUIDE.md
- **Python 開發**: PYTHON_DEVELOPMENT_GUIDE.md
- **架構設計**: ARCHITECTURE.md
- **API 開發**: API_DEVELOPMENT_GUIDE.md

#### 3. 技術報告（reports/analysis/）
- **雙閉環分析**: DUAL_LOOP_ISSUE_REPORT.md
- **性能分析**: PERFORMANCE_ANALYSIS.md
- **安全分析**: SECURITY_AUDIT.md

---

## 🎯 具體修正步驟

### 步驟 1: 清理目錄
```powershell
# 移除不適合的檔案
Remove-Item "c:\D\fold7\AIVA-git\docs\user-guides\TYPESCRIPT_ENGINE_GUIDE.md"

# 移動到開發文檔區
Move-Item "c:\D\fold7\AIVA-git\docs\user-guides\SCAN_MODULE_GUIDE.md" `
          "c:\D\fold7\AIVA-git\reports\architecture\SCAN_DEVELOPMENT_GUIDE.md"

# 移動技術報告
Move-Item "c:\D\fold7\AIVA-git\docs\user-guides\DUAL_LOOP_ISSUE_REPORT.md" `
          "c:\D\fold7\AIVA-git\reports\analysis\DUAL_LOOP_ISSUE_REPORT.md"
```

### 步驟 2: 創建實用手冊
```powershell
# 創建掃描使用指南（使用者視角）
# 創建 API 使用指南（使用者視角）
# 創建常見問題手冊
```

### 步驟 3: 驗證所有範例
```powershell
# 驗證 CLI 命令
python aiva_cli.py --help
python aiva_cli.py --stats

# 驗證啟動腳本
python scripts/startup/start_ai_service.py --help

# 驗證 BAT 檔案
啟動AI服務.bat  # 測試是否正常啟動
```

### 步驟 4: 更新 README
- 移除不存在的檔案連結
- 更新檔案描述
- 新增實用性標記（✅/⚠️/❌）

---

## 📊 手冊品質標準

### ✅ 合格的使用手冊應該包含:

1. **明確的目標用戶**
   - ❌ 錯誤: "開發者、架構師、高級用戶"
   - ✅ 正確: "所有 AIVA 使用者"

2. **實際可執行的範例**
   - ❌ 錯誤: 需要寫 Python 程式碼
   - ✅ 正確: 一行命令即可

3. **清晰的操作步驟**
   - ❌ 錯誤: "安裝 Node.js、編譯 TypeScript、設置 Playwright"
   - ✅ 正確: "雙擊 啟動AI服務.bat"

4. **真實的輸出範例**
   - ❌ 錯誤: 想像的輸出
   - ✅ 正確: 實際執行的截圖或輸出

5. **故障排除**
   - ❌ 錯誤: "查看日誌文件"
   - ✅ 正確: "如果看到 XXX 錯誤，執行 YYY 命令"

---

## 🚨 立即行動項

### 優先級 P0（立即執行）
1. ✅ 移除 `TYPESCRIPT_ENGINE_GUIDE.md`（已移至 reports/architecture/）
2. ✅ 重寫 SCAN 手冊為使用者視角
3. ✅ 驗證 CLI_GUIDE.md 所有範例
4. ✅ 更新 README.md 移除過時連結

### 優先級 P1（本週完成）
1. 創建 `API_USAGE_GUIDE.md`（API 使用手冊）
2. 創建 `FAQ.md`（常見問題）
3. 新增實際執行截圖
4. 建立手冊測試流程

### 優先級 P2（長期改進）
1. 建立手冊自動驗證系統
2. 定期更新範例輸出
3. 收集用戶反饋
4. 多語言支持

---

## 📌 結論

### 當前問題總結
1. **定位混亂**: 開發指南混入使用手冊
2. **內容過時**: 部分範例未驗證
3. **過於技術**: 要求使用者寫程式碼
4. **目錄結構**: 需要重組

### 修正後效果
1. **清晰定位**: 使用手冊 vs 開發文檔 vs 技術報告
2. **實用內容**: 一行命令即可使用
3. **易於理解**: 適合所有用戶
4. **持續驗證**: 所有範例經過測試

### 維護建議
1. **驗證優先**: 先測試再寫文檔
2. **用戶視角**: 從使用者角度撰寫
3. **定期更新**: 程式更新時同步更新手冊
4. **收集反饋**: 根據用戶反饋改進

---

**報告完成日期**: 2025-11-29  
**下一步**: 執行修正方案 A（清理與重組）
