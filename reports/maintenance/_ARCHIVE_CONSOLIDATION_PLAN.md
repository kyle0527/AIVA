# 📦 存檔目錄整合規劃

**規劃日期**: 2025-11-27
**目標**: 將專案中所有歷史/廢棄檔案集中到 **最多 2 個存檔資料夾**

---

## 📑 目錄

1. [🎯 整合目標](#-整合目標)
   - [現況](#現況)
   - [目標架構](#目標架構)
2. [📋 兩個存檔資料夾的定位](#-兩個存檔資料夾的定位)
   - [方案 A: 單一存檔 (推薦)](#方案-a-單一存檔-推薦-)
   - [方案 B: 雙存檔系統](#方案-b-雙存檔系統)
3. [🔄 整合執行計畫](#-整合執行計畫)
4. [📋 檔案模板](#-檔案模板)
5. [🗂️ 快速導航](#️-快速導航)
6. [📖 使用指南](#-使用指南)
7. [🗑️ 清理策略](#️-清理策略)
8. [📁 各分類詳細說明](#-各分類詳細說明)
9. [📁 目錄結構](#-目錄結構)
10. [⚠️ 重要提醒](#️-重要提醒)
11. [🗑️ 清理規則](#️-清理規則)
12. [📝 使用範例](#-使用範例)
13. [🎯 建議選擇](#-建議選擇)
14. [📊 整合後效益](#-整合後效益)
15. [✅ 執行檢查清單](#-執行檢查清單)
16. [🔧 執行腳本](#-執行腳本)
17. [📖 參考文件](#-參考文件)

---

## 🎯 整合目標

### 現況
- ✅ 已有 `_archive/` 目錄 (9 個子目錄, 53 個檔案)
- ⚠️ 部分歷史檔案散落在各處 (主要在 `_archive/` 的 32 個 Python 檔案)

### 目標架構
```
專案根目錄/
├── _archive/              # 📚 歷史檔案存檔 (已完成的項目、廢棄代碼)
│   ├── ARCHIVE_INDEX.md  # 📋 存檔索引 (新增)
│   └── [分類子目錄]
│
├── _temp/                # 🔄 臨時工作區 (新增 - 可選)
│   ├── TEMP_README.md   # 說明臨時檔案用途
│   ├── experiments/     # 實驗性代碼
│   ├── drafts/          # 草稿文件
│   └── wip/             # Work in Progress
│
└── [正式專案目錄]        # 只保留活躍的代碼
```

---

## 📋 兩個存檔資料夾的定位

### 方案 A: 單一存檔 (推薦) ⭐

**只保留 `_archive/`** - 所有歷史檔案統一管理

```
_archive/
├── ARCHIVE_INDEX.md                    # 📋 總索引 (快速查找)
├── EXECUTIVE_SUMMARY.md                # 📊 執行摘要 (保留)
├── ARCHITECTURE_EVOLUTION_HISTORY.md   # 🏗️ 架構演進 (保留)
│
├── 01_completed_projects/              # ✅ 已完成項目
│   ├── schema_restructuring/
│   ├── architecture_fixes/
│   ├── file_cleanup/
│   └── cache_logs_cleanup/            # 新增: 2025-11 完成
│
├── 02_deprecated_code/                 # 🗄️ 廢棄代碼 (整合)
│   ├── schema_tools/                  # 從 deprecated_schema_tools/
│   ├── legacy_components/             # 從 legacy_components/
│   ├── duplicates_cleanup/            # 從 duplicates_cleanup/
│   └── old_tests/                     # 從 old_tests/
│
├── 03_historical_reports/             # 📄 歷史報告 (整合)
│   ├── duplicate_definition_reports/  # 從原目錄移入
│   ├── analysis_reports_2025/         # 從原目錄移入
│   └── schemas/                       # 從 deprecated_schema_tools/schemas/
│
├── 04_scripts_completed/              # 📜 完成腳本
│   └── [保持現有結構]
│
└── 05_backups/                        # 💾 備份檔案
    └── [保持現有結構]
```

**優點**:
- ✅ 結構最簡單,易於理解
- ✅ 所有歷史檔案統一管理
- ✅ 降低維護成本
- ✅ 不需要判斷檔案該放哪個目錄

**缺點**:
- ⚠️ 臨時工作檔案需要放在其他地方

---

### 方案 B: 雙存檔 (進階)

**`_archive/` + `_temp/`** - 區分永久存檔和臨時檔案

#### `_archive/` - 永久歷史存檔
```
_archive/
└── [與方案 A 相同結構]
```

**用途**: 
- 已完成的項目
- 廢棄的代碼
- 歷史報告
- 不會再修改的檔案

#### `_temp/` - 臨時工作區 (新增)
```
_temp/
├── TEMP_README.md              # 📋 說明此目錄用途
├── experiments/                # 🧪 實驗性功能
│   ├── new_feature_poc/
│   └── performance_test/
├── drafts/                     # ✏️ 草稿文件
│   ├── api_design_draft.md
│   └── architecture_proposal.md
├── wip/                        # 🚧 進行中的工作
│   └── refactoring_scripts/
└── .gitignore                  # 防止提交臨時檔案
```

**用途**:
- 實驗性代碼 (POC)
- 草稿文件
- 正在進行的重構
- 需要臨時保存的檔案

**清理規則**:
- 實驗失敗 → 刪除或移到 `_archive/`
- 實驗成功 → 移到正式目錄
- 定期清理 (建議每季度)

**優點**:
- ✅ 明確區分永久 vs 臨時
- ✅ 臨時檔案有明確位置
- ✅ 不會汙染 _archive/

**缺點**:
- ⚠️ 多一個目錄需要管理
- ⚠️ 需要定期清理 _temp/

---

## 🔄 整合執行計畫

### Phase 1: 重組 _archive/ (1-2 小時)

#### Step 1: 建立新的子目錄結構
```powershell
# 建立標準化的子目錄
cd C:\D\fold7\AIVA-git\_archive

New-Item -ItemType Directory -Path "01_completed_projects" -Force
New-Item -ItemType Directory -Path "02_deprecated_code" -Force
New-Item -ItemType Directory -Path "03_historical_reports" -Force
New-Item -ItemType Directory -Path "04_scripts_completed" -Force
New-Item -ItemType Directory -Path "05_backups" -Force
```

#### Step 2: 移動現有目錄
```powershell
# 移動到對應的新分類
Move-Item "completed_projects/*" "01_completed_projects/" -Force
Move-Item "deprecated_schema_tools" "02_deprecated_code/schema_tools" -Force
Move-Item "legacy_components" "02_deprecated_code/legacy_components" -Force
Move-Item "duplicates_cleanup" "02_deprecated_code/duplicates_cleanup" -Force
Move-Item "old_tests" "02_deprecated_code/old_tests" -Force
Move-Item "duplicate_definition_reports" "03_historical_reports/" -Force
Move-Item "analysis_reports_2025" "03_historical_reports/" -Force
Move-Item "scripts_completed" "04_scripts_completed/" -Force
Move-Item "backups" "05_backups/" -Force

# 刪除空的舊目錄
Remove-Item "completed_projects" -Force -ErrorAction SilentlyContinue
```

#### Step 3: 建立存檔索引
建立 `_archive/ARCHIVE_INDEX.md` (見下方模板)

### Phase 2: 處理散落的歷史檔案 (30 分鐘)

#### 檢查需要移動的檔案
```powershell
# 掃描專案中可能的歷史檔案
$toArchive = Get-ChildItem -Recurse -File | Where-Object {
    $_.Name -match '(old|legacy|deprecated|backup|_bak|test_old)' -and
    $_.DirectoryName -notlike "*\_archive\*" -and
    $_.DirectoryName -notlike "*\node_modules\*" -and
    $_.DirectoryName -notlike "*\venv\*" -and
    $_.DirectoryName -notlike "*\services\*"
}

# 顯示找到的檔案
$toArchive | ForEach-Object {
    Write-Host $_.FullName
}
```

#### 移動到 _archive/
根據檔案類型移動到對應的子目錄

### Phase 3: (可選) 建立 _temp/ 目錄

只在選擇方案 B 時執行:

```powershell
# 建立臨時工作區
cd C:\D\fold7\AIVA-git
New-Item -ItemType Directory -Path "_temp/experiments" -Force
New-Item -ItemType Directory -Path "_temp/drafts" -Force
New-Item -ItemType Directory -Path "_temp/wip" -Force

# 建立說明文件 (見下方模板)
```

---

## 📋 檔案模板

### ARCHIVE_INDEX.md 模板

```markdown
# 📚 AIVA 歷史檔案索引

**最後更新**: 2025-11-27

## 🗂️ 快速導航

| 分類 | 路徑 | 檔案數 | 說明 |
|------|------|--------|------|
| 已完成項目 | `01_completed_projects/` | XX | 已完成的重大項目 |
| 廢棄代碼 | `02_deprecated_code/` | XX | 不再使用的代碼 |
| 歷史報告 | `03_historical_reports/` | XX | 過往的分析報告 |
| 完成腳本 | `04_scripts_completed/` | XX | 執行完畢的腳本 |
| 備份檔案 | `05_backups/` | XX | 代碼備份 |

---

## 📖 使用指南

### 查找特定項目
1. 查看 `EXECUTIVE_SUMMARY.md` 了解總體情況
2. 進入對應分類目錄查看 README.md
3. 根據需要查看詳細檔案

### 查找廢棄功能
1. 檢查 `02_deprecated_code/`
2. 查看各子目錄的 README.md
3. 了解廢棄原因和替代方案

### 查找歷史報告
1. 進入 `03_historical_reports/`
2. 按時間或主題查找
3. 參考報告了解當時的決策

---

## 🗑️ 清理策略

### 保留規則
- 重大項目的文檔 (永久保留)
- 最近 1 年的報告 (保留)
- 有參考價值的廢棄代碼 (保留)

### 刪除規則
- 3 年以上的臨時報告 (可刪除)
- 無參考價值的實驗代碼 (可刪除)
- 已被完全替代的舊版本 (可刪除)

### 定期審查
- **頻率**: 每半年
- **審查內容**: 是否還需要保留
- **執行**: 刪除不再需要的檔案

---

## 📁 各分類詳細說明

### 01_completed_projects/ - 已完成項目
重大架構改進和功能開發項目的完整記錄

**子目錄**:
- `schema_restructuring/` - Schema 模組化重構
- `architecture_fixes/` - 架構問題修復
- `file_cleanup/` - 專案清理
- `cache_logs_cleanup/` - 快取和日誌清理

### 02_deprecated_code/ - 廢棄代碼
不再使用但有參考價值的舊代碼

**子目錄**:
- `schema_tools/` - 舊版 Schema 工具
- `legacy_components/` - 舊版組件
- `duplicates_cleanup/` - 重複定義清理工具
- `old_tests/` - 舊版測試

### 03_historical_reports/ - 歷史報告
過往的分析、審計、問題排查報告

**子目錄**:
- `duplicate_definition_reports/` - 重複定義問題報告
- `analysis_reports_2025/` - 2025年分析報告
- `schemas/` - 歷史 Schema 定義

### 04_scripts_completed/ - 完成腳本
已執行完畢的一次性初始化腳本

**內容**:
- Go 環境初始化
- 服務遷移腳本

### 05_backups/ - 備份檔案
重要檔案的備份版本

**內容**:
- 關鍵組件的備份
```

### TEMP_README.md 模板 (方案 B)

```markdown
# 🔄 臨時工作區說明

**目的**: 存放實驗性代碼、草稿文件、進行中的工作

---

## 📁 目錄結構

```
_temp/
├── experiments/    # 🧪 實驗性功能 POC
├── drafts/         # ✏️ 草稿文件
└── wip/            # 🚧 進行中的重構
```

## ⚠️ 重要提醒

### 這個目錄的檔案是臨時的!

- ❌ **不要放重要代碼** - 可能會被清理
- ❌ **不要長期保存** - 最多保留 3 個月
- ✅ **實驗成功** → 移到正式目錄
- ✅ **實驗失敗** → 刪除或移到 `_archive/`

## 🗑️ 清理規則

### 自動清理 (建議設定)
```powershell
# 刪除 3 個月以上未修改的檔案
Get-ChildItem -Recurse -File | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddMonths(-3) } |
    Remove-Item -Force
```

### 手動清理
- **頻率**: 每季度
- **檢查**: 是否還需要
- **處理**: 刪除或移動

## 📝 使用範例

### 實驗新功能
```bash
# 建立實驗目錄
_temp/experiments/new_ai_engine/
├── poc_code.py
├── test_results.md
└── performance_metrics.csv

# 成功 → 移到 src/engines/
# 失敗 → 保留報告到 _archive/，刪除代碼
```

### 起草設計文件
```bash
_temp/drafts/
├── api_v2_design.md      # 草稿
└── refactor_plan.md      # 草稿

# 完成 → 移到 docs/
# 廢棄 → 刪除
```

### 進行中的重構
```bash
_temp/wip/
└── scripts_reorganization/
    ├── migration_script.py
    └── progress.md

# 完成 → 整合到主專案
# 暫停 → 保留,但定期檢查
```

---

**記住**: 這是臨時工作區,不要依賴它長期保存任何東西!
```

---

## 🎯 建議選擇

### 推薦: **方案 A (單一存檔)** ⭐

**理由**:
1. ✅ **簡單明確**: 只有一個歷史檔案位置
2. ✅ **易於維護**: 不需要在兩個目錄間判斷
3. ✅ **符合現況**: 已有完善的 `_archive/` 結構
4. ✅ **降低複雜度**: 減少決策成本

**何時考慮方案 B**:
- 團隊有大量實驗性開發
- 需要明確區分臨時 vs 永久
- 有專人維護 _temp/ 目錄

---

## 📊 整合後效益

### 專案組織改善

| 指標 | 整合前 | 整合後 | 改善 |
|------|--------|--------|------|
| 存檔目錄數 | 9 個子目錄 (扁平) | 5 個分類目錄 (層次) | +清晰度 |
| 檔案查找時間 | 5-10 分鐘 | <2 分鐘 | -70% |
| 維護成本 | 中 | 低 | -40% |
| 新人理解難度 | 中 | 低 | -50% |

### 目錄結構評分

```
整合前: 7.0/10
├── 優點: 已有分類
└── 缺點: 分類不夠清晰,缺乏索引

整合後: 9.0/10
├── 優點: 分類明確,有索引,易於查找
└── 缺點: 需要一次性整合工作
```

---

## ✅ 執行檢查清單

### Phase 1: 規劃 (30 分鐘)
- [ ] 決定採用方案 A 或 B
- [ ] 審查現有 `_archive/` 內容
- [ ] 確認哪些檔案需要移動
- [ ] 備份重要檔案

### Phase 2: 重組 _archive/ (1-2 小時)
- [ ] 建立 5 個標準化子目錄
- [ ] 移動現有 9 個子目錄到新分類
- [ ] 建立 `ARCHIVE_INDEX.md`
- [ ] 更新 `EXECUTIVE_SUMMARY.md`
- [ ] 更新 `ARCHITECTURE_EVOLUTION_HISTORY.md`
- [ ] 刪除空的舊目錄

### Phase 3: 處理散落檔案 (30 分鐘)
- [ ] 掃描專案中的歷史檔案
- [ ] 移動到 `_archive/` 對應子目錄
- [ ] 更新相關引用 (如有)
- [ ] 驗證移動後功能正常

### Phase 4: (可選) 建立 _temp/ (30 分鐘)
- [ ] 建立 `_temp/` 目錄結構
- [ ] 建立 `TEMP_README.md`
- [ ] 設定 `.gitignore`
- [ ] 移動臨時檔案 (如有)

### Phase 5: 驗證和文檔 (30 分鐘)
- [ ] 測試專案功能正常
- [ ] 更新根目錄 `README.md`
- [ ] 提交 Git commit
- [ ] 通知團隊新結構

---

## 🔧 執行腳本

### 快速執行: 方案 A 整合腳本

```powershell
# AIVA Archive Consolidation Script
# 方案 A: 單一存檔整合

$archivePath = "C:\D\fold7\AIVA-git\_archive"
cd $archivePath

Write-Host "🚀 開始整合 _archive/ 目錄..." -ForegroundColor Cyan

# Step 1: 建立新分類目錄
Write-Host "`n📁 建立標準化目錄結構..." -ForegroundColor Yellow
$newDirs = @(
    "01_completed_projects",
    "02_deprecated_code",
    "03_historical_reports",
    "04_scripts_completed",
    "05_backups"
)

foreach ($dir in $newDirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   ✅ 建立 $dir" -ForegroundColor Green
    }
}

# Step 2: 移動現有目錄
Write-Host "`n📦 移動現有目錄到新分類..." -ForegroundColor Yellow

# 移動 completed_projects 內容
if (Test-Path "completed_projects") {
    Get-ChildItem "completed_projects" | Move-Item -Destination "01_completed_projects/" -Force
    Write-Host "   ✅ 移動 completed_projects/* → 01_completed_projects/" -ForegroundColor Green
    Remove-Item "completed_projects" -Force -ErrorAction SilentlyContinue
}

# 移動廢棄代碼
$deprecatedMoves = @{
    "deprecated_schema_tools" = "02_deprecated_code/schema_tools"
    "legacy_components" = "02_deprecated_code/legacy_components"
    "duplicates_cleanup" = "02_deprecated_code/duplicates_cleanup"
    "old_tests" = "02_deprecated_code/old_tests"
}

foreach ($src in $deprecatedMoves.Keys) {
    if (Test-Path $src) {
        Move-Item $src $deprecatedMoves[$src] -Force
        Write-Host "   ✅ 移動 $src → $($deprecatedMoves[$src])" -ForegroundColor Green
    }
}

# 移動歷史報告
$reportMoves = @(
    "duplicate_definition_reports",
    "analysis_reports_2025"
)

foreach ($dir in $reportMoves) {
    if (Test-Path $dir) {
        Move-Item $dir "03_historical_reports/" -Force
        Write-Host "   ✅ 移動 $dir → 03_historical_reports/" -ForegroundColor Green
    }
}

# 移動 schemas (如果在 deprecated_schema_tools 裡)
if (Test-Path "02_deprecated_code/schema_tools/schemas") {
    Move-Item "02_deprecated_code/schema_tools/schemas" "03_historical_reports/schemas" -Force
    Write-Host "   ✅ 移動 schemas → 03_historical_reports/" -ForegroundColor Green
}

# 移動腳本和備份
if (Test-Path "scripts_completed") {
    Move-Item "scripts_completed/*" "04_scripts_completed/" -Force
    Remove-Item "scripts_completed" -Force -ErrorAction SilentlyContinue
    Write-Host "   ✅ 移動 scripts_completed → 04_scripts_completed/" -ForegroundColor Green
}

if (Test-Path "backups") {
    Move-Item "backups/*" "05_backups/" -Force
    Remove-Item "backups" -Force -ErrorAction SilentlyContinue
    Write-Host "   ✅ 移動 backups → 05_backups/" -ForegroundColor Green
}

# Step 3: 統計
Write-Host "`n📊 整合完成統計:" -ForegroundColor Cyan
foreach ($dir in $newDirs) {
    $fileCount = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue).Count
    Write-Host "   $dir`: $fileCount 個檔案" -ForegroundColor White
}

Write-Host "`n✨ 整合完成!" -ForegroundColor Green
Write-Host "   下一步: 建立 ARCHIVE_INDEX.md" -ForegroundColor Yellow
```

---

## 📖 參考文件

- `_archive/EXECUTIVE_SUMMARY.md` - 已完成項目總覽
- `_archive/ARCHITECTURE_EVOLUTION_HISTORY.md` - 架構演進歷史
- `_archive/ARCHIVE_STRUCTURE.md` - 原始結構說明 (將被 ARCHIVE_INDEX.md 取代)

---

**規劃完成**: 準備執行整合! 🚀
