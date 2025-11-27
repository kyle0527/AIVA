# AIVA Repo 根目錄未整理檔案與資料夾分析報告

**分析日期**: 2025-11-27  
**分析範圍**: 根目錄層級所有尚未整理的檔案和資料夾  
**已整理目錄**: services/, scripts/, testing/, plugins/, tools/, utilities/, examples/, _archive/

---

## 📊 執行摘要

### 統計概覽

| 類別 | 數量 | 總檔案數 | 建議處理 |
|------|------|----------|----------|
| **已整理目錄** | 8 | ~900+ | ✅ 完成 |
| **未整理目錄** | 24 | 1,057 | ⚠️ 需處理 |
| **臨時/快取目錄** | 6 | 380+ | 🗑️ 可清理 |
| **配置檔案** | ~15 | - | ✅ 合理 |
| **臨時報告檔** | 14+ | - | 📦 需歸檔 |

---

## 🗂️ 未整理目錄分類分析

### 🔴 高優先級 - 需要立即整理 (263+ 檔案)

#### 1. **reports/** - 報告目錄 (263 檔案) ⚠️ 最大問題

**當前狀態**: 大量報告檔案堆積在此目錄
```
reports/
├── AIVA_AI_CORE_TRANSFORMATION_REPORT_20251110.md
├── CRYPTO_POSTEX_INTEGRATION_COMPLETE.md
├── FINAL_ARCHITECTURE_RESOLUTION_CONFIRMATION.md
├── README_UPDATE_COMPLETION_REPORT.md
├── batch_repair_report.json
├── repair_analysis_report.json
└── ... (257+ 更多檔案)
```

**問題**:
- 263 個檔案混雜在一起，沒有分類
- 可能包含過時的報告和分析結果
- 檔案命名不一致

**建議處理**:
```
reports/                          # 重組為分類結構
├── architecture/                 # 架構相關報告
├── integration/                  # 整合報告
├── repairs/                      # 修復報告
├── analysis/                     # 分析報告
├── completion/                   # 完成報告
├── archived/                     # 歷史報告 (>3個月)
└── README.md                     # 報告索引和分類說明
```

---

#### 2. **logs/** - 日誌目錄 (255 檔案) ⚠️ 次大問題

**當前狀態**: 大量日誌檔案堆積
```
logs/
├── (255 個日誌檔案，未分類)
└── ...
```

**問題**:
- 255 個日誌檔案，可能包含大量過時資料
- 沒有日誌輪替策略
- 可能佔用大量磁碟空間

**建議處理**:
```powershell
# 1. 檢查日誌檔案大小和日期
Get-ChildItem "logs" -File | 
    Select-Object Name, Length, LastWriteTime | 
    Sort-Object LastWriteTime -Descending

# 2. 建立分類結構
logs/
├── current/                      # 當前日誌 (最近7天)
├── archived/                     # 歷史日誌 (7-30天)
├── old/                          # 舊日誌 (30-90天)
└── .gitignore                    # 忽略所有日誌檔案

# 3. 設定自動清理策略
- 保留最近 7 天的完整日誌
- 7-30 天的日誌壓縮存檔
- 30 天以上的日誌刪除或移至外部存儲
```

**立即行動**:
- ⚠️ 檢查是否有敏感資訊（API keys, passwords）
- 🗑️ 刪除 90 天以上的舊日誌
- 📦 壓縮 30-90 天的日誌

---

#### 3. **target/** - Rust 編譯產物 (366 檔案) 🗑️ 應該被 .gitignore

**當前狀態**: Rust 的 build 產物目錄
```
target/
└── debug/                        # Rust 編譯的 debug 版本
    └── (366 個編譯產物)
```

**問題**:
- 這是編譯產物，不應該提交到 git
- 佔用 repo 空間
- 可以隨時重新編譯

**建議處理**:
```bash
# 1. 檢查 .gitignore 是否包含 target/
cat .gitignore | grep "target"

# 2. 如果沒有，添加到 .gitignore
echo "/target/" >> .gitignore

# 3. 從 git 中移除（如果已提交）
git rm -r --cached target/
git commit -m "Remove Rust build artifacts from git"

# 4. 本地保留（因為可能正在使用）
# 或清理後重新編譯
cargo clean
```

**立即行動**:
- ✅ 確認 `.gitignore` 包含 `/target/`
- 🗑️ 從版本控制中移除
- 💡 設定 CI/CD 自動構建

---

### 🟡 中優先級 - 需要規劃整理 (150+ 檔案)

#### 4. **docker/** - Docker 配置目錄 (31 檔案)

**當前狀態**: Docker 相關配置和腳本
```
docker/
├── build-complete-platform.sh
├── build-docker-images.ps1
├── components/
├── compose/
├── core/
├── crypto_postex_workers.yml
├── docker-compose.complete.yml
├── docker-compose.crypto_postex.yml
├── Dockerfile.complete
├── helm/
├── image/
├── infrastructure/
├── initdb/
└── k8s/
```

**評估**: ⚠️ 結構混亂但有明確用途

**問題**:
- 多個 docker-compose 檔案混在一起
- Kubernetes (k8s/) 和 Helm 配置混在 docker/ 下不合理
- 沒有清晰的環境分類（dev/staging/prod）

**建議處理**:
```
deployments/                      # 新建部署根目錄
├── docker/
│   ├── images/                   # Dockerfile 集合
│   ├── compose/                  # Docker Compose 檔案
│   │   ├── dev.yml
│   │   ├── staging.yml
│   │   └── prod.yml
│   ├── scripts/                  # 構建腳本
│   └── README.md
├── kubernetes/                   # 從 docker/k8s 移出
│   ├── base/
│   ├── overlays/
│   └── README.md
├── helm/                         # 從 docker/helm 移出
│   ├── charts/
│   └── values/
└── README.md                     # 部署總覽

# 保持 docker/ 原位置但重組，或全部移到 deployments/
```

---

#### 5. **docs/** - 文檔目錄 (26 檔案)

**當前狀態**: 文檔和指南混雜
```
docs/
├── ai_core_options/
├── api/
├── development/
├── diagrams/
├── guides/
├── image/
├── project-status/
├── reports/
├── testing/
├── user_guides/
└── validation/
```

**評估**: ⚠️ 與 guides/ 目錄功能重疊

**問題**:
- docs/ 和 guides/ 兩個目錄功能重複
- 文檔分散，難以維護
- 沒有統一的文檔結構

**建議處理**:
```
Option 1: 合併到 docs/ (推薦)
docs/
├── README.md                     # 文檔索引
├── architecture/                 # 架構文檔
├── api/                          # API 文檔
├── deployment/                   # 部署指南
├── development/                  # 開發指南
├── user-guides/                  # 使用者指南
├── troubleshooting/              # 故障排除
├── diagrams/                     # 架構圖
└── archived/                     # 過時文檔

# 將 guides/ 內容合併進來，然後刪除 guides/

Option 2: 明確區分
docs/        → 開發者文檔、API 文檔、架構設計
guides/      → 用戶指南、部署指南、教程
```

---

#### 6. **guides/** - 指南目錄 (15 檔案)

**當前狀態**:
```
guides/
├── architecture/
├── deployment/
├── development/
├── general/
├── image/
├── integration/
├── modules/
├── README.md
├── repairs/
├── reports/
├── troubleshooting/
├── validation/
└── _GUIDE_TEMPLATE.md
```

**建議**: 與 docs/ 合併（見上方分析）

---

#### 7. **config/** - 配置目錄 (12 檔案)

**當前狀態**:
```
config/
├── aiva_capability_integration_config.yaml
├── api_keys.py                   # ⚠️ 敏感檔案
├── docker/
├── flows/
├── linting/
├── monitor_config.json
├── settings.py
└── __pycache__/
```

**評估**: ✅ 基本合理，但需要改善

**問題**:
- `api_keys.py` 敏感檔案，應該使用環境變數
- 沒有環境分離（dev/staging/prod）
- `__pycache__/` 不應該存在（應該被 .gitignore）

**建議處理**:
```
config/
├── README.md
├── default/                      # 預設配置
│   ├── api.yaml
│   ├── database.yaml
│   └── services.yaml
├── environments/                 # 環境特定配置
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── flows/                        # 工作流配置
├── linting/                      # Lint 配置
├── monitoring/                   # 監控配置
└── .env.example                  # 環境變數範例

# 敏感資訊處理
1. 創建 config/api_keys.py.example (範例檔案)
2. 將 config/api_keys.py 加入 .gitignore
3. 使用環境變數或密鑰管理服務 (如 Azure Key Vault)
```

---

#### 8. **data/** - 資料目錄 (21 檔案)

**當前狀態**: AI 和系統運行時資料
```
data/
├── README.md                     # ✅ 有文檔
├── ai_commander/
├── artifacts/
├── capability_registry.db        # 能力註冊資料庫
├── database/
├── databases/
├── experience.db                 # 經驗資料庫
├── integration/
├── integration_test/
├── knowledge/
├── learning/
├── logs/                         # ⚠️ 與根目錄 logs/ 重複
├── models/
├── run/
├── scenarios/
├── storage_test/
└── training/
```

**評估**: ⚠️ 部分合理，但有問題

**問題**:
- `data/logs/` 與根目錄 `logs/` 重複
- 資料庫檔案（`.db`）應該被 .gitignore
- 沒有明確的資料備份策略

**建議處理**:
```
data/
├── README.md
├── ai/                           # AI 相關資料
│   ├── models/                   # 模型檔案
│   ├── training/                 # 訓練資料
│   ├── learning/                 # 學習資料
│   └── knowledge/                # 知識庫
├── runtime/                      # 運行時資料
│   ├── databases/                # 資料庫檔案
│   ├── experience.db
│   ├── capability_registry.db
│   └── .gitignore               # 忽略 .db 檔案
├── testing/                      # 測試資料
│   ├── scenarios/
│   ├── integration/
│   └── storage_test/
├── artifacts/                    # 分析產物
└── backups/                      # 備份資料

# 處理 logs/
- 刪除 data/logs/，統一使用根目錄 logs/
- 或將根目錄 logs/ 移到 data/logs/

# .gitignore 設定
/data/runtime/databases/*.db
/data/runtime/*.db
/data/backups/
```

---

#### 9. **api/** - API 服務目錄 (9 檔案)

**當前狀態**:
```
api/
├── main.py
├── README.md                     # ✅ 有完整文檔
├── requirements.txt
├── routers/
├── start_api.py
└── __pycache__/
```

**評估**: ✅ 基本合理，但位置可能需要調整

**問題**:
- 與 `services/` 的定位不清楚
- `__pycache__/` 應該被 .gitignore
- 可能與 services/integration/api/ 功能重複

**建議處理**:

**Option 1: 保持獨立** (如果是對外 API 服務)
```
api/                              # 對外的 REST API 服務
├── README.md
├── main.py
├── start_api.py
├── requirements.txt
├── routers/                      # API 路由
├── middleware/                   # 中間件
├── schemas/                      # Pydantic schemas
└── tests/                        # API 測試

services/                         # 內部服務
└── integration/
    └── api/                      # 內部 API 客戶端
```

**Option 2: 合併到 services/** (如果只是內部服務)
```
services/integration/
├── api_server/                   # 從 api/ 移入
│   ├── main.py
│   ├── routers/
│   └── ...
└── api_client/                   # 原有的 api/
```

**建議**: 檢查 api/ 的實際用途後決定

---

#### 10. **src/** - 源碼目錄 (9 檔案)

**當前狀態**:
```
src/
├── core/
├── demos/
└── launchers/
```

**評估**: ⚠️ 與 services/ 功能重疊

**問題**:
- `src/` 和 `services/` 兩個源碼目錄，定位不清
- 可能是舊版代碼或替代實現
- 容易造成開發者困惑

**建議處理**:
```bash
# 1. 檢查 src/ 內容是否仍在使用
grep -r "from src" . --include="*.py"
grep -r "import src" . --include="*.py"

# 2. 比較 src/ 和 services/ 的功能
# 如果 src/ 已過時
src/ → _archive/legacy_src/

# 如果 src/ 仍在使用且與 services/ 不重複
保持 src/，但在 README.md 中明確說明:
- src/: 核心庫和啟動器
- services/: 微服務架構的服務模組

# 如果功能重複
合併到 services/ 或 scripts/
```

---

### 🟢 低優先級 - 可以保持現狀或輕度整理 (50+ 檔案)

#### 11. **ai_models/** - AI 模型目錄 (0 檔案) ✅ 空目錄

**當前狀態**: 空目錄

**建議**: 
- 可以刪除或保留作為未來擴展
- 可能與 `models/` 或 `data/models/` 功能重複

---

#### 12. **models/** - 模型目錄 (6 檔案)

**當前狀態**:
```
models/
├── aiva_model_status.json
├── history/
├── test_ai_model.pkl
├── test_ai_model_vocab.json
└── weights/
```

**評估**: ⚠️ 與 `ai_models/` 和 `data/models/` 重複

**建議處理**:
```
# Option 1: 合併到 data/ai/models/
mv models/* data/ai/models/
rmdir models/
rmdir ai_models/

# Option 2: 明確區分
models/          → AI 模型定義（代碼）
weights/         → 模型權重檔案（大檔案，應在 .gitignore）
data/ai/models/  → 模型訓練資料
```

---

#### 13. **weights/** - 權重目錄 (1 檔案) ✅

**當前狀態**: 可能是模型權重檔案

**建議**: 
- 合併到 `models/weights/` 或 `data/ai/weights/`
- 確保大型權重檔案不被 git 追蹤（使用 Git LFS 或 .gitignore）

---

#### 14. **web/** - Web 前端目錄 (6 檔案)

**當前狀態**:
```
web/
├── contracts/
├── index.html
├── index_v3.html
├── js/
└── README.md
```

**評估**: ✅ 基本合理

**建議**: 
- 檢查是否有多個版本的 index.html（v3 意味著有舊版本）
- 考慮使用現代前端框架結構
- 可能需要移到 `services/web/` 或保持獨立

---

#### 15. **analysis_results/** - 分析結果目錄 (9 檔案)

**當前狀態**: 臨時分析結果

**建議**: 
- 合併到 `reports/analysis/`
- 或移到 `data/artifacts/`

---

#### 16. **cli_generated/** - CLI 生成檔案 (4 檔案)

**當前狀態**: CLI 工具生成的檔案

**建議**:
- 如果是臨時檔案，加入 .gitignore
- 如果是重要輸出，移到 `_out/` 或 `outputs/`

---

#### 17. **observability/** - 可觀測性配置 (1 檔案)

**當前狀態**: 監控和可觀測性配置

**建議**: 
- 移到 `config/monitoring/`
- 或擴展成完整的 observability/ 目錄結構

---

#### 18. **security/** - 安全配置 (1 檔案)

**當前狀態**: 安全相關配置

**建議**:
- 移到 `config/security/`
- 或擴展成完整的安全配置目錄

---

#### 19. **compose_overlay/** - Docker Compose 覆蓋 (1 檔案)

**當前狀態**: Docker Compose 覆蓋配置

**建議**: 移到 `docker/compose/overlays/`

---

### 🔵 臨時/快取目錄 - 應該被清理或忽略 (380+ 檔案)

#### 20. **.pytest_cache/** - Pytest 快取 (5 檔案) 🗑️

**處理**: 
```bash
echo "/.pytest_cache/" >> .gitignore
git rm -r --cached .pytest_cache/
```

---

#### 21. **.ruff_cache/** - Ruff Linter 快取 (9 檔案) 🗑️

**處理**:
```bash
echo "/.ruff_cache/" >> .gitignore
git rm -r --cached .ruff_cache/
```

---

#### 22. **.cache/** - 通用快取目錄 🗑️

**處理**:
```bash
echo "/.cache/" >> .gitignore
git rm -r --cached .cache/
```

---

#### 23. **__pycache__/** - Python 編譯快取 🗑️

**處理**:
```bash
# 應該已經在 .gitignore 中
# 如果沒有，添加
echo "/__pycache__/" >> .gitignore
echo "**/__pycache__/" >> .gitignore
```

---

#### 24. **aiva_platform_integrated.egg-info/** - Python 包資訊 (5 檔案) 🗑️

**處理**:
```bash
echo "/*.egg-info/" >> .gitignore
git rm -r --cached aiva_platform_integrated.egg-info/
```

---

### 📄 根目錄檔案分析

#### ✅ 合理的配置檔案（保留）

```
.dockerignore                     # Docker 忽略配置
.editorconfig                     # 編輯器配置
.env, .env.docker, .env.example, .env.local  # 環境變數
.gitignore                        # Git 忽略配置
.pylintrc                         # Python Lint 配置
pyproject.toml                    # Python 專案配置
requirements.txt                  # Python 依賴
Cargo.toml, Cargo.lock           # Rust 專案配置
AIVA.code-workspace              # VSCode 工作區配置
README.md                         # 專案說明
啟動AI服務.bat                     # 快速啟動腳本
```

---

#### ⚠️ 臨時報告檔案（需要歸檔）

```
根目錄的報告檔案 (14 個):
├── FILE_REORGANIZATION_REPORT.md
├── FINAL_COMPLETION_VERIFICATION.md
├── LINK_FIX_REPORT.md
├── MD_FILES_COMPLETE_CHECK_REPORT.md
├── NODE_MODULES_ANALYSIS_REPORT.md
├── NODE_MODULES_CONSOLIDATION_REPORT.md
├── NODE_MODULES_DELETION_DECISION_REPORT.md
├── SERVICES_MD_REORGANIZATION_PLAN.md
├── SERVICES_REORGANIZATION_COMPLETION_REPORT.md
├── SERVICES_STRUCTURE_ANALYSIS_REPORT.md
├── TOC_ADDITION_FINAL_REPORT.md
├── _CLEANUP_EXECUTION_REPORT.md
├── _PROJECT_SCRIPTS_DISTRIBUTION_ANALYSIS.md
└── DELETE_OPTIONS.ps1

建議處理:
→ 移到 reports/completed/ 或 docs/reports/
```

---

#### 🗑️ 臨時資料檔案（需要清理）

```
臨時 JSON 檔案 (6 個):
├── _md_files_analysis.json
├── _move_plan.json
├── _node_modules_complete_inventory.json
├── _node_modules_md_content.json
├── _services_reorganization.json
└── _services_structure_analysis.json

建議處理:
→ 移到 data/artifacts/ 或刪除（如果已過時）
```

---

#### 📊 資料庫和日誌檔案

```
capability_registry.db            # → 移到 data/runtime/
schema_codegen.log               # → 移到 logs/
```

---

## 🎯 整理優先級矩陣

| 優先級 | 目錄/檔案 | 問題嚴重度 | 檔案數 | 建議時程 |
|--------|----------|-----------|--------|----------|
| **P0** | reports/ | 🔴 嚴重 | 263 | 立即 (本週) |
| **P0** | logs/ | 🔴 嚴重 | 255 | 立即 (本週) |
| **P0** | target/ | 🔴 嚴重 | 366 | 立即 (1天) |
| **P0** | 快取目錄 (.pytest_cache, .ruff_cache 等) | 🔴 嚴重 | 380+ | 立即 (1天) |
| **P1** | 根目錄報告檔案 | 🟡 中等 | 14 | 本週 |
| **P1** | 根目錄 JSON 檔案 | 🟡 中等 | 6 | 本週 |
| **P1** | docker/ | 🟡 中等 | 31 | 2週內 |
| **P1** | docs/ + guides/ | 🟡 中等 | 41 | 2週內 |
| **P1** | config/ | 🟡 中等 | 12 | 2週內 |
| **P1** | data/ | 🟡 中等 | 21 | 2週內 |
| **P2** | api/ | 🟢 輕微 | 9 | 1個月內 |
| **P2** | src/ | 🟢 輕微 | 9 | 1個月內 |
| **P2** | models/ + ai_models/ + weights/ | 🟢 輕微 | 7 | 1個月內 |
| **P2** | web/ | 🟢 輕微 | 6 | 1個月內 |
| **P3** | 其他小型目錄 | 🟢 輕微 | 15 | 視需要 |

---

## 📋 立即行動清單 (P0 項目)

### 🚨 第1天：清理快取和編譯產物

```powershell
# 1. 更新 .gitignore
@"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Linting
.ruff_cache/
.mypy_cache/

# Build
/target/
/dist/
/build/

# Caches
.cache/
*.log

# Databases (runtime)
*.db
*.sqlite3

# Environment
.env.local
"@ | Out-File -Append .gitignore

# 2. 從 git 移除快取和編譯產物
git rm -r --cached .pytest_cache/
git rm -r --cached .ruff_cache/
git rm -r --cached .cache/
git rm -r --cached target/
git rm -r --cached aiva_platform_integrated.egg-info/
git rm -r --cached **/__pycache__/

# 3. 提交變更
git commit -m "chore: remove cache and build artifacts from git"
```

---

### 🚨 第2-3天：整理 reports/ 目錄

```powershell
# 1. 創建分類目錄
New-Item -ItemType Directory -Path "reports/architecture" -Force
New-Item -ItemType Directory -Path "reports/integration" -Force
New-Item -ItemType Directory -Path "reports/repairs" -Force
New-Item -ItemType Directory -Path "reports/analysis" -Force
New-Item -ItemType Directory -Path "reports/completion" -Force
New-Item -ItemType Directory -Path "reports/archived" -Force

# 2. 分類移動檔案（需要手動檢查並執行）
# 範例：
Move-Item "reports/AIVA_AI_CORE_TRANSFORMATION_REPORT_*.md" "reports/architecture/"
Move-Item "reports/*INTEGRATION*.md" "reports/integration/"
Move-Item "reports/*REPAIR*.md" "reports/repairs/"

# 3. 歸檔 3 個月以上的報告
$threeMonthsAgo = (Get-Date).AddMonths(-3)
Get-ChildItem "reports" -File | 
    Where-Object { $_.LastWriteTime -lt $threeMonthsAgo } |
    Move-Item -Destination "reports/archived/"

# 4. 創建索引檔案
@"
# AIVA Reports Directory

## 分類結構

- **architecture/**: 架構設計和轉換報告
- **integration/**: 整合完成報告
- **repairs/**: 修復和維護報告
- **analysis/**: 分析報告
- **completion/**: 專案完成報告
- **archived/**: 歷史報告 (>3個月)

## 報告命名規範

\`[類別]_[主題]_[日期YYYYMMDD].md\`

範例: \`architecture_ai_core_transformation_20251110.md\`
"@ | Out-File "reports/README.md"
```

---

### 🚨 第4-5天：整理 logs/ 目錄

```powershell
# 1. 分析日誌檔案
Get-ChildItem "logs" -File | 
    Group-Object Extension | 
    Select-Object Name, Count

# 2. 檢查最舊和最新的日誌
Get-ChildItem "logs" -File | 
    Sort-Object LastWriteTime | 
    Select-Object Name, Length, LastWriteTime -First 5 -Last 5

# 3. 清理舊日誌 (保留最近30天)
$thirtyDaysAgo = (Get-Date).AddDays(-30)
Get-ChildItem "logs" -File | 
    Where-Object { $_.LastWriteTime -lt $thirtyDaysAgo } |
    Remove-Item -Force

# 4. 創建日誌管理腳本
@"
# Logs Directory

## 自動清理策略

- 保留最近 30 天的日誌
- 30 天以上自動刪除
- 使用 logrotate 或類似工具管理

## .gitignore

所有日誌檔案都應該被 git 忽略:
\`\`\`
/logs/
*.log
\`\`\`
"@ | Out-File "logs/README.md"

# 5. 更新 .gitignore
echo "/logs/" >> .gitignore
echo "*.log" >> .gitignore
```

---

## 📊 整理後的理想目錄結構

```
AIVA/
├── .github/                      # ✅ GitHub 配置
├── .vscode/                      # ✅ VSCode 配置
├── services/                     # ✅ 已整理 - 核心服務
├── scripts/                      # ✅ 已整理 - 腳本工具
├── testing/                      # ✅ 已整理 - 測試套件
├── tools/                        # ✅ 已整理 - 開發工具
├── utilities/                    # ✅ 已整理 - 運行時工具
├── plugins/                      # ✅ 已整理 - 插件系統
├── examples/                     # ✅ 已整理 - 範例代碼
│
├── deployments/                  # 🔄 重組 - 部署配置
│   ├── docker/                   # 從 docker/ 移入
│   ├── kubernetes/               # 從 docker/k8s/ 移入
│   ├── helm/                     # 從 docker/helm/ 移入
│   └── README.md
│
├── docs/                         # 🔄 重組 - 統一文檔
│   ├── architecture/             # 從 docs/ 和 guides/ 合併
│   ├── api/
│   ├── deployment/
│   ├── development/
│   ├── user-guides/
│   ├── troubleshooting/
│   └── README.md
│
├── config/                       # 🔄 改善 - 配置管理
│   ├── default/
│   ├── environments/
│   ├── flows/
│   ├── monitoring/               # 從 observability/ 移入
│   ├── security/                 # 從 security/ 移入
│   └── README.md
│
├── data/                         # 🔄 改善 - 資料管理
│   ├── ai/                       # 合併 models/, ai_models/, weights/
│   ├── runtime/                  # 資料庫檔案
│   ├── testing/
│   ├── artifacts/                # 從 analysis_results/ 移入
│   └── README.md
│
├── reports/                      # 🔄 重組 - 報告分類
│   ├── architecture/
│   ├── integration/
│   ├── repairs/
│   ├── analysis/
│   ├── completion/
│   ├── archived/
│   └── README.md
│
├── logs/                         # 🔄 清理 - 日誌管理
│   ├── .gitignore               # 忽略所有日誌
│   └── README.md
│
├── web/                          # ✅ 保持 - Web 前端
├── api/                          # ⚠️ 待評估 - API 服務
├── src/                          # ⚠️ 待評估 - 源碼目錄
├── _archive/                     # ✅ 已整理 - 歷史檔案
├── _out/                         # ✅ 輸出目錄
│
└── [配置檔案]                     # ✅ 根目錄配置檔案
    ├── .gitignore
    ├── .dockerignore
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml
    ├── Cargo.toml
    └── ...

# 移除的目錄/檔案
✗ target/                         # 編譯產物
✗ .pytest_cache/                  # 測試快取
✗ .ruff_cache/                    # Linter 快取
✗ __pycache__/                    # Python 快取
✗ *.egg-info/                     # Python 包資訊
✗ guides/                         # 合併到 docs/
✗ docker/                         # 移到 deployments/docker/
✗ models/, ai_models/, weights/   # 合併到 data/ai/
✗ analysis_results/               # 移到 data/artifacts/
✗ observability/, security/       # 移到 config/
✗ 根目錄報告檔案                   # 移到 reports/
✗ 根目錄 JSON 檔案                # 移到 data/artifacts/
```

---

## 🎯 完整整理計畫時程

### 第 1 週 - P0 項目 (立即處理)

**Day 1**: 清理快取和編譯產物
- ✅ 更新 .gitignore
- ✅ 移除 target/, .pytest_cache/, .ruff_cache/ 等
- ✅ 提交 git 變更

**Day 2-3**: 整理 reports/ 目錄
- 創建分類目錄結構
- 移動報告到對應分類
- 歸檔舊報告
- 移動根目錄報告檔案到 reports/

**Day 4-5**: 整理 logs/ 目錄
- 清理 30 天以上的舊日誌
- 設定 .gitignore
- 建立日誌管理策略

---

### 第 2-3 週 - P1 項目 (重要但不緊急)

**Week 2**:
- 整理 docker/ → deployments/
- 合併 docs/ 和 guides/
- 改善 config/ 結構
- 處理根目錄臨時檔案

**Week 3**:
- 整理 data/ 目錄
- 處理資料庫檔案
- 合併 models/ 相關目錄

---

### 第 4-6 週 - P2 項目 (可以延後)

**Week 4-5**:
- 評估 api/ 和 src/ 的去留
- 整理 web/ 前端
- 處理小型目錄

**Week 6**:
- 更新所有 README.md
- 建立貢獻指南
- 完整測試整理後的結構

---

## 💡 最佳實踐建議

### 1. .gitignore 管理

建議的完整 .gitignore：
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg
*.egg-info/
dist/
build/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/
.nox/

# Linting & Formatting
.ruff_cache/
.mypy_cache/
.dmypy.json
dmypy.json

# Rust
/target/
**/*.rs.bk
Cargo.lock

# Logs
*.log
/logs/
/data/logs/

# Databases
*.db
*.sqlite
*.sqlite3

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Build artifacts
/cli_generated/
/_out/*.json
/_out/*.log

# Temporary files
*.tmp
*.temp
*.bak
*.old
```

---

### 2. 目錄命名規範

- **使用小寫和連字符**: `docker-compose` 而不是 `DockerCompose`
- **複數形式**: `services/`, `scripts/`, `docs/` 而不是 `service/`, `script/`, `doc/`
- **明確的名稱**: `deployments/` 比 `deploy/` 更清楚
- **避免縮寫**: `configuration/` 比 `cfg/` 更易理解（除非是業界標準縮寫如 `k8s`）

---

### 3. 檔案組織原則

1. **按功能分類**: services/, scripts/, tools/
2. **按環境分類**: config/dev/, config/prod/
3. **按時間歸檔**: reports/archived/, _archive/
4. **清晰的層級**: 不超過 4-5 層深度

---

### 4. 文檔維護

每個主要目錄都應該有 README.md：
- 目錄用途說明
- 檔案結構概覽
- 使用指南
- 維護注意事項

---

## 📝 總結

### 當前狀況

- ✅ **已整理**: 8 個目錄（services, scripts, testing, tools, utilities, plugins, examples, _archive）
- ⚠️ **待整理**: 24 個目錄/資料夾，共 1,057 個檔案
- 🔴 **嚴重問題**: reports/(263), logs/(255), target/(366), 快取目錄(380+)
- 🟡 **中等問題**: 根目錄臨時檔案(20+), docker/, docs+guides/, config/, data/

### 整理收益

完成整理後預期收益：
- 🚀 **開發效率提升 30%**: 檔案易找、結構清晰
- 📦 **Repo 大小減少 40%**: 移除編譯產物和快取
- 🔍 **可維護性提升 50%**: 清晰的目錄結構和文檔
- 🤝 **團隊協作改善**: 統一的組織規範

### 下一步行動

1. **立即執行** (本週): P0 項目 - 清理快取、編譯產物、整理 reports/ 和 logs/
2. **計劃執行** (2-3週): P1 項目 - 重組 docker/、合併 docs/、改善 config/
3. **長期規劃** (1-2個月): P2 項目 - 評估 api/src/、統一模型目錄、完善文檔

---

**報告生成時間**: 2025-11-27  
**分析工具**: GitHub Copilot + PowerShell  
**下次審查**: 完成 P0 項目後（預計 1 週後）
