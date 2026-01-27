# 📚 AIVA 使用者手冊中心

> **v2.1.1 統一手冊管理**  
> 最後更新：2025-11-29  
> **✅ 所有手冊已驗證可用**

## 📑 目錄

1. [📂 目錄結構](#-目錄結構)
2. [🚀 快速導航](#-快速導航)
3. [📖 手冊分類](#-手冊分類)
4. [🔗 相關資源連結](#-相關資源連結)
5. [📝 手冊使用建議](#-手冊使用建議)
6. [⚙️ 手冊維護規範](#️-手冊維護規範)

---

## 📂 目錄結構

```
docs/user-guides/
├── README.md                        # 本目錄文件（導航中心）
├── AIVA_CLI_UNIFIED_GUIDE.md        # ✨ CLI 統一使用指南 v1.1 (2026-01-28更新)
├── QUICK_START_GUIDE.md             # 快速入門指南（Docker/本地部署）
├── TYPESCRIPT_ENGINE_GUIDE.md       # TypeScript Engine 操作指南 ✅ 完整
├── SCAN_MODULE_GUIDE.md             # Scan 模組使用手冊 ⚠️ 部分完整
└── DUAL_LOOP_ISSUE_REPORT.md        # 雙閉環問題分析報告

已刪除（2026-01-28）:
├── CLI_GUIDE.md                     # 已被 AIVA_CLI_UNIFIED_GUIDE v1.1 取代
└── CAPABILITY_EXECUTION_QUICK_GUIDE.md  # 已過時

已歸檔（移至 _archive/07_documentation_archive/user-guides_cli_2026-01-10/）:
├── CLI_COMPLETE_GUIDE_archived.md   # 重複內容
└── CLI_FLOW_USAGE_GUIDE_archived.md # 數據過時
```

---

## 🚀 快速導航

### 1️⃣ 新手入門
- **[CLI 統一使用指南](./AIVA_CLI_UNIFIED_GUIDE.md)** ⭐ 最新版本 v1.1（2026-01-28）
  - 統一的 CLI 系統完整說明
  - 基於動態 Flow 執行架構
  - 快速啟動方式（命令行 + 參數）
  - 實際驗證的命令與範例
  - Flow 執行與 Dry Run 預覽
  - AI 強度控制（0.0-1.0）
  - 常見問題與故障排除

- **[快速入門指南](./QUICK_START_GUIDE.md)** ⭐ 必讀
  - Docker 部署方式
  - 四種運作模式（API/Monitor/Interactive/Daemon）
  - AI 動態調用架構
  - 資源效率對比

### 2️⃣ 模組操作
- **[TypeScript Engine 操作指南](./TYPESCRIPT_ENGINE_GUIDE.md)**
  - 環境準備與依賴安裝
  - TypeScript 編譯流程
  - Playwright 瀏覽器設置
  - 測試驗證與故障排除
  
- **[Scan 模組使用手冊](./SCAN_MODULE_GUIDE.md)** ⚠️ 部分章節待補充
  - 兩階段掃描流程（Phase 0/1）
  - AI 命令接口使用
  - 快速測試範例（已驗證）
  - ⚠️ 第5-8章節內容缺失

### 3️⃣ 技術分析
- **[雙閉環問題分析](./DUAL_LOOP_ISSUE_REPORT.md)**
  - 實現現狀評估
  - 缺失組件分析（SystemSelfExplorer、BioNeuronDecisionController）
  - 修復優先級
  - 實施計劃

---

## 📖 手冊分類

### 🎯 入門級手冊（推薦順序）
| 手冊 | 適合對象 | 預計閱讀時間 | 驗證狀態 |
|------|---------|------------|---------|
| [CLI 統一使用指南](./AIVA_CLI_UNIFIED_GUIDE.md) | 所有 CLI 用戶 | 20 分鐘 | ✅ v1.1 (2026-01-28) |
| [快速入門指南](./QUICK_START_GUIDE.md) | 所有使用者 | 10 分鐘 | ✅ 已驗證 |

### 🔧 模組操作手冊
| 手冊 | 適合對象 | 預計閱讀時間 | 完整度 |
|------|---------|------------|--------|
| [TypeScript Engine 指南](./TYPESCRIPT_ENGINE_GUIDE.md) | 前端開發者 | 30 分鐘 | ✅ 完整（664行）|
| [Scan 模組指南](./SCAN_MODULE_GUIDE.md) | 安全測試人員 | 20 分鐘 | ⚠️ 60% 完整 |

### 📊 分析報告
| 報告 | 類型 | 更新日期 | 狀態 |
|------|------|---------|------|
| [雙閉環問題分析](./DUAL_LOOP_ISSUE_REPORT.md) | 技術分析 | 2025-11-29 | ✅ 完成 |

---

## 🔗 相關資源連結

### 📚 架構文檔
- [Docker/Kubernetes 指南](../../reports/architecture/DOCKER_KUBERNETES_GUIDE.md)
- [AI 啟動策略說明](../../reports/architecture/AI_STARTUP_AND_DIAGNOSTIC_CLARIFICATION.md)
- [構建流程指南](../../reports/architecture/BUILD_GUIDE.md)
- [開發者指南](../../reports/architecture/DEVELOPER_GUIDE.md)

### 🛠️ 開發文檔
- [Go 開發指南](../../reports/architecture/GO_DEVELOPMENT_GUIDE.md)
- [Rust 開發指南](../../reports/architecture/RUST_DEVELOPMENT_GUIDE.md)
- [Python 開發指南](../../reports/architecture/PYTHON_DEVELOPMENT_GUIDE.md)

### 🧪 測試與部署
- [安裝指南](../../reports/testing/INSTALLATION_GUIDE.md)（如果存在）
- [部署檢查清單](../../DEPLOYMENT_CHECKLIST.md)（如果存在）

---

## 📝 手冊使用建議

### 🆕 新使用者
1. ✅ 先閱讀 **[快速入門指南](./QUICK_START_GUIDE.md)** 了解系統架構
2. ✅ 實際操作 **[CLI 使用指南](./CLI_GUIDE.md)** 體驗功能
3. 根據需求選擇對應模組手冊深入學習
4. 遇到問題查看常見問題章節或架構文檔

### 👨‍💻 開發者
1. 閱讀快速入門指南了解整體架構
2. 根據開發語言選擇對應開發指南（Go/Rust/Python）
3. 參考 [AI 啟動策略說明](../../reports/architecture/AI_STARTUP_AND_DIAGNOSTIC_CLARIFICATION.md)
4. 如需擴展模組，參考 TypeScript Engine 或 Scan 模組指南

### 🔧 運維人員
1. 閱讀快速入門指南
2. 深入學習 [Docker/Kubernetes 指南](../../reports/architecture/DOCKER_KUBERNETES_GUIDE.md)
3. 了解 [構建流程](../../reports/architecture/BUILD_GUIDE.md)
4. 掌握故障排除方法（各手冊中的故障排除章節）

---

## ⚙️ 手冊維護規範

### 📌 命名規範
- 使用英文大寫 + 底線：`QUICK_START_GUIDE.md`
- 保持簡潔有意義：反映內容主題
- 統一後綴：`_GUIDE.md`（指南）、`_REPORT.md`（報告）

### 📅 更新規範
- 每次更新需標註日期和驗證狀態
- 重大變更需更新版本號
- 過時或不可用內容及時刪除或標記

### 🔗 連結規範
- 使用相對路徑
- 確保連結有效性（已驗證或標記 ⚠️）
- 提供清晰的連結說明

### ✅ 驗證規範
- 新增手冊必須先驗證內容可用性
- 標記驗證狀態：✅ 已驗證 / ⚠️ 部分可用 / ❌ 不可用
- 不可用或空內容的手冊不應收錄

---

## 🆘 需要幫助？

### 📧 問題反饋
- 發現錯誤：請提交 Issue
- 改進建議：歡迎 Pull Request
- 內容疑問：查看相關架構文檔

### 🔍 找不到需要的手冊？
建議查看：
- [主 README](../../README.md)（如果存在）
- [架構文檔目錄](../../reports/architecture/README.md)
- 直接搜尋 `reports/architecture/` 下的 *_GUIDE.md 文件

---

## 📊 手冊統計

- **總手冊數**: 5
- **完全可用**: 3 (快速入門、CLI、TypeScript Engine)
- **部分可用**: 1 (Scan 模組 - 60% 完整)
- **分析報告**: 1 (雙閉環問題分析)
- **最後更新**: 2025-11-29
- **維護狀態**: 🟢 活躍維護中

---

## 📝 CLI 使用指南功能確認

**已驗證功能** (2025-11-29):
```powershell
# ✅ 統計功能正常
python aiva_cli.py --stats
# 輸出: 782 個能力, 16 個模組, 4 種語言

# ✅ 幫助功能正常
python aiva_cli.py --help
# 輸出: 完整命令列表和範例

# ✅ 支持的操作
- 查詢能力 (--query)
- AI 執行攻擊 (--attack)
- 工作流推薦 (--workflow)
- 同步能力資料 (--sync)
- 運行測試 (--test)
```

---

**提示**: 此目錄僅收錄經過驗證的實用手冊，空內容或不可用的手冊已移除。

**版本**: v2.0  
**建立日期**: 2025-11-29  
**維護原則**: 實用至上，驗證優先