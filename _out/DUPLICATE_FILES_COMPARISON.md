# 重複文檔詳細比對分析

**分析日期**: 2026-01-28  
**目的**: 確認重複文檔的實際內容差異，決定保留/移除策略

---

## 📊 重複文檔組別總覽

| 組別 | 文檔對 | 結論 | 建議操作 |
|------|--------|------|---------|
| #1 | GETTING_STARTED vs QUICK_START_GUIDE | **內容不同** | 保留兩者 |
| #2 | SCAN_MODULE_GUIDE vs SCAN_USAGE_GUIDE | **內容不同** | 保留兩者 |
| #3 | CLI_GUIDE vs AIVA_CLI_UNIFIED_GUIDE | **重複（舊vs新）** | 歸檔 CLI_GUIDE |
| #4 | QUICK_REFERENCE vs QUICK_REFERENCE_GUIDE | **完全不同** | 保留兩者 |
| #5 | guides/README vs AIVA_TECHNICAL_GUIDE_INDEX | **內容不同** | 保留兩者 |

---

## 詳細比對分析

### 組別 #1: GETTING_STARTED vs QUICK_START_GUIDE

#### 📄 GETTING_STARTED.md
- **位置**: `docs/01_user_documentation/user-guides/`
- **大小**: 413 行 / 9.15 KB
- **最後更新**: 2025-11-29
- **定位**: **完整的新手入門指南**

**內容特點**:
```
✅ 詳細的目錄結構 (6 sections)
✅ 系統概述 (782個能力模組說明)
✅ 完整的架構圖 (Layer 0/1/2)
✅ Windows/Docker 兩種快速啟動方式
✅ 4種AI運作模式詳解
✅ 實際使用範例
✅ 常見問題 FAQ
✅ 下一步學習路徑
```

**目標受眾**: 所有AIVA使用者（新手友好）

---

#### 📄 QUICK_START_GUIDE.md
- **位置**: `docs/01_user_documentation/user-guides/`
- **大小**: 291 行 / 7.23 KB
- **最後更新**: 2025-12-30
- **定位**: **架構導向的快速部署指南**

**內容特點**:
```
✅ v2.1.1 架構說明（分層架構）
✅ Docker 部署流程（詳細）
✅ 4種AI核心運作模式
✅ 實際使用範例
✅ API 端點說明
✅ 進階閱讀連結
```

**目標受眾**: 理解架構的開發者/運維人員

---

#### 🔍 比對結論

| 面向 | GETTING_STARTED | QUICK_START_GUIDE |
|------|----------------|-------------------|
| **定位** | 新手入門教學 | 快速部署操作 |
| **詳細度** | 更詳細（782能力說明） | 精簡（架構重點） |
| **架構圖** | 完整 Layer 0/1/2 | 簡化分層 |
| **啟動方式** | Windows + Docker | 主要 Docker |
| **FAQ** | 有（5個常見問題） | 無 |
| **下一步** | 完整學習路徑 | 進階閱讀連結 |

**判斷**: ❌ **非重複** - 兩者服務不同目的

**建議**: ✅ **保留兩者**
- `GETTING_STARTED.md` - 新手完整入門
- `QUICK_START_GUIDE.md` - 熟悉架構者快速部署

---

### 組別 #2: SCAN_MODULE_GUIDE vs SCAN_USAGE_GUIDE

#### 📄 SCAN_MODULE_GUIDE.md
- **位置**: `docs/01_user_documentation/user-guides/`
- **大小**: 403 行 / 13.53 KB
- **最後更新**: 2025-12-30
- **定位**: **Scan 模組技術手冊**

**內容特點**:
```
✅ 架構概覽（v2.1 適配器模式）
✅ 核心組件詳解
✅ 數據流向圖
✅ 兩階段掃描流程（Phase 0/1）
✅ AI 命令接口使用
✅ 完整範例腳本

⚠️ 標註不完整（第5-8章節缺失）
⚠️ 監控、結果、故障排除、進階配置待補充
```

**目標受眾**: 系統管理員、安全測試人員（技術向）

---

#### 📄 SCAN_USAGE_GUIDE.md
- **位置**: `docs/01_user_documentation/user-guides/`
- **大小**: 537 行 / 11.29 KB
- **最後更新**: 2025-12-30
- **定位**: **掃描功能使用指南**

**內容特點**:
```
✅ 掃描功能簡介（286個能力）
✅ 快速開始（3種方式）
✅ CLI 命令使用（最簡單）
✅ 掃描方式詳解
✅ 掃描結果查看
✅ 進階配置選項
✅ 故障排除

✅ 內容完整
```

**目標受眾**: 使用掃描功能的所有使用者（操作向）

---

#### 🔍 比對結論

| 面向 | SCAN_MODULE_GUIDE | SCAN_USAGE_GUIDE |
|------|-------------------|------------------|
| **定位** | 技術手冊（架構） | 使用指南（操作） |
| **完整度** | ⚠️ 不完整（5-8章缺失） | ✅ 完整 |
| **內容重點** | 架構、組件、數據流 | 使用方式、結果、故障 |
| **技術深度** | 深（適配器模式） | 淺（操作為主） |
| **目標** | 理解模組內部 | 會使用掃描功能 |

**判斷**: ❌ **非重複** - 技術手冊 vs 使用指南

**建議**: ✅ **保留兩者**，但需要：
1. 補完 `SCAN_MODULE_GUIDE.md` 第5-8章
2. 或者標註清楚其不完整狀態
3. 兩者互相引用建立連結

---

### 組別 #3: CLI_GUIDE vs AIVA_CLI_UNIFIED_GUIDE ⚠️

#### 📄 CLI_GUIDE.md
- **位置**: `docs/01_user_documentation/user-guides/`
- **大小**: 434 行 / 13.34 KB
- **最後更新**: 2026-01-10
- **版本**: 無版本號
- **定位**: **舊版 CLI 使用指南**

**內容特點**:
```
⚠️ README.md 標註：「已標註，保留參考」
⚠️ 內部自己標註：「以下內容基於舊版本，實際功能可能已變更」
⚠️ 統計數字標註：「⚠️ [數量待確認]」

內容：
- 主選單功能（舊版）
- 常用操作
- Rich CLI 整合
- 命令行參數
- 故障排除
```

**目標受眾**: （已過時）

---

#### 📄 AIVA_CLI_UNIFIED_GUIDE.md
- **位置**: `docs/01_user_documentation/user-guides/`
- **大小**: 630 行 / 17.84 KB
- **最後更新**: 2026-01-10
- **版本**: v1.0
- **狀態**: ✅ 已驗證可用
- **定位**: **官方統一 CLI 指南**

**內容特點**:
```
✅ v1.0 正式版本
✅ 已驗證可用（2026-01-10）
✅ 詳細的系統概述
✅ 兩套CLI系統完整說明
✅ 系統一：AI 能力查詢系統
✅ 系統二：Flow 執行系統
✅ 常見問題 FAQ
✅ 故障排除
```

**目標受眾**: 所有AIVA使用者（官方標準）

---

#### 🔍 比對結論

| 面向 | CLI_GUIDE | AIVA_CLI_UNIFIED_GUIDE |
|------|-----------|------------------------|
| **狀態** | ⚠️ 已過時 | ✅ 當前版本 |
| **版本** | 無 | v1.0 |
| **驗證** | 未驗證 | ✅ 2026-01-10 |
| **內容** | 舊版選單 | 兩套CLI完整說明 |
| **標註** | 自己標過時 | 官方統一指南 |
| **README 標註** | 「保留參考」 | 正常 |

**判斷**: ✅ **重複（舊版被新版取代）**

**建議**: 🗑️ **歸檔 CLI_GUIDE.md**
```
操作：
_archive/06_documentation_archive/2026-01/
└── CLI_GUIDE_archived_20260128.md

原因：
1. 自己內部標註過時
2. README.md 標註「保留參考」（可歸檔）
3. 已有官方統一指南取代
4. 避免用戶混淆（兩個CLI指南存在）
```

---

### 組別 #4: QUICK_REFERENCE vs QUICK_REFERENCE_GUIDE

#### 📄 QUICK_REFERENCE.md (guides/general/)
- **位置**: `guides/general/`
- **大小**: ~60 行 / 1.88 KB
- **定位**: **Python Engine 快速參考卡**

**內容特點**:
```
✅ 專注 Python Engine
✅ 3步驟安裝
✅ 1命令測試
✅ 故障排查（BeautifulSoup、Playwright）
✅ 驗證結果（2025-11-19）
✅ 文檔導航連結
```

**目標受眾**: Python Engine 使用者（極簡參考）

---

#### 📄 QUICK_REFERENCE_GUIDE.md (guides/general/)
- **位置**: `guides/general/`
- **大小**: 193 行 / 5.35 KB
- **版本**: v2.2.0
- **日期**: 2026-01-14
- **定位**: **AIVA 外部模塊快速參考**

**內容特點**:
```
✅ 外部模塊系統（8個模塊）
✅ 快速啟動命令
✅ XSS 測試完整參數
✅ 系統狀態一覽（210 flows）
✅ 可用模塊清單（生產就緒 vs Worker需求）
✅ 測試結果摘要
✅ 故障排除
✅ 完整文檔連結
```

**目標受眾**: 外部模塊使用者（全面參考）

---

#### 🔍 比對結論

| 面向 | QUICK_REFERENCE | QUICK_REFERENCE_GUIDE |
|------|-----------------|----------------------|
| **範圍** | Python Engine | 外部模塊系統 |
| **內容** | 安裝+測試+故障 | 完整模塊使用 |
| **長度** | 極簡（60行） | 完整（193行） |
| **目的** | 快速參考卡 | 完整參考指南 |

**判斷**: ❌ **完全不同主題**

**建議**: ✅ **保留兩者**，但需改名：
```
建議改名：
guides/general/QUICK_REFERENCE.md
→ guides/general/PYTHON_ENGINE_QUICK_REFERENCE.md

guides/general/QUICK_REFERENCE_GUIDE.md
→ guides/general/EXTERNAL_MODULES_QUICK_REFERENCE.md

原因：
1. 兩者主題完全不同（Python Engine vs 外部模塊）
2. 避免名稱混淆
3. 明確指示內容範圍
```

---

### 組別 #5: guides/README vs AIVA_TECHNICAL_GUIDE_INDEX

#### 📄 guides/README.md
- **位置**: `guides/`
- **大小**: 391 行 / 26.97 KB
- **定位**: **AIVA 指南中心（全面索引）**

**內容特點**:
```
✅ 完整指南架構總覽
✅ v2.1.2 系統架構說明
✅ 六大核心服務詳解
✅ 雙閉環系統說明
✅ 指南分類目錄（8大類）
✅ 使用建議與學習路徑（6條路徑）
✅ 文檔維護原則
✅ 相關報告連結
```

**組織方式**: 按文檔類型分類（架構、開發、部署、模組...）

**目標受眾**: 所有角色（入口索引）

---

#### 📄 AIVA_TECHNICAL_GUIDE_INDEX.md
- **位置**: `guides/`
- **大小**: 247 行 / 9.25 KB
- **版本**: v1.0
- **日期**: 2026-01-12
- **定位**: **按 AI 運作流程的技術指南索引**

**內容特點**:
```
✅ 按 AI 8 個運作階段組織
✅ 每階段的核心模組
✅ 每階段的職責說明
✅ 已完成指南 vs 待建指南
✅ 已知問題標記
✅ 適合追蹤技術問題
```

**組織方式**: 按 AI 執行流程階段（Stage 1-8）

**目標受眾**: 技術開發者（理解 AI 內部流程）

---

#### 🔍 比對結論

| 面向 | guides/README | AIVA_TECHNICAL_GUIDE_INDEX |
|------|---------------|---------------------------|
| **組織方式** | 按文檔類型 | 按 AI 執行階段 |
| **視角** | 全面（架構+開發+運維） | 技術（AI 內部流程） |
| **內容** | 指南分類+學習路徑 | 階段式技術指南 |
| **詳細度** | 完整（391行） | 專注技術（247行） |
| **目標** | 統一入口 | 技術追蹤 |

**判斷**: ❌ **非重複** - 不同組織視角

**建議**: ✅ **保留兩者**
```
理由：
1. README.md - 全面入口索引（適合所有角色）
2. AIVA_TECHNICAL_GUIDE_INDEX.md - 技術流程索引（適合深入開發）
3. 兩者互補，服務不同需求
4. 可在 README.md 中引用 TECHNICAL_GUIDE_INDEX
```

---

## 📋 最終建議彙總

### ✅ 保留所有文檔（非重複）

| 文檔 | 原因 |
|------|------|
| `GETTING_STARTED.md` | 新手完整入門（詳細） |
| `QUICK_START_GUIDE.md` | 快速部署（精簡） |
| `SCAN_MODULE_GUIDE.md` | Scan 模組技術手冊 |
| `SCAN_USAGE_GUIDE.md` | Scan 使用指南 |
| `guides/README.md` | 全面指南索引 |
| `AIVA_TECHNICAL_GUIDE_INDEX.md` | 技術流程索引 |

### 🗑️ 歸檔過時文檔（確認重複）

| 文檔 | 原因 | 目標位置 |
|------|------|---------|
| `CLI_GUIDE.md` | 被 AIVA_CLI_UNIFIED_GUIDE 取代 | `_archive/06_documentation_archive/2026-01/` |
| `CAPABILITY_EXECUTION_QUICK_GUIDE.md` | README 標註已過時 | `_archive/06_documentation_archive/2026-01/` |

### 🔄 需要改名（避免混淆）

| 原檔名 | 新檔名 | 原因 |
|--------|--------|------|
| `guides/general/QUICK_REFERENCE.md` | `PYTHON_ENGINE_QUICK_REFERENCE.md` | 明確範圍 |
| `guides/general/QUICK_REFERENCE_GUIDE.md` | `EXTERNAL_MODULES_QUICK_REFERENCE.md` | 明確範圍 |

### ⚠️ 需要補完或標註

| 文檔 | 問題 | 建議 |
|------|------|------|
| `SCAN_MODULE_GUIDE.md` | 第5-8章缺失 | 補完或明確標註不完整 |

---

## 🎯 執行優先順序

### Priority 1: 立即執行（避免混淆）
```bash
# 1. 歸檔 CLI_GUIDE.md
Move-Item "docs/01_user_documentation/user-guides/CLI_GUIDE.md" `
          "_archive/06_documentation_archive/2026-01/CLI_GUIDE_archived_20260128.md"

# 2. 歸檔 CAPABILITY_EXECUTION_QUICK_GUIDE.md
Move-Item "docs/01_user_documentation/user-guides/CAPABILITY_EXECUTION_QUICK_GUIDE.md" `
          "_archive/06_documentation_archive/2026-01/CAPABILITY_EXECUTION_QUICK_GUIDE_archived_20260128.md"
```

### Priority 2: 建議改名（提升清晰度）
```bash
# 改名 QUICK_REFERENCE 相關文檔
Rename-Item "guides/general/QUICK_REFERENCE.md" `
            "PYTHON_ENGINE_QUICK_REFERENCE.md"

Rename-Item "guides/general/QUICK_REFERENCE_GUIDE.md" `
            "EXTERNAL_MODULES_QUICK_REFERENCE.md"
```

### Priority 3: 補充說明（長期維護）
```markdown
# 在 SCAN_MODULE_GUIDE.md 開頭添加醒目標註
> ⚠️ **文檔狀態**: 部分完整（第1-4章已完成，第5-8章待補充）
> 📖 **使用建議**: 
> - 了解架構和基本使用：閱讀本文檔（第1-4章）
> - 完整使用指南：參考 [SCAN_USAGE_GUIDE.md](./SCAN_USAGE_GUIDE.md)
```

---

## 📊 統計摘要

| 項目 | 數量 | 百分比 |
|------|------|--------|
| **原識別為重複** | 5 組 (10 個文檔) | 100% |
| **實際重複（需歸檔）** | 1 組 (2 個文檔) | 20% |
| **非重複（保留）** | 4 組 (8 個文檔) | 80% |
| **需改名** | 2 個文檔 | - |
| **需補完** | 1 個文檔 | - |

---

## 結論

經過詳細比對，**原本識別的 5 組"重複"文檔中，只有 1 組是真正的重複**：

✅ **真重複**: `CLI_GUIDE.md` vs `AIVA_CLI_UNIFIED_GUIDE.md`（舊版 vs 新版）

❌ **非重複**: 其他 4 組都是服務不同目的或不同範圍的獨立文檔

**建議立即執行**: 歸檔 2 個過時文檔，改名 2 個混淆文檔

