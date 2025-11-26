# 📚 AIVA 使用者手冊中心

## 📑 目錄

- [📂 目錄結構](#目錄結構)
- [📖 手冊分類](#手冊分類)
  - [🌐 00. 通用指南](#00-通用指南)
  - [🤖 01. Core 模組手冊](#01-core-模組手冊)
  - [🔗 02. Common 模組手冊](#02-common-模組手冊)
  - [🎯 03. Features 模組手冊](#03-features-模組手冊)
  - [🔄 04. Integration 模組手冊](#04-integration-模組手冊)
  - [🔍 05. Scan 模組手冊](#05-scan-模組手冊)
- [🎯 快速導航](#快速導航)
  - [按角色查找](#按角色查找)
- [📝 文檔維護原則](#文檔維護原則)
  - [集中管理的手冊](#集中管理的手冊)
  - [分散在模組中的文檔](#分散在模組中的文檔)
  - [文檔更新規範](#文檔更新規範)
- [🔗 相關資源](#相關資源)
  - [📚 其他指南目錄](#其他指南目錄)
  - [🛠️ 技術參考](#技術參考)

---

## 📂 目錄結構

```
docs/user_guides/
├── README.md                           # 本索引文件
├── 00_general/                         # 通用指南
├── 01_core/                            # Core 模組手冊
├── 02_common/                          # Common 模組手冊
├── 03_features/                        # Features 模組手冊
├── 04_integration/                     # Integration 模組手冊
└── 05_scan/                            # Scan 模組手冊
```

---

## 📖 手冊分類

### 🌐 00. 通用指南

| 文檔名稱 | 路徑 | 適用對象 | 說明 |
|---------|------|----------|------|
| **AIVA 使用者手冊** | [`00_general/AIVA_USER_MANUAL.md`](00_general/AIVA_USER_MANUAL.md) | 所有使用者 | AIVA 系統整體使用指南 |
| **AIVA 模型指南** | [`00_general/AIVA_MODEL_GUIDE.md`](00_general/AIVA_MODEL_GUIDE.md) | AI 工程師 | AI 模型使用和配置 |

### 🤖 01. Core 模組手冊

| 文檔名稱 | 路徑 | 適用對象 | 說明 |
|---------|------|----------|------|
| **AIVA Core 使用者手冊** | [`01_core/AIVA_CORE_使用者手冊.md`](01_core/AIVA_CORE_使用者手冊.md) | Core 開發者 | Core 模組完整使用指南 |
| **真實 AI 核心操作手冊** | [`01_core/REAL_AI_CORE_OPERATIONS_MANUAL.md`](01_core/REAL_AI_CORE_OPERATIONS_MANUAL.md) | AI 架構師 | 建立真實神經網路核心 |
| **AIVA AI 使用者手冊** | [`01_core/AIVA_AI_USER_MANUAL.md`](01_core/AIVA_AI_USER_MANUAL.md) | AI 使用者 | AI 功能使用指南 |
| **AI 服務使用指南** | [`01_core/AI_SERVICES_USER_GUIDE.md`](01_core/AI_SERVICES_USER_GUIDE.md) | 開發者 | AI 系統實際使用指南 |

### 🔗 02. Common 模組手冊

| 文檔位置 | 說明 |
|---------|------|
| **Common 模組文檔** | 請參考 [`services/aiva_common/README.md`](../../services/aiva_common/README.md) |

> **說明**: Common 模組的主要文檔保持在 `services/aiva_common/README.md`，作為共享庫的核心參考文檔。

### 🎯 03. Features 模組手冊

| 文檔位置 | 說明 |
|---------|------|
| **Features 模組文檔** | 請參考 [`services/features/README.md`](../../services/features/README.md) |
| **功能分類文檔** | 請參考 `services/features/docs/` 目錄 |

> **說明**: Features 模組的文檔保持在原位置，包含多語言功能架構和各功能模組的詳細文檔。

### 🔄 04. Integration 模組手冊

| 文檔位置 | 說明 |
|---------|------|
| **Integration 模組文檔** | 請參考 [`services/integration/README.md`](../../services/integration/README.md) |
| **核心實現文檔** | 請參考 [`services/integration/aiva_integration/README.md`](../../services/integration/aiva_integration/README.md) |

> **說明**: Integration 模組的文檔保持在原位置，涵蓋企業級整合架構和詳細的 API 參考。

### 🔍 05. Scan 模組手冊

| 文檔位置 | 說明 |
|---------|------|
| **Scan 模組文檔** | 請參考 [`services/scan/README.md`](../../services/scan/README.md) |
| **協調器文檔** | 請參考 [`services/scan/coordinators/README.md`](../../services/scan/coordinators/README.md) |
| **引擎文檔** | 請參考 `services/scan/engines/` 目錄 |

> **說明**: Scan 模組的文檔保持在原位置，包含多語言掃描引擎和協調器的完整文檔。

---

## 🎯 快速導航

### 按角色查找

**🆕 新手入門**:
1. 閱讀 [`00_general/AIVA_USER_MANUAL.md`](00_general/AIVA_USER_MANUAL.md)
2. 瀏覽各服務模組的 README.md

**👨‍💻 開發者**:
1. [`01_core/AI_SERVICES_USER_GUIDE.md`](01_core/AI_SERVICES_USER_GUIDE.md) - AI 系統使用
2. [`services/aiva_common/README.md`](../../services/aiva_common/README.md) - 共享庫參考
3. 各模組的 README.md - 具體開發指南

**🏗️ 架構師**:
1. [`01_core/REAL_AI_CORE_OPERATIONS_MANUAL.md`](01_core/REAL_AI_CORE_OPERATIONS_MANUAL.md) - AI 核心架構
2. 各服務模組的架構文檔
3. [`guides/architecture/`](../../guides/architecture/) - 架構指南

**🤖 AI 工程師**:
1. [`00_general/AIVA_MODEL_GUIDE.md`](00_general/AIVA_MODEL_GUIDE.md) - AI 模型指南
2. [`01_core/REAL_AI_CORE_OPERATIONS_MANUAL.md`](01_core/REAL_AI_CORE_OPERATIONS_MANUAL.md) - 神經網路核心
3. [`01_core/AIVA_AI_USER_MANUAL.md`](01_core/AIVA_AI_USER_MANUAL.md) - AI 功能使用

---

## 📝 文檔維護原則

### 集中管理的手冊
- **User Guides** (`docs/user_guides/`) - 面向使用者的綜合手冊
- **Core 模組手冊** - AI 核心相關的操作和使用指南
- **通用指南** - 跨模組的整體系統指南

### 分散在模組中的文檔
- **Services README** (`services/*/README.md`) - 各服務模組的主要文檔
- **技術文檔** - 保持在對應模組的 `docs/` 目錄
- **API 參考** - 保持在對應模組的源代碼附近

### 文檔更新規範
1. **使用者手冊** - 在 `docs/user_guides/` 中統一更新
2. **模組文檔** - 在對應的 `services/*/` 目錄中更新
3. **架構文檔** - 在 `guides/architecture/` 中更新
4. **開發指南** - 在 `guides/development/` 中更新

---

## 🔗 相關資源

### 📚 其他指南目錄
- **[Guides 總覽](../../guides/README.md)** - 開發、架構、部署指南
- **[Services 總覽](../../services/README.md)** - 六大核心服務架構
- **[開發者指南](../../reports/documentation/DEVELOPER_GUIDE.md)** - 開發環境和流程

### 🛠️ 技術參考
- **[架構指南](../../guides/architecture/)** - 系統架構設計
- **[開發指南](../../guides/development/)** - 開發環境和工具
- **[部署指南](../../guides/deployment/)** - 部署和運維
- **[問題排查](../../guides/troubleshooting/)** - 疑難排解

---

**📝 文檔版本**: v2.0  
**🔄 最後更新**: 2025年11月22日  
**👥 維護團隊**: AIVA Documentation Team
