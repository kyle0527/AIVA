# 📖 AIVA Features 功能模組文檔中心

**導航**: [← 返回主模組](../README.md)

---

## 📑 目錄

- [文檔架構說明](#文檔架構說明)
- [核心功能模組](#核心功能模組)
- [按功能分類瀏覽](#按功能分類瀏覽)
- [按語言分類瀏覽](#按語言分類瀏覽)
- [快速導航](#快速導航)

---

## 🏗️ 文檔架構說明

AIVA Features 模組採用分散式文檔架構，每個子模組在其自己的目錄中維護README：

```
services/features/
├── README.md                    # 主導航文檔
├── docs/                        # 分類文檔中心
│   ├── README.md               # 本文檔 - 文檔中心
│   ├── security/README.md      # 安全功能分類
│   ├── development/README.md   # 開發中功能
│   ├── python/README.md        # Python模組集合
│   └── go/README.md            # Go模組集合
├── function_sqli/README.md     # SQL注入檢測模組
├── function_xss/README.md      # XSS檢測模組
├── function_ssrf/README.md     # SSRF檢測模組
├── function_idor/README.md     # IDOR檢測模組
├── function_authn_go/README.md # 認證檢測模組
├── function_crypto/README.md   # 密碼學檢測模組
└── function_postex/README.md   # 後滲透模組
```

---

## 🛡️ 核心功能模組

### ✅ **完整實現模組**

#### [🗃️ SQL注入檢測](../function_sqli/README.md)
- **文件數**: 20個Python文件
- **實現程度**: 95%
- **主要功能**: 布爾盲注、時間盲注、聯合查詢注入、錯誤注入檢測
- **引擎**: 4個專業檢測引擎

#### [🔍 XSS檢測](../function_xss/README.md)
- **文件數**: 10個Python文件  
- **實現程度**: 90%
- **主要功能**: Reflected XSS、Stored XSS、DOM XSS檢測
- **特色**: 支援盲注XSS檢測

#### [🌐 SSRF檢測](../function_ssrf/README.md)
- **文件數**: 12個Python文件
- **實現程度**: 90%
- **主要功能**: 內網地址檢測、OAST調度、參數語義分析
- **特色**: 智能SSRF檢測器

#### [🔐 IDOR檢測](../function_idor/README.md)
- **文件數**: 12個Python文件
- **實現程度**: 85%
- **主要功能**: 垂直權限提升、水平權限繞過檢測
- **特色**: 跨用戶測試、資源ID提取

### 🔹 **部分實現模組**

#### [🔑 認證檢測](../function_authn_go/README.md)
- **文件數**: 4個Go文件
- **實現程度**: 60%
- **主要功能**: 認證繞過、令牌測試、弱配置檢測
- **特色**: Go高併發實現

#### [🔒 密碼學檢測](../function_crypto/README.md)
- **文件數**: 8個Python文件
- **實現程度**: 40%
- **主要功能**: 密碼學漏洞檢測 (開發中)

#### [⚡ 後滲透](../function_postex/README.md)
- **文件數**: 9個Python文件
- **實現程度**: 30%
- **主要功能**: 滲透後利用模組 (開發中)

---

## 📂 按功能分類瀏覽

### [🛡️ 安全功能](security/README.md)
完整實現的安全檢測功能，包括四大核心檢測引擎：
- SQL注入檢測 (function_sqli)
- XSS檢測 (function_xss)  
- SSRF檢測 (function_ssrf)
- IDOR檢測 (function_idor)

### [🔧 開發中功能](development/README.md)
正在開發或部分實現的功能模組：
- 認證檢測 (function_authn_go)
- 密碼學檢測 (function_crypto)
- 後滲透 (function_postex)

---

## 💻 按語言分類瀏覽

### [🐍 Python 模組](python/README.md)
**主力實現語言** - 75個文件，12,002行代碼
- 所有安全檢測引擎的主要實現
- 系統管理和協調功能
- 智能檢測策略

### [🐹 Go 模組](go/README.md)  
**高效能實現** - 11個文件，1,796行代碼
- 認證檢測的高併發實現
- 共享通用組件
- 跨語言橋接

---

## 🚀 快速導航

### **新用戶推薦路徑**
1. 📖 閱讀 [主模組README](../README.md) 了解整體架構
2. 🛡️ 查看 [安全功能](security/README.md) 了解核心檢測能力
3. 🗃️ 深入 [SQL注入檢測](../function_sqli/README.md) 了解實現細節

### **開發者推薦路徑**
1. 🐍 查看 [Python模組](python/README.md) 了解主要實現
2. 🔧 參考 [開發中功能](development/README.md) 了解擴展方向
3. 📝 選擇具體模組深入開發

### **架構師推薦路徑**
1. 📊 研讀主README中的架構設計
2. 🏗️ 查看各語言模組的協作方式
3. 🎯 規劃功能擴展和優化方向

---

## 📞 支援與聯繫

- 🐛 **問題回報**: 在對應功能模組目錄下提交Issue
- 📖 **文檔貢獻**: 直接編輯對應模組的README
- 💬 **技術討論**: 參考各模組的開發指南

---

*最後更新: 2025年11月7日*  
*維護團隊: AIVA Features Development Team*