# Git 推送規範和安全指南

## 📑 目錄

- [概述](#概述)
- [🚫 禁止推送的內容](#禁止推送的內容)
  - [1. 真實的敏感資訊](#1-真實的敏感資訊)
  - [2. 測試用敏感資料（即使是假的）](#2-測試用敏感資料即使是假的)
  - [3. 個人資訊和內部資訊](#3-個人資訊和內部資訊)
  - [4. 大型或臨時檔案](#4-大型或臨時檔案)
- [✅ 推送前檢查清單](#推送前檢查清單)
  - [1. 代碼審查](#1-代碼審查)
  - [2. 檔案審查](#2-檔案審查)
  - [3. 提交資訊](#3-提交資訊)
- [🛡️ 安全最佳實踐](#安全最佳實踐)
  - [1. 環境變數使用](#1-環境變數使用)
  - [2. 配置檔案管理](#2-配置檔案管理)
  - [3. 測試資料規範](#3-測試資料規範)
- [🔍 推送前自動檢查](#推送前自動檢查)
  - [使用 Git Hooks](#使用-git-hooks)
  - [使用命令行檢查](#使用命令行檢查)
- [📋 特定檔案類型規範](#特定檔案類型規範)
  - [1. 文檔檔案](#1-文檔檔案)
  - [2. 測試檔案](#2-測試檔案)
  - [3. 配置檔案](#3-配置檔案)
- [🚨 意外推送的應對措施](#意外推送的應對措施)
  - [1. 立即措施](#1-立即措施)
  - [2. 通知程序](#2-通知程序)
  - [3. 後續行動](#3-後續行動)
- [📚 相關資源](#相關資源)
- [🤝 團隊責任](#團隊責任)
- [🔗 相關資源](#相關資源-1)
  - [開發指南](#開發指南)
  - [架構指南](#架構指南)
  - [故障排除](#故障排除)
  - [使用者手冊](#使用者手冊)

---
## 概述

為了保護項目安全並避免敏感資訊洩露，所有開發者在推送代碼到遠端倉庫前必須遵循以下規範。

## 🚫 禁止推送的內容

### 1. 真實的敏感資訊
- 實際的 API 密鑰、訪問令牌
- 真實的密碼、憑證
- 私鑰文件（RSA、EC、PGP 等）
- 數據庫連接字符串（包含真實密碼）
- 第三方服務的實際憑證

### 2. 測試用敏感資料（即使是假的）
- 包含大量測試密鑰示例的檔案
- 密鑰檢測規則檔案（含有密鑰格式範例）
- 包含假密鑰但可能被誤認為真實的測試程式碼

### 3. 個人資訊和內部資訊
- 個人電子郵件地址、電話號碼
- 內部系統 IP 地址、主機名
- 公司內部文檔、會議記錄
- 客戶資訊、商業機密

### 4. 大型或臨時檔案
- 編譯後的二進制檔案（除非必要）
- 日誌檔案（超過 1MB）
- 臨時檔案、緩存檔案
- IDE 配置檔案（個人設定）

## ✅ 推送前檢查清單

### 1. 代碼審查
- [ ] 檢查所有新增或修改的檔案
- [ ] 確認沒有硬編碼的敏感資訊
- [ ] 驗證測試資料使用明顯的假值
- [ ] 檢查註釋中是否包含敏感資訊

### 2. 檔案審查
- [ ] 確認 `.gitignore` 規則正確
- [ ] 檢查檔案大小，避免推送大檔案
- [ ] 驗證檔案路徑和命名規範
- [ ] 確認不包含個人配置檔案

### 3. 提交資訊
- [ ] 提交訊息清晰描述變更內容
- [ ] 避免在提交訊息中包含敏感資訊
- [ ] 使用適當的提交類型前綴（feat, fix, docs 等）

## 🛡️ 安全最佳實踐

### 1. 環境變數使用
```bash
# ✅ 正確：使用環境變數
API_KEY=${API_KEY:-"your-api-key-here"}

# ❌ 錯誤：硬編碼
API_KEY="sk_live_1234567890abcdef"
```

### 2. 配置檔案管理
```yaml
# ✅ 正確：使用佔位符
database:
  host: ${DB_HOST:-localhost}
  password: ${DB_PASSWORD:-password}

# ❌ 錯誤：真實憑證
database:
  host: prod-db.company.com
  password: RealPassword123!
```

### 3. 測試資料規範
```python
# ✅ 正確：明顯的測試資料
TEST_API_KEY = "test_key_12345"
EXAMPLE_TOKEN = "example_token_abcdef"

# ❌ 錯誤：看似真實的資料
API_KEY = "sk_live_abcd1234567890"
```

## 🔍 推送前自動檢查

### 使用 Git Hooks
建議設置 pre-commit hook 進行自動檢查：

```bash
#!/bin/sh
# .git/hooks/pre-commit

# 檢查敏感資訊
if git diff --cached --name-only | xargs grep -l "AKIA\|sk_live\|-----BEGIN" 2>/dev/null; then
    echo "❌ 發現可能的敏感資訊，請檢查後再提交"
    exit 1
fi

# 檢查大檔案
if git diff --cached --name-only | xargs du -h 2>/dev/null | awk '$1 ~ /[0-9]+M/ {print $2}'; then
    echo "⚠️  發現大檔案，請確認是否需要推送"
fi

echo "✅ 預提交檢查通過"
```

### 使用命令行檢查
推送前手動執行：

```bash
# 檢查暫存的變更中是否包含敏感內容
git diff --cached | grep -i -E "(password|secret|key|token|api_key)" | head -5

# 檢查檔案大小
git diff --cached --name-only | xargs ls -lh | awk '$5 ~ /[0-9]+M/'

# 檢查是否有二進制檔案
git diff --cached --numstat | awk '$1 == "-" && $2 == "-"'
```

## 📋 特定檔案類型規範

### 1. 文檔檔案
- **可以推送**：README.md、API 文檔、使用指南
- **需要審查**：包含示例 API 呼叫的文檔
- **禁止推送**：包含真實憑證的配置示例

### 2. 測試檔案
- **可以推送**：單元測試、集成測試
- **需要審查**：包含外部服務調用的測試
- **禁止推送**：包含真實測試帳號的測試

### 3. 配置檔案
- **可以推送**：範本配置檔案、開發環境配置
- **需要審查**：包含預設密碼的配置
- **禁止推送**：生產環境配置檔案

## 🚨 意外推送的應對措施

### 1. 立即措施
```bash
# 如果剛推送且沒有其他人拉取
git reset --hard HEAD~1
git push --force-with-lease

# 如果已經有其他人拉取
git revert <commit-hash>
git push
```

### 2. 通知程序
1. 立即通知團隊負責人
2. 評估洩露範圍和影響
3. 如有必要，輪換相關憑證
4. 更新安全措施和檢查流程

### 3. 後續行動
- 檢查並修復導致洩露的流程
- 更新 `.gitignore` 和檢查腳本
- 團隊培訓和意識提升

## 📚 相關資源

- [Git 安全最佳實踐](https://git-scm.com/book/en/v2)
- [GitHub 安全指南](https://docs.github.com/en/code-security)
- [敏感資料檢測工具](https://github.com/trufflesecurity/trufflehog)

## 🤝 團隊責任

每個團隊成員都有責任：
- 遵循這些規範
- 在代碼審查中檢查安全問題
- 發現問題時及時提出
- 持續改進安全流程

---

**記住：安全是每個人的責任，預防勝於治療！**

最後更新：2025年10月26日

---

## 🔗 相關資源

### 開發指南
- 📖 [開發快速指南](./DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [開發任務指南](./DEVELOPMENT_TASKS_GUIDE.md)
- 📖 [開發者指南](./DEVELOPER_GUIDE.md)
- 📖 [依賴管理指南](./DEPENDENCY_MANAGEMENT_GUIDE.md)
- 📖 [API 驗證指南](./API_VERIFICATION_GUIDE.md)
- 📖 [Schema 導入指南](./SCHEMA_IMPORT_GUIDE.md)

### 架構指南
- 📖 [Schema 統一指南](../architecture/SCHEMA_GUIDE.md)
- 📖 [架構兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 故障排除
- 📖 [導入問題解決](../troubleshooting/IMPORT_ISSUES_RESOLUTION_GUIDE.md)
- 📖 [性能優化指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)

### 使用者手冊
- 📚 [Core 模組手冊](../../docs/user_guides/01_core/AIVA_CORE_使用者手冊.md)

