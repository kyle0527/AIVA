# 🚀 AIVA 外部模塊快速參考

**版本**: v2.2.0 | **日期**: 2026-01-14 | **狀態**: 生產就緒（部分）

---


## 📑 目錄

- [⚡ 快速啟動](#-快速啟動)
  - [啟動交互式選單](#啟動交互式選單)
  - [直接執行 XSS 測試](#直接執行-xss-測試)
  - [運行完整測試](#運行完整測試)
- [📊 系統狀態一覽](#-系統狀態一覽)
- [🎯 可用模塊](#-可用模塊)
  - [✅ 生產就緒](#-生產就緒)
  - [⏳ 需要 Worker](#-需要-worker)
- [🛠️ XSS 模塊參數](#-xss-模塊參數)
- [📝 常用命令](#-常用命令)
  - [XSS 測試](#xss-測試)
  - [執行器命令](#執行器命令)
- [🎯 測試靶場](#-測試靶場)
- [📂 關鍵檔案](#-關鍵檔案)
- [🔧 故障排除](#-故障排除)
- [📊 測試結果摘要](#-測試結果摘要)
  - [XSS 模塊（最新測試）](#xss-模塊最新測試)
  - [HTTP 狀態碼](#http-狀態碼)
- [🚀 下一步](#-下一步)
- [📚 完整文檔](#-完整文檔)

---
## ⚡ 快速啟動

### 啟動交互式選單
```bash
.\啟動外部能力選單.bat
```

### 直接執行 XSS 測試
```bash
python -m services.features.function_xss --url "http://localhost:3000" --param q --type reflected
```

### 運行完整測試
```bash
.\test_multi_capabilities.ps1
```

---

## 📊 系統狀態一覽

| 指標 | 數值 | 狀態 |
|------|------|------|
| 總流程數 | 210 | ✅ |
| 總模塊數 | 8 | ✅ |
| CLI 就緒 | 1 (XSS) | ⚠️ |
| Worker 需求 | 7 | ⏳ |
| 測試通過 | XSS 100% | ✅ |

---

## 🎯 可用模塊

### ✅ 生產就緒
| 模塊 | 語言 | 流程數 | CLI | 測試 |
|------|------|--------|-----|------|
| **function_xss** | Python | 195 | ✅ | ✅ |

### ⏳ 需要 Worker
| 模塊 | 語言 | 流程數 | CLI | Worker |
|------|------|--------|-----|--------|
| function_ssrf | Python | 2 | ❌ | ✅ |
| function_sqli | Python | 2 | ❌ | ✅ |
| function_idor | Python | 2 | ❌ | ✅ |
| function_bizlogic | Python | 2 | ❌ | ✅ |
| function_authn_go | Go | 4 | ❌ | ✅ |
| typescript_engine | TypeScript | 3 | ❌ | ✅ |

---

## 🛠️ XSS 模塊參數

| 參數 | 必填 | 值 | 說明 |
|------|------|-----|------|
| `--url` | ✅ | URL | 目標網址 |
| `--type` | ❌ | reflected/dom/stored | XSS 類型 |
| `--param` | ❌ | string | 參數名稱 |
| `--method` | ❌ | GET/POST | HTTP 方法 |
| `--location` | ❌ | query/body/header | 參數位置 |
| `--timeout` | ❌ | seconds | 超時時間 |
| `--view-url` | ❌ | URL | Stored XSS 查看頁 |

---

## 📝 常用命令

### XSS 測試
```bash
# Reflected XSS (GET)
python -m services.features.function_xss --url "http://target.com" --param q --type reflected

# DOM XSS
python -m services.features.function_xss --url "http://target.com/#/page" --type dom

# Stored XSS (POST)
python -m services.features.function_xss --url "http://target.com/api/comment" --param text --type stored --method POST --location body --view-url "http://target.com/view"
```

### 執行器命令
```bash
# 列出所有能力
python aiva_external_executor.py --list

# 列出特定語言
python aiva_external_executor.py --list --lang python

# 啟動選單
python aiva_external_executor.py --menu

# 生成文檔
python aiva_external_executor.py --generate-doc md
python aiva_external_executor.py --generate-doc json
```

---

## 🎯 測試靶場

| 靶場 | URL | 用途 |
|------|-----|------|
| Juice Shop | http://localhost:3000 | Web 漏洞 |
| WebGoat | http://localhost:8080 | 教學型 |

---

## 📂 關鍵檔案

| 檔案 | 位置 | 用途 |
|------|------|------|
| 統一執行器 | `services/core/aiva_core/internal_exploration/aiva_external_executor.py` | 主執行器 |
| 分類數據 | `services/core/aiva_core/internal_exploration/classification_data.json` | 210 flows |
| 測試腳本 | `test_multi_capabilities.ps1` | 自動化測試 |
| CLI 參考 | `features_classification/EXTERNAL_CLI_COMMANDS_REFERENCE.md` | 命令手冊 |
| 測試報告 | `docs/MULTI_LANGUAGE_CAPABILITY_TESTING_REPORT.md` | 詳細報告 |

---

## 🔧 故障排除

| 問題 | 解決方案 |
|------|----------|
| `No module named __main__` | 該模塊需要 Worker，無法直接 CLI 執行 |
| `Connection refused` | 確認靶場服務已啟動 (`docker ps`) |
| `Worker 模塊無法執行` | 需部署 RabbitMQ 和 Worker 系統 |

---

## 📊 測試結果摘要

### XSS 模塊（最新測試）
- ✅ **Reflected XSS**: 2 個端點，9 個 HTTP 請求
- ✅ **DOM XSS**: 1 個端點，1 個 HTTP 請求
- ✅ **Stored XSS**: 1 個端點，2 個 HTTP 請求
- 📊 **總計**: 15+ HTTP 請求，100% 成功率

### HTTP 狀態碼
- 200 OK: 9 次 ✅
- 302 Found: 3 次 ✅
- 500 Error: 3 次 ⚠️

---

## 🚀 下一步

1. ⏳ **部署 Worker 系統** - 解鎖 7 個模塊
2. ⏳ **完善執行器** - 統一所有模塊接口
3. ⏳ **自動化報告** - HTML/PDF 報告生成
4. ⏳ **性能優化** - 提升掃描速度

---

## 📚 完整文檔

- [多語言能力測試報告](docs/MULTI_LANGUAGE_CAPABILITY_TESTING_REPORT.md)
- [外部模塊系統狀態](logs/status_reports/status_20260114_external_modules.md)
- [更新日誌](CHANGELOG.md)
- [主 README](README.md)

---

**快速參考卡片** | 打印此頁面作為桌面參考 | 最後更新: 2026-01-14
