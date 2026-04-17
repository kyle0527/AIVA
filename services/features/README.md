# AIVA Features 模組 (v3.0.0)

> **版本**: v3.0.0 | **狀態**: ✅ 架構重構與標準化完成 | **更新**: 2026-03-27

## 🚀 模組概述

`services/features/` 目錄包含了 AIVA (Autonomous Intelligence Virtual Assistant) 所有的核心資安檢測、漏洞驗證與支援工具。這裡的每一個 `function_*` 資料夾都代表一種特定的攻防能力或分析技術。

AIVA Features 模組經過 v3.0.0 的架構標準化，**全面廢棄了舊版的 `BaseCapability`、`FeatureRegistry` 與過渡期的 `CommandHandler` 架構**。目前所有模組均採取**「CLI 驅動為主，Python API 匯入為輔」**的輕量化設計，由上層的 AI Commander (`AttackCoordinator`) 或外部執行器 (`aiva_external_executor.py`) 根據任務動態調度。

## 📐 設計原則 (v3.0.0)

1. **模組自治 (Self-Contained)**:
   每個功能模組維護自己的目錄結構與依賴，使用最適合該技術領域的實作方式 (例如：純 Python、Go 引擎、Rust CLI binary)。不再強制繼承龐大且不實用的基底類別。
2. **統一介面 (Uniform Interface)**:
   - **獨立 CLI**: 多數成熟模組提供 `__main__.py` 或獨立執行檔，支援直接由 Terminal 傳入參數並回傳 JSON 結構。
   - **Python API**: 每個模組的 `__init__.py` 匯出一個管理類別 (如 `Manager` 或 `Detector`)，提供同步/非同步的 `.scan()` 或 `.analyze()` 方法供 AIVA Core 匯入。
3. **無相對導入黑魔法**:
   所有模組內的檔案引用均改用乾淨的 Relative Imports (`from . import ...` 或 `from .. import ...`)，確保無論是 CLI 執行或是被當作套件匯入都不會出現 `ImportError`。

---

## 📚 模組目錄 (功能分類)

所有功能模組已依照**完成度**與**自動化可行性**進行分類。請點擊各模組的 `README.md` 查看詳細的技術架構與操作說明。

### 🛡️ 高完成度 - 實戰自動化檢測 (主力引擎)

這些模組具備高度自動化能力，可無縫整合進 AIVA 的全自動掃描流程中：

- ⚡️ **[SQL 注入檢測 (function_sqli)](function_sqli/README.md)**: 6 種引擎 (Error, Boolean, Time, Union, OOB, HackingTool)，涵蓋 WAF 繞過與資料庫指紋識別。
- 🎭 **[XSS 檢測 (function_xss)](function_xss/README.md)**: 原生靜態與動態分析，支援 Reflected, Stored, DOM-based, Blind XSS。
- 🌐 **[SSRF 檢測 (function_ssrf)](function_ssrf/README.md)**: 內網探測、雲端 Metadata、DNS Rebinding 與 OAST 驗證。
- 🔐 **[IDOR 檢測 (function_idor)](function_idor/README.md)**: 水平與垂直越權檢測，具備 URL ID 自動萃取。
- 💼 **[業務邏輯檢測 (function_bizlogic)](function_bizlogic/README.md)**: 價格操控、競態條件、流程與支付繞過測試。
- 📄 **[敏感資訊檢測 (function_info_leak)](function_info_leak/README.md)**: 50+ 種 API 密鑰、JWT、憑證檢測與香農熵分析。
- 🔍 **[Web 應用掃描器 (function_web_scanner)](function_web_scanner/README.md)**: 子域名枚舉、目錄爆破、端口掃描與漏洞基礎探測。

### ⚡ 跨語言高效能引擎 (需編譯)

為了突破 Python 效能瓶頸，這些模組採用 Go 或 Rust 撰寫，需先行編譯二進位檔：

- 🔑 **[認證繞過檢測 (function_authn_go)](function_authn_go/README.md)**: 基於 Go 的高效能認證測試，支援弱密碼、Session 安全與 2FA 繞過。
- 🔒 **[密碼學配置分析 (function_crypto)](function_crypto/README.md)**: 純 Rust 實作的 CLI 工具，專注於網路層 TLS、Cookie 與安全標頭配置分析。

### ⚠️ 高風險與進階利用 (需授權)

這些模組具備破壞性或觸及後滲透階段，強烈要求必須在**具備書面授權**的紅隊演練或受控環境中使用：

- 💥 **[攻擊執行與利用 (function_exploit)](function_exploit/README.md)**: 漏洞利用、動態 Payload 生成與攻擊鏈執行引擎。
- 🕵️ **[後滲透測試 (function_postex)](function_postex/README.md)**: 橫向移動、權限提升與持久化路徑偵測。

### 🛠️ 輔助與人工操作框架

這些模組牽涉大量的外部工具依賴或複雜的人工互動，自動化程度較低，主要作為框架或輔助腳本：

- 🎣 **[社交工程 (function_social_engineering)](function_social_engineering/README.md)**: 釣魚活動與憑證收集 (框架)。
- 🧩 **[逆向工程 (function_reverse_engineering)](function_reverse_engineering/README.md)**: 封裝 radare2/jadx/apktool 等外部二進位分析工具。
- 🖼️ **[隱寫術分析 (function_steganography)](function_steganography/README.md)**: StegX 嵌入提取與基礎 CNN 圖片分析。
- 📝 **[字典生成器 (function_wordlist_generator)](function_wordlist_generator/README.md)**: CUPP 個人化密碼本、字元組合生成。
- 🔬 **[數位鑑識 (function_forensic)](function_forensic/README.md)**: 鑑識案件管理、證據 Hash 登錄與時間線產生。

---

## 🛠️ 開發與維護指南

### 1. 新增功能模組
- **獨立封裝**: 建立新的 `function_<name>` 資料夾，並撰寫專屬的 `README.md`。
- **對外介面**: 在 `__init__.py` 匯出主要類別 (如 `MyScanner`) 或提供 `__main__.py` 作為 CLI 入口。
- **避免依賴 BaseCapability**: 不再使用已被棄用的 `FeatureRegistry`。
- **使用 AIVA Common**: 結果結構應回傳 `aiva_common` 內定義之標準 schemas 或 dict，切勿自行發明新的 `FeatureResult` 格式。

### 2. 執行與測試
在修改任何模組後，請確保該模組能正確被匯入，且沒有遺留導致崩潰的絕對路徑：

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/services
python tests/test_direct_import.py
```
