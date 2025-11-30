# 🔍 Services 目錄深度分析報告

---
**分析時間**: 2025年11月27日

## 📑 目錄

- [模組分析](#模組分析)
- [空資料夾](#空資料夾)
- [README 問題](#readme-問題)
- [建議行動](#建議行動)

---

## 📦 模組分析

### aiva_common

**路徑**: `services/aiva_common`

**代碼文件**:
- Python: 137 個
- Rust: 1 個
- TypeScript: 2 個

**README 分析**:
- 標題: *AIVA Common - Bug Bounty 專業化共享庫*
- 大小: 58,067 bytes (2507 行)
- 目錄: ✅
- 連結: 63 個

⚠️ **損壞的連結** (16 個):
  - `[📖 文檔中心](../../docs/README.md)`
  - `[Python 工具 (22個)](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-1-python-開發生態-22-個)`
  - `[核心插件速查表](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-核心插件速查表)`
  - `[文檔工具 (8個)](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-8-文檔與標記語言-8-個)`
  - `[品質工具 (5個)](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-7-程式碼品質與-linting-5-個)`
  - ... 還有 11 個

⚠️ **建議拆分**: README 內容過多，建議提取為獨立文件

**子目錄** (17 個):
- `.ruff_cache/`
- `ai/`
- `aiva_common.egg-info/`
- `async_utils/`
- `cli/`
- `config/`
- `cross_language/`
- `enums/`
- `messaging/`
- `observability/`
- ... 還有 7 個

### core

**路徑**: `services/core`

**代碼文件**:
- Python: 136 個

**README 分析**:
- 標題: *AIVA Core 模組 - AI驅動核心引擎架構*
- 大小: 45,911 bytes (1891 行)
- 目錄: ✅
- 連結: 206 個

⚠️ **損壞的連結** (29 個):
  - `[📖 文檔中心](../../docs/README.md)`
  - `[架構修復完成報告](../../ARCHITECTURE_FIXES_COMPLETION_REPORT.md)`
  - `[AI語義能力審計](../../AI_CODE_ANALYSIS_CAPABILITY_AUDIT.md)`
  - `[語義編碼測試腳本](../../test_ai_semantic_encoding.py)`
  - `[三層AI決策架構](aiva_core/README.md#-ai決策系統)`
  - ... 還有 24 個

⚠️ **建議拆分**: README 內容過多，建議提取為獨立文件

**子目錄** (3 個):
- `aiva_core/`
- `tests/`
- `tools/`

### features

**路徑**: `services/features`

**代碼文件**:
- Python: 131 個
- Go: 11 個
- Rust: 1 個

**README 分析**:
- 標題: *AIVA Features 模組 - 多語言安全功能架構*
- 大小: 28,842 bytes (1074 行)
- 目錄: ✅
- 連結: 98 個

⚠️ **損壞的連結** (44 個):
  - `[📖 文檔中心](../../docs/README.md)`
  - `[05_A_Social_Engineering_Technical_Integration.md](../../../Users/User/Downloads/新增資料夾%20(6)`
  - `[05_B_Payload_Generator_Technical_Integration.md](../../../Users/User/Downloads/新增資料夾%20(6)`
  - `[C:\Users\User\Downloads\新增資料夾 (6)\AIVA_Enhancement_Plan\00_INDEX.md](../../../Users/User/Downloads/新增資料夾%20(6)`
  - `[執行摘要與現狀分析](../../../Users/User/Downloads/新增資料夾%20(6)`
  - ... 還有 39 個

⚠️ **建議拆分**: README 內容過多，建議提取為獨立文件

**子目錄** (20 個):
- `base/`
- `common/`
- `docs/`
- `function_authn_go/`
- `function_bizlogic/`
- `function_crypto/`
- `function_ddos/`
- `function_exploit_framework/`
- `function_forensic/`
- `function_idor/`
- ... 還有 10 個

### integration

**路徑**: `services/integration`

**代碼文件**:
- Python: 87 個

**README 分析**:
- 標題: *AIVA 整合模組 - 企業級安全整合中樞*
- 大小: 37,685 bytes (1353 行)
- 目錄: ✅
- 連結: 37 個

⚠️ **損壞的連結** (11 個):
  - `[📖 文檔中心](../../docs/README.md)`
  - `[建立報告](../../reports/INTEGRATION_DATA_STORAGE_SETUP_REPORT.md)`
  - `[Python 工具 (22個)](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-1-python-開發生態-22-個)`
  - `[資料庫工具 (4個)](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-11-資料庫與連線-3-個)`
  - `[開發工具 (7個)](../../_out/VSCODE_EXTENSIONS_INVENTORY.md#-10-開發工具與測試-7-個)`
  - ... 還有 6 個

⚠️ **建議拆分**: README 內容過多，建議提取為獨立文件

**子目錄** (8 個):
- `aiva_integration/`
- `alembic/`
- `api_gateway/`
- `capability/`
- `coordinators/`
- `docs/`
- `scripts/`
- `tools/`

### scan

**路徑**: `services/scan`

**代碼文件**:
- Python: 62 個
- Go: 18 個
- Rust: 19 個
- TypeScript: 16 個

**README 分析**:
- 標題: *🎯 AIVA Scan - 多語言統一掃描引擎*
- 大小: 19,606 bytes (758 行)
- 目錄: ✅
- 連結: 70 個

⚠️ **損壞的連結** (20 個):
  - `[📖 文檔中心](../../docs/README.md)`
  - `[🔬 引擎驗證指南](./ENGINE_VERIFICATION_AND_FIX_PLAN.md)`
  - `[能力分析報告](_out/SCAN_MODULE_CAPABILITY_ANALYSIS.md)`
  - `[修復計劃文檔](./SCAN_MODULE_RESTORATION_PLAN.md)`
  - `[SCAN_USER_GUIDE.md](./SCAN_USER_GUIDE.md)`
  - ... 還有 15 個

⚠️ **建議拆分**: README 內容過多，建議提取為獨立文件

**子目錄** (4 個):
- `archived_docs/`
- `coordinators/`
- `engines/`
- `image/`

## 📂 空資料夾

找到 86 個空資料夾:

- `core\aiva_core\service_backbone\monitoring`
- `features\function_payload_generator\delivery`
- `features\function_payload_generator\obfuscation`
- `features\function_payload_generator\templates`
- `features\function_payload_generator\tests`
- `integration\capability\payload_templates`
- `scan\archived_docs`
- `scan\engines\go_engine\internal\sca\analyzer`
- `scan\engines\rust_engine\target\debug\build\amq-protocol-fdfd691ebcb582d2\out`
- `scan\engines\rust_engine\target\debug\build\async-io-8c51801eec189c02\out`
- `scan\engines\rust_engine\target\debug\build\async-io-928df9b864210498\out`
- `scan\engines\rust_engine\target\debug\build\crossbeam-utils-5bdd24400ce79ee2\out`
- `scan\engines\rust_engine\target\debug\build\doc-comment-b2163b1f21725850\out`
- `scan\engines\rust_engine\target\debug\build\generic-array-bd9a897cbdd7380d\out`
- `scan\engines\rust_engine\target\debug\build\getrandom-86bb09d2f0863fbe\out`
- `scan\engines\rust_engine\target\debug\build\getrandom-def61ea3598e4fb0\out`
- `scan\engines\rust_engine\target\debug\build\httparse-63879826a15e254c\out`
- `scan\engines\rust_engine\target\debug\build\icu_normalizer_data-5ed2155a5bbaf299\out`
- `scan\engines\rust_engine\target\debug\build\icu_normalizer_data-6836c0c4841cc0e1\out`
- `scan\engines\rust_engine\target\debug\build\icu_properties_data-635c33be391ecf25\out`
- `scan\engines\rust_engine\target\debug\build\icu_properties_data-798699f2627003bf\out`
- `scan\engines\rust_engine\target\debug\build\lapin-7d073a005fa9c9e3\out`
- `scan\engines\rust_engine\target\debug\build\libc-881a18ffa071e6c2\out`
- `scan\engines\rust_engine\target\debug\build\native-tls-0ed90f334d35c0d8\out`
- `scan\engines\rust_engine\target\debug\build\num-traits-9b30dc7c4323aadc\out`
- `scan\engines\rust_engine\target\debug\build\num-traits-c60d8ff17850a0b2\out`
- `scan\engines\rust_engine\target\debug\build\parking_lot_core-39ace72b11582f2c\out`
- `scan\engines\rust_engine\target\debug\build\polling-fde28af5ee421551\out`
- `scan\engines\rust_engine\target\debug\build\proc-macro2-50c313c3960bf994\out`
- `scan\engines\rust_engine\target\debug\build\proc-macro2-9471494de1994910\out`
- `scan\engines\rust_engine\target\debug\build\quote-102a82c13fe8e5c0\out`
- `scan\engines\rust_engine\target\debug\build\quote-7521154f582b5b45\out`
- `scan\engines\rust_engine\target\debug\build\rayon-core-a23592f83a3a4418\out`
- `scan\engines\rust_engine\target\debug\build\rustix-22d050289ef542c6\out`
- `scan\engines\rust_engine\target\debug\build\rustix-7b0fe748525fab62\out`
- `scan\engines\rust_engine\target\debug\build\rustls-4d05d375e0272876\out`
- `scan\engines\rust_engine\target\debug\build\rustls-53e9370ff2862c3e\out`
- `scan\engines\rust_engine\target\debug\build\rustls-b7ab7635e54c42fe\out`
- `scan\engines\rust_engine\target\debug\build\serde_json-e4cd23a57b9d499a\out`
- `scan\engines\rust_engine\target\debug\build\winapi-c79fc9717760a494\out`
- `scan\engines\rust_engine\target\debug\build\windows_x86_64_msvc-8f1c70f04d068791\out`
- `scan\engines\rust_engine\target\debug\build\windows_x86_64_msvc-a640bc323e97bdd2\out`
- `scan\engines\rust_engine\target\debug\build\windows_x86_64_msvc-f32bcc1c4fe1ac9e\out`
- `scan\engines\rust_engine\target\debug\build\zerocopy-6b50d5fe36dc624b\out`
- `scan\engines\rust_engine\target\debug\examples`
- `scan\engines\rust_engine\target\release\build\amq-protocol-5996bbff6ee6ea9b\out`
- `scan\engines\rust_engine\target\release\build\async-io-6678368ac23573ba\out`
- `scan\engines\rust_engine\target\release\build\async-io-7d56e0f8aed563ab\out`
- `scan\engines\rust_engine\target\release\build\crossbeam-utils-40590b5a02e822bd\out`
- `scan\engines\rust_engine\target\release\build\doc-comment-223414b9340eb780\out`
- `scan\engines\rust_engine\target\release\build\generic-array-711b5ff35ef59623\out`
- `scan\engines\rust_engine\target\release\build\getrandom-456d9e4562f9c38c\out`
- `scan\engines\rust_engine\target\release\build\getrandom-bf369dcd5f42aadc\out`
- `scan\engines\rust_engine\target\release\build\getrandom-d21328e1b31be343\out`
- `scan\engines\rust_engine\target\release\build\httparse-7add6b829bb6489c\out`
- `scan\engines\rust_engine\target\release\build\icu_normalizer_data-2b77b01e50045fbf\out`
- `scan\engines\rust_engine\target\release\build\icu_normalizer_data-a22d9682d01888e9\out`
- `scan\engines\rust_engine\target\release\build\icu_properties_data-001224e12c6fe390\out`
- `scan\engines\rust_engine\target\release\build\icu_properties_data-0db65fee186051a7\out`
- `scan\engines\rust_engine\target\release\build\lapin-27fff3ddcf704762\out`
- `scan\engines\rust_engine\target\release\build\libc-35ac515a332b0d67\out`
- `scan\engines\rust_engine\target\release\build\native-tls-2bfbd1b6338057d4\out`
- `scan\engines\rust_engine\target\release\build\num-traits-568d831ab88a1e83\out`
- `scan\engines\rust_engine\target\release\build\num-traits-cec8c9595ea30b04\out`
- `scan\engines\rust_engine\target\release\build\parking_lot_core-f69e2468bcd124cd\out`
- `scan\engines\rust_engine\target\release\build\polling-c71ce4eb10e5ad4f\out`
- `scan\engines\rust_engine\target\release\build\proc-macro2-9217f20cd4594178\out`
- `scan\engines\rust_engine\target\release\build\proc-macro2-f74596eab596e9d8\out`
- `scan\engines\rust_engine\target\release\build\quote-2d85ff4e35150fff\out`
- `scan\engines\rust_engine\target\release\build\quote-946cf514411a8599\out`
- `scan\engines\rust_engine\target\release\build\rayon-core-7abc2e88be6d5199\out`
- `scan\engines\rust_engine\target\release\build\rustix-934382b65a930b2c\out`
- `scan\engines\rust_engine\target\release\build\rustix-ea8746dc7e4124e1\out`
- `scan\engines\rust_engine\target\release\build\rustls-bb94d2c2699a3209\out`
- `scan\engines\rust_engine\target\release\build\rustls-eb019dc278aa63b2\out`
- `scan\engines\rust_engine\target\release\build\serde_json-fe7b3cfc745a57f8\out`
- `scan\engines\rust_engine\target\release\build\winapi-88f1f884dc17cf98\out`
- `scan\engines\rust_engine\target\release\build\windows_x86_64_msvc-1c8e724b4cf38d3a\out`
- `scan\engines\rust_engine\target\release\build\windows_x86_64_msvc-4c7f2cdf348001e2\out`
- `scan\engines\rust_engine\target\release\build\windows_x86_64_msvc-5ce82068ca8e5a8e\out`
- `scan\engines\rust_engine\target\release\build\zerocopy-2c8ca854c15b5a58\out`
- `scan\engines\rust_engine\target\release\examples`
- `scan\engines\rust_engine\target\release\incremental`
- `scan\engines\typescript_engine\dist\utils`
- `scan\engines\typescript_engine\docs`
- `scan\engines\typescript_engine\src\utils`

## 📋 建議行動

### 1. 刪除空資料夾

```powershell
Remove-Item -Path "C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\monitoring" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\features\function_payload_generator\delivery" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\features\function_payload_generator\obfuscation" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\features\function_payload_generator\templates" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\features\function_payload_generator\tests" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\integration\capability\payload_templates" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\archived_docs" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\go_engine\internal\sca\analyzer" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\amq-protocol-fdfd691ebcb582d2\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\async-io-8c51801eec189c02\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\async-io-928df9b864210498\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\crossbeam-utils-5bdd24400ce79ee2\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\doc-comment-b2163b1f21725850\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\generic-array-bd9a897cbdd7380d\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\getrandom-86bb09d2f0863fbe\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\getrandom-def61ea3598e4fb0\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\httparse-63879826a15e254c\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\icu_normalizer_data-5ed2155a5bbaf299\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\icu_normalizer_data-6836c0c4841cc0e1\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\icu_properties_data-635c33be391ecf25\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\icu_properties_data-798699f2627003bf\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\lapin-7d073a005fa9c9e3\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\libc-881a18ffa071e6c2\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\native-tls-0ed90f334d35c0d8\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\num-traits-9b30dc7c4323aadc\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\num-traits-c60d8ff17850a0b2\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\parking_lot_core-39ace72b11582f2c\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\polling-fde28af5ee421551\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\proc-macro2-50c313c3960bf994\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\proc-macro2-9471494de1994910\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\quote-102a82c13fe8e5c0\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\quote-7521154f582b5b45\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\rayon-core-a23592f83a3a4418\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\rustix-22d050289ef542c6\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\rustix-7b0fe748525fab62\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\rustls-4d05d375e0272876\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\rustls-53e9370ff2862c3e\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\rustls-b7ab7635e54c42fe\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\serde_json-e4cd23a57b9d499a\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\winapi-c79fc9717760a494\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\windows_x86_64_msvc-8f1c70f04d068791\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\windows_x86_64_msvc-a640bc323e97bdd2\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\windows_x86_64_msvc-f32bcc1c4fe1ac9e\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\build\zerocopy-6b50d5fe36dc624b\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\debug\examples" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\amq-protocol-5996bbff6ee6ea9b\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\async-io-6678368ac23573ba\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\async-io-7d56e0f8aed563ab\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\crossbeam-utils-40590b5a02e822bd\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\doc-comment-223414b9340eb780\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\generic-array-711b5ff35ef59623\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\getrandom-456d9e4562f9c38c\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\getrandom-bf369dcd5f42aadc\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\getrandom-d21328e1b31be343\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\httparse-7add6b829bb6489c\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\icu_normalizer_data-2b77b01e50045fbf\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\icu_normalizer_data-a22d9682d01888e9\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\icu_properties_data-001224e12c6fe390\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\icu_properties_data-0db65fee186051a7\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\lapin-27fff3ddcf704762\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\libc-35ac515a332b0d67\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\native-tls-2bfbd1b6338057d4\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\num-traits-568d831ab88a1e83\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\num-traits-cec8c9595ea30b04\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\parking_lot_core-f69e2468bcd124cd\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\polling-c71ce4eb10e5ad4f\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\proc-macro2-9217f20cd4594178\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\proc-macro2-f74596eab596e9d8\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\quote-2d85ff4e35150fff\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\quote-946cf514411a8599\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\rayon-core-7abc2e88be6d5199\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\rustix-934382b65a930b2c\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\rustix-ea8746dc7e4124e1\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\rustls-bb94d2c2699a3209\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\rustls-eb019dc278aa63b2\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\serde_json-fe7b3cfc745a57f8\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\winapi-88f1f884dc17cf98\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\windows_x86_64_msvc-1c8e724b4cf38d3a\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\windows_x86_64_msvc-4c7f2cdf348001e2\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\windows_x86_64_msvc-5ce82068ca8e5a8e\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\build\zerocopy-2c8ca854c15b5a58\out" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\examples" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\rust_engine\target\release\incremental" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine\dist\utils" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine\docs" -Force -Recurse
Remove-Item -Path "C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine\src\utils" -Force -Recurse
```

### 2. 修正損壞的連結

**aiva_common**: 16 個損壞連結
**core**: 29 個損壞連結
**features**: 44 個損壞連結
**integration**: 11 個損壞連結
**scan**: 20 個損壞連結

### 3. 拆分過大的 README

以下模組的 README 建議拆分:

- `aiva_common/README.md`
- `core/README.md`
- `features/README.md`
- `integration/README.md`
- `scan/README.md`

---

*報告生成時間: 2025年11月27日*
