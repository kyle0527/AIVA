# 中等完成度模組完善計劃

**日期**: 2025-12-11  
**狀態**: ⚠️ 部分已過時 - 許多問題已在後續開發中解決  
**最新狀態**: 請參考 [README.md](README.md) 了解當前實際狀態

---

## 📋 模組現狀分析

### 1. function_web_scanner (完成度: 35% - 需補充文檔)

**現有架構**：
- ✅ `WebScannerManager` - 主控制類（scanner_manager.py）
- ✅ `SubdomainEnumerator` - 子域名枚舉
- ✅ `DirectoryScanner` - 目錄掃描
- ✅ `VulnerabilityScanner` - 漏洞掃描
- ✅ `TechnologyDetector` - 技術檢測

**已有功能**：
- 子域名枚舉（crt.sh + DNS暴力破解）
- 目錄掃描（字典攻擊）
- 基礎漏洞掃描（XSS, SQLi, 目錄遍歷）
- 安全標頭檢測
- 技術棧檢測

**需完善內容**：
1. ✅ 同步 WebScannerManager.scan() 已實現
2. ❌ **缺少 README.md**（主要問題）
3. ⏳ 增強檢測邏輯準確性
4. ⏳ 減少誤報

**架構評估**：✅ 良好 - 已有基礎架構，缺文檔

---

### 2. function_authn_go (完成度: 70%)

**現有架構**：
- ✅ Go 語言實現（internal/engine.go）
- ✅ AMQP 消息隊列集成
- ✅ Python 橋接層（authn_manager.py 已完成）

**已有功能**（Go 端）：
- 認證協議測試
- 會話管理
- 消息隊列通信

**需完善內容**：
1. ⏳ 編譯 Go 二進制文件（主要剩餘工作）
2. ✅ Python 調用接口已完善
3. ⏳ 添加測試用例
4. ✅ README.md 文檔已完整

**架構評估**：✅ 優秀 - Go 高性能實現

---

### 3. function_crypto (完成度: 95% - 已大幅提升)

**現有架構**：
- ✅ 純 Rust CLI 實現 (v2.0)
- ✅ Rust 核心引擎（rust_core/）
- ✅ 獨立可執行文件

**已有功能**：
- ✅ JavaScript 密碼學問題檢測
- ✅ TLS/SSL 配置檢測
- ✅ Cookie 安全性檢測
- ✅ HTTP 安全標頭檢測
- ✅ 硬編碼 API 金鑰檢測
- ✅ 弱加密算法檢測
- ✅ 不安全隨機數檢測

**需完善內容**：
1. ✅ Rust 核心已完成（rust_core/src/）
2. ✅ 不需要 Python 回退（純 CLI 架構）
3. ✅ README.md 文檔完整（786 行）
4. ⏳ 確保 Rust 二進制文件已編譯

**架構評估**：⭐⭐⭐⭐⭐ 優秀 - 純 Rust 高性能實現

---

### 4. function_postex (完成度: 50%)

**現有架構**：
- ✅ `PostExDetector` - 主檢測類
- ✅ `PrivilegeEscalationTester` - 權限提升測試
- ✅ `LateralMovementTester` - 橫向移動測試
- ✅ `PersistenceChecker` - 持久化檢測
- ✅ 安全模式機制

**已有功能**：
- 權限提升向量檢測
- 橫向移動測試
- 持久化機制檢測
- 安全模式控制

**需完善內容**：
1. ⏳ 完善各個引擎的實際檢測邏輯
2. ⏳ 增加更多測試場景
3. ⏳ 增強系統權限分析
4. ⏳ 添加測試用例

**架構評估**：✅ 良好 - 多引擎架構

---

## 🎯 完善策略

### 關鍵原則

根據 [SIMPLE_ARCHITECTURE.md](./SIMPLE_ARCHITECTURE.md)：
- ✅ 保持每個模組現有的架構設計
- ✅ 不強制統一命名或結構
- ✅ 確保提供簡單的同步接口
- ✅ 專注於完善檢測邏輯本身

### 完善優先級

**Phase 1 - 立即完善**（已有良好基礎）：
1. function_crypto - 增強密碼學檢測規則
2. function_postex - 完善各引擎的檢測邏輯
3. function_web_scanner - 減少誤報，提升準確性

**Phase 2 - 中期完善**（需要編譯/構建）：
4. function_authn_go - 編譯 Go 引擎並集成

---

## 📝 詳細完善計劃

### 1. function_crypto 完善計劃

**目標**：增強密碼學分析能力，提升檢測準確性

**任務列表**：
- [ ] 添加更多弱加密算法模式（MD5, DES, RC4 等）
- [ ] 增強硬編碼密鑰檢測（支持更多格式）
- [ ] 添加證書驗證問題檢測
- [ ] 增加密碼強度檢測
- [ ] 支持多種編程語言的模式匹配
- [ ] 減少誤報（使用上下文分析）

**文件修改**：
- `detector/crypto_detector.py` - 增強檢測規則
- `python_wrapper/engine_bridge.py` - 完善掃描邏輯

---

### 2. function_postex 完善計劃

**目標**：完善後滲透測試引擎，提供更全面的檢測

**任務列表**：
- [ ] 完善 PrivilegeEscalationTester 的實際檢測邏輯
  - SUID/SGID 文件檢查
  - Sudo 配置檢查
  - 內核漏洞檢測
- [ ] 完善 LateralMovementTester 的檢測邏輯
  - 憑證重用檢測
  - 網絡共享檢測
  - 服務漏洞檢測
- [ ] 完善 PersistenceChecker 的檢測邏輯
  - 啟動項檢查
  - 計劃任務檢查
  - 服務持久化檢查
- [ ] 增強 EnhancedPrivilegeAnalyzer
- [ ] 添加更多測試場景

**文件修改**：
- `engines/privilege_engine.py` - 完善權限提升檢測
- `engines/lateral_engine.py` - 完善橫向移動檢測
- `engines/persistence_engine.py` - 完善持久化檢測

---

### 3. function_web_scanner 完善計劃

**目標**：提升漏洞掃描準確性，減少誤報

**任務列表**：
- [ ] 改進 XSS 檢測邏輯
  - 添加 Context-aware 檢測
  - 減少誤報
- [ ] 改進 SQLi 檢測邏輯
  - 添加盲注檢測
  - 支持更多資料庫類型
- [ ] 改進目錄掃描
  - 智能過濾404頁面
  - 指紋識別
- [ ] 改進技術檢測
  - 增加更多框架識別
  - 版本檢測

**文件修改**：
- `integration_tools/web_tools.py` - 改進各個掃描器

---

### 4. function_authn_go 完善計劃

**目標**：編譯 Go 引擎並提供 Python 調用接口

**任務列表**：
- [ ] 編譯 Go 二進制文件
  - Windows: `go build -o bin/authn-worker.exe cmd/worker/main.go`
  - Linux: `go build -o bin/authn-worker cmd/worker/main.go`
- [ ] 測試 Go 引擎功能
- [ ] 創建 Python 橋接層（subprocess 調用）
- [ ] 添加測試用例
- [ ] 文檔完善

**文件創建/修改**：
- `bridge.py` - Python 橋接層（可選）
- `README.md` - 編譯和使用說明

---

## 🚀 實施步驟

### 立即行動（不需編譯）

1. ✅ 完善 function_crypto 的檢測規則
2. ✅ 完善 function_postex 的引擎邏輯
3. ✅ 改進 function_web_scanner 的掃描準確性

### 後續行動（需要編譯）

4. 編譯 function_authn_go 的 Go 引擎
5. 編譯 function_crypto 的 Rust 核心（可選）

---

## 📊 完成標準

每個模組完善後應滿足：
- ✅ 提供清晰的同步接口
- ✅ 檢測邏輯準確（減少誤報）
- ✅ 錯誤處理完善
- ✅ 日誌記錄清晰
- ✅ 代碼注釋完整
- ✅ 基本測試通過

---

## 🎯 目標完成度

完善後的目標完成度：
- function_crypto: 60% → 75%
- function_postex: 50% → 70%
- function_web_scanner: 45% → 65%
- function_authn_go: 70% → 85%（編譯後）
