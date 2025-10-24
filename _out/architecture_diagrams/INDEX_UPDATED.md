# AIVA Architecture Diagrams - Updated Index

> **更新時間**: 2025-10-24  
> **生成工具**: py2mermaid.py + mermaid_optimizer.py  
> **圖表數量**: 307+ 詳細流程圖

## 🏗️ 核心架構圖表

### 1. 系統級架構圖
- [`01_overall_architecture.mmd`](01_overall_architecture.mmd) - 系統整體架構
- [`02_modules_overview.mmd`](02_modules_overview.mmd) - 模組概覽圖  
- [`14_deployment_architecture.mmd`](14_deployment_architecture.mmd) - 部署架構圖

### 2. 模組架構圖
- [`03_core_module.mmd`](03_core_module.mmd) - 核心模組架構
- [`04_scan_module.mmd`](04_scan_module.mmd) - 掃描模組架構
- [`05_function_module.mmd`](05_function_module.mmd) - 功能模組架構
- [`06_integration_module.mmd`](06_integration_module.mmd) - 整合模組架構

### 3. 工作流程圖
- [`11_complete_workflow.mmd`](11_complete_workflow.mmd) - 完整工作流程
- [`13_data_flow.mmd`](13_data_flow.mmd) - 數據流程圖
- [`12_language_decision.mmd`](12_language_decision.mmd) - 語言選擇決策

### 4. 漏洞檢測流程圖
- [`07_sqli_flow.mmd`](07_sqli_flow.mmd) - SQL 注入檢測流程
- [`08_xss_flow.mmd`](08_xss_flow.mmd) - XSS 檢測流程  
- [`09_ssrf_flow.mmd`](09_ssrf_flow.mmd) - SSRF 檢測流程
- [`10_idor_flow.mmd`](10_idor_flow.mmd) - IDOR 檢測流程

## 🔍 掃描模組詳細圖表 (新增)

### 核心掃描組件
- [`aiva_scan_scan_orchestrator_Module.mmd`](aiva_scan_scan_orchestrator_Module.mmd) - 掃描編排器
- [`aiva_scan_scan_context_Module.mmd`](aiva_scan_scan_context_Module.mmd) - 掃描上下文
- [`unified_scan_engine_Module.mmd`](unified_scan_engine_Module.mmd) - 統一掃描引擎

### 爬取引擎
- [`aiva_scan_core_crawling_engine_http_client_hi_Module.mmd`](aiva_scan_core_crawling_engine_http_client_hi_Module.mmd) - HTTP 客戶端
- [`aiva_scan_core_crawling_engine_static_content_parser_Module.mmd`](aiva_scan_core_crawling_engine_static_content_parser_Module.mmd) - 靜態內容解析器
- [`aiva_scan_core_crawling_engine_url_queue_manager_Module.mmd`](aiva_scan_core_crawling_engine_url_queue_manager_Module.mmd) - URL 隊列管理器

### 動態掃描引擎
- [`aiva_scan_dynamic_engine_dynamic_content_extractor_Module.mmd`](aiva_scan_dynamic_engine_dynamic_content_extractor_Module.mmd) - 動態內容提取器
- [`aiva_scan_dynamic_engine_headless_browser_pool_Module.mmd`](aiva_scan_dynamic_engine_headless_browser_pool_Module.mmd) - 無頭瀏覽器池
- [`aiva_scan_dynamic_engine_js_interaction_simulator_Module.mmd`](aiva_scan_dynamic_engine_js_interaction_simulator_Module.mmd) - JS 交互模擬器

### 資訊收集器
- [`aiva_scan_info_gatherer_sensitive_info_detector_Module.mmd`](aiva_scan_info_gatherer_sensitive_info_detector_Module.mmd) - 敏感資訊檢測器
- [`aiva_scan_info_gatherer_javascript_source_analyzer_Module.mmd`](aiva_scan_info_gatherer_javascript_source_analyzer_Module.mmd) - JavaScript 源碼分析器
- [`aiva_scan_info_gatherer_passive_fingerprinter_Module.mmd`](aiva_scan_info_gatherer_passive_fingerprinter_Module.mmd) - 被動指紋識別

### 掃描器系列
- [`aiva_scan_service_detector_Module.mmd`](aiva_scan_service_detector_Module.mmd) - 服務檢測器
- [`aiva_scan_network_scanner_Module.mmd`](aiva_scan_network_scanner_Module.mmd) - 網路掃描器
- [`aiva_scan_vulnerability_scanner_Module.mmd`](aiva_scan_vulnerability_scanner_Module.mmd) - 漏洞掃描器
- [`aiva_scan_sensitive_data_scanner_Module.mmd`](aiva_scan_sensitive_data_scanner_Module.mmd) - 敏感數據掃描器

### 管理組件
- [`aiva_scan_authentication_manager_Module.mmd`](aiva_scan_authentication_manager_Module.mmd) - 認證管理器
- [`aiva_scan_scope_manager_Module.mmd`](aiva_scan_scope_manager_Module.mmd) - 範圍管理器
- [`aiva_scan_fingerprint_manager_Module.mmd`](aiva_scan_fingerprint_manager_Module.mmd) - 指紋管理器
- [`aiva_scan_strategy_controller_Module.mmd`](aiva_scan_strategy_controller_Module.mmd) - 策略控制器

### 配置中心
- [`aiva_scan_config_control_center_Module.mmd`](aiva_scan_config_control_center_Module.mmd) - 配置控制中心
- [`aiva_scan_header_configuration_Module.mmd`](aiva_scan_header_configuration_Module.mmd) - 標頭配置

## 📊 統計資訊

### 圖表生成統計
- **總圖表數**: 307+ 個
- **模組級圖表**: 38 個主要模組
- **函數級圖表**: 269+ 個詳細函數流程
- **核心架構圖**: 14 個系統級圖表

### 涵蓋範圍
- ✅ **掃描模組**: 100% 覆蓋
- ✅ **核心引擎**: 完整流程圖
- ✅ **動態引擎**: 詳細交互圖
- ✅ **資訊收集**: 完整檢測邏輯
- ✅ **管理組件**: 配置和策略流程

### 技術層面
- **Python 模組**: 38 個已分析
- **函數流程**: 269+ 個詳細圖表
- **複雜度覆蓋**: 從系統到函數級別
- **更新狀態**: 基於最新代碼結構

## 🛠️ 使用指南

### 查看圖表
1. **VS Code**: 安裝 Mermaid 插件後直接查看
2. **線上預覽**: 訪問 [mermaid.live](https://mermaid.live/) 貼上代碼
3. **文檔系統**: 使用支援 Mermaid 的文檔平台

### 圖表更新流程
```bash
# 1. 更新掃描模組圖表
python tools/common/development/py2mermaid.py -i services/scan -o _out/architecture_diagrams

# 2. 更新其他模組
python tools/common/development/py2mermaid.py -i services/core -o _out/architecture_diagrams
python tools/common/development/py2mermaid.py -i services/integration -o _out/architecture_diagrams

# 3. 優化圖表格式
python -c "
from tools.features.mermaid_optimizer import MermaidOptimizer
optimizer = MermaidOptimizer()
# 批量優化圖表
"
```

### 問題記錄和解決

#### 已知問題 ✅
1. **路徑問題**: generate_mermaid_diagrams.py 輸出路徑計算錯誤 - 已使用 py2mermaid 替代
2. **圖表數量**: 生成了 307 個詳細圖表 - 已建立索引便於導航
3. **文件組織**: 大量細粒度圖表 - 已按功能模組分類整理

#### 解決方案
- ✅ 使用 py2mermaid.py 直接生成到正確目錄
- ✅ 創建分層索引便於查找
- ✅ 保留原有核心架構圖不變
- ✅ 新增詳細模組和函數級流程圖

## 🔄 更新日誌

### 2025-10-24 更新
- ✅ 修復了所有代碼質量問題 (147→0 錯誤)
- ✅ 重新生成了 services/scan 模組的完整流程圖
- ✅ 新增 307+ 個詳細的函數和模組級圖表
- ✅ 建立了完整的圖表索引和導航系統
- ✅ 確保所有圖表基於最新的代碼結構

### 下一步計劃
- 🔄 擴展到其他模組 (core, integration, function)
- 🔄 整合 TypeScript 和 Rust 模組的圖表
- 🔄 使用 mermaid_optimizer.py 優化圖表樣式
- 🔄 建立自動更新機制

---

📝 **維護者**: AIVA Development Team  
🛠️ **工具**: py2mermaid.py, mermaid_optimizer.py  
📅 **更新頻率**: 隨代碼結構變更而更新