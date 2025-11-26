# 📁 AIVA 目錄結構說明

## 📑 目錄
- [🗂️ 主要目錄結構](#️-主要目錄結構)
- [📱 源代碼目錄 (src/)](#-源代碼目錄-src)
- [🔧 服務目錄 (services/)](#-服務目錄-services)
- [📚 文檔目錄 (docs/)](#-文檔目錄-docs)
- [⚙️ 配置目錄 (config/)](#️-配置目錄-config)
- [📊 數據目錄 (data/)](#-數據目錄-data)
- [🧪 測試目錄 (tests/)](#-測試目錄-tests)
- [📋 其他重要目錄](#-其他重要目錄)

---

## 🗂️ **主要目錄結構**

```
AIVA/
├── 📱 src/                          # 源代碼目錄
│   ├── core/                        # AI核心引擎
│   │   ├── real_ai_core.py         # 真實AI神經網路核心
│   │   ├── aiva_capability_orchestrator.py  # 能力編排器
│   │   ├── aiva_model_manager.py    # 模型管理器
│   │   └── aiva_5M_replacement_evaluation.py  # 5M模型評估
│   ├── launchers/                   # 啟動器目錄
│   │   ├── aiva_launcher.py         # 主啟動器
│   │   ├── start_rich_cli.py        # CLI啟動器
│   │   └── start_ui_auto.py         # UI自動啟動器
│   └── demos/                       # 演示程序
│       ├── demo_5m_neural_network.py  # 5M神經網路演示
│       └── weight_integration_demo.py  # 權重整合演示
├── 🧠 models/                       # AI模型目錄
│   ├── weights/                     # 模型權重文件
│   │   └── aiva_real_*.pth         # AI核心權重文件
│   ├── history/                     # 訓練歷史
│   │   └── aiva_real_ai_core_history.json  # 核心訓練歷史
│   ├── aiva_model_status.json      # 模型狀態
│   ├── test_ai_model.pkl           # 測試模型
│   └── test_ai_model_vocab.json    # 模型詞彙
├── 📖 docs/                        # 文檔目錄
│   ├── guides/                     # 使用指南
│   │   ├── integration/            # 整合指南
│   │   │   ├── AIVA_5M_*.md       # 5M模型整合文檔
│   │   │   ├── AIVA_WEB_RESEARCH_INTEGRATION_GUIDE.md  # 網路研究整合
│   │   │   └── AIVA_CORE_5M_CAPABILITY_REQUIREMENTS.md  # 能力需求
│   │   └── AIVA_AI_REPAIR_GUIDE.md  # AI修復指南
│   ├── reports/                    # 報告目錄
│   │   ├── testing/                # 測試報告
│   │   │   ├── aiva_ai_analysis_test_report_*.json  # AI測試報告
│   │   │   └── aiva_ai_analysis_test.log  # 測試日誌
│   │   ├── mermaid/                # Mermaid圖表報告
│   │   │   └── MERMAID_*.md        # Mermaid相關報告
│   │   ├── batch_repair_report.json  # 批次修復報告
│   │   └── repair_*.json           # 修復報告
│   └── project-status/             # 專案狀態
│       ├── AIVA_PROJECT_STATUS.md  # 專案狀態報告
│       ├── AIVA_PROJECT_PROGRESS_REPORT_20251110.md  # 進度報告
│       ├── AIVA_SYSTEM_ANALYSIS_REPORT_2025-11-10.md  # 系統分析
│       └── integration_status_report.md  # 整合狀態
├── ⚙️ config/                      # 配置目錄
│   └── aiva_capability_integration_config.yaml  # 能力整合配置
├── 📊 data/                        # 數據目錄
│   └── capability_registry.db      # 能力註冊資料庫
├── 🧪 tests/                       # 測試目錄
│   ├── test_5m_integration.py      # 5M整合測試
│   ├── test_direct_ai_core.py      # AI核心直接測試
│   ├── test_integration.py         # 整合測試
│   ├── test_real_ai_core.py        # 真實AI核心測試
│   ├── ai_autonomous_testing_loop.py  # 自主測試循環
│   ├── ai_program_analysis_test.py  # AI程序分析測試
│   └── validate_integration.py     # 整合驗證
├── 🔧 scripts/                     # 實用腳本
│   ├── safe_batch_repair.py        # 安全批次修復
│   ├── health_check.py             # 健康檢查
│   └── setup_python_path.py        # Python路徑設置
├── 📋 _archive/                    # 歸檔目錄
│   └── backups/                    # 備份文件
│       ├── *.backup                # 各種備份文件
│       └── *batch_backup           # 批次備份文件
└── 🌐 [其他現有目錄]               # 保持原有結構
    ├── services/                   # 微服務
    ├── api/                        # API接口
    ├── web/                        # 網頁界面
    ├── tools/                      # 工具集
    ├── utilities/                  # 實用工具
    ├── guides/                     # 原有指南
    ├── examples/                   # 示例代碼
    ├── docker/                     # 容器配置
    ├── security/                   # 安全模組
    ├── observability/              # 可觀察性
    ├── plugins/                    # 插件
    └── weights/                    # 原有權重目錄
```

## 🎯 **整理原則**

### **📂 按功能分類**
- **src/**: 所有源代碼按功能分組
- **models/**: 統一管理AI模型相關文件
- **docs/**: 結構化的文檔組織
- **tests/**: 集中管理所有測試文件

### **🏷️ 按用途分組**
- **core/**: 核心AI引擎和邏輯
- **launchers/**: 各種啟動方式
- **demos/**: 演示和示例程序
- **guides/**: 使用和整合指南

### **📊 按報告類型**
- **testing/**: 測試相關報告
- **mermaid/**: 圖表和架構報告
- **project-status/**: 專案狀態追蹤

## 🔧 **使用指南**

### **開發者**
1. **核心開發**: 主要在 `src/core/` 目錄工作
2. **測試開發**: 在 `tests/` 目錄添加測試
3. **功能演示**: 在 `src/demos/` 創建演示

### **用戶**
1. **快速啟動**: 使用 `src/launchers/` 中的啟動器
2. **文檔查閱**: 參考 `docs/` 中的結構化文檔
3. **狀態追蹤**: 查看 `docs/project-status/` 了解最新進展

### **維護者**
1. **模型管理**: `models/` 目錄統一管理所有AI模型
2. **配置管理**: `config/` 目錄管理系統配置
3. **報告生成**: 結構化存儲在 `docs/reports/` 中

## 📈 **整理效果**

### **✅ 改善結果**
- **目錄清晰**: 根目錄文件從80+減少到20+
- **分類明確**: 按功能和用途清晰分組
- **查找便利**: 文件位置邏輯清晰，易於定位
- **維護簡化**: 結構化組織便於項目維護

### **🎯 核心優勢**
1. **開發效率**: 開發者可快速定位相關文件
2. **項目管理**: 項目狀態和進展追蹤更清晰
3. **新人友善**: 新加入者更容易理解項目結構
4. **擴展性**: 為未來功能擴展預留了清晰的目錄結構

---

*整理完成時間: 2025年11月10日*  
*整理原則: 功能分組 + 用途分類 + 邏輯清晰*