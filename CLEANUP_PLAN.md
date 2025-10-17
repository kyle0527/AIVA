# 🧹 文件整理與歸檔計畫

## 目標
- **減少 50%+ 文件** - 清理冗餘、過時、臨時文件
- **建立清晰結構** - 歸檔重要文件,刪除不必要內容
- **Git 里程碑** - 提交並標記重要版本

## 文件整理策略

### ✅ 保留文件 (核心功能)

#### 1. 核心代碼
- `services/` - 所有服務代碼
- `tools/` - 工具腳本 (新創建的安全分析工具)
- `tests/` - 測試文件
- `schemas/` - Schema 定義

#### 2. 配置文件
- `pyproject.toml`
- `requirements.txt`
- `pyrightconfig.json`
- `mypy.ini`
- `ruff.toml`

#### 3. 重要文檔
- `README.md`
- `QUICK_START.md`
- 最新的架構文檔

#### 4. 數據文件
- `_out/cli_training/` - CLI 訓練狀態
- `_out/core_cli_possibilities.json`
- 最新生成的報告

### 🗑️ 刪除/歸檔文件

#### 1. 歸檔到 `_archive/`
- 所有 `*_ANALYSIS.md` 舊分析文檔
- `*_REPORT.md` 完成的報告
- `*_SUMMARY.md` 總結文檔
- `*_COMPLETE.md` 已完成項目文檔
- 舊的計畫文檔

#### 2. 刪除臨時文件
- `temp_*.py` - 臨時腳本
- `test_*.py` (根目錄的測試,應在 tests/ 下)
- `train_*.py` (已整合到 tools/)
- `final_report.py` (一次性腳本)
- `benchmark_performance.py` (已完成)

#### 3. 清理 `_out/` 輸出
- 保留最新報告
- 刪除舊的 tree 輸出
- 合併重複的輸出目錄

#### 4. 清理 `_archive/`
- 移除深度嵌套的舊文件夾
- 保留有價值的歷史文檔

## 執行步驟

### Step 1: 歸檔文檔 (移動到 _archive/)
```
AI_ARCHITECTURE_ANALYSIS.md
AI_ARRAY_ANALYSIS_CONCLUSION.md
AI_COMPETITIVE_ANALYSIS.md
ARCHITECTURE_CONTRACT_COMPLIANCE_REPORT.md
CLI_AND_AI_TRAINING_GUIDE.md
CLI_COMMAND_REFERENCE.md
CLI_CORE_MODULE_FLOWS.md
CLI_CROSS_MODULE_GUIDE.md
CLI_IMPLEMENTATION_COMPLETE.md
CLI_QUICK_REFERENCE.md
CLI_UNIFIED_SETUP_GUIDE.md
FILE_ORGANIZATION_REPORT.md
PROJECT_ORGANIZATION_COMPLETE.md
SCHEMA_DEFINITION_CLEANUP_REPORT.md
SERVICES_ARCHITECTURE_COMPLIANCE_REPORT.md
SERVICES_ORGANIZATION_SUMMARY.md
SPECIALIZED_AI_CORE_DESIGN.md
SPECIALIZED_AI_IMPLEMENTATION_PLAN.md
```

### Step 2: 刪除臨時腳本
```
temp_generate_stats.py
test_ai_core.py (→ tests/)
test_ai_real_data.py (→ tests/)
test_internal_communication.py (→ tests/)
test_message_system.py (→ tests/)
test_simple_matcher.py (→ tests/)
train_ai_with_cli.py (已有 tools/train_cli_with_memory.py)
train_cli_matching.py (已整合)
final_report.py
benchmark_performance.py
```

### Step 3: 清理 _out/
- 保留 `cli_training/`
- 保留最新安全報告
- 刪除舊 tree 輸出
- 清理 `_out1101016/` (舊備份)
- 清理 `emoji_backups*/` (emoji 備份)

### Step 4: 整理 _archive/
- 保留重要歷史文檔
- 刪除深層嵌套文件夾

## Git 操作

### 提交信息
```
🎯 Major Cleanup & Security Analysis Integration

- ✅ 新增安全分析工具套件
  - security_log_analyzer.py (OWASP 日誌分析)
  - attack_pattern_trainer.py (攻擊模式 AI 訓練)
  - real_time_threat_detector.py (實時威脅檢測)
  - run_security_analysis.py (一鍵執行)

- 🧹 大規模文件整理
  - 歸檔 17+ 個完成的文檔到 _archive/
  - 刪除 10+ 個臨時/重複腳本
  - 清理輸出目錄,減少 50%+ 文件
  - 優化項目結構

- 📊 CLI 訓練進度
  - 924/978 組合已訓練 (94.5%)
  - 12 種使用模式已學習
  - 83% 成功率

減少文件數: 50%+
新增功能: 安全威脅檢測與 AI 訓練
```

### Git Tag (里程碑)
```
v1.0.0-security-milestone
- 完成 CLI 訓練系統
- 整合 OWASP 安全分析
- 項目結構大幅優化
```

## 預期結果

### 減少的文件類型
- 文檔: ~17 個 → _archive/
- 腳本: ~10 個 → 刪除/整合
- 輸出: ~20+ 個舊文件 → 刪除
- 總計: **50-60% 文件減少**

### 保留的核心
- 所有 services/ 代碼
- 新的 tools/ 工具
- 關鍵配置文件
- 最新訓練數據
- 重要文檔

---

**準備執行**: 等待確認後開始自動化整理流程
