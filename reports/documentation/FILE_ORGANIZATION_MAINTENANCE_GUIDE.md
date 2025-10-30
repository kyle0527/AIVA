# AIVA 檔案組織維護指南

---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Guide
---

## 檔案組織維護指南

### 🎯 維護原則

1. **定期整理**: 每月運行一次自動整理工具
2. **時間戳標準**: 所有新報告必須包含時間戳
3. **分類規則**: 按功能分類，保持目錄結構清晰
4. **版本追蹤**: 修改報告時更新 Last Modified 時間

### 🔧 維護工具使用

#### 自動整理工具
```bash
# 運行檔案整理
python scripts/misc/organize_aiva_files.py

# 查看整理報告
cat reports/project_status/FILE_ORGANIZATION_REPORT_*.md
```

#### 手動維護
```bash
# 為新報告添加時間戳標頭
---
Created: YYYY-MM-DD
Last Modified: YYYY-MM-DD
Document Type: Report/Guide/Data
---
```

### 📁 標準目錄結構

```
AIVA-git/
├── scripts/                           # 執行腳本
│   ├── ai_analysis/                   # AI 分析腳本
│   │   ├── ai_system_explorer_v3.py   # 系統探索
│   │   ├── analyze_ai_performance.py  # 性能分析
│   │   └── ...
│   ├── testing/                       # 測試腳本
│   │   ├── comprehensive_*.py         # 綜合測試
│   │   ├── test_*.py                  # 單元測試
│   │   └── ...
│   ├── analysis/                      # 分析工具
│   │   ├── analyze_cross_language_warnings.py
│   │   ├── check_*.py                 # 檢查工具
│   │   └── ...
│   ├── utilities/                     # 系統工具
│   │   ├── aiva_launcher.py           # 系統啟動
│   │   ├── aiva_package_validator.py  # 依賴驗證
│   │   └── ...
│   └── misc/                          # 其他腳本
├── reports/                           # 報告文件
│   ├── ai_analysis/                   # AI 分析報告
│   ├── architecture/                  # 架構報告
│   ├── schema/                        # Schema 報告
│   ├── testing/                       # 測試報告
│   ├── documentation/                 # 文檔報告
│   ├── project_status/                # 專案狀態
│   ├── data/                          # JSON 數據
│   └── misc/                          # 其他報告
└── logs/                              # 日誌檔案
```

### 📋 檔案命名規範

#### 腳本檔案
- AI 分析: `ai_*.py`, `analyze_ai_*.py`, `*ai_manager*.py`
- 測試: `test_*.py`, `*_test.py`, `comprehensive_*test*.py`
- 分析: `analyze_*.py`, `check_*.py`, `verify_*.py`
- 工具: `aiva_*.py`, `*_validator.py`, `health_*.py`

#### 報告檔案
- 架構: `ARCHITECTURE_*.md`, `SYSTEM_*.md`
- AI: `AI_*.md`, `AIVA_AI_*.md`
- Schema: `SCHEMA_*.md`, `CROSS_LANGUAGE_*.md`
- 測試: `TEST_*.md`, `TESTING_*.md`
- 文檔: `*GUIDE*.md`, `DOCUMENTATION_*.md`

### 🔄 定期維護檢查清單

#### 每周檢查
- [ ] 檢查根目錄是否有新的散落檔案
- [ ] 確認新報告包含正確的時間戳
- [ ] 檢查日誌檔案大小，必要時清理

#### 每月維護
- [ ] 運行自動整理工具
- [ ] 檢查整理報告
- [ ] 更新維護文檔
- [ ] 清理過期的臨時檔案

#### 季度檢查
- [ ] 檢查分類規則是否需要更新
- [ ] 評估目錄結構優化需求
- [ ] 更新維護指南
- [ ] 備份重要報告

### 🚨 注意事項

1. **不要移動的檔案**:
   - `README.md`
   - `pyproject.toml`
   - `requirements.txt`
   - `docker-compose.yml`
   - `.env.example`

2. **特殊處理檔案**:
   - 配置檔案保持在根目錄
   - 重要文檔不自動移動
   - 系統檔案跳過處理

3. **版本控制**:
   - 整理後檢查 git 狀態
   - 確認沒有意外刪除重要檔案
   - 提交時使用描述性訊息

### 📞 問題處理

如果整理過程中出現問題：

1. **檔案移動錯誤**: 檢查目標目錄權限
2. **時間戳添加失敗**: 檢查檔案編碼格式
3. **分類錯誤**: 更新分類規則然後重新運行

---

**維護負責**: AIVA 開發團隊  
**下次檢查**: 2025-11-30