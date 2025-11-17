# Services README 更新計劃

**更新日期**: 2025-11-16  
**更新原因**: 整合模組資料儲存結構建立完成,需要更新所有相關文件

---

## ✅ 已確認現況

### services/README.md 檢查結果
- ✅ **五大模組連結**: 已完整連結到 Core、Common、Features、Integration、Scan
- ✅ **文檔結構**: 完整的 README 架構圖和導航系統
- ✅ **版本資訊**: v6.1, 最後更新 2025-11-15
- ⚠️ **需要更新**: 整合模組資料儲存結構尚未反映

---

## 📋 更新待辦事項 (按優先級排序)

### Priority 1: 核心文件更新 (必須)

#### ✅ TODO-1: services/README.md
**狀態**: 需更新  
**原因**: 反映整合模組最新資料儲存結構  
**更新內容**:
- [ ] 更新版本號: v6.1 → v6.2
- [ ] 更新日期: 2025-11-15 → 2025-11-16
- [ ] 新增整合模組資料儲存說明
- [ ] 更新 Integration 模組描述,加入資料儲存架構
- [ ] 新增 2025年11月16日更新摘要

**預估時間**: 15 分鐘

#### ✅ TODO-2: services/integration/README.md
**狀態**: 需更新  
**原因**: 反映資料儲存結構建立完成  
**更新內容**:
- [ ] 更新版本號: v6.1 → v6.2
- [ ] 更新日期: 2025-11-13 → 2025-11-16
- [ ] 新增資料儲存結構章節
- [ ] 更新環境變數配置 (加入整合模組變數)
- [ ] 新增維護腳本說明 (backup.py, cleanup.py)
- [ ] 更新架構圖,反映資料儲存層
- [ ] 新增 Neo4j→NetworkX 遷移說明

**預估時間**: 30 分鐘

#### ✅ TODO-3: services/integration/aiva_integration/README.md
**狀態**: 需更新  
**原因**: 反映核心模組配置變更  
**更新內容**:
- [ ] 更新 config.py 章節 (新的標準化配置)
- [ ] 新增攻擊路徑分析器資料儲存說明
- [ ] 更新經驗資料庫配置說明
- [ ] 新增訓練資料集管理章節

**預估時間**: 20 分鐘

### Priority 2: 資料儲存文件優化 (重要)

#### ✅ TODO-4: data/integration/README.md
**狀態**: 已建立,需優化  
**原因**: 新建檔案,需確保完整性  
**檢查內容**:
- [x] 目錄結構說明完整
- [x] 資料庫說明詳細
- [x] 配置方式清晰
- [x] 使用範例充足
- [x] 備份策略明確
- [ ] 加入交叉連結 (與 services/integration/README.md 互聯)

**預估時間**: 10 分鐘

#### ✅ TODO-5: services/integration/scripts/README.md
**狀態**: 已建立,需優化  
**原因**: 新建檔案,需確保完整性  
**檢查內容**:
- [x] 腳本列表完整
- [x] 用法說明清晰
- [x] 排程建議詳細
- [x] 故障排除指引
- [ ] 加入交叉連結

**預估時間**: 10 分鐘

### Priority 3: 子模組 README 更新 (次要)

#### 🔹 TODO-6: services/integration/aiva_integration/attack_path_analyzer/README.md
**狀態**: 需檢查  
**原因**: NetworkX 遷移完成後需確認文檔更新  
**檢查內容**:
- [ ] 檢查是否還有 Neo4j 相關說明
- [ ] 確認 NetworkX 使用範例正確
- [ ] 更新配置路徑說明

**預估時間**: 15 分鐘

#### 🔹 TODO-7: services/integration/aiva_integration/reception/README.md
**狀態**: 需檢查是否存在  
**原因**: 經驗資料庫為核心功能,應有文檔  
**行動**:
- [ ] 檢查檔案是否存在
- [ ] 如不存在,建立完整文檔
- [ ] 包含 experience_repository.py 使用指南

**預估時間**: 20 分鐘 (如需建立)

### Priority 4: 主專案文件同步 (可選)

#### 🔹 TODO-8: AIVA-git/README.md
**狀態**: 待確認  
**原因**: 主專案文件可能需反映整合模組更新  
**行動**:
- [ ] 檢查主 README 是否需要更新
- [ ] 確認版本號與日期

**預估時間**: 5 分鐘

#### 🔹 TODO-9: reports/INTEGRATION_DATA_STORAGE_SETUP_REPORT.md
**狀態**: 已建立  
**原因**: 完整建立報告,可作為歷史記錄  
**行動**:
- [x] 已建立完整報告
- [ ] 確認交叉連結完整

**預估時間**: 5 分鐘

---

## 📊 更新統計

### 需更新檔案數量
- **Priority 1 (必須)**: 3 個檔案
- **Priority 2 (重要)**: 2 個檔案
- **Priority 3 (次要)**: 2 個檔案
- **Priority 4 (可選)**: 2 個檔案

**總計**: 9 個檔案

### 預估總時間
- Priority 1: 65 分鐘
- Priority 2: 20 分鐘
- Priority 3: 35 分鐘
- Priority 4: 10 分鐘

**總計**: 130 分鐘 (~2.2 小時)

---

## 🎯 執行順序建議

### Phase 1: 核心文件 (立即執行)
1. services/README.md
2. services/integration/README.md
3. services/integration/aiva_integration/README.md

### Phase 2: 資料儲存文件 (同步執行)
4. data/integration/README.md (加入交叉連結)
5. services/integration/scripts/README.md (加入交叉連結)

### Phase 3: 子模組文件 (後續執行)
6. attack_path_analyzer/README.md (檢查更新)
7. reception/ (建立或更新文檔)

### Phase 4: 主專案同步 (最後確認)
8. AIVA-git/README.md (如需)
9. 報告交叉連結確認

---

## 🔄 更新版本資訊

### 統一更新項目
- **版本號**: v6.1 → v6.2
- **更新日期**: 2025-11-16
- **更新原因**: 整合模組資料儲存結構建立完成
- **關鍵變更**:
  - Neo4j → NetworkX 遷移完成
  - 資料儲存標準化 (attack_paths, experiences, training_datasets, models)
  - 統一配置系統 (config.py)
  - 維護腳本建立 (backup.py, cleanup.py)

---

## ✅ 完成檢查清單

### Phase 1
- [ ] services/README.md 更新完成
- [ ] services/integration/README.md 更新完成
- [ ] services/integration/aiva_integration/README.md 更新完成

### Phase 2
- [ ] data/integration/README.md 交叉連結完成
- [ ] services/integration/scripts/README.md 交叉連結完成

### Phase 3
- [ ] attack_path_analyzer/README.md 檢查完成
- [ ] reception/ 文檔確認完成

### Phase 4
- [ ] AIVA-git/README.md 同步確認
- [ ] 所有報告交叉連結完整

---

**建立時間**: 2025-11-16  
**預計完成**: 2025-11-16  
**負責人**: GitHub Copilot
