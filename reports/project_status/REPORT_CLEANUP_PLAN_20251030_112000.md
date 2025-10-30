# AIVA 報告清理計劃

---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Plan
---

## 清理摘要

- **分析日期**: 2025-10-30
- **清理標準**: 超過 7 天且已完成的報告
- **總報告數**: 76

## 分類統計

### 📊 報告分類
- **最近報告** (< 7天): 74 個
- **舊重要報告** (> 7天, 重要): 0 個
- **舊已完成報告** (> 7天, 已完成): 2 個
- **日期不明報告**: 0 個
- **清理候選**: 0 個

### 🗑️ 建議清理的報告

✅ 沒有發現需要清理的報告

### 🔒 保留的重要報告

### ⚠️ 日期不明的報告

## 清理操作

### 自動清理
```bash
# 執行清理計劃
python scripts/misc/cleanup_old_reports.py --execute

# 僅預覽清理
python scripts/misc/cleanup_old_reports.py --preview
```

### 手動清理
建議的清理步驟：
1. 備份重要報告
2. 移動到 `_archive` 目錄而非直接刪除
3. 30天後再永久刪除

## 安全措施

- ✅ 自動備份到 `_archive/cleanup_{timestamp}/`
- ✅ 保留所有重要文檔和指南
- ✅ 不刪除最近 7 天內的報告
- ✅ 提供復原機制

---

**清理工具**: `cleanup_old_reports.py`  
**備份位置**: `_archive/cleanup_{timestamp}/`
