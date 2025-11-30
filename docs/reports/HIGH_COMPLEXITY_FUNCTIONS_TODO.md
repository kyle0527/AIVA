# 高複雜度函數記錄 - 待後續重構

## 記錄目的
記錄需要重構的高複雜度函數，正式發佈前統一處理

## 高複雜度函數清單

### 1. internal_loop_connector.py

#### `_categorize_capability` (Line 270)
- **當前複雜度**: 44
- **建議複雜度**: ≤15
- **檔案**: `services/core/aiva_core/cognitive_core/internal_loop_connector.py`
- **功能**: 能力分類 (根據模組路徑和名稱判斷類別和子類別)
- **重構建議**: 
  - 拆分為多個小函數：`_categorize_by_module()`, `_categorize_by_name()`, `_get_subcategory()`
  - 使用策略模式或映射表替代多層 if-else
- **優先級**: 中
- **影響範圍**: 能力掃描和分類邏輯

#### `_convert_to_documents` (Line 566)
- **當前複雜度**: 45
- **建議複雜度**: ≤15
- **檔案**: `services/core/aiva_core/cognitive_core/internal_loop_connector.py`
- **功能**: 將能力轉換為 ChromaDB 文檔格式
- **重構建議**:
  - 拆分為 `_build_document_content()`, `_build_document_metadata()`, `_format_parameters()`
  - 提取參數和返回值處理為獨立方法
- **優先級**: 中
- **影響範圍**: RAG 知識庫數據格式

### 2. real_neural_core.py

#### 重複的條件判斷 (Line 214+)
- **問題**: 多個相同的 `if AIVA_COMMON_AVAILABLE:` 區塊
- **檔案**: `services/core/aiva_core/cognitive_core/neural/real_neural_core.py`
- **重構建議**: 合併重複邏輯或使用裝飾器模式
- **優先級**: 低
- **影響範圍**: 神經核心功能切換

## 重構計劃

### 階段 1: 準備 (正式發佈前)
- [ ] 為高複雜度函數添加完整單元測試
- [ ] 建立效能基準測試
- [ ] 記錄現有行為和邊界條件

### 階段 2: 重構 (正式發佈前)
- [ ] `_categorize_capability`: 拆分為策略模式
- [ ] `_convert_to_documents`: 拆分為多個小函數
- [ ] `real_neural_core.py`: 合併重複條件

### 階段 3: 驗證 (正式發佈前)
- [ ] 確保所有測試通過
- [ ] 驗證效能無退化
- [ ] 代碼審查

## 備註

- ✅ 這些函數目前功能正常，不影響系統運行
- ✅ 重構僅為提升代碼可維護性
- ✅ 建議在正式發佈前統一處理，避免打斷開發流程

---

**記錄時間**: 2025年11月29日  
**記錄原因**: 用戶要求記錄高複雜度函數，待全部改善完成後再處理  
**狀態**: 📝 待重構（非阻塞性）
