# 重複定義清理報告
## AIVA v5.0 自動生成文件歸檔

### 移除的自動生成文件
**來源目錄**: `services/aiva_common/schemas/generated/`
**移動日期**: 2024-12-19

1. **async_utils.py** - 異步工具定義 (與手動維護版本衝突)
2. **base_types.py** - 基礎類型定義 (包含重複的 Target, RiskLevel 等)
3. **cli.py** - CLI介面定義 (自動生成版本)
4. **findings.py** - 發現結果定義 (與掃描模組衝突)
5. **messaging.py** - 消息傳遞定義 (自動生成版本)
6. **plugins.py** - 插件定義 (自動生成版本)
7. **tasks.py** - 任務定義 (自動生成版本)
8. **__init__.py** - 初始化文件 (自動生成版本)

### 移除的備份工具文件
**來源目錄**: `services/aiva_common/tools/`

1. **schema_codegen_tool_backup.py** - Schema代碼生成器備份版本
2. **cross_language_interface_backup.py** - 跨語言介面備份版本

### 清理原因
- 自動生成的schema文件與手動維護版本產生重複定義衝突
- 備份文件不再需要，造成代碼維護困擾
- 遵循AIVA v5.0統一架構原則，避免重複實現

### 後續行動
1. ✅ 移除自動生成版本
2. 🔄 重新分析實際需要修復的重複定義問題
3. ⏳ 更新修復工具只處理真正的重複定義
4. ⏳ 驗證清理結果不影響現有功能

### 保留的手動維護文件
- `services/aiva_common/schemas/` (主要schema定義)
- `services/aiva_common/scan/` (掃描相關定義)
- `services/aiva_common/business/` (業務邏輯定義)
- `services/aiva_common/tools/schema_codegen_tool.py` (主要工具)

### 注意事項
此清理操作遵循AIVA Common開發指南，確保：
- 不影響核心功能
- 保留手動維護的高品質代碼
- 消除自動生成造成的衝突
- 簡化項目結構和維護工作