# Schema 生成修復計劃

## 問題摘要
通過應用 CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md，發現了系統性的 Schema 生成問題，需要從根本修復。

## 錯誤分類

### 1. 重複類型定義（編譯阻斷）
```
services/features/common/go/aiva_common_go/schemas/generated/schemas.go:176:6: FindingPayload redeclared in this block
services/features/common/go/aiva_common_go/schemas/generated/schemas.go:350:6: FindingEvidence redeclared in this block
services/features/common/go/aiva_common_go/schemas/generated/schemas.go:364:6: FindingImpact redeclared in this block  
services/features/common/go/aiva_common_go/schemas/generated/schemas.go:378:6: FindingRecommendation redeclared in this block
```

### 2. 未定義類型引用（編譯阻斷）
```
services/features/common/go/aiva_common_go/schemas/generated/schemas.go:106:9: undefined: AsyncTaskStatus
services/features/common/go/aiva_common_go/schemas/generated/schemas.go:302:7: undefined: PluginType
services/features/common/go/aiva_common_go/schemas/generated/schemas.go:306:9: undefined: PluginStatus
```

### 3. 跨語言字段不匹配（功能缺失）
- Python `FunctionTaskTarget.url` 字段在 Go 版本中缺失

## 修復方案

### 階段 1：清理重複定義
1. **分析重複原因**：檢查 Schema 生成管道是否多次生成相同類型
2. **安全移除**：保留第一個定義，移除重複的定義
3. **驗證完整性**：確保所有字段都在保留的定義中

### 階段 2：解決未定義類型
1. **追蹤源頭**：在 Python Schema 中找到對應的類型定義
2. **生成缺失類型**：確保 Go 生成包含所有必要的類型
3. **更新引用**：驗證所有類型引用的正確性

### 階段 3：跨語言同步
1. **字段對齊**：確保 Go `FunctionTaskTarget` 包含 Python 版本的 `url` 字段
2. **命名統一**：統一使用 `TaskId`、`FindingId` 等命名約定
3. **類型驗證**：確保所有類型在兩種語言中保持一致

### 階段 4：自動化驗證
1. **生成測試**：創建跨語言一致性測試
2. **CI 集成**：確保未來的 Schema 變更不會破壞一致性
3. **文檔更新**：更新 Schema 生成和維護文檔

## 執行順序
1. 修復重複定義（立即解除編譯阻斷）
2. 添加缺失類型定義（恢復功能完整性）
3. 同步跨語言字段（確保功能一致性）
4. 建立長期維護機制（預防未來問題）

## 風險評估
- **低風險**：移除重複定義（明確的重複）
- **中風險**：添加缺失類型（需要驗證依賴關係）
- **高風險**：修改現有類型結構（可能影響現有功能）

## 成功標準
- [x] 所有 Go 文件編譯成功
- [x] 跨語言 Schema 測試通過
- [x] 功能測試保持綠色
- [x] 代碼生成管道產生一致結果