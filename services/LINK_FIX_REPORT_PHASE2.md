# 連結修復報告 - Phase 2

## 執行摘要

- **修改檔案**: 7 個
- **修復連結**: 28 處
- **執行時間**: 2025-11-23

## 修復類型

### 1. 絕對路徑修復（14 處）

修復 features/README.md 中的絕對路徑：
- `C:\Users\User\Downloads\新增資料夾 (6)\AIVA_Enhancement_Plan\*.md`
- `../../../Users/User/Downloads/新增資料夾%20(6)/AIVA_Enhancement_Plan/*.md`

**替換為**: `../../../docs/reports/*.md`

### 2. 不存在文檔修復（7 處）

修復指向不存在的 `../../docs/README.md`：
- 在 aiva_common/README.md (1 處)
- 在 core/README.md (1 處)
- 在 features/README.md (1 處)
- 在 integration/README.md (1 處)
- 在 scan/README.md (1 處)
- 在 function_payload_generator/README.md (1 處)
- 在 core/aiva_core/README.md (1 處)

**替換為**: `../../docs/guides/services/` 或 `../../docs/`

### 3. _out 目錄修復（7 處）

修復指向舊 `_out` 目錄的連結：
- 將 `../../_out/*.md` 更新為 `../../docs/reports/*.md`
- 影響多個模組的 README

## 修改檔案清單

1. `aiva_common/README.md` - 3 處修復
2. `core/README.md` - 2 處修復
3. `features/README.md` - 16 處修復（包含 12 個絕對路徑）
4. `integration/README.md` - 2 處修復
5. `scan/README.md` - 2 處修復
6. `features/function_payload_generator/README.md` - 2 處修復
7. `core/aiva_core/README.md` - 1 處修復

## 後續行動

- [ ] 將 features/README.md 大型內容提取到獨立文件
- [ ] 建立 docs/reports/ 目錄並移動相關文件
- [ ] 驗證所有新連結指向正確
- [ ] 建立層級導航結構
- [ ] 確保每個 README 只保留概覽和導航

## 注意事項

部分連結修復後可能指向尚未建立的文件，需要：
1. 確認 `docs/reports/` 目錄存在
2. 確認被引用的報告文件已正確放置
3. 如有缺失，需建立對應的文件或調整連結
