# CLI 文檔歸檔說明

**歸檔日期**: 2026-01-10  
**原因**: 整合為統一文檔，舊版本內容過時

## 歸檔文件清單

| 文件 | 問題 | 處理方式 |
|------|------|---------|
| CLI_COMPLETE_GUIDE.md | 與 CLI_GUIDE.md 內容 90% 重複 | 歸檔保留 |
| CLI_GUIDE.md | 聲稱 782 個能力（數量待確認），內容部分正確 | 已標註後歸檔 |
| CLI_FLOW_USAGE_GUIDE.md | 聲稱 840 flows（實際 ~276），命令格式簡化 | 已標註後歸檔 |
| CAPABILITY_EXECUTION_QUICK_GUIDE.md | 聲稱 926 flows（實際 ~276），統計過時 | 已標註後歸檔 |

## 新文檔

**統一文檔**: `AIVA_CLI_UNIFIED_GUIDE.md`

**改進點**:
1. ✅ 清晰區分兩套 CLI 系統
2. ✅ 修正數字錯誤（276 flows）
3. ✅ 統一命令格式
4. ✅ 整合所有有用內容
5. ✅ 實際驗證可用性

## 如何使用舊文檔

如需參考舊版內容：
1. 文檔已標註不準確部分（⚠️ 標記）
2. 建議以新統一文檔為準
3. 舊文檔僅供歷史參考

## 實際驗證結果

```
實際文件位置:
✅ scripts/common/aiva_cli.py - 存在且可用
✅ services/core/aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py - 存在且可用
✅ 啟動能力選單.bat - 存在且可用
✅ 執行Flow.bat - 存在且可用
✅ 預覽Flow.bat - 存在且可用

實際數據:
✅ Flow 數量: ~276（2026-01-10 驗證）
❌ 文檔聲稱: 782能力, 840flows, 926flows（均不準確）
```

---

**歸檔人員**: GitHub Copilot  
**審核狀態**: 已標註問題並創建新文檔
