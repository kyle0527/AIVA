# AIVA 隊列命名標準

## 概述

為確保 AIVA 各語言 workers 的一致性，所有隊列名稱必須遵循統一標準。

## 命名規範

### 1. 任務隊列 (Task Queues)
```
tasks.{module}.{function}
```

**範例:**
- `tasks.scan.sensitive_info` - 敏感信息掃描
- `tasks.function.authn` - 認證功能測試
- `tasks.function.ssrf` - SSRF 功能測試
- `tasks.function.sast` - SAST 功能測試
- `tasks.function.cspm` - CSPM 功能測試

### 2. 結果隊列 (Result Queues)
**統一標準**: 所有掃描結果使用相同隊列
```
findings.new
```

**原因:**
- 簡化結果收集
- 統一結果處理管道
- 便於監控和統計

### 3. 專用隊列 (Specialized Queues)
```
{purpose}.{specific}
```

**範例:**
- `statistics.metrics` - 統計數據
- `notifications.alerts` - 警報通知
- `deadletter.tasks` - 任務死信隊列
- `deadletter.findings` - 結果死信隊列

## 當前隊列對照表

| Worker | 語言 | 舊隊列名 | 新隊列名 |
|--------|------|----------|----------|
| info_gatherer | Rust | `findings.new` | ✅ 已符合標準 |
| authn | Go | `findings.new` | ✅ 已符合標準 |
| ssrf | Go | `findings.new` | ✅ 已符合標準 |
| sast | Rust | `findings` | ❌ 需修改為 `findings.new` |
| scan_node | TypeScript | `results.scan.completed` | ❌ 需修改為 `findings.new` |

## 實施計畫

### 階段 1: 修復不符合標準的隊列
1. ✅ **info_gatherer_rust**: 已使用正確隊列名
2. ✅ **function_authn_go**: 已使用正確隊列名  
3. ✅ **function_ssrf_go**: 已使用正確隊列名
4. ❌ **function_sast_rust**: 需要從 `findings` 改為 `findings.new`
5. ❌ **aiva_scan_node**: 需要從 `results.scan.completed` 改為 `findings.new`

### 階段 2: 環境變數標準化

所有 workers 應該支援以下環境變數:
```bash
AIVA_RESULT_QUEUE=findings.new  # 結果隊列（預設）
AIVA_TASK_QUEUE=tasks.{module}.{function}  # 任務隊列
```

### 階段 3: 配置驗證

1. 確保所有 workers 使用統一的配置模式
2. 支援環境變數覆蓋
3. 實施配置驗證邏輯

## 遷移注意事項

1. **向後相容**: 在遷移期間保持舊隊列監聽
2. **漸進式部署**: 先部署消費者，再切換生產者
3. **監控確認**: 確認新隊列正常接收和處理消息
4. **清理舊資源**: 遷移完成後清理舊隊列

## 實施狀態

- [x] 制定命名標準
- [ ] 修復 function_sast_rust 隊列名
- [ ] 修復 aiva_scan_node 隊列名
- [ ] 更新配置文件
- [ ] 驗證隊列一致性
- [ ] 部署和測試

---

*此文檔是 AIVA 架構標準化項目的一部分，確保跨語言一致性。*