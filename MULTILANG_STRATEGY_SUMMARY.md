# 快速摘要: AIVA 多語言發展策略

## 🎯 核心建議

### 1. **立即行動項 (本週)**

#### ✅ Go - 建立共用模組 (已完成框架)
```powershell
# 執行初始化
.\init_go_common.ps1
```

**已建立:**
- `aiva_common_go/mq/client.go` - 統一 RabbitMQ 客戶端
- `aiva_common_go/logger/logger.go` - 標準化日誌
- `aiva_common_go/config/config.go` - 配置管理
- `aiva_common_go/schemas/message.go` - 與 Python 對應的 Schema

**預期效果:**
- 消除 60% 的重複程式碼
- 統一錯誤處理和日誌格式
- 加速新 Go 服務開發

#### 📝 Python - 強化類型檢查
```bash
mypy services/core services/integration --strict
```

#### 🌐 TypeScript - 確認已完全替代 Python Playwright
```bash
grep -r "playwright" services/core/ services/integration/
# 如果有結果,應全部移除
```

---

### 2. **語言職責確認**

| 語言 | 負責領域 | 優先發展方向 |
|------|---------|------------|
| **Python** | 核心協調、AI、資料庫 | 深化 FastAPI、整合 AI 到生命週期管理 |
| **Go** | 併發 I/O、雲端安全 | 使用共用模組、提升併發效能 |
| **Rust** | SAST、秘密掃描 | 規則外部化、PyO3 整合 |
| **TypeScript** | 動態掃描、SPA 測試 | API 發現、智慧表單填充 |

---

### 3. **關鍵問題與解決方案**

#### ❌ 問題: Go 服務重複程式碼多
**✅ 解決:** 已建立 `aiva_common_go`,下一步遷移各服務

#### ❌ 問題: Schema 跨語言同步困難
**✅ 解決:** 
- 短期: 手動維護對應的 struct/interface
- 長期: 考慮遷移到 Protocol Buffers

#### ❌ 問題: SAST 規則更新需要重新編譯
**✅ 解決:** 實作規則外部化 (YAML 載入)

---

### 4. **實施時間表**

```
第1週: 遷移 function_sca_go 使用共用模組
第2週: 遷移所有 Go 服務
第3週: 強化 TypeScript 動態掃描
第4週: 優化 Rust SAST 規則引擎
第5-6週: 建立跨語言整合測試
第7-8週: 評估 Protocol Buffers 可行性
```

---

### 5. **成功指標**

| 指標 | 當前 | 目標 (3個月) |
|------|-----|------------|
| Go 程式碼重複率 | 60% | < 10% |
| 動態掃描 API 發現率 | 30% | > 80% |
| SAST 規則數量 | 15 | > 50 |
| Python 類型覆蓋率 | 60% | > 90% |

---

## 📚 完整文件

詳細策略請參考: **[MULTILANG_STRATEGY.md](./MULTILANG_STRATEGY.md)**

---

## 🚀 快速開始

1. **初始化 Go 共用模組:**
   ```powershell
   .\init_go_common.ps1
   ```

2. **查看實施指南:**
   ```powershell
   code MULTILANG_STRATEGY.md
   ```

3. **開始遷移第一個服務:**
   參考 `MULTILANG_STRATEGY.md` 中的範例程式碼

---

**關鍵原則: 每種語言做它最擅長的事,透過統一的契約協作**
