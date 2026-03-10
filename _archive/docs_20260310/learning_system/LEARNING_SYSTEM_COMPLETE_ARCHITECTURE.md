# 學習系統完整架構文檔

**更新時間**: 2026-01-20  
**狀態**: ✅ **架構完整，數據流已打通**

---

## 📊 核心架構設計

### 三層入口 + 雙路分離 + 安全防護

```
外部 HTTP 請求
   ↓
main.py (第一道安全防線)
   ↓ 檢測木馬、注入攻擊、異常參數
   ↓ 速率限制、訪問控制
   ↓ 通過檢測後轉發
   ↓
app.py (程序與 AI 的溝通接口 - 可插拔設計)
   ↓
   ├─────────────────┬─────────────────┐
   ↓                 ↓
路徑1: 整合模組存儲  路徑2: 任務規劃 AI
(實時記錄所有數據)   (分析決策下令)
   ↓                 ↓
JSONL 文件           EnhancedDecisionAgent
按能力分類           ↓
(xss, sqli, ssrf)    掃描/功能模組執行
```

**分層意義**：
- main.py 挡住攻击 → 保護 AI 系統不受污染
- app.py 雙路分離 → 數據存儲與決策獨立
- AI 只負責規劃 → 不直接接觸外部，不負責執行

---

## 📚 學習系統的三個數據源

### 1. 整合模組 - 本次執行記錄

**位置**: `services/integration/data/experiences/*.jsonl`

**格式**: JSONL（每行一個 JSON）
```json
{
  "timestamp": "2026-01-20T10:30:00",
  "task_id": "scan_abc123",
  "capability": "xss",
  "target": "http://example.com",
  "request": {
    "payload": "<script>alert(1)</script>",
    "method": "GET",
    "encoding": "url"
  },
  "response": {
    "status_code": 403,
    "waf_detected": true
  },
  "result": {
    "xss_triggered": false,
    "blocked_by": "Cloudflare WAF"
  },
  "metadata": {
    "trace_id": "trace_xyz789",
    "source": "external_request"
  }
}
```

**分類**: 按能力類型分別存儲
- `xss.jsonl` - XSS 檢測記錄
- `sqli.jsonl` - SQL 注入記錄
- `ssrf.jsonl` - SSRF 記錄
- `phase0.jsonl` - 快速掃描記錄
- `phase1.jsonl` - 深度掃描記錄

### 2. 整合模組 - 歷史數據

**位置**: 同上，但讀取歷史時間段

**作用**: 
- 作為"預期響應"的參考
- 分析同樣請求在不同情況下的響應變化
- 識別目標防護措施的升級/變化

**示例**: 
```python
# 讀取過去30天的 XSS 攻擊記錄
history = data_manager.load_capability_data(
    capability="xss",
    start_time=datetime.now() - timedelta(days=30),
    limit=500
)
```

### 3. 能力知識庫 - 分析報告

**位置**: `cognitive_core/learning_system/knowledge/`

**內容**: Markdown 格式的完整分析報告
- `XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md`
- `SQLI_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md`
- `SSRF_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md`
- 更多能力的報告...

**包含信息**:
- 攻擊手法和技術細節
- 繞過技巧（WAF、過濾器）
- Payload 變形方法
- 預期響應模式（成功/可疑/失敗）
- 已知 WAF 特徵（Cloudflare、Akamai 等）
- 因果關係分析

---

## 🔄 完整學習流程

### 階段1: 數據收集（任務執行期間）

```
1. 外部請求 → main.py（安全檢測）
2. main.py → app.py（通過檢測後轉發）
3. app.py 雙路處理：
   ├─ 路徑1: 存儲到整合模組
   │   SimpleDataManager.save_task_data()
   │   → experiences/phase0.jsonl
   │
   └─ 路徑2: AI 分析決策
       EnhancedDecisionAgent → 下令執行
       → 掃描模組/功能模組
       → 執行檢測
       → 結果發布 MQ
       → app.py 訂閱處理
       → 繼續存儲（按能力分類）
```

### 階段2: 學習分析（任務結束後，異步獨立）

```
1. 讀取三個數據源
   ├─ ExperienceManager.load_from_integration()
   │   - 本次記錄：最新的執行數據
   │   - 歷史數據：過去的執行記錄
   │
   └─ ModuleKnowledgeManager.load_all_knowledge()
       - 各能力的分析報告
       - 已知攻擊模式

2. 三路比對評估
   ├─ 本次 vs 歷史
   │   - 發現差異
   │   - 識別變化趨勢
   │   - 目標防護是否升級
   │
   ├─ 本次 vs 知識庫
   │   - 匹配已知模式
   │   - 識別成功/失敗原因
   │   - 確認因果關係
   │
   └─ 歷史 vs 知識庫
       - 驗證知識庫準確性
       - 更新過時信息
   ↓
   判斷：是否為已知情況？
   ├─ ✅ 已知情況
   │   - 直接使用知識庫建議
   │   - 生成優化方案
   │
   └─ ❌ 未知情況（不在任何數據中）
       ↓
       觸發 RAG 搜索
       ├─ 搜索範圍：
       │   - 更廣泛的知識庫
       │   - 技術文檔
       │   - CVE 數據庫
       │   - 安全研究報告
       │   - 外部資源
       │
       └─ RAG 返回：
           - 相似案例
           - 解決方案建議
           - 相關技術資料
   ↓
   ├─ 參數優化
   │   - timeout 調整
   │   - depth 優化
   │   - threads 配置
   │   - 檢測閾值
   │
   └─ 方法優化
       - Payload 變形建議
       - 繞過技巧選擇
       - 攻擊手法調整
       - 編碼方式改進

4. 生成新權重/新策略

5. 驗證新權重效果
   - 在測試環境中驗證
   - 使用歷史場景測試
   - 評估性能提升

6. 決策
   ├─ ✅ 效果好
   │   - 保存新權重
   │   - 更新模型
   │   - 更新知識庫
   │
   └─ ❌ 效果差
       - 丟棄新權重
       - 保留原有權重
       - 記錄失敗原因
       - 分析失敗模式
```

---

## 🔐 安全設計

### 1. 分層防護

```
main.py (第一道防線)
  ↓ 阻擋惡意輸入
app.py (業務層)
  ↓ 處理乾淨數據
整合模組 (存儲層)
  ↓ 記錄安全數據
學習系統 (學習層)
  ↓ 從安全數據學習
```

**保證**: 學習系統不會學到惡意數據

### 2. 異步獨立運行

- ✅ 學習系統在任務**結束後**才啟動
- ✅ 不在任務執行期間介入
- ✅ 避免修改運行中的代碼
- ✅ 保證系統穩定性

### 3. 權重驗證

- ✅ 新權重必須驗證後才保存
- ✅ 效果差的權重會被丟棄
- ✅ 保留原有權重作為 fallback
- ✅ 記錄失敗嘗試供後續分析

---

## 📂 關鍵文件路徑

### 數據存儲
```
services/integration/
├── data/experiences/
│   ├── xss.jsonl          # XSS 檢測記錄
│   ├── sqli.jsonl         # SQL 注入記錄
│   ├── ssrf.jsonl         # SSRF 記錄
│   ├── phase0.jsonl       # 快速掃描記錄
│   └── phase1.jsonl       # 深度掃描記錄
└── simple_data_manager.py # 數據管理器
```

### 學習系統
```
services/core/aiva_core/cognitive_core/learning_system/
├── experience_manager.py            # 經驗管理器
├── knowledge/
│   ├── module_knowledge_manager.py  # 知識庫管理器
│   └── README.md                    # 知識庫文檔
└── learning/                        # 訓練相關
```

### 入口層
```
services/core/
├── main.py    # 對外入口（安全檢測）
└── aiva_core/service_backbone/api/
    └── app.py # AI 接口（雙路分離）
```

---

## ✅ 總結

**完整數據流已打通**：
1. ✅ main.py 安全檢測（第一道防線）
2. ✅ app.py 雙路分離（存儲 + AI）
3. ✅ 整合模組統一存儲（按能力分類，JSONL）
4. ✅ 學習系統三源讀取（本次 + 歷史 + 知識庫）
5. ✅ 三路比對評估（全面分析）
6. ✅ 權重驗證保存（效果好才保存）

**學習系統應該沒什麼問題了！** 🎉
