# AIVA 能力整合方案 - 最終建議

## 🎯 核心目標

**讓 AI 和人類在執行前就能理解和選擇能力**

## ✅ 已完成整合

### 1. 數據層整合
- ✅ **enriched_classification.json** (840 flows)
  - 每個 Flow 都包含 `capability` 欄位
  - 包含：能力名稱、描述、CLI 指令、標籤、模組、複雜度

### 2. 工具層整合
- ✅ **aiva_capability_cli.py** - AI 友好的查詢工具
  - 支援能力搜尋（`--search`）
  - 支援詳細資訊查看（`--info`）
  - 支援能力列表（`--list`）

### 3. 文檔層整合
- ✅ **AI_Capability_Reference.md** - 人類可讀手冊
- ✅ **capability_index.json** - AI 快速檢索索引

---

## 📋 AI 使用流程（推薦）

### 階段 1：能力發現
```bash
# 場景：AI 需要找 XSS 掃描能力
python aiva_capability_cli.py --search xss

# 場景：AI 需要找編排相關能力
python aiva_capability_cli.py --search 編排

# 場景：AI 需要找認知核心模組的能力
python aiva_capability_cli.py --search cognitive --search-by module
```

### 階段 2：能力理解
```bash
# AI 找到 Flow 313 後，查看詳細資訊
python aiva_capability_cli.py --info 313

# 輸出包含：
# - 能力名稱：編排器
# - 描述：【認知核心】編排器 - 流程長度 2 步
# - CLI 指令：python -m ...
# - 標籤：workflow, automation, cognitive_core
# - 模組：認知核心
# - 複雜度：simple
# - 完整路徑：session_state_manager -> capability_orchestrator
```

### 階段 3：能力執行
```bash
# 預覽執行計畫（不實際運行）
python aiva_cli_implementation.py --flow 313 --dry-run

# 實際執行
python aiva_cli_implementation.py --flow 313
```

---

## 🔄 與現有系統的關係

### 舊系統（保留）
1. **latest_classification.json**
   - 原始數據來源
   - 包含完整 Flow 定義
   - 用於執行

2. **Manifest JSON 系統**（9 個檔案）
   - 位於 `core_capabilities/manifests/capabilities/`
   - 設計中的 AI 認知層協議
   - **建議**：暫時保留，未來可與 enriched 數據合併

3. **MinimalManifest.py**（範例）
   - 僅為參考範例
   - **建議**：保留作為文檔

### 新系統（推薦使用）
1. **enriched_classification.json**
   - 包含所有 840 flows 的能力資訊
   - **主要數據源**

2. **aiva_capability_cli.py**
   - AI 查詢工具
   - **主要查詢接口**

3. **aiva_cli_implementation.py**
   - 執行工具
   - **主要執行接口**

---

## 💡 建議的架構決策

### 選項 A：雙軌並行（推薦）✅
```
enriched_classification.json  →  AI 執行層（現在）
         ↓
manifest.json (未來)          →  AI 認知層（設計中）
```

**優點**：
- 立即可用：840 個能力都有描述
- 不破壞現有設計：Manifest 系統可繼續發展
- 平滑過渡：未來可將 enriched 數據遷移到 Manifest

### 選項 B：完全統一（長期）
將 enriched 數據轉換為 840 個 manifest.json
- 優點：架構統一
- 缺點：需要大量工作（840 個檔案）

---

## 🚀 立即可用的 AI 工作流

### 範例 1：AI 需要掃描能力
```bash
# 1. 搜尋
$ python aiva_capability_cli.py --search scan --search-by tag

# 2. 查看詳情（假設找到 Flow 50）
$ python aiva_capability_cli.py --info 50

# 3. 執行
$ python aiva_cli_implementation.py --flow 50
```

### 範例 2：AI 需要了解認知核心能力
```bash
# 列出所有認知核心能力
$ python aiva_capability_cli.py --search cognitive_core --search-by module

# 逐一查看感興趣的 Flow
$ python aiva_capability_cli.py --info <flow_id>
```

---

## 📊 數據完整性

| 項目 | 數量 | 狀態 |
|------|------|------|
| 總 Flows | 840 | ✅ 全部有能力資訊 |
| 模組覆蓋 | 6/6 | ✅ 完整 |
| 複雜度標記 | 840 | ✅ 全部標記 |
| CLI 指令 | 840 | ✅ 全部生成 |

---

## 🔧 維護建議

### 日常維護
```bash
# 當 latest_classification.json 更新後
cd C:\D\fold7\AIVA-git\scripts

# 重新生成 enriched 版本
python enrich_flows_with_capabilities.py

# 重新生成參考手冊
python generate_ai_capability_reference.py
```

### 數據同步
- latest_classification.json → enriched_classification.json（自動）
- enriched → AI_Capability_Reference.md（自動）
- 手動維護：Manifest JSON（如需要）

---

## 🎯 結論

### 當前最佳實踐
1. **AI 查詢**：使用 `aiva_capability_cli.py`
2. **執行**：使用 `aiva_cli_implementation.py`
3. **數據源**：`enriched_classification.json`

### 未來方向
- 可選：逐步為關鍵能力添加詳細 manifest.json
- 優先級：頻繁使用的 158 個能力優先
- 最終目標：完整的 Manifest 驅動架構

---

## 📝 快速參考

### AI 必知的 3 個指令
```bash
# 1. 搜尋能力
python aiva_capability_cli.py --search <關鍵字>

# 2. 查看詳情
python aiva_capability_cli.py --info <flow_id>

# 3. 執行能力
python aiva_cli_implementation.py --flow <flow_id>
```

### 檔案位置
- 數據：`C:/Users/User/Downloads/data/internal_exploration/enriched_classification.json`
- 工具：`C:/D/fold7/AIVA-git/services/core/aiva_core/internal_exploration/python_tools/`
- 文檔：`C:/Users/User/Downloads/data/internal_exploration/capability_references/`
