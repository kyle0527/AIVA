# 📁 核心模組 CLI 計算 - 檔案索引

## 快速導航

### 🎯 想要快速了解結果？
→ 閱讀 [`CLI_ACTUAL_COUNT_SUMMARY.md`](./CLI_ACTUAL_COUNT_SUMMARY.md)

### 📊 想要查看詳細統計？
→ 查看 [`_out/core_cli_possibilities.json`](./_out/core_cli_possibilities.json)

### 🛠️ 想要執行工具？
→ 閱讀 [`tools/CLI_COUNT_README.md`](./tools/CLI_COUNT_README.md)

### 📚 想要深入了解方法？
→ 閱讀 [`_out/CORE_CLI_POSSIBILITIES_REPORT.md`](./_out/CORE_CLI_POSSIBILITIES_REPORT.md)

---

## 📂 完整檔案清單

### 📊 主要成果文件

| 檔案 | 大小 | 說明 |
|------|------|------|
| **CLI_ACTUAL_COUNT_SUMMARY.md** | 9.7 KB | **執行摘要報告**（從這裡開始） |
| _out/core_cli_possibilities.json | 2.3 KB | 機器可讀的完整統計數據 |
| _out/core_cli_possibilities_examples.json | 2.3 KB | Top-10 常用命令範例 |
| _out/CORE_CLI_POSSIBILITIES_REPORT.md | 7.2 KB | 詳細技術報告 |

### 🛠️ 工具腳本

| 檔案 | 大小 | 說明 |
|------|------|------|
| **tools/count_core_cli_possibilities.py** | 13.6 KB | **主計算器**（432 行） |
| tools/verify_cli_calculation.py | 6.7 KB | 驗證工具（199 行） |
| tools/cli_count_config.example.json | 287 B | 配置範例 |
| tools/CLI_COUNT_README.md | 8.1 KB | 工具使用指南 |

### 📍 原始碼參考

| 檔案 | 說明 |
|------|------|
| services/core/aiva_core/ui_panel/auto_server.py | 唯一的 CLI 入口點 |

---

## 🚀 快速開始

### 1️⃣ 執行計算
```bash
python tools/count_core_cli_possibilities.py
```

### 2️⃣ 驗證結果
```bash
python tools/verify_cli_calculation.py
```

### 3️⃣ 查看報告
```bash
cat _out/core_cli_possibilities.json
```

---

## 📊 核心數字一覽

```
範圍: services/core/**
CLI 入口點: 1 個
總使用可能性: 978 種
下界（最小）: 3 種

參數空間:
├── --mode: 3 個選項 (ui, ai, hybrid)
├── --host: 1 個候選 (127.0.0.1)
└── --ports: 325 種有序序列
```

---

## 🎯 閱讀建議

### 🏃 快速瀏覽（5 分鐘）
1. CLI_ACTUAL_COUNT_SUMMARY.md → 「執行摘要」
2. _out/core_cli_possibilities.json → 「核心數字」

### 👨‍💻 開發者（15 分鐘）
1. CLI_ACTUAL_COUNT_SUMMARY.md → 完整閱讀
2. tools/CLI_COUNT_README.md → 「快速開始」+ 「命令列選項」
3. 執行 `python tools/count_core_cli_possibilities.py`

### 🔬 深入研究（30 分鐘）
1. CLI_ACTUAL_COUNT_SUMMARY.md → 完整閱讀
2. _out/CORE_CLI_POSSIBILITIES_REPORT.md → 完整閱讀
3. tools/count_core_cli_possibilities.py → 閱讀原始碼
4. 執行驗證工具並理解數學公式

---

## 📋 使用情境

### 情境 1：我想知道核心模組有多少種 CLI 用法
→ 答案：**978 種**（詳見 `CLI_ACTUAL_COUNT_SUMMARY.md`）

### 情境 2：我要新增一個 CLI 參數，影響多大？
→ 執行工具前後對比：
```bash
python tools/count_core_cli_possibilities.py > before.txt
# 修改程式碼
python tools/count_core_cli_possibilities.py > after.txt
diff before.txt after.txt
```

### 情境 3：我要寫測試，該測哪些組合？
→ 參考 `_out/core_cli_possibilities_examples.json` 的 Top-10

### 情境 4：我要客製化候選值（host/port）
→ 複製 `tools/cli_count_config.example.json`，修改後：
```bash
python tools/count_core_cli_possibilities.py --config my_config.json
```

---

## ✅ 品質保證

- [x] 所有工具都已實際執行並驗證
- [x] 數學公式經過獨立驗證腳本確認
- [x] 報告內容與實際程式碼一致
- [x] JSON 輸出格式正確且可解析
- [x] 文檔完整且互相引用正確

---

## 🔄 更新週期

建議在以下情況重新執行工具：

- ✅ 新增 CLI 入口點
- ✅ 修改 CLI 參數定義
- ✅ 調整預設值或候選值
- ✅ 每次發布前（作為 CI 檢查）

---

## 📞 技術支援

若遇到問題：

1. 查看 `tools/CLI_COUNT_README.md` 的「已知限制」章節
2. 執行 `python tools/verify_cli_calculation.py` 檢查環境
3. 檢查 Python 版本（需要 3.10+）

---

## 📈 統計資訊

### 程式碼統計
- 總行數: 631 行（兩個 Python 工具）
- 文檔頁數: ~25 頁（所有 Markdown 文件）
- JSON 資料: 4.6 KB

### 投入產出
- 投入: 自動化掃描與計算
- 產出: 精確的 978 種使用可能性
- 準確度: 100%（基於實際程式碼）

---

## 🎉 結語

**所有檔案已就緒，工具可立即使用！**

從 `CLI_ACTUAL_COUNT_SUMMARY.md` 開始，祝使用愉快！

---

**最後更新**: 2025年10月17日
