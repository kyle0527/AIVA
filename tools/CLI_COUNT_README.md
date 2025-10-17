# 核心模組 CLI 使用可能性計算工具

## 📦 專案概述

這個工具集用於**精確計算**核心模組（`services/core/**`）中所有 CLI 入口點的使用可能性數量。基於實際程式碼分析，提供機器可讀的統計報告。

---

## 🎯 計算結果摘要

### 核心發現

- **CLI 入口點數量**: 1 個
- **唯一入口**: `services/core/aiva_core/ui_panel/auto_server.py`
- **總使用可能性**: **978 種**（基於預設配置）

### 參數空間

| 參數 | 類型 | 可選值/候選 | 數量 |
|------|------|-------------|------|
| `--mode` | 枚舉 | `ui`, `ai`, `hybrid` | 3 |
| `--host` | 字串 | `127.0.0.1` | 1 |
| `--ports` | 列表（有序） | `[3000, 8000, 8080, 8888, 9000]` | 325 種序列 |

### 計算公式

```
總數 = 3 (modes) × 1 (hosts) × (325 (port sequences) + 1 (auto))
     = 978
```

---

## 📁 產出檔案

### 1. 主工具：計算器
- **檔案**: `tools/count_core_cli_possibilities.py`
- **功能**: 掃描核心模組，分析參數空間，計算使用可能性
- **輸出**: JSON 格式報告 + 命令列摘要

### 2. 驗證工具
- **檔案**: `tools/verify_cli_calculation.py`
- **功能**: 驗證數學公式正確性，列舉樣本組合
- **用途**: 確保計算邏輯無誤

### 3. 設定檔範例
- **檔案**: `tools/cli_count_config.example.json`
- **內容**: 主機和端口候選集合的配置範例
- **用途**: 自訂參數候選值

### 4. 輸出報告
- **主報告**: `_out/core_cli_possibilities.json`（機器可讀）
- **範例集**: `_out/core_cli_possibilities_examples.json`（Top-K 常用組合）
- **詳細文檔**: `_out/CORE_CLI_POSSIBILITIES_REPORT.md`（人類可讀）

---

## 🚀 快速開始

### 基本使用

```bash
# 使用預設配置執行
python tools/count_core_cli_possibilities.py

# 查看報告
cat _out/core_cli_possibilities.json

# 驗證計算
python tools/verify_cli_calculation.py
```

### 使用自訂配置

1. 建立配置檔 `my_config.json`：
   ```json
   {
     "host_candidates": ["127.0.0.1", "0.0.0.0"],
     "port_candidates": [3000, 8080, 9000]
   }
   ```

2. 執行計算：
   ```bash
   python tools/count_core_cli_possibilities.py --config my_config.json
   ```

### 生成更多範例

```bash
# 生成 Top-20 常用組合
python tools/count_core_cli_possibilities.py --examples 20
```

---

## 📊 計算邏輯說明

### Port Sequences 計算

由於 `--ports` 參數：
- 接受 1 個或多個整數
- **順序有意義**（會依序嘗試端口）
- 不重複（同一個端口不會列兩次）

從 N=5 個候選端口中的所有可能序列數：

$$
\sum_{k=1}^{5} P(5,k) = \sum_{k=1}^{5} \frac{5!}{(5-k)!} = 5 + 20 + 60 + 120 + 120 = 325
$$

### 總數分解

```
978 = 3 (modes) × 326 (每種模式的組合)

每種模式的 326 種組合：
  - 325 種：指定不同的 --ports 序列
  - 1 種：不指定 --ports（自動選擇）
```

---

## 🔍 實際範例

### 下界（最小可區分）

最簡單的 3 種使用方式：

```bash
python -m services.core.aiva_core.ui_panel.auto_server --mode ui
python -m services.core.aiva_core.ui_panel.auto_server --mode ai
python -m services.core.aiva_core.ui_panel.auto_server --mode hybrid
```

### 指定單一端口

```bash
python -m services.core.aiva_core.ui_panel.auto_server --mode ui --ports 8080
```

### 指定多端口順序

```bash
python -m services.core.aiva_core.ui_panel.auto_server --mode ui --ports 3000 8000 8080
```

說明：優先嘗試 3000，若被佔用則嘗試 8000，再不行則 8080。

---

## 📈 參數影響分析

不同配置對總數的影響：

| 配置 | 主機數 | 端口數 | 端口序列 | 總計 |
|------|--------|--------|----------|------|
| 預設 | 1 | 5 | 325 | **978** |
| 增加主機 | 3 | 5 | 325 | **2,934** |
| 減少端口 | 1 | 3 | 15 | **48** |
| 增加端口 | 1 | 7 | 13,699 | **41,100** |

---

## 🛠️ 命令列選項

### count_core_cli_possibilities.py

```bash
python tools/count_core_cli_possibilities.py [OPTIONS]

選項:
  --config PATH            設定檔路徑（JSON 格式）
  --output PATH            輸出報告路徑（預設: _out/core_cli_possibilities.json）
  --examples N             生成 Top-N 範例（預設: 10）
  --examples-output PATH   範例輸出路徑
```

### 範例

```bash
# 使用自訂配置和輸出路徑
python tools/count_core_cli_possibilities.py \
  --config my_config.json \
  --output reports/cli_stats.json \
  --examples 20
```

---

## 🔬 驗證與測試

### 執行驗證

```bash
python tools/verify_cli_calculation.py
```

驗證項目：
1. ✓ 排列公式正確性
2. ✓ 樣本組合列舉
3. ✓ 總數計算邏輯
4. ✓ 參數縮放影響

### 輸出示例

```
驗證排列公式 P(n,k) = n! / (n-k)!
  k=1: P(5,1) = 5
  k=2: P(5,2) = 20
  k=3: P(5,3) = 60
  k=4: P(5,4) = 120
  k=5: P(5,5) = 120
  總計: 325

✓ 驗證通過：總使用可能性 = 978 種
```

---

## 🎓 數學背景

### 排列 vs 組合

**為什麼用排列（Permutation）而非組合（Combination）？**

因為 `--ports` 參數的順序會影響行為：
- `--ports 3000 8000`：優先 3000，次選 8000
- `--ports 8000 3000`：優先 8000，次選 3000

這兩種是**不同的使用方式**，因此需要用排列計算。

### 公式推導

從 N 個元素中選取 k 個元素的有序排列數：

$$
P(N, k) = \frac{N!}{(N-k)!}
$$

所有非空序列總數：

$$
\sum_{k=1}^{N} P(N, k) = \sum_{k=1}^{N} \frac{N!}{(N-k)!}
$$

---

## 📚 相關文檔

- [詳細報告](_out/CORE_CLI_POSSIBILITIES_REPORT.md)
- [機器可讀報告](_out/core_cli_possibilities.json)
- [使用範例](_out/core_cli_possibilities_examples.json)
- [CLI 入口點原始碼](services/core/aiva_core/ui_panel/auto_server.py)

---

## 🔄 CI/CD 整合

### GitHub Actions 範例

```yaml
name: Count CLI Possibilities

on: [push, pull_request]

jobs:
  count-cli:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Count CLI possibilities
        run: |
          python tools/count_core_cli_possibilities.py
          python tools/verify_cli_calculation.py
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: cli-reports
          path: _out/core_cli_*
```

---

## 🚧 已知限制與未來改進

### 已知限制

1. **靜態分析**: 基於程式碼文本分析，不執行動態追蹤
2. **單一入口點**: 目前核心模組只有一個 CLI 入口
3. **佔位符號**: 某些檔案包含 `...` 佔位，可能有未公開的 CLI

### 未來改進

1. [ ] 支援參數依賴關係（某些組合無效）
2. [ ] 整合使用日誌分析（實際使用頻率）
3. [ ] 自動生成測試案例
4. [ ] 支援更多 CLI 框架（click, typer）
5. [ ] 生成互動式文檔（HTML/Mermaid）

---

## 📝 版本歷史

### v1.0.0 (2025-10-17)
- ✨ 初始版本
- ✨ 支援 auto_server.py 分析
- ✨ 完整數學驗證
- ✨ 機器可讀 JSON 輸出
- ✨ Top-K 範例生成

---

## 🙏 貢獻指南

若要新增其他 CLI 入口點的分析：

1. 在 `CLIAnalyzer` 類別中新增分析方法
2. 在 `analyze_all()` 中調用
3. 更新測試和驗證腳本
4. 執行驗證確保正確性

---

## 📄 授權

與主專案相同。

---

## ✨ 結論

這個工具集提供了**精準、可追蹤、可擴展**的核心模組 CLI 使用可能性計算。

- ✅ 所有數字都有數學證明
- ✅ 基於實際程式碼，非估算
- ✅ 機器可讀，易於 CI 整合
- ✅ 完整驗證，確保正確性

**核心模組當前 CLI 使用可能性總數：978 種**
