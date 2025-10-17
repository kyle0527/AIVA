# 核心模組 CLI 使用可能性計算報告

## 📊 執行摘要

### 證據來源
- **掃描範圍**: `services/core/**`
- **CLI 入口點數量**: **1**
- **唯一入口**: `services/core/aiva_core/ui_panel/auto_server.py`

### 實際計算結果

基於預設配置（5 個候選端口），計算結果如下：

```
總使用可能性 = 978 種
├── 下界（最小可區分）= 3 種
├── 有指定 --ports = 975 種
└── 無指定 --ports = 3 種
```

---

## 🔍 參數空間分析

### 1. `--mode` 參數
- **類型**: 枚舉（enum）
- **可選值**: `ui`, `ai`, `hybrid`
- **計數**: **3**
- **證據**: 
  ```python
  parser.add_argument('--mode', choices=['ui', 'ai', 'hybrid'])
  ```

### 2. `--host` 參數
- **類型**: 字串
- **預設值**: `127.0.0.1`
- **候選值**: `["127.0.0.1"]`（預設配置僅一個）
- **計數**: **1**
- **證據**:
  ```python
  parser.add_argument('--host', default='127.0.0.1')
  ```

### 3. `--ports` 參數
- **類型**: `list[int]`
- **屬性**: `nargs='+'`（一個或多個）
- **順序**: **有意義**（會依序嘗試端口）
- **可選**: **是**
- **候選集合**: `[3000, 8000, 8080, 8888, 9000]`
- **候選數量 (N)**: **5**
- **證據**:
  ```python
  parser.add_argument('--ports', nargs='+', type=int)
  ```

---

## 📐 計算公式

### Port Sequences 計算

由於 `--ports` 參數：
1. 接受一個或多個整數
2. 順序有意義（優先嘗試前面的端口）
3. 不重複（同一個端口不會列兩次）

因此，從 N=5 個候選端口中選取的所有可能序列數為：

$$
\sum_{k=1}^{N} P(N,k) = \sum_{k=1}^{5} \frac{5!}{(5-k)!}
$$

展開計算：
- k=1: P(5,1) = 5!/(5-1)! = 5
- k=2: P(5,2) = 5!/(5-2)! = 20
- k=3: P(5,3) = 5!/(5-3)! = 60
- k=4: P(5,4) = 5!/(5-4)! = 120
- k=5: P(5,5) = 5!/(5-5)! = 120

**總計**: 5 + 20 + 60 + 120 + 120 = **325 種序列**

### 總使用可能性計算

```
總數 = mode數量 × host數量 × (port序列數 + 不指定port的情況)
     = 3 × 1 × (325 + 1)
     = 3 × 326
     = 978
```

### 依模式分組

每種模式（ui / ai / hybrid）：
- 有指定 `--ports`: 1 (host) × 325 (port sequences) = **325**
- 無指定 `--ports`: 1 (host) × 1 (auto) = **1**
- **小計**: **326**

---

## 📋 Top-10 常用範例

### 最小組合（下界）

這些是最基本的使用方式，只變更 `--mode`，其他參數使用預設：

1. **UI 模式（預設）**
   ```bash
   python -m services.core.aiva_core.ui_panel.auto_server --mode ui
   ```

2. **AI 模式**
   ```bash
   python -m services.core.aiva_core.ui_panel.auto_server --mode ai
   ```

3. **混合模式**
   ```bash
   python -m services.core.aiva_core.ui_panel.auto_server --mode hybrid
   ```

### 指定單一偏好端口

4. **UI + 端口 3000**
   ```bash
   python -m services.core.aiva_core.ui_panel.auto_server --mode ui --ports 3000
   ```

5. **UI + 端口 8000**
   ```bash
   python -m services.core.aiva_core.ui_panel.auto_server --mode ui --ports 8000
   ```

6. **UI + 端口 8080**
   ```bash
   python -m services.core.aiva_core.ui_panel.auto_server --mode ui --ports 8080
   ```

### 多端口順序組合

10. **UI + 多端口順序**
    ```bash
    python -m services.core.aiva_core.ui_panel.auto_server --mode ui --ports 3000 8000 8080
    ```
    說明：依序嘗試 3000 → 8000 → 8080，若前者被佔用則嘗試下一個

---

## 🛠️ 工具使用

### 基本執行
```bash
python tools/count_core_cli_possibilities.py
```

### 使用自訂配置
```bash
python tools/count_core_cli_possibilities.py --config tools/cli_count_config.json
```

### 指定輸出路徑
```bash
python tools/count_core_cli_possibilities.py --output _out/my_report.json
```

### 生成更多範例
```bash
python tools/count_core_cli_possibilities.py --examples 20
```

---

## 📝 配置檔格式

建立 `cli_count_config.json`：

```json
{
  "host_candidates": [
    "127.0.0.1",
    "0.0.0.0",
    "localhost"
  ],
  "port_candidates": [
    3000,
    8000,
    8080,
    8888,
    9000
  ],
  "scope": "services/core/**"
}
```

### 配置影響

若將 `host_candidates` 增加到 3 個，`port_candidates` 保持 5 個：

```
新總數 = 3 (modes) × 3 (hosts) × (325 (port sequences) + 1 (auto))
       = 3 × 3 × 326
       = 2,934 種可能
```

---

## 🔬 驗證與邊界條件

### 保守性原則

1. **範圍限定**: 僅掃描 `services/core/**`，不包含其他模組
2. **實際證據**: 所有計算基於實際程式碼，而非假設
3. **未來擴展**: 若新增 CLI 入口點，重新執行工具即可更新

### 已知限制

1. **佔位符號**: 某些檔案（如 `ai_commander.py`）包含 `...` 佔位，表示內容未完整公開
2. **動態參數**: 若未來參數可動態擴展（如從設定檔讀取），需更新計算邏輯
3. **互斥組**: 目前無參數互斥組（mutually exclusive groups）

---

## 🎯 實務應用

### CI/CD 整合

在 `.github/workflows` 或 CI 腳本中：

```yaml
- name: Count CLI Possibilities
  run: |
    python tools/count_core_cli_possibilities.py
    cat _out/core_cli_possibilities.json
```

### 文檔生成

可以將輸出的 JSON 轉換為：
- Markdown 表格
- Mermaid 圖表
- 互動式 HTML 文檔

### 測試覆蓋

根據計算結果，設計測試案例：
- 優先測試前 10 個常用組合
- 確保邊界情況（單端口、多端口、無端口）
- 驗證所有三種模式

---

## 📈 擴展建議

### 1. 加入其他 CLI 入口點

若在 `services/core/**` 下新增其他 CLI 腳本：

```python
# 在 CLIAnalyzer 類別中新增方法
def analyze_new_cli(self) -> dict[str, Any]:
    # 分析新 CLI 的參數空間
    pass

# 在 analyze_all 中調用
cli_entries.append(self.analyze_new_cli())
```

### 2. 支援參數依賴關係

若某些參數組合無效（如 `--mode ai` 時不支援某些 port）：

```python
def validate_combination(mode, host, ports) -> bool:
    # 實作驗證邏輯
    if mode == "ai" and ports and 3000 in ports:
        return False  # AI 模式不支援 3000 端口
    return True
```

### 3. 加入使用頻率權重

基於日誌或使用統計，為組合賦予權重：

```json
{
  "command": "--mode ui",
  "probability": 0.45,
  "rank": 1
}
```

---

## 📚 相關檔案

- **工具腳本**: `tools/count_core_cli_possibilities.py`
- **設定範例**: `tools/cli_count_config.example.json`
- **輸出報告**: `_out/core_cli_possibilities.json`
- **使用範例**: `_out/core_cli_possibilities_examples.json`
- **CLI 入口**: `services/core/aiva_core/ui_panel/auto_server.py`

---

## ✅ 結論

1. **精準計算**: 基於實際程式碼，非估算
2. **可追蹤**: 所有數字都有數學公式和證據支持
3. **可擴展**: 工具設計支援未來新增 CLI 入口點
4. **CI 友好**: 可整合到自動化流程中持續追蹤

**核心模組當前 CLI 使用可能性總數**: **978 種**
