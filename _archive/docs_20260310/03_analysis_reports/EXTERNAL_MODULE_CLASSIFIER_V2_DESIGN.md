# 外部模組分類器 v2.0 - 數據驅動設計文檔

## 核心設計理念

**從數據動態提取模組信息，而非預先定義**

##設計原理

### 問題
原版本（v1.0）預先定義了所有模組：
```python
FUNCTION_MODULES = {
    "function_sqli": {...},
    "function_xss": {...},
    ...
}
```

這種方式的問題：
1. 每次新增模組都需要修改代碼
2. 模組定義與實際數據可能不同步
3. 無法自動發現新模組

### 解決方案

**從分類器輸入路徑中動態提取模組名稱**

```python
# 用戶執行：
python aiva_external_module_classifier_v2.py module_analysis/function_sqli/

# 分類器自動識別：
module_name = "function_sqli"
module_category = "features"
```

## 實現策略

### 1. 從 input_dir 提取模組名稱

```python
def __init__(self, input_dir: str, output_dir: str):
    self.input_dir = Path(input_dir)
    
    # 策略：從路徑中提取 function_* 或 *_engine 模式
    path_str = str(self.input_dir).replace('\\', '/')
    
    # 匹配 function_*
    if match := re.search(r'function_([a-z_0-9]+)', path_str):
        self.module_name = f"function_{match.group(1)}"
        self.module_category = "features"
    
    # 匹配 scan/*
    elif '/scan/' in path_str:
        if 'typescript' in path_str:
            self.module_name = "typescript_engine"
        # ...
        self.module_category = "scan"
```

### 2. 從模組名稱推斷屬性

使用模式匹配推斷攻擊類型：

```python
ATTACK_TYPE_PATTERNS = {
    "sqli": {"type": "injection", "category": "database_security"},
    "xss": {"type": "injection", "category": "web_security"},
    "ssrf": {"type": "ssrf", "category": "network_security"},
    # ...
}

def _infer_module_info(self, module_name: str):
    for pattern, info in ATTACK_TYPE_PATTERNS.items():
        if pattern in module_name.lower():
            return info
    
    # 未知模組返回通用描述
    return {'name': module_name.title(), 'type': 'unknown'}
```

### 3. 支援多種數據格式

不同語言分析工具產生不同格式：

| 語言 | 格式鍵 | 流程結構 |
|------|--------|----------|
| Python | `flow_chains` | `List[List[str]]` (路徑列表的列表) |
| Rust/Go/TS | `flows` | `List[Dict]` (流程對象列表) |

分類器自動適配：

```python
def load_flow_data(self):
    data = json.load(open(input_file))
    
    # 優先使用 flows
    if 'flows' in data:
        self.flows_data = data
    # 回退到 flow_chains
    elif 'flow_chains' in data:
        # 需要轉換格式（待實現）
        pass
```

## 數據位置對應

### 功能模組 (features)

```
module_analysis/function_*/
├── analysis_results.json    → function_sqli, function_xss, etc.
    
services/features/.../function_*/
├── analysis_output/analysis_results.json → function_authn_go
├── rust_core_analysis/analysis_results.json → function_crypto
```

### 掃描引擎 (scan)

```
services/scan/
├── typescript_engine/analysis_output/  → typescript_engine
├── rust_engine/rust_analysis_output/   → rust_engine
├── analysis_output/                    → scan_engine
```

## 使用方式

### 基本用法

```bash
# 分類 SQL 注入模組
python aiva_external_module_classifier_v2.py module_analysis/function_sqli

# 分類 TypeScript 掃描引擎
python aiva_external_module_classifier_v2.py services/scan/typescript_engine/analysis_output

# 指定輸出目錄
python aiva_external_module_classifier_v2.py module_analysis/function_xss -o custom_output
```

### 批次處理

```bash
# 分類所有功能模組
for dir in module_analysis/function_*/; do
    python aiva_external_module_classifier_v2.py "$dir"
done

# 分類所有掃描引擎
for dir in services/scan/*/analysis_output/; do
    python aiva_external_module_classifier_v2.py "$dir"
done
```

## 輸出報告

### 1. classification_summary.md

- 總體統計（總流程數、發現模組數）
- 發現的模組表格（含攻擊類型、支援語言）
- 攻擊類型分布
- 語言分布

### 2. complete_flow_details.md

- 按模組分組
- 按語言子分組
- 詳細流程路徑、入口點、CLI命令

### 3. classification_data.json

- 完整 JSON 數據
- 包含 discovered_modules 詳情
- 統計數據

## 優勢

### ✅ 自動發現模組
不需要預先定義，自動從數據中發現新模組

### ✅ 數據驅動
模組信息從實際數據路徑提取，保證一致性

### ✅ 易於擴展
新增模組無需修改代碼，只需放入正確目錄

### ✅ 多語言支援
統一處理 Python/Rust/Go/TypeScript 分析結果

## 與內部分類器的對比

| 特性 | 內部分類器 (aiva_flow_classifier.py) | 外部分類器 v2.0 |
|------|---------------------------------------|----------------|
| 目標模組 | AI Core 5大模組 | 功能模組 + 掃描引擎 |
| 模組定義 | 預先定義（MODULES字典） | 動態提取（從路徑） |
| 分類維度 | AI能力（AI內部/對外/程式） | 攻擊類型 + 語言 |
| 數據來源 | features_classification/ | module_analysis/ + services/ |
| 更新方式 | 修改代碼 | 添加數據目錄 |

## 後續改進

### 待實現功能

1. **flow_chains 轉換**
   - 將 Python 的 `flow_chains` 轉換為統一的 `flows` 格式
   - 提取文件路徑和函數名稱

2. **多層級模組**
   - 支援 `function_crypto/rust_core` 這種子模組
   - 顯示模組層級關係

3. **模組關聯分析**
   - 分析模組間的調用關係
   - 生成模組依賴圖

4. **CLI 整合**
   - 與 aiva_external_module_cli.py 整合
   - 自動生成執行命令

## 總結

v2.0 實現了真正的**數據驅動分類**：
- ✅ 不再需要預先定義模組列表
- ✅ 從實際數據路徑自動提取
- ✅ 支援動態發現新模組
- ✅ 保證數據與分類的一致性

這個設計完全符合您的要求：**純粹從數據資料來分類**！
