# AIVA Internal Executor 執行驗證報告
生成時間: 2026-02-01

## 一、執行器狀態總結

### ✅ 成功修復的功能
1. **數據讀取**: 正確載入 `internal_classification.json` (364 flows)
2. **Merged Flow 處理**: 成功跳過 merged_into 記錄，避免 KeyError
3. **安全字段訪問**: 所有 7 個關鍵位置使用 `.get()` 防禦性編程
4. **模組載入**: 成功動態導入 Python 模組
5. **錯誤處理**: 容錯機制運作正常，能優雅處理錯誤

### ⚠️ 發現的問題

## 二、問題1: 類別命名轉換規則不完善

### 問題描述
執行器使用 `_snake_to_camel()` 方法將文件名轉換為類別名，但規則過於簡單：

```python
# 當前實現（第 375 行）
def _snake_to_camel(self, snake_str: str) -> str:
    return ''.join(x.title() for x in snake_str.split('_'))
```

### 實際案例

| 文件名 | 執行器猜測 | 實際類別名 | 結果 |
|-------|----------|----------|-----|
| `analyze_dataflow_breakpoints.py` | `AnalyzeDataflowBreakpoints` | `DataFlowBreakpointAnalyzer` | ❌ 找不到 |
| `event_listener.py` | `EventListener` | `ExternalLearningListener` | ❌ 找不到 |
| `sync_experiences.py` | `SyncExperiences` | 無類別（只有函數） | ❌ 找不到 |

### 發生原因
- Python 類別命名沒有強制規範
- 開發者可能使用不同命名風格：
  - 文件名基礎: `EventListener` ✅
  - 功能描述: `DataFlowBreakpointAnalyzer` ❌
  - 更具體名稱: `ExternalLearningListener` ❌
  - 無類別設計: 純函數模組 ❌

### 當前容錯機制
```python
# 第 598-602 行
if not hasattr(module, class_name):
    # 容錯:搜尋模組內定義的任何類別
    classes = [m[1] for m in inspect.getmembers(module, inspect.isclass) 
               if m[1].__module__ == module.__name__]
    if classes:
        cls = classes[0]  # 使用第一個找到的類別
```

**問題**: 選擇第一個類別可能不正確（如 Flow 4 選到 `BreakpointIssue` 數據類而非 `DataFlowBreakpointAnalyzer` 執行類）

## 三、問題2: 缺少執行入口方法

### 問題描述
執行器尋找入口方法的優先順序:
```python
# 第 392-402 行
priority_methods = ['execute', 'process', 'run', 'analyze', 'start']

if step_index == 0:
    priority_methods = ['train', 'generate', 'load', 'initialize'] + priority_methods
```

### 實際情況統計

檢查了 10 個模組，結果如下:

| 模組類型 | 數量 | 入口方法情況 |
|---------|-----|------------|
| 有完整執行入口 | 2 | `run()`, `execute()` 等標準方法 |
| 有特殊執行入口 | 3 | `run_full_analysis()`, `start_listening()` 等 |
| 只有數據類別 | 2 | `BreakpointIssue`, `ExecutionContext` (無執行邏輯) |
| 純函數模組 | 3 | 無類別，只有頂層函數 |

### 具體案例

#### Flow 4: analyze_dataflow_breakpoints
**文件**: `analyze_dataflow_breakpoints.py`

**包含的類別**:
1. `BreakpointIssue` - 數據類（@dataclass）
   - 無執行方法
   - 用途: 存儲分析結果
   
2. `DataFlowBreakpointAnalyzer` - 執行類 ✅
   - 有執行方法: `run_full_analysis()` 
   - 問題: 方法名不在優先列表中

**執行器行為**:
1. 猜測類別名: `AnalyzeDataflowBreakpoints` ❌
2. 找不到，啟動容錯機制
3. 找到第一個類別: `BreakpointIssue` ❌ (錯誤選擇)
4. 嘗試實例化: 失敗（需要4個必填參數）

#### Flow 2: event_listener
**文件**: `event_listener.py`

**包含的類別**:
1. `ExternalLearningListener` - 執行類 ✅
   - 有執行方法: `start_listening()`
   - 問題: 
     - 類別名與文件名不匹配
     - 方法名不在優先列表中
     - 方法是阻塞式同步監聽（不適合流程中間步驟）

#### Flow 11: sync_experiences
**文件**: `sync_experiences.py`

**結構**: 純函數模組
```python
async def sync_experiences_to_vector_store(...):
    """主函數"""
    pass
```

**問題**: 
- 無類別定義
- 無法使用當前執行器架構執行

## 四、問題3: 深層依賴缺失

### 錯誤信息
```
[Import Error] 導入模組失敗: No module named 'services.integration.capability'
```

### 根本原因

#### 1. 導入路徑不一致
檢查代碼庫後發現多種導入風格混用:

```python
# 風格1: 相對導入（推薦）✅
from ...service_backbone.messaging.message_broker import MessageBroker

# 風格2: aiva_core 根導入 ✅
from aiva_core.cognitive_core.learning_system import xxx

# 風格3: services 全路徑導入（問題來源）❌
from services.integration.capability import CapabilityManager
```

**問題**: 風格3 需要 PYTHONPATH 包含項目根目錄 `C:\D\fold7\AIVA-git`

#### 2. PYTHONPATH 設置對比

**批次腳本設置** (執行Flow.bat):
```batch
set PYTHONPATH=C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services
```

**實際測試設置**:
```powershell
$env:PYTHONPATH="C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services;..."
```

**分析**: 
- 批次腳本缺少根目錄，會導致相同錯誤
- 需要添加項目根目錄支持 `services.xxx` 導入風格

#### 3. 缺少的模組文件

搜尋結果:
```bash
services/integration/capability/  # 此目錄不存在
```

**可能原因**:
- 模組已遷移/重命名
- 代碼未同步更新導入語句
- Git 忽略了某些文件

### 受影響的模組
至少包含以下文件:
- `event_listener.py`
- `analyze_dataflow_breakpoints.py`
- 其他未測試的模組（可能更多）

## 五、依賴檢查結果

### Python 包檢查
```bash
# 核心依賴已安裝 ✅
fastapi, uvicorn, httpx, pydantic - 已就緒
```

### 項目內部模組檢查

| 模組 | 狀態 | 說明 |
|-----|------|------|
| `aiva_common` | ✅ | 存在於 `services/common/aiva_common` |
| `aiva_core` | ✅ | 存在於 `services/core/aiva_core` |
| `services.integration` | ⚠️ | 存在，但子模組結構不完整 |
| `services.integration.capability` | ❌ | 不存在 |

## 六、解決方案建議

### 短期修復（立即可用）

#### 1. 更新批次腳本 PYTHONPATH
```batch
REM 修改所有 .bat 文件
set PYTHONPATH=C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services;C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services\integration
```

#### 2. 修復或移除 services.integration.capability 導入
搜尋並替換所有使用該模組的地方:
```python
# 舊導入（失敗）
from services.integration.capability import CapabilityManager

# 可能的修復:
# 方案A: 使用相對導入
from ...integration.capability import CapabilityManager

# 方案B: 移除該依賴（如果不需要）
# from services.integration.capability import CapabilityManager
```

#### 3. 改進執行器類別查找邏輯
```python
def _find_best_class(self, module, expected_name: str):
    """智能查找最佳類別"""
    # 1. 嘗試預期名稱
    if hasattr(module, expected_name):
        return getattr(module, expected_name)
    
    # 2. 查找所有定義的類別
    classes = [m for m in inspect.getmembers(module, inspect.isclass) 
               if m[1].__module__ == module.__name__]
    
    if not classes:
        return None
    
    # 3. 優先選擇有執行方法的類別
    for name, cls in classes:
        methods = [m for m, _ in inspect.getmembers(cls, predicate=inspect.ismethod)]
        if any(m in ['execute', 'run', 'process', 'analyze'] for m in methods):
            return cls
    
    # 4. 過濾掉數據類（@dataclass 或無方法）
    for name, cls in classes:
        # 跳過數據類
        if hasattr(cls, '__dataclass_fields__'):
            continue
        # 選擇第一個非數據類
        return cls
    
    # 5. 最後才返回第一個
    return classes[0][1]
```

### 中期改進（需要重構）

#### 1. 在分類數據中添加類別信息
修改 `aiva_internal_classifier.py` 添加元數據:
```python
flow_info = {
    "id": flow_id,
    "path": ["script1", "script2"],
    "full_paths": ["path/to/script1.py", "path/to/script2.py"],
    "classes": ["Script1Class", "Script2Class"],  # 新增
    "methods": ["execute", "run"],  # 新增
}
```

#### 2. 標準化內部模組結構
為所有可執行模組添加統一接口:
```python
class ExecutableModule(ABC):
    @abstractmethod
    def execute(self, context: Any) -> Any:
        """統一執行入口"""
        pass
```

### 長期規劃（架構層面）

#### 1. 統一導入風格
全局搜尋替換，統一使用相對導入或 aiva_core 根導入

#### 2. 完善分類器
- AST 分析時提取類別名和方法名
- 生成完整的可執行性元數據
- 標記哪些 flow 可以實際執行

#### 3. 添加執行器測試套件
為每個模組類型創建測試:
- 數據類模組 - 跳過執行
- 函數模組 - 特殊處理
- 執行類模組 - 正常執行

## 七、當前執行器能力評估

### 已驗證功能 ✅
1. **列表顯示**: `--list` 正常顯示 203 個 flow
2. **Dry-Run**: `--dry-run` 正確生成執行計畫
3. **模組載入**: 能成功導入 Python 模組
4. **錯誤處理**: 優雅處理各種異常情況
5. **數據結構**: 正確處理 merged flows

### 限制與問題 ⚠️
1. **類別名猜測準確率**: ~30% (基於簡單命名轉換)
2. **入口方法查找**: 無法識別非標準方法名 (`run_full_analysis` 等)
3. **依賴問題**: 約 50% 模組有導入錯誤
4. **純函數模組**: 無法執行（需要類別架構）
5. **數據類誤判**: 可能選擇錯誤的類別

### 預估成功執行率
基於當前問題分析:
- 理論可執行 flow: 203
- 有類別且有標準方法: ~60 (30%)
- 依賴正常: ~30 (15%)
- **實際可執行**: 約 20-30 flows (10-15%)

## 八、建議的驗證步驟

1. **修復 PYTHONPATH** - 立即修復所有 .bat 文件
2. **定位並修復依賴** - 搜尋 `services.integration.capability` 所有引用
3. **改進類別查找** - 實現智能類別選擇邏輯
4. **批量測試** - 自動測試所有 203 flows 的 dry-run
5. **生成報告** - 統計哪些可執行，哪些需要修復

## 九、結論

執行器的核心架構和錯誤處理機制 **運作正常** ✅，但存在以下阻礙實際執行的問題:

1. **高優先級**: 依賴路徑問題（阻止 ~50% flows 執行）
2. **中優先級**: 類別查找邏輯不完善（影響準確率）
3. **低優先級**: 入口方法查找可擴展性

建議先解決依賴問題，然後改進類別查找，最後考慮架構重構。
