# 🌐 AIVA 跨語言模式同步修復指南

## 📋 目標與範圍

本指南基於 AIVA Common 架構原則，提供跨語言模式同步問題的系統性修復方法。整合了實際修復經驗（Python、Rust、Go），並遵循 AIVA 的四層優先級架構設計。

### 🎯 核心目標
- **統一性**: 確保跨語言模式定義的一致性
- **合規性**: 符合 AIVA Common SOT (Single Source of Truth) 原則
- **安全性**: 遵循批量處理安全協議
- **可維護性**: 建立可追蹤的同步修復流程

### 🔍 適用範圍
- Schema 跨語言轉換問題
- 模組依賴路徑不一致
- 命名規範差異修復
- 類型系統映射問題

---

## 🏗️ AIVA 架構整合原則

### 📊 四層優先級系統

遵循 AIVA Common 的四層優先級架構：

```
1️⃣ 國際標準 (International Standards)
   └─ SARIF, CVSS, OpenAPI 等國際標準
   
2️⃣ 語言標準 (Language Standards)  
   └─ Python PEP, Rust RFC, Go Spec 等官方標準
   
3️⃣ aiva_common 規範 (AIVA Common Standards)
   └─ 項目通用枚舉、Schema、設計原則
   
4️⃣ 模組專屬規範 (Module-specific Standards)
   └─ 各服務模組的特殊需求
```

### 🌐 網路搜索新類型驗證原則

⚠️ **重要新增原則**: 當發現新的類型或命名問題時，**必須先搜索網路**，確認是否有官方標準或規範：

1. **搜索官方文檔**: 查找語言官方標準（如 Go Wiki, PEPs, RFCs）
2. **確認標準存在性**: 驗證是否為官方定義或認可的標準
3. **無標準時新增**: 如果官方沒有定義，則按照 AIVA 規範新增
4. **有標準時遵循**: 如果官方有標準，嚴格按照官方標準執行

**範例**: Go Initialisms 標準
- 🔍 **網路確認**: `https://go.dev/wiki/CodeReviewComments#initialisms`

---

## 🐍 Python 環境配置與導入原則

### ⚠️ **重要發現**: 直接使用現實路徑優於虛擬環境

基於實際修復經驗，發現以下關鍵原則：

#### 🎯 **現實vs虛擬環境選擇策略**

**✅ 推薦方案: 直接使用現實路徑**
```python
import sys
import os

# 1. 添加services路徑到Python path
sys.path.insert(0, r'C:\D\fold7\AIVA-git\services')

# 2. 設置必要的環境變數（從.env文件獲取）
os.environ['AIVA_RABBITMQ_URL'] = 'amqp://guest:guest@localhost:5672/'
os.environ['AIVA_ENVIRONMENT'] = 'development'

# 3. 直接導入aiva_common
import aiva_common
from aiva_common.schemas import FindingPayload
from aiva_common.enums import Severity, Confidence
```

**❌ 虛擬環境可能出現的問題**
- `pip install -e` editable install 路徑映射失敗
- `MAPPING = {}` 空字典導致模組找不到
- 複雜的 `.pth` 文件和 finder 機制故障

#### 📋 **環境變數配置原則**

**遵循現有配置，絕對不自創**：
1. **讀取現有.env**: 使用項目根目錄的 `.env` 文件
2. **必要變數列表**: 
   ```bash
   AIVA_RABBITMQ_URL=amqp://guest:guest@localhost:5672/
   AIVA_ENVIRONMENT=development
   AIVA_DATABASE_URL=postgresql://postgres:aiva123@localhost:5432/aiva_db
   ```
3. **編碼注意**: `.env` 文件使用UTF-8編碼，需要適當處理
4. **不創建新變數**: 所有環境變數必須來自現有配置文件

#### 🔧 **導入問題修復模式**

**階段1: 環境準備**
```python
# 標準化的環境設置腳本
def setup_aiva_environment():
    """設置AIVA開發環境 - 基於實際修復經驗"""
    import sys
    import os
    from pathlib import Path
    
    # 1. 項目根路徑
    project_root = Path(__file__).resolve().parents[2]  # 根據實際層級調整
    services_path = project_root / "services"
    
    # 2. 添加到Python路徑
    if str(services_path) not in sys.path:
        sys.path.insert(0, str(services_path))
        print(f"✅ Added to Python path: {services_path}")
    
    # 3. 讀取現有環境變數（不自創）
    env_file = project_root / ".env" 
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("✅ Environment variables loaded from .env")
    
    # 4. 設置必要的預設值（僅當不存在時）
    default_vars = {
        'AIVA_ENVIRONMENT': 'development',
        'AIVA_RABBITMQ_URL': 'amqp://guest:guest@localhost:5672/'
    }
    
    for key, value in default_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"✅ Set default: {key}")
    
    return True

# 使用方式
if __name__ == "__main__":
    setup_aiva_environment()
    
    # 現在可以安全導入aiva_common
    import aiva_common
    from aiva_common.schemas import FindingPayload
    from aiva_common.enums import Severity, Confidence
```

**階段2: 導入驗證**
```python
# 設置路徑和環境變數
sys.path.insert(0, 'services目錄的絕對路徑')
# 從.env載入必要變數
```

**階段2: 逐步導入測試**
```python
try:
    import aiva_common  # 基礎模組
    from aiva_common.enums import Severity  # 枚舉
    from aiva_common.schemas import FindingPayload  # Schema
except ImportError as e:
    # 記錄具體的導入失敗原因
    print(f"Import failed: {e}")
```

**階段3: 問題分類修復**
- **Missing imports**: 添加缺失的導入語句
- **Path resolution**: 修復相對路徑為絕對路徑
- **Type annotation**: 解決類型標註問題
- ✅ **官方標準**: `TaskId` → `TaskID`, `FindingId` → `FindingID`
- 🚀 **執行修復**: 按照官方標準進行修復

### 🔄 SOT (Single Source of Truth) 實現

```mermaid
graph TD
    A[aiva_common/schemas] --> B[Python 實現]
    A --> C[Rust 實現] 
    A --> D[Go 實現]
    
    B --> E[services/core]
    B --> F[services/features]
    C --> G[services/scan/info_gatherer_rust]
    D --> H[services/features/function_authn_go]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
```

---

## 🎯 **系統性錯誤修復方法論** (實戰總結)

### 📊 **錯誤分析與分類框架**

基於實際修復 ~500 個錯誤的經驗，建立以下系統性方法論：

#### **第一階段：錯誤盤點與分類**
```bash
# 1. 獲取完整錯誤清單
get_errors() | tee error_inventory.log

# 2. 按錯誤類型分類
grep "Pydantic" error_inventory.log | wc -l    # 配置錯誤
grep "無法解析匯入" error_inventory.log | wc -l  # 導入錯誤
grep "類型部分未知" error_inventory.log | wc -l  # 類型推斷錯誤
grep "覆寫類別" error_inventory.log | wc -l     # 繼承/接口錯誤

# 3. 識別錯誤集中的文件
awk '{print $2}' error_inventory.log | sort | uniq -c | sort -nr
```

#### **第二階段：優先級排序原則**
1. **🔥 高優先級** - 阻塞性錯誤 (編譯/導入失敗)
2. **⚡ 中優先級** - 類型安全錯誤 (運行時可能出錯)  
3. **📝 低優先級** - 代碼風格錯誤 (不影響功能)

#### **第三階段：批量修復策略**
- **按文件分組**: 同一文件的錯誤一起修復
- **按類型分組**: 相同類型的錯誤使用統一修復模式
- **逐步驗證**: 每修復一類錯誤就運行驗證

### 🔄 **五步驟修復循環**

```mermaid
graph LR
    A[1.錯誤識別] --> B[2.根因分析]
    B --> C[3.修復實施]
    C --> D[4.驗證測試]
    D --> E[5.文檔更新]
    E --> A
    
    style A fill:#ffcdd2
    style B fill:#fff3e0
    style C fill:#e8f5e8
    style D fill:#e3f2fd
    style E fill:#f3e5f5
```

#### **步驟詳解與實戰經驗**

**1️⃣ 錯誤識別 (Error Identification)**
```python
# 使用系統性錯誤收集
def collect_error_inventory():
    """收集並分類所有錯誤"""
    errors = get_errors()
    
    categories = {
        'pydantic_config': [],
        'schema_mismatch': [],
        'type_inference': [],
        'async_context': [],
        'import_resolution': []
    }
    
    for error in errors:
        if 'Field' in error.message and 'env' in error.message:
            categories['pydantic_config'].append(error)
        elif 'attribute' in error.message.lower():
            categories['schema_mismatch'].append(error)
        # ... 其他分類邏輯
    
    return categories
```

**2️⃣ 根因分析 (Root Cause Analysis)**
```python
# 針對 Schema 屬性不匹配的根因分析範例
def analyze_schema_mismatch(error):
    """分析 Schema 屬性不匹配的根本原因"""
    
    # 檢查實際 Schema 定義
    actual_schema = find_schema_definition(error.file_path)
    
    # 檢查代碼中的期望用法
    expected_usage = extract_attribute_usage(error.file_path, error.line)
    
    # 對比差異
    missing_attributes = expected_usage - actual_schema.attributes
    
    return {
        'cause': 'Schema definition mismatch',
        'missing': missing_attributes,
        'solution': 'Update attribute access to match actual schema'
    }
```

**3️⃣ 修復實施 (Fix Implementation)**
```python
# 批量修復模式範例
def batch_fix_pydantic_fields(file_paths):
    """批量修復 Pydantic Field 配置錯誤"""
    
    fixes_applied = 0
    
    for file_path in file_paths:
        # 讀取文件內容
        content = read_file(file_path)
        
        # 應用修復模式
        # 移除過時的 env 參數
        content = re.sub(
            r'Field\([^)]*env=["\']\w+["\']\s*,?\s*\)',
            'Field(',
            content
        )
        
        # 修復 default_factory 語法
        content = re.sub(
            r'default_factory=(\w+)',
            r'default_factory=lambda: \1()',
            content
        )
        
        # 寫回文件
        write_file(file_path, content)
        fixes_applied += 1
    
    return fixes_applied
```

**4️⃣ 驗證測試 (Validation & Testing)**
```python
# 修復後的系統性驗證
def validate_fixes():
    """系統性驗證修復結果"""
    
    validation_results = {}
    
    # 1. 語法驗證
    try:
        import ast
        for py_file in get_python_files():
            with open(py_file) as f:
                ast.parse(f.read())
        validation_results['syntax'] = 'PASS'
    except SyntaxError as e:
        validation_results['syntax'] = f'FAIL: {e}'
    
    # 2. 導入驗證
    try:
        import aiva_common
        from aiva_common.schemas import EnhancedFindingPayload
        from aiva_common.config import Settings
        validation_results['imports'] = 'PASS'
    except ImportError as e:
        validation_results['imports'] = f'FAIL: {e}'
    
    # 3. 類型檢查驗證
    remaining_errors = len(get_errors())
    validation_results['type_check'] = f'{remaining_errors} errors remaining'
    
    return validation_results
```

**5️⃣ 文檔更新 (Documentation Update)**
```python
# 自動生成修復報告
def generate_fix_report(before_errors, after_errors, fixes_applied):
    """生成修復成果報告"""
    
    report = f"""
## 修復成果報告 - {datetime.now().strftime('%Y年%m月%d日')}

### 📊 數據統計
- **修復前錯誤**: {len(before_errors)} 個
- **修復後錯誤**: {len(after_errors)} 個  
- **修復率**: {((len(before_errors) - len(after_errors)) / len(before_errors) * 100):.1f}%
- **應用修復**: {len(fixes_applied)} 個

### 🔧 修復分類
"""
    
    for category, fixes in fixes_applied.items():
        report += f"- **{category}**: {len(fixes)} 個修復\n"
    
    return report
```

### 📈 **修復效率優化技巧**

#### **並行修復策略**
1. **按模組並行**: 不同模組可以同時修復
2. **按類型串行**: 同類型錯誤需要串行修復避免衝突
3. **增量驗證**: 每修復一批立即驗證，避免積累錯誤

#### **修復模式複用**
```python
# 建立修復模式庫
REPAIR_PATTERNS = {
    'pydantic_v2_field': {
        'pattern': r'Field\([^)]*env=["\']\w+["\']\s*,?\s*\)',
        'replacement': 'Field(',
        'description': 'Remove deprecated env parameter from Pydantic Field'
    },
    
    'schema_attribute_mapping': {
        'mappings': {
            'vulnerability.cwe.cwe_id': 'vulnerability.vulnerability_id',
            'vulnerability.name.value': 'vulnerability.title',
            'vulnerability.confidence.value': 'vulnerability.ai_confidence'
        },
        'description': 'Fix EnhancedVulnerability attribute access'
    }
}

def apply_repair_pattern(content, pattern_name):
    """應用預定義的修復模式"""
    pattern = REPAIR_PATTERNS[pattern_name]
    
    if 'pattern' in pattern:
        return re.sub(pattern['pattern'], pattern['replacement'], content)
    elif 'mappings' in pattern:
        for old_attr, new_attr in pattern['mappings'].items():
            content = content.replace(old_attr, new_attr)
        return content
```

---

## 🛠️ 通用修復方法論

### 🔄 六階段修復流程

#### 1️⃣ 架構合規檢查 (Architecture Compliance)
```bash
# 檢查是否遵循四層優先級
grep -r "from aiva_common" services/*/
grep -r "schemas/generated" services/*/

# 驗證 SOT 原則
find services/ -name "*.py" -exec grep -l "TaskStatus\|Severity\|Confidence" {} \;
```

#### 2️⃣ 語法修復 (Syntax Repair)
```bash
# Python: 前向引用問題
# 使用漸進式類型標註
Dict[str, Any]  # 替代複雜類型推導

# Rust: 關鍵字衝突
pub type -> pub r#type  # 使用原始標識符

# Go: 模組路徑問題  
import "schemas/generated"  # 標準化路徑
```

#### 3️⃣ 類型系統映射 (Type System Mapping)
```bash
# 跨語言類型對應表
Python Dict[str, Any] ↔ Rust HashMap<String, serde_json::Value> ↔ Go map[string]interface{}
Python Enum ↔ Rust enum ↔ Go const/iota
Python dataclass ↔ Rust struct ↔ Go struct

# Go 官方命名標準 (Go Wiki: Initialisms)
# 縮寫必須統一大小寫，全大寫或全小寫
URL -> URL (正確) 或 url (正確)，絕不能是 Url (錯誤)
HTTP -> HTTP (正確) 或 http (正確)，絕不能是 Http (錯誤)  
ID -> ID (正確) 或 id (正確)，絕不能是 Id (錯誤)
API -> API (正確) 或 api (正確)，絕不能是 Api (錯誤)
JSON -> JSON (正確) 或 json (正確)，絕不能是 Json (錯誤)
```

#### 3️⃣.1 **Go Interface{} 類型斷言模式**
```go
// ❌ 常見錯誤 - 直接使用 interface{} 作為具體類型
func processTask(task schemas.TaskPayload) error {
    url := task.Target.URL  // 編譯錯誤: cannot use interface{} as string
    return processURL(url)
}

// ✅ 正確模式 - 類型斷言 + 錯誤處理
func processTask(task schemas.TaskPayload) error {
    // 模式 1: 基本類型斷言
    url, ok := task.Target.URL.(string)
    if !ok {
        return fmt.Errorf("target URL is not a string: %T", task.Target.URL)
    }
    
    // 模式 2: 帶日誌的安全斷言
    authHeader, exists := task.Target.Headers["Authorization"]
    if !exists {
        return fmt.Errorf("authorization header missing")
    }
    
    authHeaderStr, ok := authHeader.(string)
    if !ok {
        logger.Error("Authorization header is not a string", 
                     zap.Any("header", authHeader),
                     zap.String("type", fmt.Sprintf("%T", authHeader)))
        return fmt.Errorf("authorization header is not a string")
    }
    
    return processAuth(authHeaderStr)
}

// 模式 3: Map 類型安全處理
func createTarget(url string) schemas.Target {
    return schemas.Target{
        URL: url,  // interface{} 可以接受任何類型
        Headers: map[string]interface{}{  // 注意: 不是 map[string]string
            "Content-Type": "application/json",
            "User-Agent":   "AIVA-Scanner/1.0",
        },
        Params: map[string]interface{}{
            "timeout": 30,        // int
            "verify":  true,      // bool  
            "retry":   "3",       // string
        },
    }
}
```

#### 3️⃣.2 **Schema 字段缺失自動檢測**
```bash
# 檢測程式碼使用但 Schema 未定義的字段
# 1. 掃描 Go 程式碼中的字段訪問
grep -r "result\." services/features/*/internal/ | \
    grep -oE "result\.[A-Za-z][A-Za-z0-9]*" | \
    sort | uniq > /tmp/used_fields.txt

# 2. 掃描 Schema 定義的字段
grep -A 20 "TokenTestResult:" services/aiva_common/core_schema_sot.yaml | \
    grep -oE "[a-z_]+:" | \
    sed 's/://' > /tmp/defined_fields.txt

# 3. 找出缺失的字段
comm -23 /tmp/used_fields.txt /tmp/defined_fields.txt

# 常見缺失模式檢查
echo "=== 檢查常見缺失字段 ==="
grep -r "\.Severity\|\.TestType\|\.Description" services/features/*/internal/ || echo "未發現缺失字段使用"
```

#### 3️⃣.3 **批量類型修復腳本**
```bash
# PowerShell 批量修復腳本
function Fix-GoNamingStandards {
    param($Directory)
    
    Get-ChildItem -Path $Directory -Recurse -Filter "*.go" | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        
        # 修復常見命名問題
        $content = $content -replace '\.TaskId\b', '.TaskID'
        $content = $content -replace '\.FindingId\b', '.FindingID'  
        $content = $content -replace '\.ScanId\b', '.ScanID'
        $content = $content -replace '\.Url\b', '.URL'
        $content = $content -replace '\.Cwe\b', '.CWE'
        
        # 修復結構體字段
        $content = $content -replace 'TaskId:', 'TaskID:'
        $content = $content -replace 'FindingId:', 'FindingID:'
        $content = $content -replace 'ScanId:', 'ScanID:'
        $content = $content -replace 'Cwe:', 'CWE:'
        $content = $content -replace 'Url:', 'URL:'
        
        Set-Content -Path $_.FullName -Value $content
        Write-Host "Fixed: $($_.FullName)"
    }
}

# 使用範例
# Fix-GoNamingStandards -Directory "services\features\function_authn_go"
```

#### 4️⃣ 依賴同步 (Dependency Synchronization)
```bash
# 檢查跨模組依賴一致性
services/aiva_common/schemas/ -> 各語言實現
schemas/generated/ -> 統一生成路徑
```

#### 5️⃣ 命名規範統一 (Naming Convention Unification)
```bash
# 遵循各語言標準
Python: snake_case (task_id)
Rust: snake_case (task_id) 
Go: camelCase (TaskID) - 注意縮寫大寫規則
```

#### 6️⃣ 驗證與測試 (Validation & Testing)
```bash
# 各語言編譯/執行測試
python -m py_compile *.py
cargo check --all-targets
go build ./...
```

---

## 📋 語言特定修復模式

### 🐍 **Python 修復模式** (基於實戰經驗)

#### **1. Pydantic v2 配置現代化**
```python
# ❌ 過時的 Pydantic v1 語法 (導致 11 個配置錯誤)
class Settings(BaseSettings):
    log_level: str = Field(default="INFO", env="AIVA_LOG_LEVEL")
    database_url: str = Field(env="AIVA_DATABASE_URL")
    debug_mode: bool = Field(default=False, env="AIVA_DEBUG")
    
    class Config:
        env_file = ".env"

# ✅ Pydantic v2 現代語法 (完全修復)
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIVA_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    log_level: str = Field(default="INFO", description="Log level")
    database_url: str = Field(description="Database connection URL")
    debug_mode: bool = Field(default=False, description="Debug mode")
```

#### **2. Schema 屬性訪問修復流程**
```python
# 步驟 1: 識別錯誤的屬性訪問
# ❌ 基於錯誤假設的屬性訪問
vulnerability.cwe.cwe_id          # CWE 對象不存在
vulnerability.name.value          # Name 對象不存在
vulnerability.confidence.value    # Confidence 對象不存在

# 步驟 2: 查找實際 Schema 定義
# 檢查 services/aiva_common/schemas/ai.py 中的實際結構
grep -A 20 "class EnhancedVulnerability" services/aiva_common/schemas/ai.py

# 步驟 3: 修復屬性映射
# ✅ 基於實際 Schema 的正確訪問
vulnerability.vulnerability_id    # 實際字段名
vulnerability.title              # 實際字段名  
vulnerability.ai_confidence      # 實際字段名
```

#### **3. 類型推斷與類型安全**
```python
# ❌ 複雜類型推導問題 (導致 6 個類型警告)
evidence_data_list: List[Any] = execution_evidence["evidence_list"]  # Unknown 類型
evidence_dict: Dict[str, Any] = evidence_data           # Unknown 類型

# ✅ 使用 cast 進行安全類型轉換
from typing import cast, List, Dict, Any

evidence_data_list = cast(List[Any], execution_evidence["evidence_list"])
evidence_dict = cast(Dict[str, Any], evidence_data)

# ✅ Literal 類型的正確處理 (解決枚舉映射問題)
from typing import Literal, cast

level_mapping = {
    "critical": "error",
    "high": "error", 
    "medium": "warning",
    "low": "warning",
    "informational": "note",
}

level = cast(Literal["error", "warning", "info", "note"], 
             level_mapping.get(severity.lower(), "warning"))
```

#### **4. 異步上下文管理器修復**
```python
# ❌ 返回類型不匹配 (導致 5 個異步錯誤)
class IAIContext(AsyncContextManager[None]):  # 錯誤：返回 None
    async def __aenter__(self):  # 缺少返回類型註解
        return self

# ✅ 正確的異步上下文管理器實現
class IAIContext(AsyncContextManager['IAIContext']):  # 返回自身類型
    async def __aenter__(self) -> 'AIVAContext':  # 明確返回類型
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        await self.cleanup()
```

#### **5. 動態導入類型處理**
```python
# ❌ 動態導入類型未知 (導致 3 個類型錯誤)
from services.core.aiva_core.dialog.assistant import dialog_assistant
component_class = type(dialog_assistant)  # 類型未知

# ✅ 安全的動態導入處理
from typing import cast, Type, Any

try:
    from services.core.aiva_core.dialog.assistant import dialog_assistant  # type: ignore
    if dialog_assistant is not None:
        dialog_assistant_typed = cast(Any, dialog_assistant)
        component_class = cast(Type[Any], type(dialog_assistant_typed))
        # 現在可以安全使用 component_class
except (ImportError, AttributeError) as e:
    logger.debug(f"Optional component not available: {e}")
```

#### **6. 修復驗證與測試流程**
```python
# 修復後的驗證腳本
def verify_aiva_common_fixes():
    """驗證 AIVA Common 修復成果"""
    import sys
    sys.path.insert(0, r'C:\D\fold7\AIVA-git\services')
    
    try:
        # 測試基礎導入
        from aiva_common.enums.common import Severity, Confidence
        print('✅ 枚舉導入成功')
        
        # 測試 Schema 導入
        from aiva_common.schemas.enhanced import EnhancedFindingPayload
        print('✅ Enhanced Schema 導入成功')
        
        # 測試配置導入
        from aiva_common.config.settings import Settings
        print('✅ Pydantic v2 配置導入成功')
        
        # 測試異步組件
        from aiva_common.ai.registry import AIVAContext
        print('✅ 異步上下文管理器導入成功')
        
        print('🎉 所有 AIVA Common 組件驗證通過！')
        return True
        
    except Exception as e:
        print(f'❌ 驗證失敗: {e}')
        return False

# 使用驗證
if __name__ == "__main__":
    verify_aiva_common_fixes()
```

#### **前向引用處理** (保留原有內容)
```python
# ❌ 循環導入
from .models import TaskResult

# ✅ 字符串註解 + TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .models import TaskResult

def process_task(self) -> "TaskResult":
    pass
```

### 🦀 Rust 修復模式

#### 關鍵字衝突處理
```rust
// ❌ 關鍵字衝突 (36 個編譯錯誤)
pub type: String,

// ✅ 原始標識符修復
pub r#type: String,
```

#### 枚舉命名規範
```rust
// ❌ 不符合 Rust 標準
pub enum Confidence {
    Certain,
    likely,
}

// ✅ SCREAMING_SNAKE_CASE
pub enum Confidence {
    CONFIRMED,
    LIKELY,
}
```

#### 類型匹配修復
```rust
// ❌ 類型不匹配
let mut findings: Vec<Finding> = HashMap::new();

// ✅ 正確集合類型
let mut findings: HashMap<String, Finding> = HashMap::new();
```

### 🐹 Go 修復模式

#### 模組路徑標準化
```go
// ❌ 路徑不一致
import "schemas"

// ✅ 標準化生成路徑
import "schemas/generated"
```

#### 命名規範修復
```go
// ❌ 不一致命名
TaskId  // 混合風格

// ✅ Go 標準縮寫規則
TaskID  // 縮寫全大寫
```

---

## 🔧 AIVA 特定修復協議

### 📦 批量處理安全協議

遵循 AIVA Common 的批量修復安全原則：

```bash
# 1. 建立修復分支
git checkout -b fix/cross-language-schema-sync

# 2. 按模組分批修復 (避免大規模同時修改)
# 第一批: Python 相關模組
git add services/aiva_common/ai/
git commit -m "fix(python): Apply gradual typing for complex inference"

# 第二批: Rust 相關模組  
git add services/scan/info_gatherer_rust/
git commit -m "fix(rust): Resolve keyword conflicts and enum naming"

# 第三批: Go 相關模組
git add services/features/function_authn_go/
git commit -m "fix(go): Standardize module paths and naming conventions"
```

### 🔍 SOT 驗證檢查清單

```bash
# ✅ 檢查 aiva_common 作為單一真相源
find services/ -name "*.py" -exec grep -l "TaskStatus" {} \; | \
  xargs grep -L "from aiva_common"  # 應該為空

# ✅ 檢查跨語言枚舉一致性
grep -r "CONFIRMED\|LIKELY" services/scan/info_gatherer_rust/
grep -r "Confirmed\|Likely" services/aiva_common/enums/

# ✅ 檢查 Schema 生成路徑一致性
find services/ -name "*.go" -exec grep -l "schemas" {} \; | \
  xargs grep "import.*schemas"  # 應該都是 schemas/generated
```

### 📊 文檔同步要求

根據 AIVA Common 文檔同步機制：

```bash
# 修復後必須同步更新的文檔
services/aiva_common/README.md        # 問題記錄與修復狀態
services/core/README.md               # 如涉及 TaskStatus 等
services/features/README.md           # 如涉及安全枚舉  
services/scan/README.md               # 如涉及掃描相關
services/integration/README.md        # 如涉及資產枚舉
```

---

## 📈 實際修復成效記錄

### 🐍 **Python 修復成效** (2025年10月最新)
- **總體錯誤**: ~500 → 2 個錯誤 (99.6% 改善) 🎉
- **AIVA Common 模組**: 100% 無錯誤 ✨
- **主要修復類別**:
  - **Pydantic Field 配置** (11個): ✅ 100% 修復
  - **Schema 屬性不匹配** (10個): ✅ 100% 修復  
  - **異步上下文管理器** (5個): ✅ 100% 修復
  - **類型推斷警告** (6個): ✅ 100% 修復
  - **動態導入類型** (3個): ✅ 100% 修復

#### 🔧 **關鍵修復技術記錄**

**1. Pydantic v2 升級模式**
```python
# ❌ 舊版本 Pydantic Field 使用
log_level: str = Field(default="INFO", env="AIVA_LOG_LEVEL")
debug_mode: bool = Field(default=False, env="AIVA_DEBUG")

# ✅ 新版本 - 移除 env 參數，使用 SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIVA_",
        case_sensitive=False,
        env_file=".env"
    )
    
    log_level: str = Field(default="INFO", description="Log level")
    debug_mode: bool = Field(default=False, description="Debug mode")
```

**2. Schema 屬性映射修復**
```python
# ❌ 錯誤的屬性訪問 (基於過時的 Schema 定義)
rule_id = self.vulnerability.cwe.cwe_id
message = self.vulnerability.name.value
confidence = self.vulnerability.confidence.value

# ✅ 正確的屬性映射 (基於實際 EnhancedVulnerability 結構)
rule_id = self.vulnerability.vulnerability_id
message = self.vulnerability.title  
confidence = self.vulnerability.ai_confidence
```

**3. 異步上下文管理器類型修復**
```python
# ❌ 接口類型定義錯誤
class IAIContext(AsyncContextManager[None]):  # 錯誤返回類型

# ✅ 正確的接口定義
class IAIContext(AsyncContextManager['IAIContext']):  # 返回自身類型

# ✅ 實現類正確返回類型
async def __aenter__(self) -> 'AIVAContext':
    await self.initialize()
    return self
```

**4. 類型安全轉換技術**
```python
# ❌ 不安全的類型推斷
evidence_data_list: List[Any] = execution_evidence["evidence_list"]  # 類型未知警告

# ✅ 使用 cast 進行安全類型轉換
from typing import cast
evidence_data_list = cast(List[Any], execution_evidence["evidence_list"])

# ✅ Literal 類型的正確處理
from typing import Literal, cast
level = cast(Literal["error", "warning", "info", "note"], 
             level_mapping.get(severity, "warning"))
```

### 🦀 Rust 修復成效  
- **編譯錯誤**: 36 → 0 個錯誤 (100% 解決)
- **主要問題**: 關鍵字衝突、枚舉命名、類型匹配
- **修復方法**: 原始標識符、SCREAMING_SNAKE_CASE、正確集合類型

### 🐹 Go 問題分析
- **模組依賴**: `schemas` → `schemas/generated` 路徑標準化
- **命名不一致**: `TaskId` → `TaskID` 縮寫規範
- **狀態**: 已識別，待修復實施

---

## 🚀 最佳實踐與建議

### 🎯 架構設計原則

1. **遵循四層優先級**: 國際標準 > 語言標準 > aiva_common > 模組專屬
2. **SOT 原則**: aiva_common 作為跨語言模式的單一真相源
3. **漸進式修復**: 避免大規模同時修改，按模組分批處理
4. **文檔同步**: 修復後必須同步更新相關模組文檔

### 🔍 驗證與品質保證

```bash
# 自動化驗證流程
python scripts/validate_cross_language_sync.py
cargo check --all-targets  
go build ./...

# 檔案同步檢查
python scripts/check_doc_sync.py --source services/aiva_common/README.md
```

#### 🔍.1 **Schema 與程式碼一致性驗證工具**
```python
#!/usr/bin/env python3
"""
Schema 與程式碼同步驗證工具
檢查程式碼使用的字段是否在 Schema 中定義
"""
import re
import yaml
import glob
from pathlib import Path

def extract_go_field_usage(go_files):
    """提取 Go 程式碼中使用的結構體字段"""
    field_usage = {}
    field_pattern = re.compile(r'(\w+)\.(\w+)\s*[=\.\(]')
    
    for go_file in go_files:
        with open(go_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        matches = field_pattern.findall(content)
        for struct_name, field_name in matches:
            if struct_name not in field_usage:
                field_usage[struct_name] = set()
            field_usage[struct_name].add(field_name)
    
    return field_usage

def extract_schema_definitions(yaml_file):
    """提取 YAML Schema 中定義的字段"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    
    definitions = {}
    if 'definitions' in schema:
        for name, definition in schema['definitions'].items():
            if 'properties' in definition:
                definitions[name] = set(definition['properties'].keys())
    
    return definitions

def validate_field_consistency():
    """驗證字段一致性"""
    print("🔍 Schema 與程式碼一致性驗證")
    print("=" * 50)
    
    # 1. 掃描 Go 程式碼
    go_files = glob.glob("services/features/*/internal/**/*.go", recursive=True)
    field_usage = extract_go_field_usage(go_files)
    
    # 2. 載入 Schema 定義
    schema_file = "services/aiva_common/core_schema_sot.yaml"
    if Path(schema_file).exists():
        schema_definitions = extract_schema_definitions(schema_file)
    else:
        print(f"❌ Schema 文件不存在: {schema_file}")
        return
    
    # 3. 檢查一致性
    issues_found = 0
    
    for struct_name, used_fields in field_usage.items():
        if struct_name in schema_definitions:
            defined_fields = schema_definitions[struct_name]
            
            # 檢查缺失字段
            missing_fields = used_fields - defined_fields
            if missing_fields:
                print(f"⚠️  {struct_name} 缺失字段: {sorted(missing_fields)}")
                issues_found += 1
            
            # 檢查未使用字段
            unused_fields = defined_fields - used_fields
            if unused_fields:
                print(f"ℹ️  {struct_name} 未使用字段: {sorted(unused_fields)}")
        else:
            print(f"❓ {struct_name} 在 Schema 中未定義")
            issues_found += 1
    
    # 4. 總結
    if issues_found == 0:
        print("✅ 所有字段一致性檢查通過")
    else:
        print(f"❌ 發現 {issues_found} 個一致性問題")
    
    return issues_found == 0

# 使用範例
if __name__ == "__main__":
    validate_field_consistency()
```

#### 🔍.2 **自動化錯誤檢測腳本**
```bash
#!/bin/bash
# cross_language_error_detector.sh
# 自動檢測跨語言同步問題

echo "🔍 AIVA 跨語言錯誤自動檢測"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="

# 1. 檢測 Go Initialisms 違規
echo "1️⃣ 檢測 Go Initialisms 違規..."
grep -r "\.TaskId\|\.FindingId\|\.ScanId\|\.Url[^L]\|\.Cwe" services/features/*/internal/ && {
    echo "❌ 發現 Go Initialisms 違規"
} || {
    echo "✅ Go Initialisms 檢查通過"
}

# 2. 檢測類型斷言缺失  
echo "2️⃣ 檢測類型斷言缺失..."
grep -r "task\.Target\.URL\|authHeader\)" services/features/*/internal/ | grep -v "\.(" && {
    echo "❌ 發現缺失類型斷言"
} || {
    echo "✅ 類型斷言檢查通過"
}

# 3. 檢測 Map 類型不匹配
echo "3️⃣ 檢測 Map 類型不匹配..."
grep -r "map\[string\]string" services/features/*/internal/ | grep -i header && {
    echo "❌ 發現 Map 類型不匹配 (應為 map[string]interface{})"
} || {
    echo "✅ Map 類型檢查通過"
}

# 4. 編譯驗證
echo "4️⃣ 執行編譯驗證..."
for module in function_authn_go function_sca_go function_ssrf_go function_cspm_go; do
    echo "  檢查 $module..."
    cd "services/features/$module" 2>/dev/null && {
        go build ./... 2>/dev/null && {
            echo "  ✅ $module 編譯成功"
        } || {
            echo "  ❌ $module 編譯失敗"
        }
        cd ../../..
    } || {
        echo "  ⚠️ $module 目錄不存在"
    }
done

echo "🎯 自動檢測完成"
```

### 📋 持續改進機制

1. **問題發現**: 在各模組 README 記錄發現的同步問題
2. **修復追蹤**: 在 aiva_common README 更新修復狀態  
3. **效果評估**: 記錄修復前後的錯誤數量變化
4. **經驗總結**: 將成功模式納入本指南的最佳實踐

---

## 📞 支援與貢獻

### 🛠️ 故障排除

遇到跨語言同步問題時：

1. 檢查是否遵循四層優先級原則
2. 確認 aiva_common 作為 SOT 的使用
3. 驗證語言特定的命名和類型規範
4. 參考本指南的實際修復案例

### 📸 修復文檔化原則

為確保修復過程的可追溯性和經驗傳承：

#### 必要截圖時機
1. **修復前狀態**: 錯誤訊息、編譯失敗畫面
2. **官方標準查證**: 語言官方文檔截圖（如 Go Wiki: Initialisms）
3. **修復後驗證**: 成功編譯、測試通過畫面
4. **對比結果**: 修復前後的具體差異

#### 截圖品質要求
- 包含完整錯誤訊息和上下文
- 標註關鍵修復點和官方標準依據
- 使用高對比度和清晰字體
- 保存為 PNG 格式以確保文字清晰度

#### 文檔整合
- 截圖應與文字說明配對使用
- 在 markdown 中使用適當的標題和說明
- 建立修復案例資料庫供後續參考

### 🔗 跨語言繼承處理建議

#### 繼承模式識別
1. **YAML extends 語法**: `extends: BaseType` 表示繼承關係
2. **Python 實現**: 使用 Pydantic BaseModel 繼承或組合
3. **Go 實現**: 使用 struct 嵌入或字段複製
4. **Rust 實現**: 使用 trait 或 struct 組合

#### 常見繼承問題
1. **Go 繼承缺失**: 
   - 問題: `FunctionTaskTarget` 缺少 `Target` 基類字段
   - 現象: `task.Target.URL undefined` 編譯錯誤
   - 解決: 在 Go struct 中嵌入基類或複製所有字段

2. **字段名稱不一致**:
   - YAML: `url` (小寫)
   - Go: 應該是 `URL` (符合 Initialisms 標準)
   - Python: `url` (小寫，符合 snake_case)

3. **類型映射差異**:
   - YAML: `Any` 類型
   - Python: `Any` 
   - Go: `interface{}`
   - Rust: `serde_json::Value`

#### 修復策略
1. **檢查繼承完整性**: 確保子類包含所有父類字段
2. **統一命名規範**: 按語言標準調整字段名稱
3. **驗證類型映射**: 確保類型轉換正確
4. **測試跨語言互操作**: 驗證序列化/反序列化一致性

#### 🚨 代碼生成器缺陷檢查
**新發現問題類型**: 代碼生成器本身可能有實現缺陷

1. **函數缺失**: 
   - 現象: 代碼中調用 `_get_all_fields()` 但函數未實現
   - 影響: 繼承關係完全無法處理
   - 檢查方法: `grep -n "_get_all_fields" schema_codegen_tool.py`

2. **邏輯不完整**:
   - 現象: 只有註解 `# 繼承自: BaseClass` 但沒有實際字段合併
   - 影響: 子類缺少父類字段，導致編譯錯誤
   - 檢查方法: 對比生成的 Schema 與預期的繼承結構

3. **修復優先級**:
   - **必須先修復代碼生成器**，再修復 Schema
   - 實現 `_get_all_fields()` 函數以正確處理繼承
   - 驗證生成的代碼符合各語言標準

### 🤝 貢獻指南

1. 發現新的跨語言同步問題時，請更新相應的模組 README
2. 成功修復後，請將經驗總結添加到本指南
3. 遵循 AIVA Common 的文檔同步機制
4. 提交 PR 時請包含修復前後的對比數據

---

## 🚀 **快速參考手冊** (實戰速查)

### ⚡ **緊急修復命令清單**

#### **Python 錯誤快速修復**
```bash
# 1. 檢查當前錯誤狀態
python -c "
import sys
sys.path.insert(0, 'services')
from get_errors import get_errors
errors = get_errors()
print(f'當前錯誤數: {len(errors)}')
"

# 2. Pydantic v2 批量修復
find services/aiva_common -name "*.py" -exec sed -i 's/Field([^)]*env=["'"'"'][^"'"'"']*["'"'"'][^)]*)/Field(/g' {} \;

# 3. Schema 屬性訪問批量修復  
find services/aiva_common -name "*.py" -exec sed -i 's/vulnerability\.cwe\.cwe_id/vulnerability.vulnerability_id/g' {} \;
find services/aiva_common -name "*.py" -exec sed -i 's/vulnerability\.name\.value/vulnerability.title/g' {} \;

# 4. 驗證修復結果
python -c "
import sys
sys.path.insert(0, 'services')
try:
    from aiva_common.schemas import EnhancedFindingPayload
    from aiva_common.config import Settings
    print('✅ 修復驗證通過')
except Exception as e:
    print(f'❌ 修復驗證失敗: {e}')
"
```

#### **Go 命名標準批量修復**
```bash
# Go Initialisms 批量修復
find services/features -name "*.go" -exec sed -i 's/\.TaskId\b/.TaskID/g' {} \;
find services/features -name "*.go" -exec sed -i 's/\.FindingId\b/.FindingID/g' {} \;
find services/features -name "*.go" -exec sed -i 's/\.Url\b/.URL/g' {} \;

# 驗證編譯
for module in function_authn_go function_sca_go function_ssrf_go; do
    echo "檢查 $module..."
    cd "services/features/$module" && go build ./... && echo "✅ $module 編譯成功"
    cd ../../..
done
```

### 📋 **修復檢查清單**

#### **🐍 Python 修復檢查清單**
- [ ] **Pydantic v2 配置**
  - [ ] 移除所有 `Field(env="...")` 參數
  - [ ] 使用 `SettingsConfigDict` 替代 `Config` 類
  - [ ] 修復 `default_factory` 語法
  
- [ ] **Schema 屬性映射**
  - [ ] 檢查 `EnhancedVulnerability` 屬性訪問
  - [ ] 修復 `SARIFLocation` 參數使用
  - [ ] 驗證所有 Schema 導入正常
  
- [ ] **類型系統修復**
  - [ ] 使用 `cast()` 解決類型推斷警告
  - [ ] 修復異步上下文管理器返回類型
  - [ ] 處理動態導入的類型問題

#### **🦀 Rust 修復檢查清單**
- [ ] **關鍵字衝突**
  - [ ] 使用 `r#type` 等原始標識符
  - [ ] 檢查所有 Rust 關鍵字使用
  
- [ ] **枚舉命名**
  - [ ] 使用 `SCREAMING_SNAKE_CASE`
  - [ ] 確保枚舉值一致性

#### **🐹 Go 修復檢查清單**
- [ ] **Initialisms 標準**
  - [ ] `TaskId` → `TaskID`
  - [ ] `FindingId` → `FindingID`
  - [ ] `Url` → `URL`
  - [ ] `Cwe` → `CWE`
  
- [ ] **類型斷言**
  - [ ] 所有 `interface{}` 使用都有類型斷言
  - [ ] 添加適當的錯誤處理
  
- [ ] **模組路徑**
  - [ ] 統一使用 `schemas/generated`
  - [ ] 檢查所有導入路徑

### 🔍 **問題診斷速查表**

| 錯誤症狀 | 可能原因 | 快速修復 |
|---------|---------|---------|
| `無法解析匯入 "aiva_common"` | Python 路徑問題 | `sys.path.insert(0, 'services')` |
| `Field() missing 1 required positional argument` | Pydantic v2 升級問題 | 移除 `env` 參數 |
| `attribute 'cwe' not found` | Schema 屬性不匹配 | 檢查實際 Schema 定義 |
| `__aenter__ 返回類型不符` | 異步上下文管理器問題 | 修正接口返回類型 |
| `TaskId undefined` | Go Initialisms 違規 | 改為 `TaskID` |
| `cannot use interface{} as string` | Go 類型斷言缺失 | 添加類型斷言 |

### 🎯 **修復成功指標**

#### **量化指標**
- **錯誤數量**: 目標 < 5 個非關鍵錯誤
- **模組覆蓋**: aiva_common 模組 100% 無錯誤
- **編譯成功**: 所有語言模組編譯通過

#### **質量指標**  
- **類型安全**: 所有類型推斷警告解決
- **標準遵循**: 符合各語言官方命名標準
- **架構一致**: 遵循 AIVA 四層優先級原則

### 📞 **故障排除聯絡清單**

1. **Python 相關問題**: 檢查 `services/aiva_common/` 模組
2. **Schema 定義問題**: 查看 `services/aiva_common/schemas/ai.py`
3. **跨語言同步問題**: 參考本指南的語言特定修復模式
4. **架構設計問題**: 遵循四層優先級原則

---

**AIVA 跨語言模式同步指南** - 確保多語言架構的一致性與可維護性 🌐

*最後更新: 2025年10月30日 - 整合實戰修復經驗，錯誤修復率達 99.6%* ✨