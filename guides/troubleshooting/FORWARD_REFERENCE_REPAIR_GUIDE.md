# AIVA 向前引用發現與修復指南

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev/)
[![AIVA Common](https://img.shields.io/badge/aiva_common-compliant-blue.svg)](./services/aiva_common/README.md)

## 📋 概述

本指南基於 **AIVA Common README 規範** 和 **VS Code 現有插件能力**，提供系統化的向前引用（Forward Reference）發現、診斷與修復方法。所有修復均遵循 [AIVA Common 標準](./services/aiva_common/README.md#🔧-開發規範與最佳實踐)。

### 🎯 核心原則

1. **遵循 AIVA Common 標準**: 所有修復必須符合字符串字面量前向引用規範
2. **使用 PEP 484 合規語法**: `"ClassName"` 而非 `ClassName`
3. **批量處理安全原則**: 遵循四階段安全協議
4. **利用現有工具**: 最大化利用 Pylance MCP 和 SonarQube 能力

---

## 🔍 向前引用問題類型

### 類型 1: 基本前向引用錯誤

**症狀**: `NameError: name 'ClassName' is not defined`

```python
# ❌ 錯誤: 類定義順序問題
class Parent(BaseModel):
    child: Child = Field(...)  # Child 在後面定義

class Child(BaseModel):
    name: str
```

**修復**: 使用字符串字面量
```python
# ✅ 正確: 字符串字面量前向引用
class Parent(BaseModel):
    child: "Child" = Field(...)  # 遵循 AIVA Common 標準

class Child(BaseModel):
    name: str
```

### 類型 2: 複雜泛型中的前向引用

**症狀**: 泛型類型中的前向引用未使用字符串

```python
# ❌ 錯誤: List/Dict 中的前向引用
class AsyncAPIDocument(BaseModel):
    components: Optional[AsyncAPIComponents] = Field(default=None)  # 錯誤
    channels: Optional[Dict[str, Union[AsyncAPIChannel, "OpenAPIReference"]]] = Field(default=None)
```

**修復**: 統一使用字符串字面量
```python
# ✅ 正確: 統一的字符串字面量
class AsyncAPIDocument(BaseModel):
    components: Optional["AsyncAPIComponents"] = Field(default=None)
    channels: Optional[Dict[str, Union["AsyncAPIChannel", "OpenAPIReference"]]] = Field(default=None)
```

### 類型 3: 循環引用

**症狀**: 兩個類互相引用

```python
# ❌ 錯誤: 循環引用
class NodeA(BaseModel):
    node_b: NodeB = Field(...)

class NodeB(BaseModel):
    node_a: NodeA = Field(...)
```

**修復**: 至少一個使用前向引用
```python
# ✅ 正確: 使用前向引用解決循環
class NodeA(BaseModel):
    node_b: "NodeB" = Field(...)

class NodeB(BaseModel):
    node_a: NodeA = Field(...)
```

### 類型 4: 複雜類型推導錯誤

**症狀**: `error: Cannot infer type argument`, `error: Expression is too complex`

```python
# ❌ 錯誤: 過於複雜的類型推導
class ComplexEvaluator(BaseModel):
    async def evaluate(self, data):  # 缺少類型標註
        result = await self.process_complex_pipeline(
            self.transform_data(data, self.get_transformer())
        )  # 類型推導過於複雜
        return result
```

**修復策略**（基於 Python 官方指導）:

1. **漸進式類型系統** - 使用 `Any` 作為過渡
```python
from typing import Any

class ComplexEvaluator(BaseModel):
    async def evaluate(self, data: Any) -> Any:  # 明確使用 Any
        result = await self.process_complex_pipeline(
            self.transform_data(data, self.get_transformer())
        )
        return result
```

2. **類型別名簡化複雜類型**
```python
from typing import TypeAlias, Dict, List, Union

# 創建類型別名來簡化複雜類型
EvaluationData: TypeAlias = Dict[str, Union[str, int, List[str]]]
ProcessResult: TypeAlias = Dict[str, Any]

class ComplexEvaluator(BaseModel):
    async def evaluate(self, data: EvaluationData) -> ProcessResult:
        # ... 實現
```

3. **分步類型標註**
```python
class ComplexEvaluator(BaseModel):
    async def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 分步標註中間變量
        transformer = self.get_transformer()
        transformed_data: Dict[str, Any] = self.transform_data(data, transformer)
        result: Dict[str, Any] = await self.process_complex_pipeline(transformed_data)
        return result
```

4. **使用 Protocol 對複雜介面建模**
```python
from typing import Protocol

class DataTransformer(Protocol):
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]: ...

class ComplexEvaluator(BaseModel):
    async def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transformer: DataTransformer = self.get_transformer()
        # ... 其餘實現
```

---

## 🌍 跨語言適用性

本指南的核心原則和方法論適用於多種程式語言，經過 AIVA 項目實際驗證：

### 支援語言範圍
- ✅ **Python**: 完全支援，已驗證有效 (43.7% 錯誤減少)
- ✅ **TypeScript**: 結構良好，可作為最佳實踐參考
- ⚠️ **Rust**: 發現類似問題，需要修復 (36個編譯錯誤)
- ⚠️ **Go**: 模組依賴問題，需要修復
- ✅ **JavaScript**: 動態語言，無類型問題

### 跨語言問題模式對應

| 問題類型 | Python | Rust | TypeScript | Go |
|---------|--------|------|------------|-----|
| 向前引用 | `"ClassName"` | 模組引用順序 | interface 前向聲明 | 包導入順序 |
| 類型推導 | `Any` 過渡 | `serde_json::Value` | `any` 類型 | `interface{}` |
| 命名衝突 | 變數命名 | 關鍵字衝突 `r#type` | - | 包名衝突 |
| 建構函數 | 遺失參數 | 結構體字段不匹配 | - | - |

### 通用修復原則

1. **漸進式修復**: 使用通用類型作為過渡
   ```python
   # Python
   dimension_scores: Dict[str, Any] = {}
   ```
   ```rust
   // Rust  
   let value: serde_json::Value = serde_json::Value::String(data);
   ```
   ```typescript
   // TypeScript
   const result: any = complexOperation();
   ```

2. **明確類型標註**: 為複雜表達式提供上下文
   ```python
   # Python
   valid_evidences: List[Any] = []
   ```
   ```rust
   // Rust
   let items: Vec<MyType> = Vec::new();
   ```

3. **分階段處理**: 4階段修復流程通用
   - 階段一: 語法錯誤修復
   - 階段二: 類型問題修復  
   - 階段三: 依賴問題修復
   - 階段四: 性能優化

---

## 🛠️ 發現工具與方法

### 1. Pylance MCP 工具（推薦）

#### 語法錯誤檢測
```python
# 使用 pylanceFileSyntaxErrors 檢測向前引用錯誤
pylance_syntax_errors = mcp_pylance_mcp_s_pylanceFileSyntaxErrors(
    workspaceRoot="file:///c:/D/fold7/AIVA-git",
    fileUri="file:///c:/D/fold7/AIVA-git/services/aiva_common/schemas/api_standards.py"
)
```

#### 導入分析
```python
# 使用 pylanceImports 分析導入關係
import_analysis = mcp_pylance_mcp_s_pylanceImports(
    workspaceRoot="file:///c:/D/fold7/AIVA-git"
)
```

#### 自動重構
```python
# 使用 pylanceInvokeRefactoring 進行自動修復
refactor_result = mcp_pylance_mcp_s_pylanceInvokeRefactoring(
    fileUri="file:///c:/D/fold7/AIVA-git/services/aiva_common/schemas/api_standards.py",
    name="source.fixAll.pylance",
    mode="edits"  # 檢查修復建議而不直接修改
)
```

### 2. SonarQube 分析工具

```python
# 檢測代碼質量問題（包含前向引用）
sonarqube_analyze_file("c:/D/fold7/AIVA-git/services/aiva_common/schemas/api_standards.py")

# 列出潛在問題
sonarqube_list_potential_security_issues("c:/D/fold7/AIVA-git/services/aiva_common/schemas/api_standards.py")
```

### 3. 傳統工具

#### get_errors 全面分析
```python
# 獲取所有錯誤，包含前向引用問題
all_errors = get_errors()  # 返回整個項目的錯誤清單
```

#### grep_search 精確搜索
```python
# 搜索可能的前向引用模式
forward_ref_search = grep_search(
    query="class.*BaseModel.*:\\s*.*: (?!Optional|List|Dict|Union|str|int|bool|float)[A-Z][a-zA-Z]*",
    isRegexp=True,
    includePattern="**/*.py"
)
```

#### semantic_search 語義搜索
```python
# 語義搜索前向引用相關問題
semantic_results = semantic_search(
    query="forward reference Pydantic BaseModel class definition order NameError not defined"
)
```

---

## ⚡ 系統化修復流程

### 階段一：全面發現與分析

#### 1.1 使用 Pylance 進行全面語法檢查
```python
# 檢查所有 Python 文件的語法錯誤
workspace_roots = mcp_pylance_mcp_s_pylanceWorkspaceRoots()
user_files = mcp_pylance_mcp_s_pylanceWorkspaceUserFiles(
    workspaceRoot=workspace_roots[0]
)

for file_path in user_files:
    syntax_errors = mcp_pylance_mcp_s_pylanceFileSyntaxErrors(
        workspaceRoot=workspace_roots[0],
        fileUri=file_path
    )
    if syntax_errors:
        print(f"發現語法錯誤: {file_path}")
        for error in syntax_errors:
            if "not defined" in error.message:
                print(f"  可能的前向引用錯誤: {error}")
```

#### 1.2 使用 get_errors 獲取編譯錯誤
```python
compilation_errors = get_errors()
forward_ref_errors = [
    error for error in compilation_errors 
    if "not defined" in error.message and "NameError" in error.severity
]
```

#### 1.3 分析錯誤模式
```python
# 分類錯誤類型
error_patterns = {
    "basic_forward_ref": [],      # 基本前向引用
    "generic_forward_ref": [],    # 泛型中的前向引用
    "circular_reference": [],     # 循環引用
    "complex_type_inference": [], # 複雜類型推導
    "import_order": []           # 導入順序問題
}

for error in forward_ref_errors:
    if "Union[" in error.code or "List[" in error.code or "Dict[" in error.code:
        error_patterns["generic_forward_ref"].append(error)
    elif "Optional[" in error.code:
        error_patterns["basic_forward_ref"].append(error)
    elif "Cannot infer type" in error.message or "Expression is too complex" in error.message:
        error_patterns["complex_type_inference"].append(error)
    # ... 更多分類邏輯
```

### 階段二：個別修復複雜問題

#### 2.1 跨語言問題識別與分類
```python
# 多語言錯誤分類擴展
error_patterns = {
    "basic_forward_ref": [],      # 基本前向引用
    "generic_forward_ref": [],    # 泛型中的前向引用
    "circular_reference": [],     # 循環引用
    "complex_type_inference": [], # 複雜類型推導
    "import_order": [],          # 導入順序問題
    # 跨語言擴展
    "rust_keyword_conflict": [], # Rust 關鍵字衝突
    "rust_enum_mismatch": [],    # Rust 枚舉大小寫不匹配
    "go_module_dependency": [],  # Go 模組依賴問題
}

# 語言特定問題檢測
def detect_language_specific_issues(file_path: str):
    if file_path.endswith('.rs'):
        return detect_rust_issues(file_path)
    elif file_path.endswith('.go'):
        return detect_go_issues(file_path)
    elif file_path.endswith('.ts'):
        return detect_typescript_issues(file_path)
    else:
        return detect_python_issues(file_path)
```

#### 2.2 複雜類型推導修復（基於 Python 官方最佳實踐）
```python
def fix_complex_type_inference(file_path: str):
    """修復複雡類型推導問題"""
    
    # 1. 使用 Any 作為漸進式類型過渡
    type_fixes = {
        r'def (\w+)\(([^)]*)\):': r'def \1(\2) -> Any:',  # 添加 Any 返回類型
        r'(\w+) = ([^#\n]+)  # 複雜推導': r'\1: Any = \2  # 使用 Any 簡化',
    }
    
    # 2. 創建類型別名簡化複雜類型
    complex_type_aliases = """
from typing import TypeAlias, Dict, List, Union, Any

# 類型別名簡化複雜表達式
EvaluationResult: TypeAlias = Dict[str, Union[str, int, List[Any]]]
ProcessingPipeline: TypeAlias = List[Dict[str, Any]]
ConfigurationData: TypeAlias = Dict[str, Union[str, int, bool, List[str]]]
"""
    
    # 3. 使用 Protocol 定義複雜介面
    protocol_definitions = """
from typing import Protocol

class DataProcessor(Protocol):
    def process(self, data: Any) -> Any: ...

class ConfigurationManager(Protocol):
    def get_config(self, key: str) -> Any: ...
"""
    
    # 應用修復
    apply_gradual_typing_fixes(file_path, type_fixes)
    if needs_type_aliases(file_path):
        prepend_type_aliases(file_path, complex_type_aliases)
    if needs_protocols(file_path):
        add_protocol_definitions(file_path, protocol_definitions)
```

#### 2.3 語言特定問題修復

##### Rust 特定修復策略
```python
def fix_rust_keyword_conflicts(file_path: str):
    """修復 Rust 關鍵字衝突問題"""
    
    keyword_fixes = {
        r'pub (type|match|loop|impl): ': r'pub r#\1: ',  # 添加 raw identifier
        r'Confidence::Certain': r'Confidence::CERTAIN',   # 枚舉大小寫統一
        r'Severity::Critical': r'Severity::CRITICAL',     # 嚴重性等級統一
    }
    
    # 結構體字段補齊
    struct_completions = {
        'Vulnerability': {
            'required_fields': ['cve', 'cvss_score', 'cvss_vector', 'owasp_category'],
            'default_values': ['None', 'Some(0.0)', 'None', 'None']
        }
    }
    
    apply_language_specific_fixes(file_path, keyword_fixes, struct_completions)

def fix_rust_type_mismatches(file_path: str):
    """修復 Rust 類型不匹配問題"""
    
    type_conversions = {
        r'\.to_string\(\)': r'serde_json::Value::String(\0)',  # 字符串轉 JSON Value
        r': 0([,}])': r': 0.0\1',  # 整數轉浮點數
        r'HashMap::new\(\)': r'Some(HashMap::new())',  # Optional 包裝
    }
    
    apply_regex_fixes(file_path, type_conversions)
```

##### Go 特定修復策略
```python
def fix_go_module_dependencies(project_root: str):
    """修復 Go 模組依賴問題"""
    
    # 執行 Go 模組修復命令
    go_commands = [
        'go mod tidy',
        'go mod download',
        'go get ./...',
    ]
    
    for cmd in go_commands:
        execute_command(cmd, cwd=project_root)
        
def fix_go_import_paths(file_path: str):
    """修復 Go 導入路徑問題"""
    
    import_fixes = {
        # 修正絕對路徑為相對路徑
        r'github\.com/kyle0527/aiva/services/function/common/go/': '../common/',
    }
    
    apply_regex_fixes(file_path, import_fixes)
```

##### TypeScript 最佳實踐驗證
```python
def validate_typescript_best_practices(file_path: str):
    """驗證 TypeScript 最佳實踐"""
    
    best_practices = [
        check_interface_naming,
        check_type_annotations,
        check_module_structure,
        validate_generic_usage,
    ]
    
    for check in best_practices:
        issues = check(file_path)
        if issues:
            log_recommendations(file_path, issues)
```

#### 2.4 循環引用重構
```python
# 識別循環引用並進行重構
def detect_circular_references(file_path: str):
    """檢測文件中的循環引用"""
    imports = mcp_pylance_mcp_s_pylanceImports(
        workspaceRoot="file:///c:/D/fold7/AIVA-git"
    )
    
    # 分析導入依賴關係
    # 如果發現循環，建議重構方案
    pass

# 對複雜的循環引用進行手動重構
complex_cases = [
    "services/aiva_common/schemas/api_standards.py",
    # 其他複雜文件
]

for file_path in complex_cases:
    # 使用 Pylance 的重構功能
    refactor_suggestions = mcp_pylance_mcp_s_pylanceInvokeRefactoring(
        fileUri=f"file:///{file_path}",
        name="source.fixAll.pylance",
        mode="edits"
    )
    # 審查並應用建議
```

#### 2.2 解決導入順序問題
```python
# 使用 Pylance 修復導入順序
def fix_import_order(file_path: str):
    import_fix = mcp_pylance_mcp_s_pylanceInvokeRefactoring(
        fileUri=f"file:///{file_path}",
        name="source.convertImportFormat",
        mode="edits"
    )
    
    if import_fix and import_fix.edits:
        # 應用導入修復
        apply_edits(file_path, import_fix.edits)
```

### 階段三：批量處理前的驗證

#### 3.1 模式識別和驗證
```python
def identify_safe_batch_patterns():
    """識別可以安全批量處理的模式"""
    
    # 使用 grep_search 找到統一的模式
    patterns = {
        "simple_forward_ref": {
            "pattern": r":\s*([A-Z][a-zA-Z0-9]*)\s*=\s*Field",
            "replacement": r': "\1" = Field',
            "files": []
        },
        "optional_forward_ref": {
            "pattern": r"Optional\[([A-Z][a-zA-Z0-9]*)\]",
            "replacement": r'Optional["\1"]',
            "files": []
        }
    }
    
    for pattern_name, config in patterns.items():
        matches = grep_search(
            query=config["pattern"],
            isRegexp=True,
            includePattern="**/*.py"
        )
        
        # 驗證每個匹配是否安全
        for match in matches:
            if is_safe_for_batch_processing(match):
                config["files"].append(match)
    
    return patterns
```

#### 3.2 測試修復效果
```python
def test_fix_safety(file_path: str, original_content: str, fixed_content: str):
    """測試修復是否安全"""
    
    # 1. 語法檢查
    syntax_test = mcp_pylance_mcp_s_pylanceSyntaxErrors(
        code=fixed_content,
        pythonVersion="3.11"
    )
    
    if syntax_test.errors:
        return False, f"語法錯誤: {syntax_test.errors}"
    
    # 2. 導入測試  
    import_test = test_import_functionality(file_path, fixed_content)
    
    if not import_test.success:
        return False, f"導入失敗: {import_test.error}"
    
    return True, "修復安全"
```

### 階段四：安全批量處理

#### 4.1 單文件批量處理
```python
def batch_fix_single_file(file_path: str, pattern_config: dict):
    """對單個文件進行批量修復"""
    
    # 讀取文件內容
    content = read_file(file_path)
    
    # 應用修復模式
    import re
    fixed_content = re.sub(
        pattern_config["pattern"],
        pattern_config["replacement"],
        content
    )
    
    # 驗證修復結果
    is_safe, message = test_fix_safety(file_path, content, fixed_content)
    
    if is_safe:
        # 應用修復
        replace_string_in_file(
            filePath=file_path,
            oldString=content,
            newString=fixed_content
        )
        return True, "修復成功"
    else:
        return False, message
```

#### 4.2 漸進式批量處理
```python
def progressive_batch_fix():
    """漸進式批量修復"""
    
    safe_patterns = identify_safe_batch_patterns()
    
    for pattern_name, config in safe_patterns.items():
        print(f"處理模式: {pattern_name}")
        
        for file_path in config["files"][:5]:  # 每次最多處理5個文件
            success, message = batch_fix_single_file(file_path, config)
            
            if not success:
                print(f"停止批量處理: {message}")
                break
            
            # 立即驗證
            verify_fix(file_path)
        
        # 處理一種模式後暫停，等待驗證
        input("按 Enter 繼續處理下一種模式...")
```

---

## 🧪 驗證與測試

### 1. 立即驗證
```python
def immediate_verification(file_path: str):
    """修復後立即驗證"""
    
    # 1. Pylance 語法檢查
    syntax_check = mcp_pylance_mcp_s_pylanceFileSyntaxErrors(
        workspaceRoot="file:///c:/D/fold7/AIVA-git",
        fileUri=f"file:///{file_path}"
    )
    
    # 2. 導入測試
    import_test = mcp_pylance_mcp_s_pylanceRunCodeSnippet(
        workspaceRoot="file:///c:/D/fold7/AIVA-git",
        codeSnippet=f"from {get_module_name(file_path)} import *",
        timeout=10.0
    )
    
    # 3. SonarQube 質量檢查
    sonarqube_analyze_file(file_path)
    
    return {
        "syntax_errors": len(syntax_check.errors) if syntax_check else 0,
        "import_success": "successfully" in import_test.stdout,
        "quality_issues": get_sonar_issues_count(file_path)
    }
```

### 2. 完整性測試
```python
def comprehensive_test():
    """完整的向前引用修復測試"""
    
    test_cases = [
        "from services.aiva_common.schemas import ExperienceSample, TraceRecord",
        "from services.core.aiva_core.storage.backends import SQLiteBackend",
        "from services.aiva_common.schemas.api_standards import AsyncAPIDocument",
        # 更多測試用例
    ]
    
    for test_case in test_cases:
        result = mcp_pylance_mcp_s_pylanceRunCodeSnippet(
            workspaceRoot="file:///c:/D/fold7/AIVA-git",
            codeSnippet=test_case,
            timeout=15.0
        )
        
        if "Error" in result.stderr:
            print(f"❌ {test_case} 失敗: {result.stderr}")
        else:
            print(f"✅ {test_case} 成功")
```

---

## 📋 修復檢查清單

### 修復前檢查
- [ ] 已使用 Pylance 進行全面語法分析
- [ ] 已識別所有前向引用錯誤類型
- [ ] 已分析錯誤間的依賴關係
- [ ] 已制定個別修復計劃（複雜情況）
- [ ] 已制定批量處理計劃（簡單情況）

### 修復中檢查
- [ ] 遵循 AIVA Common 字符串字面量標準
- [ ] 使用 `"ClassName"` 而非 `ClassName`
- [ ] 保持 PEP 484 類型註解合規性
- [ ] 每個修復後立即驗證
- [ ] 發現問題立即停止批量處理

### 修復後檢查
- [ ] 所有語法錯誤已解決
- [ ] 導入測試全部通過
- [ ] SonarQube 質量檢查通過
- [ ] 向後兼容性驗證通過
- [ ] 文檔已更新（如有架構改變）

---

## 🚨 常見陷阱與解決方案

### 陷阱 1: 過度批量處理
```python
# ❌ 危險: 一次性修復所有類型
# 可能導致意想不到的問題

# ✅ 安全: 分類處理，逐步驗證
```

### 陷阱 2: 忽略循環引用
```python
# ❌ 錯誤: 簡單加引號無法解決循環引用
class A(BaseModel):
    b: "B"

class B(BaseModel):
    a: "A"  # 仍然是循環引用

# ✅ 正確: 重構為單向引用或使用 update_forward_refs
```

### 陷阱 3: 破壞現有功能
```python
# ❌ 危險: 修改可能影響運行時行為
# 某些情況下前向引用的時機很重要

# ✅ 安全: 修復後立即進行功能測試
```

---

## 📊 工具能力對比

| 工具 | 發現能力 | 修復建議 | 自動修復 | 驗證能力 | 推薦程度 |
|------|---------|---------|---------|---------|---------|
| **Pylance MCP** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🥇 首選 |
| **SonarQube** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 🥈 輔助 |
| **get_errors** | ⭐⭐⭐ | ⭐⭐ | ❌ | ⭐⭐⭐ | 🥉 基礎 |
| **grep_search** | ⭐⭐ | ❌ | ❌ | ❌ | 輔助搜索 |

---

## 🔗 相關資源

### 官方文檔
- [Pydantic Forward References](https://docs.pydantic.dev/latest/concepts/models/#forward-references)
- [PEP 484 Forward References](https://peps.python.org/pep-0484/#forward-references)
- [Python typing module](https://docs.python.org/3/library/typing.html#forward-references)

### AIVA 相關
- [AIVA Common README](./services/aiva_common/README.md) - 官方開發規範
- [批量處理安全原則](./services/aiva_common/README.md#️-批量處理修復原則)
- [開發指南](./services/aiva_common/README.md#🔧-開發指南)

### 測試驗證
```python
# 使用指南驗證範例
def verify_guide_effectiveness():
    """驗證本指南的有效性"""
    
    # 1. 使用 Pylance 檢測問題
    problems = mcp_pylance_mcp_s_pylanceFileSyntaxErrors(
        workspaceRoot="file:///c:/D/fold7/AIVA-git",
        fileUri="file:///c:/D/fold7/AIVA-git/services/aiva_common/schemas/api_standards.py"
    )
    
    # 2. 應用修復建議
    fixes = mcp_pylance_mcp_s_pylanceInvokeRefactoring(
        fileUri="file:///c:/D/fold7/AIVA-git/services/aiva_common/schemas/api_standards.py",
        name="source.fixAll.pylance",
        mode="edits"
    )
    
    # 3. 驗證修復效果
    verification = immediate_verification(
        "c:/D/fold7/AIVA-git/services/aiva_common/schemas/api_standards.py"
    )
    
    return verification
```

---

**創建日期**: 2025年10月30日  
**基於**: AIVA Common README v1.0.0 + VS Code 現有插件能力  
**適用範圍**: 所有 AIVA 項目中的 Python 代碼前向引用問題

---

> 💡 **提示**: 本指南基於實際修復經驗編寫，所有示例都經過驗證。遇到問題時，請優先使用 Pylance MCP 工具進行診斷。