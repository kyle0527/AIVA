# AIVA 通用 AI 修復指南
**版本**: 2.0  
**日期**: 2025-11-10  
**適用範圍**: 通用性AI代碼修復方法論  
**設計原則**: 系統化分析、分類處理、漸進式修復

---

## 🎯 修復指南總覽

本指南為通則性修復方法論，適用於任何AI系統的代碼品質提升。基於AIVA五大模組最佳實踐，整合複雜度管理、重構技術和品質標準，建立系統化修復流程。

### 🔄 核心修復原則

1. **先全面分析並且將問題分類** - 系統性問題發現與歸類
2. **無法批量處理的先進行** - 優先處理複雜個案
3. **完成後才進行批量處理** - 避免批量修復衝突
4. **原則上一次一個腳本** - 避免錯誤累積
5. **單一事實原則** - 確保函數和名稱一致性

---

## � 階段一：全面分析與問題分類

### 1.1 系統化代碼掃描

#### 📊 **多維度問題發現**
```python
class UniversalCodeAnalyzer:
    """通用代碼分析器 - 基於AIVA最佳實踐"""
    
    def comprehensive_analysis(self, target_path: str) -> Dict[str, List]:
        """全面分析並分類問題"""
        problems = {
            'syntax_errors': [],      # 語法錯誤
            'type_issues': [],        # 型別問題  
            'complexity_issues': [],  # 複雜度問題
            'architecture_issues': [],# 架構問題
            'import_issues': [],      # 匯入問題
            'async_issues': [],       # 異步問題
            'unused_issues': [],      # 未使用問題
        }
        
        # 使用 Pylance 進行深度分析
        syntax_errors = self._check_syntax_errors(target_path)
        type_errors = self._analyze_type_consistency(target_path)
        complexity_metrics = self._calculate_complexity_metrics(target_path)
        
        # 按嚴重程度和處理複雜度分類
        return self._categorize_by_urgency_and_complexity(problems)
```

#### 🏷️ **問題分類維度**

**按修復複雜度分類:**
- **🟢 簡單批量** - 可自動化批量處理
  - 空F-string修復
  - 未使用import清理
  - 基礎型別註解統一
  
- **🟡 中等個別** - 需要邏輯判斷
  - 簡單async函數調整
  - 基礎匯入路徑修復
  - 未使用參數處理
  
- **🔴 複雜手動** - 需要深度重構
  - 高複雜度函數分解 (>15認知複雜度)
  - 架構解耦重設計
  - 循環引用重構

**按影響範圍分類:**
- **局部影響** - 單文件內修復
- **模組影響** - 影響同一模組
- **系統影響** - 跨模組依賴修復

### 1.2 複雜度評估標準

#### 📏 **基於AIVA品質標準**
```python
def assess_complexity_level(analysis_result: Dict) -> str:
    """評估複雜度等級 - 基於AIVA五大模組標準"""
    
    cognitive_complexity = analysis_result.get('cognitive_complexity', 0)
    cyclomatic_complexity = analysis_result.get('cyclomatic_complexity', 0)
    function_length = analysis_result.get('function_length', 0)
    nesting_depth = analysis_result.get('nesting_depth', 0)
    
    # AIVA企業級品質標準
    if cognitive_complexity <= 15 and cyclomatic_complexity <= 10:
        return "SIMPLE_BATCH"      # 可批量處理
    elif cognitive_complexity <= 25 and function_length <= 100:
        return "MODERATE_INDIVIDUAL"  # 需個別處理
    else:
        return "COMPLEX_MANUAL"    # 需手動重構
```

#### 🎯 **重構觸發點 (基於AIVA實踐)**
- **建議重構**: 複雜度 >10
- **必須重構**: 複雜度 >15 
- **強制拆分**: 複雜度 >25

---

## ⚡ 階段二：個別處理複雜問題

### 2.1 高複雜度函數重構

#### 🔧 **Extract Method Pattern (基於AIVA core模組實踐)**
```python
# ❌ 重構前: 高複雜度函數 (複雜度 > 15)
def complex_ai_analysis(self, data: Dict) -> Dict:
    # 118行代碼，認知複雜度 29
    result = {}
    
    # 資料預處理 (15行)
    if data.get('type') == 'neural':
        # 複雜預處理邏輯...
        
    # 特徵提取 (25行)  
    if data.get('features'):
        # 複雜特徵提取...
        
    # AI推理 (30行)
    if self.ai_model:
        # 複雜推理邏輯...
        
    # 結果後處理 (20行)
    if result.get('predictions'):
        # 複雜後處理...
        
    return result

# ✅ 重構後: 職責分離，複雜度 ≤15
def complex_ai_analysis(self, data: Dict) -> Dict:
    """主控函數 - 複雜度降至 8"""
    preprocessed_data = self._preprocess_input_data(data)
    features = self._extract_advanced_features(preprocessed_data) 
    predictions = self._perform_ai_inference(features)
    result = self._postprocess_predictions(predictions)
    return result

def _preprocess_input_data(self, data: Dict) -> Dict:
    """資料預處理 - 複雜度 5"""
    # 15行專門處理邏輯...

def _extract_advanced_features(self, data: Dict) -> np.ndarray:
    """特徵提取 - 複雜度 8"""
    # 25行特徵提取邏輯...

def _perform_ai_inference(self, features: np.ndarray) -> Dict:
    """AI推理 - 複雜度 12"""
    # 30行推理邏輯...

def _postprocess_predictions(self, predictions: Dict) -> Dict:
    """結果後處理 - 複雜度 6"""
    # 20行後處理邏輯...
```

#### 🏗️ **Strategy Pattern 應用**
```python
# 基於AIVA決策模組最佳實踐
class ComplexityReductionStrategy:
    """複雜度降低策略模式"""
    
    def __init__(self):
        self.strategies = {
            'extract_method': self._extract_method_refactoring,
            'strategy_pattern': self._apply_strategy_pattern,
            'early_return': self._apply_early_return,
            'delegate_pattern': self._apply_delegation
        }
    
    def reduce_complexity(self, function_node: ast.FunctionDef, 
                         complexity_score: int) -> List[str]:
        """選擇適當的複雜度降低策略"""
        
        if complexity_score > 25:
            return ['extract_method', 'strategy_pattern']
        elif complexity_score > 15:
            return ['extract_method', 'early_return']
        else:
            return ['early_return']
```

### 2.2 架構解耦重設計

#### 🔗 **循環引用解決 (基於AIVA integration模組)**
```python
# ❌ 問題: 循環引用
# module_a.py
from module_b import ClassB

class ClassA:
    def __init__(self):
        self.b = ClassB()

# module_b.py  
from module_a import ClassA  # 循環引用!

class ClassB:
    def __init__(self):
        self.a = ClassA()

# ✅ 解決: 依賴倒轉 + 接口抽象
# interfaces.py
from abc import ABC, abstractmethod

class ComponentInterface(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

# module_a.py
from interfaces import ComponentInterface

class ClassA:
    def __init__(self, component_b: ComponentInterface):
        self.b = component_b  # 依賴注入

# module_b.py
from interfaces import ComponentInterface

class ClassB(ComponentInterface):
    def process(self, data: Any) -> Any:
        return f"Processed: {data}"
```

### 2.3 深度型別推導修復

#### 🔍 **複雜型別推導簡化**
```python
# ❌ 複雜型別推導問題
def complex_type_function(
    data: Dict[str, Union[List[Optional[Dict[str, Any]]], 
                         Callable[[str, int], Optional[Tuple[str, ...]]]]],
    callback: Optional[Callable[[Dict[str, Any]], 
                               Future[Optional[List[Dict[str, Union[str, int]]]]]]]
) -> Optional[Dict[str, Union[str, List[Dict[str, Any]]]]]:
    # 過於複雜的型別推導

# ✅ 使用類型別名簡化
from typing import TypeAlias, Dict, List, Union, Optional, Callable, Any

# 創建類型別名
DataValue: TypeAlias = Union[List[Optional[Dict[str, Any]]], 
                            Callable[[str, int], Optional[Tuple[str, ...]]]]
InputData: TypeAlias = Dict[str, DataValue]
ProcessCallback: TypeAlias = Callable[[Dict[str, Any]], 
                                     Future[Optional[List[Dict[str, Union[str, int]]]]]]
ResultData: TypeAlias = Dict[str, Union[str, List[Dict[str, Any]]]]

def simplified_type_function(
    data: InputData,
    callback: Optional[ProcessCallback] = None
) -> Optional[ResultData]:
    """簡化後的型別註解，提高可讀性"""
    # 清晰的邏輯實現...
```

---

## 🔄 階段三：批量處理標準化問題

### 3.1 批量修復規則

#### ⚡ **可安全批量處理的問題類型**

**🟢 低風險批量操作:**
```python
class BatchProcessor:
    """安全的批量修復處理器"""
    
    def safe_batch_operations(self, file_list: List[str]) -> Dict[str, int]:
        """安全的批量操作清單"""
        return {
            'empty_f_strings': self._fix_empty_f_strings_batch,
            'unused_imports': self._remove_unused_imports_batch,
            'basic_type_hints': self._add_basic_type_hints_batch,
            'docstring_format': self._standardize_docstrings_batch,
            'import_sorting': self._sort_imports_batch
        }
    
    def _fix_empty_f_strings_batch(self, files: List[str]) -> int:
        """批量修復空F-string - 低風險操作"""
        fixes = 0
        patterns = [
            (r'f"([^{]*)"', r'"\1"'),  # f"text" -> "text"
            (r"f'([^{]*)'", r"'\1'"),  # f'text' -> 'text'
        ]
        
        for file_path in files:
            content = self._read_file(file_path)
            for pattern, replacement in patterns:
                if self._is_safe_f_string_replacement(content, pattern):
                    content = re.sub(pattern, replacement, content)
                    fixes += 1
            self._write_file(file_path, content)
        
        return fixes
```

#### 🟡 **需要驗證的批量操作**
```python
def cautious_batch_operations(self, file_list: List[str]) -> Dict[str, int]:
    """需要逐一驗證的批量操作"""
    return {
        'simple_async_removal': self._remove_simple_async_batch,
        'import_path_standardization': self._standardize_imports_batch,
        'unused_parameter_marking': self._mark_unused_parameters_batch
    }

def _remove_simple_async_batch(self, files: List[str]) -> int:
    """批量移除簡單async - 需要驗證每個案例"""
    fixes = 0
    
    for file_path in files:
        # 逐個文件分析，確保安全
        if self._is_safe_async_removal(file_path):
            content = self._read_file(file_path)
            content = self._remove_unnecessary_async(content)
            
            # 語法驗證
            if self._validate_syntax(content):
                self._write_file(file_path, content)
                fixes += 1
            else:
                print(f"⚠️ 語法驗證失敗，跳過: {file_path}")
    
    return fixes
```

### 3.2 批量處理驗證機制

#### 🧪 **多層驗證檢查**
```python
class BatchValidationPipeline:
    """批量處理驗證管道"""
    
    def validate_batch_changes(self, file_path: str, 
                             original: str, modified: str) -> bool:
        """多層驗證批量修改"""
        
        # 第一層：語法檢查
        if not self._syntax_check(modified):
            return False
            
        # 第二層：導入檢查  
        if not self._import_resolution_check(modified):
            return False
            
        # 第三層：行為一致性檢查
        if not self._behavior_consistency_check(original, modified):
            return False
            
        # 第四層：型別檢查
        if not self._type_check(modified):
            return False
            
        return True
    
    def _behavior_consistency_check(self, original: str, modified: str) -> bool:
        """確保修改不改變程序行為"""
        try:
            # 編譯兩個版本，比較AST結構
            original_ast = ast.parse(original)
            modified_ast = ast.parse(modified)
            
            # 檢查關鍵結構是否一致
            return self._compare_ast_structure(original_ast, modified_ast)
        except:
            return False
```

---

## 🎯 階段四：單一事實原則修復流程

### 4.1 函數名稱與接口一致性

#### 🔗 **AI核心連接一致性檢查**
```python
class SingleTruthValidator:
    """單一事實原則驗證器"""
    
    def __init__(self):
        # AIVA核心接口標準
        self.core_interfaces = {
            'RealAICore': {
                'methods': ['forward', 'forward_with_aux', 'predict'],
                'expected_signature': {
                    'forward': 'forward(self, x: torch.Tensor) -> torch.Tensor',
                    'forward_with_aux': 'forward_with_aux(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]'
                }
            }
        }
    
    def validate_interface_consistency(self, orchestrator_path: str, 
                                     core_path: str) -> Dict[str, List[str]]:
        """驗證接口調用一致性"""
        issues = {
            'method_mismatches': [],
            'signature_mismatches': [],
            'missing_methods': []
        }
        
        # 分析orchestrator中的調用
        orchestrator_calls = self._extract_method_calls(orchestrator_path)
        
        # 分析core中的實際定義
        core_definitions = self._extract_method_definitions(core_path)
        
        # 檢查一致性
        for call in orchestrator_calls:
            if call['method'] not in core_definitions:
                issues['missing_methods'].append(call['method'])
            elif not self._signatures_match(call, core_definitions[call['method']]):
                issues['signature_mismatches'].append(call['method'])
        
        return issues
    
    def generate_consistency_fixes(self, issues: Dict[str, List[str]]) -> List[str]:
        """生成一致性修復建議"""
        fixes = []
        
        for missing_method in issues['missing_methods']:
            fixes.append(f"需要在AI核心中實現方法: {missing_method}")
        
        for mismatched_method in issues['signature_mismatches']:
            fixes.append(f"需要統一方法簽名: {mismatched_method}")
        
        return fixes
```

#### 🏗️ **統一命名標準**
```python
class NamingStandardizer:
    """命名標準化器 - 確保單一事實原則"""
    
    def __init__(self):
        # AIVA命名規範
        self.naming_standards = {
            'ai_core_instance': 'ai_core',          # 統一AI核心實例名
            'capability_prefix': 'execute_',        # 能力方法前綴
            'analysis_prefix': 'analyze_',          # 分析方法前綴
            'data_suffix': '_data',                 # 資料變數後綴
            'result_suffix': '_result'              # 結果變數後綴
        }
    
    def standardize_naming(self, file_content: str) -> Tuple[str, List[str]]:
        """標準化命名，確保一致性"""
        changes = []
        
        # 統一AI核心實例命名
        patterns = [
            (r'self\.real_ai_core', 'self.ai_core'),
            (r'self\.neural_core', 'self.ai_core'),
            (r'self\.ai_engine', 'self.ai_core'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, file_content):
                file_content = re.sub(pattern, replacement, file_content)
                changes.append(f"統一AI核心實例名: {pattern} -> {replacement}")
        
        return file_content, changes
```

### 4.2 依賴關係單一化

#### 🔄 **依賴注入標準化**
```python
# ✅ 標準化的依賴注入模式
class CapabilityOrchestrator:
    """能力協調器 - 單一事實原則設計"""
    
    def __init__(self, ai_core: Optional['RealAICore'] = None):
        """構造函數：明確依賴關係"""
        self.ai_core = ai_core or self._initialize_ai_core()
        self.capabilities = self._register_capabilities()
        
    def _initialize_ai_core(self) -> 'RealAICore':
        """統一的AI核心初始化 - 單一事實來源"""
        try:
            from services.core.aiva_core.ai_engine.real_neural_core import RealAICore
            return RealAICore()
        except ImportError:
            # 優雅降級，保持接口一致
            return MockAICore()
    
    def execute_capability(self, capability_name: str, **kwargs) -> CapabilityResult:
        """統一的能力執行接口 - 避免重複定義"""
        if capability_name not in self.capabilities:
            return CapabilityResult(
                success=False,
                error=f"未知能力: {capability_name}"
            )
        
        # 單一事實：所有能力都通過相同的接口執行
        capability = self.capabilities[capability_name]
        return capability.execute(**kwargs)
```

---

## 📊 修復品質驗證標準

### 5.1 基於AIVA品質標準

#### 🏆 **企業級品質指標**
```python
class QualityMetricsValidator:
    """品質指標驗證器 - 基於AIVA五大模組標準"""
    
    def __init__(self):
        # AIVA品質里程碑標準
        self.quality_standards = {
            'cognitive_complexity': {'max': 15, 'target': 10},
            'cyclomatic_complexity': {'max': 10, 'target': 6},
            'function_length': {'max': 50, 'target': 30},
            'nesting_depth': {'max': 4, 'target': 3},
            'test_coverage': {'min': 80, 'target': 90},
            'documentation_coverage': {'min': 75, 'target': 85}
        }
    
    def validate_repair_quality(self, file_path: str) -> Dict[str, Any]:
        """驗證修復後的品質"""
        metrics = self._calculate_metrics(file_path)
        
        quality_report = {
            'overall_score': 0,
            'passed_standards': [],
            'failed_standards': [],
            'recommendations': []
        }
        
        for metric_name, values in self.quality_standards.items():
            current_value = metrics.get(metric_name, 0)
            
            if 'max' in values and current_value <= values['max']:
                quality_report['passed_standards'].append(metric_name)
            elif 'min' in values and current_value >= values['min']:
                quality_report['passed_standards'].append(metric_name)
            else:
                quality_report['failed_standards'].append({
                    'metric': metric_name,
                    'current': current_value,
                    'required': values,
                    'action': self._get_improvement_action(metric_name, current_value, values)
                })
        
        quality_report['overall_score'] = (
            len(quality_report['passed_standards']) / 
            len(self.quality_standards) * 100
        )
        
        return quality_report
```

### 5.2 持續品質保證

#### 🔄 **迭代改進機制**
```python
class ContinuousQualityImprovement:
    """持續品質改進系統"""
    
    def __init__(self):
        self.quality_history = []
        self.improvement_patterns = []
    
    def track_quality_evolution(self, repair_session: Dict) -> None:
        """追蹤品質演進"""
        session_metrics = {
            'timestamp': datetime.now(),
            'files_modified': repair_session['files_count'],
            'issues_fixed': repair_session['issues_fixed'],
            'quality_improvement': repair_session['quality_delta'],
            'successful_batch_operations': repair_session['batch_success_rate']
        }
        
        self.quality_history.append(session_metrics)
        self._analyze_improvement_patterns()
    
    def generate_next_iteration_plan(self) -> Dict[str, Any]:
        """基於歷史數據生成下一輪改進計劃"""
        if len(self.quality_history) < 2:
            return self._default_improvement_plan()
        
        latest = self.quality_history[-1]
        previous = self.quality_history[-2]
        
        plan = {
            'focus_areas': self._identify_focus_areas(latest, previous),
            'batch_strategy': self._optimize_batch_strategy(),
            'individual_priorities': self._prioritize_individual_fixes(),
            'risk_mitigation': self._update_risk_mitigation()
        }
        
        return plan
```

---

## 🚀 實施執行計劃

### 階段化實施策略

#### �📋 **第一階段：分析與準備 (30分鐘)**
1. **全面系統掃描** - 使用Pylance進行完整分析
2. **問題分類歸檔** - 按複雜度和影響範圍分類
3. **修復策略制定** - 決定批量vs個別處理順序
4. **風險評估** - 識別高風險修復項目

#### ⚡ **第二階段：個別複雜問題處理 (2小時)**
1. **高複雜度重構** - 處理>15複雜度函數
2. **架構調整** - 解決循環引用和設計問題
3. **接口一致性** - 確保AI核心連接正確
4. **逐一驗證** - 每個修復都進行完整測試

#### 🔄 **第三階段：批量標準化處理 (1小時)**
1. **安全批量操作** - 處理低風險標準化問題
2. **分批驗證** - 每批修復後進行驗證
3. **品質檢查** - 確保批量修復不引入新問題

#### 🏆 **第四階段：品質驗證與文檔 (30分鐘)**
1. **全面品質評估** - 對照AIVA品質標準
2. **修復報告生成** - 詳細記錄修復過程和效果
3. **未來改進計劃** - 為下一輪修復做準備

### 執行檢查清單

#### ✅ **每個修復階段必須完成的檢查**
- [ ] 語法正確性檢查
- [ ] 型別一致性驗證  
- [ ] 功能行為保持不變
- [ ] 接口調用正確性
- [ ] 認知複雜度符合標準
- [ ] 單一事實原則遵循
- [ ] 備份文件已創建
- [ ] 修復前後對比記錄

### 🚨 **P0 - 緊急修復** (立即執行)
1. **Async函數濫用** - 移除不必要的async標記
2. **匯入路徑錯誤** - 修正相對路徑匯入
3. **未使用參數** - 移除或標記未使用參數
4. **型別註解不一致** - 統一型別標記

### ⚡ **P1 - 高優先度** (今天完成)
1. **認知複雜度過高** - 簡化複雜函數
2. **F-string濫用** - 修正空的f-string
3. **未使用變量** - 清理未使用局部變量

### 🔧 **P2 - 中優先度** (本週完成)
1. **架構簡化** - 簡化過度複雜的設計
2. **錯誤處理** - 強化異常處理機制

---

## 🛠️ 具體修復規則

### 1. **Async函數修復規則**

#### ❌ 錯誤模式
```python
# 錯誤: async函數內沒有任何await調用
async def execute_static_analysis(self, target_code: str) -> CapabilityResult:
    # 沒有任何異步操作
    return CapabilityResult(...)
```

#### ✅ 正確修復
```python
# 方案A: 移除async (推薦)
def execute_static_analysis(self, target_code: str) -> CapabilityResult:
    """執行靜態分析"""
    return CapabilityResult(...)

# 方案B: 添加真實的異步操作
async def execute_static_analysis(self, target_code: str) -> CapabilityResult:
    """執行異步靜態分析"""
    await asyncio.sleep(0)  # 讓出控制權
    # 或者調用真實的異步API
    return CapabilityResult(...)
```

### 2. **匯入路徑修復規則**

#### ❌ 錯誤模式
```python
# 錯誤: 直接匯入找不到的模組
from real_neural_core import RealAICore

# 錯誤: services路徑匯入
from services.core.aiva_core.ai_engine.real_neural_core import RealAICore
```

#### ✅ 正確修復
```python
# 正確: 使用相對路徑或動態路徑
try:
    # 嘗試相對路徑
    from .real_neural_core import RealAICore
except ImportError:
    try:
        # 嘗試絕對路徑
        from services.core.aiva_core.ai_engine.real_neural_core import RealAICore
    except ImportError:
        # 優雅降級
        RealAICore = None
        logger.warning("無法載入RealAICore，將使用模擬模式")
```

### 3. **未使用參數修復規則**

#### ❌ 錯誤模式
```python
def execute_static_analysis(self, target_code: str) -> CapabilityResult:
    # target_code未使用
    return CapabilityResult(...)
```

#### ✅ 正確修復
```python
# 方案A: 使用參數
def execute_static_analysis(self, target_code: str) -> CapabilityResult:
    # 實際使用參數
    analysis_result = analyze_code(target_code)
    return CapabilityResult(data={'code': target_code, 'result': analysis_result})

# 方案B: 標記為未使用 (適用於接口要求)
def execute_static_analysis(self, target_code: str) -> CapabilityResult:
    _ = target_code  # 明確標記為未使用
    return CapabilityResult(...)

# 方案C: 改為通用參數
def execute_static_analysis(self, **kwargs) -> CapabilityResult:
    target_code = kwargs.get('target_code', '')
    return CapabilityResult(...)
```

### 4. **型別註解修復規則**

#### ❌ 錯誤模式
```python
async def make_ai_decision(self, feature_vector: np.ndarray) -> Dict:
    # 可能返回None，但標記為必需Dict
    if not self.ai_core:
        return None
```

#### ✅ 正確修復
```python
from typing import Dict, Optional, Any

async def make_ai_decision(self, feature_vector: np.ndarray) -> Optional[Dict[str, Any]]:
    """AI決策，可能返回None"""
    if not self.ai_core:
        return None
    
    return {
        'decision': 'analysis_complete',
        'confidence': 0.95
    }
```

### 5. **F-string修復規則**

#### ❌ 錯誤模式
```python
logger.info(f"✅ 提取512維特徵向量完成")  # 空的f-string
print(f"\n📊 分析結果總覽:")
```

#### ✅ 正確修復
```python
logger.info("✅ 提取512維特徵向量完成")
print("\n📊 分析結果總覽:")

# 或者添加實際變量
dimension = 512
logger.info(f"✅ 提取{dimension}維特徵向量完成")
```

---

## 🔄 修復執行流程

### 階段1: 自動修復腳本
```python
#!/usr/bin/env python3
"""
AIVA AI修復腳本
基於修復指南自動修復常見問題
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple

class AIVACodeFixer:
    def __init__(self, target_file: str):
        self.target_file = Path(target_file)
        self.backup_file = self.target_file.with_suffix('.py.backup')
        
    def fix_async_functions(self, content: str) -> Tuple[str, int]:
        """修復不必要的async函數"""
        fixes = 0
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    # 檢查函數體是否有await調用
                    has_await = any(
                        isinstance(n, ast.Await) for n in ast.walk(node)
                    )
                    
                    if not has_await:
                        # 移除async關鍵字
                        func_line = node.lineno - 1
                        if func_line < len(lines):
                            lines[func_line] = lines[func_line].replace('async def', 'def')
                            fixes += 1
                            
        except SyntaxError:
            pass  # 語法錯誤，跳過
            
        return '\n'.join(lines), fixes
    
    def fix_import_paths(self, content: str) -> Tuple[str, int]:
        """修復匯入路徑"""
        fixes = 0
        
        # 修復real_neural_core匯入
        if 'from real_neural_core import' in content:
            content = content.replace(
                'from real_neural_core import',
                'try:\n    from .real_neural_core import'
            )
            content += '\nexcept ImportError:\n    RealAICore = None'
            fixes += 1
            
        return content, fixes
    
    def fix_unused_parameters(self, content: str) -> Tuple[str, int]:
        """修復未使用參數"""
        fixes = 0
        
        # 簡單的未使用參數檢測和修復
        patterns = [
            (r'def (\w+)\(self, (target_code|target_url|target_host): str\)', 
             r'def \1(self, **kwargs)'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes += 1
                
        return content, fixes
    
    def fix_type_annotations(self, content: str) -> Tuple[str, int]:
        """修復型別註解"""
        fixes = 0
        
        # 修復返回型別
        content = content.replace(
            ') -> Dict:',
            ') -> Optional[Dict[str, Any]]:'
        )
        
        # 確保有正確的import
        if 'Optional[Dict' in content and 'from typing import' in content:
            if 'Optional' not in content.split('from typing import')[1].split('\n')[0]:
                content = content.replace(
                    'from typing import',
                    'from typing import Optional,'
                )
                fixes += 1
                
        return content, fixes
    
    def fix_f_strings(self, content: str) -> Tuple[str, int]:
        """修復空的f-string"""
        fixes = 0
        
        # 檢測空的f-string
        empty_f_patterns = [
            (r'f"([^{]*)"', r'"\1"'),  # f"text" -> "text"
            (r"f'([^{]*)'", r"'\1'"),  # f'text' -> 'text'
        ]
        
        for pattern, replacement in empty_f_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if '{' not in match.group(1):  # 確實是空f-string
                    content = content.replace(match.group(0), match.group(1))
                    fixes += 1
                    
        return content, fixes
    
    def apply_all_fixes(self) -> Dict[str, int]:
        """應用所有修復"""
        if not self.target_file.exists():
            return {'error': 'File not found'}
            
        # 備份原文件
        with open(self.target_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        with open(self.backup_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
            
        # 應用修復
        content = original_content
        fixes_summary = {}
        
        content, fixes = self.fix_async_functions(content)
        fixes_summary['async_functions'] = fixes
        
        content, fixes = self.fix_import_paths(content)
        fixes_summary['import_paths'] = fixes
        
        content, fixes = self.fix_unused_parameters(content)
        fixes_summary['unused_parameters'] = fixes
        
        content, fixes = self.fix_type_annotations(content)
        fixes_summary['type_annotations'] = fixes
        
        content, fixes = self.fix_f_strings(content)
        fixes_summary['f_strings'] = fixes
        
        # 寫入修復後的內容
        if sum(fixes_summary.values()) > 0:
            with open(self.target_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
        return fixes_summary

def main():
    # 修復目標文件
    target_files = [
        'C:/D/fold7/AIVA-git/aiva_capability_orchestrator.py',
        'C:/D/fold7/AIVA-git/services/core/aiva_core/ai_engine/real_neural_core.py'
    ]
    
    total_fixes = 0
    
    for file_path in target_files:
        print(f"🔧 修復檔案: {file_path}")
        fixer = AIVACodeFixer(file_path)
        fixes = fixer.apply_all_fixes()
        
        if 'error' in fixes:
            print(f"❌ 修復失敗: {fixes['error']}")
            continue
            
        file_total = sum(fixes.values())
        total_fixes += file_total
        
        print(f"   ✅ 修復 {file_total} 個問題:")
        for fix_type, count in fixes.items():
            if count > 0:
                print(f"      - {fix_type}: {count}")
                
    print(f"\n🎉 修復完成! 共修復 {total_fixes} 個問題")
    return total_fixes > 0

if __name__ == "__main__":
    main()
```

### 階段2: 手動驗證
1. **執行修復腳本**
2. **檢查語法錯誤**
3. **運行基本測試**
4. **驗證功能完整性**

### 階段3: 迭代改進
1. **收集修復效果**
2. **更新修復規則**
3. **完善自動化工具**

---

## 📊 修復驗證標準

### 成功標準
- [ ] Lint錯誤 < 5個
- [ ] 所有async函數都有實際await調用
- [ ] 匯入路徑正確解析
- [ ] 型別註解一致性 > 90%
- [ ] 認知複雜度 < 15

### 回歸測試
```python
def validate_fixes():
    """驗證修復效果"""
    # 1. 語法檢查
    try:
        import aiva_capability_orchestrator
        print("✅ aiva_capability_orchestrator 語法正確")
    except SyntaxError as e:
        print(f"❌ 語法錯誤: {e}")
        
    # 2. 匯入檢查
    try:
        from services.core.aiva_core.ai_engine import real_neural_core
        print("✅ real_neural_core 匯入成功")
    except ImportError as e:
        print(f"❌ 匯入錯誤: {e}")
        
    # 3. 基本功能檢查
    # ... 添加具體功能測試
```

---

## 🚀 執行計劃

### 立即執行 (接下來30分鐘)
1. ✅ 創建修復指南
2. 🔧 執行自動修復腳本
3. 🧪 驗證修復效果
4. 📊 生成修復報告

### 後續優化 (今天內)
1. 🔍 深度代碼檢查
2. 🛠️ 手動修復複雜問題
3. 📈 性能優化
4. 📚 文檔更新

---

## 📚 修復技術參考

### 基於AIVA五大模組最佳實踐

#### 🔧 **重構技術應用清單**
- **Extract Method Pattern**: 大型函數分解為專門化小函數
- **Strategy Pattern**: 複雜條件判斷用策略模式替代  
- **Early Return Pattern**: 減少嵌套層級和認知負擔
- **Delegation Pattern**: 委託模式降低耦合度
- **Interface Segregation**: 接口分離提高模組化程度

#### 📊 **複雜度控制策略**
```python
# AIVA認知複雜度標準
COMPLEXITY_THRESHOLDS = {
    'SIMPLE': 5,       # 簡單函數
    'MODERATE': 10,    # 中等複雜度 - 建議重構觸發點
    'COMPLEX': 15,     # 複雜函數 - 企業標準上限
    'CRITICAL': 20,    # 危險區域 - 必須重構
    'EMERGENCY': 25    # 緊急重構 - 立即處理
}

def assess_refactoring_urgency(complexity_score: int) -> str:
    """基於AIVA標準評估重構急迫性"""
    if complexity_score >= COMPLEXITY_THRESHOLDS['EMERGENCY']:
        return "立即重構 - 使用Extract Method + Strategy Pattern"
    elif complexity_score >= COMPLEXITY_THRESHOLDS['CRITICAL']:
        return "本週內重構 - 使用Extract Method Pattern"
    elif complexity_score >= COMPLEXITY_THRESHOLDS['COMPLEX']:
        return "本月內重構 - 使用Early Return Pattern"
    elif complexity_score >= COMPLEXITY_THRESHOLDS['MODERATE']:
        return "建議重構 - 簡化邏輯結構"
    else:
        return "品質良好 - 維持現狀"
```

#### 🏆 **AIVA品質里程碑參考**
| 模組類型 | 重構前最高複雜度 | 重構後複雜度 | 改善幅度 | 應用技術 |
|---------|------------------|--------------|----------|----------|
| Bio Neuron Core | 97 | ≤15 | 84% | Extract Method + Strategy |
| AI Controller | 77 | ≤12 | 84% | Delegation + Early Return |
| Decision Agent | 75 | ≤10 | 86% | Strategy Pattern |
| Perception Module | 29 | ≤15 | 48% | Extract Method |
| Knowledge Module | 25 | ≤8 | 68% | Interface Segregation |

---

## 🛡️ 風險防範與回復機制

### 修復風險等級

#### 🟢 **低風險操作** (可放心批量處理)
- 空F-string清理
- 未使用import移除  
- docstring格式統一
- 基礎型別註解添加

#### 🟡 **中風險操作** (需要逐一驗證)
- async函數調整
- 匯入路徑修復
- 未使用參數處理
- 簡單重構操作

#### 🔴 **高風險操作** (需要手動處理)
- 複雜函數重構
- 架構設計調整
- 循環引用解決
- 接口定義變更

### 回復與回滾機制

#### 💾 **多層備份策略**
```python
class RepairBackupManager:
    """修復備份管理器"""
    
    def create_comprehensive_backup(self, target_path: str) -> str:
        """創建全面備份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{target_path}_repair_backup_{timestamp}"
        
        # 完整目錄備份
        shutil.copytree(target_path, backup_dir)
        
        # 創建修復點記錄
        repair_log = {
            'timestamp': timestamp,
            'target_path': target_path,
            'backup_path': backup_dir,
            'git_commit': self._get_current_git_commit(),
            'quality_metrics_before': self._measure_quality(target_path)
        }
        
        with open(f"{backup_dir}/repair_session.json", 'w') as f:
            json.dump(repair_log, f, indent=2)
        
        return backup_dir
    
    def rollback_if_quality_degraded(self, backup_path: str, 
                                   current_path: str) -> bool:
        """品質下降時自動回滾"""
        
        # 載入修復前品質指標
        with open(f"{backup_path}/repair_session.json", 'r') as f:
            repair_log = json.load(f)
        
        before_quality = repair_log['quality_metrics_before']
        after_quality = self._measure_quality(current_path)
        
        # 檢查是否品質下降
        if self._is_quality_degraded(before_quality, after_quality):
            print("⚠️ 檢測到品質下降，自動回滾...")
            shutil.rmtree(current_path)
            shutil.copytree(backup_path, current_path)
            return True
        
        return False
```

---

## 📈 成效追蹤與持續改進

### 品質改進追蹤

#### 📊 **修復成效指標**
```python
class RepairEffectivenessTracker:
    """修復成效追蹤器"""
    
    def calculate_repair_roi(self, repair_session: Dict) -> Dict[str, float]:
        """計算修復投資回報率"""
        
        # 技術債務降低程度
        technical_debt_reduction = (
            repair_session['complexity_before'] - 
            repair_session['complexity_after']
        ) / repair_session['complexity_before']
        
        # 維護成本降低估算
        maintenance_cost_reduction = technical_debt_reduction * 0.3
        
        # 開發效率提升
        dev_efficiency_gain = (
            repair_session['lint_errors_fixed'] * 0.1 +
            repair_session['type_errors_fixed'] * 0.15 +
            repair_session['complexity_improvements'] * 0.25
        )
        
        return {
            'technical_debt_reduction': technical_debt_reduction,
            'maintenance_cost_reduction': maintenance_cost_reduction,
            'dev_efficiency_gain': dev_efficiency_gain,
            'overall_roi': (technical_debt_reduction + dev_efficiency_gain) / 2
        }
    
    def generate_improvement_insights(self, history: List[Dict]) -> List[str]:
        """生成改進洞察"""
        insights = []
        
        if len(history) >= 3:
            recent_sessions = history[-3:]
            
            # 分析修復效率趨勢
            efficiency_trend = [s['overall_roi'] for s in recent_sessions]
            if all(efficiency_trend[i] < efficiency_trend[i+1] for i in range(len(efficiency_trend)-1)):
                insights.append("🎯 修復效率持續提升，建議繼續當前策略")
            
            # 分析常見問題模式
            common_issues = Counter()
            for session in recent_sessions:
                common_issues.update(session['issue_types'])
            
            most_common = common_issues.most_common(3)
            insights.append(f"🔍 最常見問題類型: {[issue for issue, count in most_common]}")
            
        return insights
```

### 知識庫累積

#### 🧠 **修復模式學習**
```python
class RepairPatternLearning:
    """修復模式學習系統"""
    
    def __init__(self):
        self.successful_patterns = {}
        self.failed_patterns = {}
        
    def record_repair_outcome(self, pattern: str, files: List[str], 
                            success: bool, context: Dict):
        """記錄修復結果，積累經驗"""
        
        pattern_record = {
            'pattern': pattern,
            'files_applied': files,
            'success': success,
            'context': context,
            'timestamp': datetime.now(),
            'quality_impact': context.get('quality_delta', 0)
        }
        
        if success:
            if pattern not in self.successful_patterns:
                self.successful_patterns[pattern] = []
            self.successful_patterns[pattern].append(pattern_record)
        else:
            if pattern not in self.failed_patterns:
                self.failed_patterns[pattern] = []
            self.failed_patterns[pattern].append(pattern_record)
    
    def recommend_repair_strategy(self, current_issues: List[str], 
                                context: Dict) -> Dict[str, float]:
        """基於歷史經驗推薦修復策略"""
        
        recommendations = {}
        
        for pattern, records in self.successful_patterns.items():
            # 計算模式適用性分數
            applicability_score = self._calculate_applicability(
                pattern, current_issues, context, records
            )
            
            if applicability_score > 0.6:  # 高信心度才推薦
                recommendations[pattern] = applicability_score
        
        # 按信心度排序
        return dict(sorted(recommendations.items(), 
                          key=lambda x: x[1], reverse=True))
```

---

**指南版本控制**: 本通用指南將根據實際應用效果持續優化，確保與AI系統演進和最佳實踐保持同步。基於單一事實原則，所有修復決策都以此指南為準，避免修復標準不一致問題。