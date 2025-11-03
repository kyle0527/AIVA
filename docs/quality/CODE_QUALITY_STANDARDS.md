# AIVA 程式碼品質標準

> **企業級品質標準**: 基於 SonarQube + 認知複雜度最佳實踐

## 📋 目錄

- [🎯 品質標準概述](#-品質標準概述)
- [🔧 認知複雜度標準](#-認知複雜度標準)
- [🛡️ SonarQube 合規要求](#️-sonarqube-合規要求)
- [📊 品質檢查工具](#-品質檢查工具)
- [🏆 品質里程碑參考](#-品質里程碑參考)
- [📚 最佳實踐指南](#-最佳實踐指南)
- [🔍 常見問題排查](#-常見問題排查)

---

## 🎯 品質標準概述

### 🏛️ **基本原則**
AIVA 採用企業級程式碼品質標準，確保所有代碼符合以下核心要求：

1. **可維護性優先**: 代碼結構清晰，易於理解和修改
2. **穩定性保證**: 通過自動化檢查確保代碼可靠性
3. **一致性標準**: 跨模組統一的品質標準和規範
4. **持續改進**: 建立持續的品質監控和改進機制

### 🎖️ **品質等級定義**
| 等級 | 要求 | 適用範圍 |
|------|------|---------|
| **企業級** | 0 錯誤 + 複雜度 ≤15 | 核心模組、公共庫 |
| **生產級** | 0 錯誤 + 複雜度 ≤20 | 業務邏輯模組 |
| **開發級** | 0 嚴重錯誤 | 工具腳本、測試代碼 |

---

## 🔧 認知複雜度標準

> **基準**: 基於 SonarQube Cognitive Complexity 標準

### 📐 **複雜度限制**
- **強制上限**: ≤15 (企業級標準)
- **建議上限**: ≤10 (推薦重構觸發點)
- **警告閾值**: >8 (開始關注)

### 🔍 **複雜度計算規則**
認知複雜度基於以下結構計算：

| 結構類型 | 複雜度 | 範例 |
|---------|--------|------|
| **線性結構** | +0 | 順序語句 |
| **條件分支** | +1 | `if`, `elif`, `else` |
| **循環結構** | +1 | `for`, `while` |
| **異常處理** | +1 | `try`, `except` |
| **嵌套結構** | +n | 每層嵌套 +1 |
| **邏輯運算** | +1 | `and`, `or` |
| **遞歸調用** | +1 | 函數自調用 |

### 🛠️ **重構策略**

#### **1. Extract Method Pattern**
```python
# ❌ 高複雜度函數 (複雜度 > 15)
def process_security_finding(data: dict) -> FindingPayload:
    # 驗證邏輯 (複雜度 +5)
    if not data or not isinstance(data, dict):
        raise ValueError("Invalid data")
    if 'vulnerability' not in data:
        raise KeyError("Missing vulnerability")
    
    # 業務邏輯處理 (複雜度 +8)
    finding_type = data.get('type', 'unknown')
    if finding_type == 'sql_injection':
        severity = 'high'
        category = 'injection'
    elif finding_type == 'xss':
        severity = 'medium'
        category = 'injection'
    elif finding_type == 'csrf':
        severity = 'medium'
        category = 'broken_access'
    # ... 更多條件判斷
    
    return FindingPayload(**processed_data)

# ✅ 重構後 (複雜度 ≤ 15)
def process_security_finding(data: dict) -> FindingPayload:
    """主處理函數 - 保持簡潔 (複雜度 ≤5)"""
    validated_data = _validate_input_data(data)
    processed_data = _apply_business_rules(validated_data)
    return _create_finding_payload(processed_data)

def _validate_input_data(data: dict) -> dict:
    """驗證輸入數據 (複雜度 ≤5)"""
    if not data or not isinstance(data, dict):
        raise ValueError("Invalid data")
    if 'vulnerability' not in data:
        raise KeyError("Missing vulnerability")
    return data

def _apply_business_rules(data: dict) -> dict:
    """應用業務規則 (複雜度 ≤5)"""
    finding_type = data.get('type', 'unknown')
    severity, category = _determine_finding_classification(finding_type)
    return {**data, 'severity': severity, 'category': category}

def _determine_finding_classification(finding_type: str) -> tuple[str, str]:
    """分類邏輯 (複雜度 ≤5)"""
    classification_rules = {
        'sql_injection': ('high', 'injection'),
        'xss': ('medium', 'injection'),
        'csrf': ('medium', 'broken_access'),
        # ... 字典映射替代複雜條件
    }
    return classification_rules.get(finding_type, ('unknown', 'other'))
```

#### **2. Strategy Pattern**
```python
# ✅ 使用策略模式降低複雜度
from abc import ABC, abstractmethod

class ValidationStrategy(ABC):
    @abstractmethod
    def validate(self, data: dict) -> bool:
        pass

class SQLInjectionValidator(ValidationStrategy):
    def validate(self, data: dict) -> bool:
        # 單一職責驗證邏輯
        pass

class XSSValidator(ValidationStrategy):
    def validate(self, data: dict) -> bool:
        # 單一職責驗證邏輯
        pass

def validate_finding(data: dict, strategy: ValidationStrategy) -> bool:
    """使用策略模式 - 複雜度大幅降低"""
    return strategy.validate(data)
```

#### **3. Early Return Pattern**
```python
# ❌ 深層嵌套 (高複雜度)
def process_data(data):
    if data:
        if isinstance(data, dict):
            if 'key' in data:
                if data['key']:
                    return process_value(data['key'])
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None

# ✅ Early Return (低複雜度)
def process_data(data):
    """Early Return 模式降低嵌套"""
    if not data:
        return None
    if not isinstance(data, dict):
        return None
    if 'key' not in data:
        return None
    if not data['key']:
        return None
    
    return process_value(data['key'])
```

---

## 🛡️ SonarQube 合規要求

### 🚨 **錯誤等級要求**
| 等級 | 要求 | 處理方式 |
|------|------|---------|
| **Blocker** | 0 個 | 強制修復，阻止提交 |
| **Critical** | 0 個 | 必須修復，影響功能 |
| **Major** | 0 個 | 重要問題，及時修復 |
| **Minor** | ≤5 個 | 一般問題，計劃修復 |
| **Info** | 不限 | 信息提示，可選修復 |

### 🔍 **主要檢查規則**
#### **安全性 (Security)**
- 不使用危險函數 (`eval`, `exec`)
- 避免硬編碼密碼和密鑰
- 正確處理用戶輸入驗證

#### **可靠性 (Reliability)**
- 避免空指針引用
- 正確的異常處理
- 資源正確釋放

#### **可維護性 (Maintainability)**
- 認知複雜度 ≤15
- 函數長度適中 (≤50 行)
- 避免重複代碼

#### **可讀性 (Readability)**
- 有意義的變數和函數命名
- 適當的注釋和文檔
- 一致的代碼格式

### 🛠️ **常見修復方案**
```python
# ❌ SonarQube 問題範例

# 1. 重複字符串常量
def format_optional(field_name):
    if condition:
        return f"Optional[{field_name}]"  # 重複
    return f"Optional[{field_name}]"      # 重複

# 2. 過深的嵌套
def complex_logic(data):
    if data:
        if data.valid:
            if data.processed:
                # 深層嵌套邏輯
                pass

# ✅ 修復後

# 1. 提取字符串常量
OPTIONAL_TEMPLATE = "Optional[{}]"

def format_optional(field_name):
    return OPTIONAL_TEMPLATE.format(field_name)

# 2. 使用 Early Return
def complex_logic(data):
    if not data:
        return
    if not data.valid:
        return
    if not data.processed:
        return
    
    # 主要邏輯
    pass
```

---

## 📊 品質檢查工具

### 🔧 **自動化檢查工具**
| 工具 | 用途 | 使用方式 |
|------|------|---------|
| **SonarLint** | IDE 即時檢查 | VS Code 插件自動運行 |
| **SonarQube** | 深度分析 | `sonarqube_analyze_file` |
| **Pylance** | 型別檢查 | VS Code Python 擴展 |
| **Ruff** | 快速 Linting | `ruff check <file>` |
| **Black** | 代碼格式化 | `black <file>` |

### 📋 **檢查命令範例**
```bash
# 1. SonarQube 分析
python -c "
from sonarqube_analyze_file import analyze_file
analyze_file('path/to/file.py')
"

# 2. 認知複雜度檢查
radon cc path/to/file.py -s -n B

# 3. 型別檢查
python -m mypy path/to/file.py

# 4. 語法檢查
python -m py_compile path/to/file.py

# 5. 快速 Linting
ruff check path/to/file.py

# 6. 格式化
black path/to/file.py
```

### 🤖 **CI/CD 整合**
```yaml
# GitHub Actions 範例
name: Code Quality Check
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install sonarqube-api ruff black mypy
      
      - name: Run quality checks
        run: |
          # 複雜度檢查
          radon cc . -n B --total-average
          
          # SonarQube 檢查
          python tools/quality_check.py --all
          
          # 格式檢查
          black --check .
          ruff check .
```

---

## 🏆 品質里程碑參考

> **基於 AIVA v5.1 品質保證成果**

### ✅ **成功案例分析**
#### **Schema 代碼生成工具重構**
- **文件**: `plugins/aiva_converters/core/schema_codegen_tool.py`
- **重構函數**: 6 個核心函數
- **複雜度改善**: 從 18-29 降至 ≤15
- **新增輔助函數**: 45+ 個
- **重構技術**: Extract Method + Strategy Pattern

| 函數名稱 | 重構前複雜度 | 重構後複雜度 | 改善幅度 |
|---------|-------------|-------------|----------|
| `_render_rust_struct` | 29 | ≤15 | 48%+ |
| `_generate_python_field` | 29 | ≤15 | 48%+ |
| `_render_go_schemas` | 20 | ≤15 | 25%+ |
| `_convert_to_rust_type` | 20 | ≤15 | 25%+ |
| `validate_schemas` | 20 | ≤15 | 25%+ |
| `_get_rust_default_value` | 18 | ≤15 | 17%+ |

#### **AI 模型管理器重構**
- **文件**: `services/core/aiva_core/ai_engine/ai_model_manager.py`
- **重構函數**: `train_models`
- **複雜度改善**: 從 18 降至 ≤15
- **重構技術**: Extract Method + 職責分離

### 📊 **品質指標達成**
| 品質指標 | 目標 | 實際達成 | 達成率 |
|---------|------|---------|--------|
| **認知複雜度合規** | ≤15 | 7/7 函數 | 100% |
| **SonarQube 零錯誤** | 0 錯誤 | 0 錯誤 | 100% |
| **代碼覆蓋範圍** | 核心模組 | 3 個文件 | 100% |
| **文檔完整性** | 完整記錄 | 詳細文檔 | 100% |

### 🎯 **重構技術效果**
1. **Extract Method Pattern**: 大型函數分解為專門小函數
2. **Strategy Pattern**: 複雜條件判斷用策略模式替代
3. **Early Return Pattern**: 減少嵌套層級和認知負擔
4. **字符串常量管理**: 統一常量定義，提升維護性
5. **職責分離**: 每個函數專注單一職責

---

## 📚 最佳實踐指南

### 🎯 **函數設計原則**
#### **單一職責原則 (SRP)**
```python
# ✅ 正確：每個函數專注單一職責
def validate_user_input(data: dict) -> bool:
    """只負責驗證"""
    return all(key in data for key in ['name', 'email'])

def transform_user_data(data: dict) -> UserModel:
    """只負責轉換"""
    return UserModel(**data)

def save_user_data(user: UserModel) -> bool:
    """只負責儲存"""
    return database.save(user)

# ❌ 錯誤：一個函數承擔多個職責
def process_user(data: dict) -> bool:
    """違反單一職責 - 驗證+轉換+儲存"""
    # 驗證邏輯
    if not validate_data(data):
        return False
    # 轉換邏輯  
    user = UserModel(**data)
    # 儲存邏輯
    return database.save(user)
```

#### **命名規範**
```python
# ✅ 清晰的命名
def calculate_risk_score(vulnerability_data: dict) -> float:
    """動詞開頭，清楚表達功能"""
    pass

def extract_finding_metadata(raw_data: dict) -> dict:
    """明確表達提取操作"""
    pass

# ❌ 模糊的命名  
def process(data):  # 太generic
    pass

def handle_stuff(x):  # 不明確
    pass
```

### 🔧 **錯誤處理模式**
```python
# ✅ 良好的錯誤處理
def process_security_finding(data: dict) -> FindingPayload:
    """明確的錯誤處理"""
    try:
        validated_data = _validate_finding_data(data)
        return FindingPayload(**validated_data)
    except ValidationError as e:
        logger.error(f"Finding validation failed: {e}")
        raise ProcessingError(f"Invalid finding data: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in finding processing: {e}")
        raise

# ❌ 過度寬泛的異常處理
def process_data(data):
    try:
        # 大量邏輯
        pass
    except:  # 捕獲所有異常
        return None  # 丟失錯誤信息
```

### 📝 **文檔標準**
```python
def analyze_vulnerability_impact(
    vulnerability: VulnerabilityPayload,
    context: SecurityContext
) -> RiskAssessment:
    """分析漏洞影響程度並評估風險等級。
    
    Args:
        vulnerability: 漏洞詳細信息，包含類型、嚴重性等
        context: 安全上下文，包含環境和威脅模型
        
    Returns:
        RiskAssessment: 包含風險等級、影響評估和建議措施
        
    Raises:
        ValidationError: 當輸入數據格式不正確時
        ProcessingError: 當風險評估過程中發生錯誤時
        
    Example:
        >>> vuln = VulnerabilityPayload(type="sql_injection", severity="high")
        >>> ctx = SecurityContext(environment="production")
        >>> risk = analyze_vulnerability_impact(vuln, ctx)
        >>> assert risk.level in ["low", "medium", "high", "critical"]
    """
    pass
```

---

## 🔍 常見問題排查

### ❓ **Q1: 如何快速檢查認知複雜度？**
```bash
# 使用 radon 檢查複雜度
pip install radon
radon cc path/to/file.py -s -n B

# 或使用 AIVA 內建工具
python -c "
from tools.complexity_analyzer import check_complexity
check_complexity('path/to/file.py')
"
```

### ❓ **Q2: SonarQube 報告錯誤如何解讀？**
```bash
# 運行 SonarQube 分析
python -c "
from sonarqube_analyze_file import analyze_file
result = analyze_file('path/to/file.py')
print(result)  # 查看詳細錯誤報告
"

# 常見錯誤類型：
# - Cognitive Complexity: 函數過於複雜
# - Duplicated String Literals: 重複字符串
# - Nested Control Flow: 嵌套過深
```

### ❓ **Q3: 重構後如何驗證功能完整性？**
```bash
# 1. 運行單元測試
pytest tests/ -v

# 2. 語法檢查
python -m py_compile path/to/refactored_file.py

# 3. 型別檢查
mypy path/to/refactored_file.py

# 4. 功能回歸測試
python -c "
# 導入並測試重構後的函數
from module import refactored_function
result = refactored_function(test_data)
assert result == expected_result
"
```

### ❓ **Q4: 如何平衡重構範圍和風險？**
**建議策略**:
1. **從最小範圍開始**: 一次只重構一個函數
2. **保持接口穩定**: 不改變函數簽名和返回值
3. **充分測試**: 每次重構後立即驗證功能
4. **分階段提交**: 小步提交，便於回滾

---

## 📈 持續改進機制

### 🎯 **品質監控儀表板**
建議建立以下監控指標：

| 指標類別 | 具體指標 | 目標值 |
|---------|---------|--------|
| **複雜度** | 平均認知複雜度 | ≤8 |
| **品質** | SonarQube 錯誤數 | 0 |
| **覆蓋率** | 代碼覆蓋率 | ≥80% |
| **維護性** | 技術債務比率 | ≤5% |

### 🔄 **定期品質審查**
- **週期**: 每月一次
- **範圍**: 新增和修改的代碼
- **流程**: 自動掃描 + 人工審查
- **改進**: 制定改進計劃和執行跟蹤

### 🎓 **團隊培訓計劃**
1. **品質意識培訓**: 品質標準和重要性
2. **工具使用培訓**: SonarQube、Pylance 等工具
3. **重構技術培訓**: 常用重構模式和技巧
4. **最佳實踐分享**: 定期分享成功案例

---

## 📞 支援與聯絡

### 🛠️ **技術支援**
- **品質問題諮詢**: 開發團隊 Slack #code-quality
- **工具使用問題**: 參考 [VS Code 插件指南](../../_out/VSCODE_EXTENSIONS_INVENTORY.md)
- **重構技術指導**: 架構團隊 code-review@aiva.com

### 📚 **參考資源**
- [Martin Fowler - Refactoring](https://refactoring.com/)
- [SonarQube Rules](https://rules.sonarsource.com/)
- [Clean Code - Robert Martin](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)

---

**文檔版本**: v1.0  
**建立日期**: 2025-11-03  
**最後更新**: 2025-11-03  
**維護團隊**: AIVA 品質保證團隊  
**適用範圍**: 全專案開發團隊  
**基準來源**: AIVA v5.1 認知複雜度修復成果  

> **🎯 品質目標**: 建立可持續的企業級程式碼品質標準，確保 AIVA 平台的長期可維護性和穩定性。