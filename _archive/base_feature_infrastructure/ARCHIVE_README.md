# base 功能基礎設施 - 歸檔說明

**歸檔日期**: 2026-02-09  
**原路徑**: `services/features/base/`  
**歸檔原因**: 設計未被實際使用，所有功能模組使用各自架構

---

## 📋 歸檔內容

### 1. **base/** 資料夾（666 行代碼）
- `feature_registry.py` (137 行) - 功能模組註冊表
- `result_schema.py` (198 行) - 標準結果數據結構
- `integration_helper.py` (331 行) - 整合助手
- `__init__.py` - 模組導出

### 2. **feature_step_executor.py** (338 行)
- 功能步驟執行器
- 依賴 FeatureRegistry 和 FeatureResult
- 無法工作（註冊表為空）

---

## 🔍 歸檔原因分析

### **核心問題：設計與實現脫節**

#### ❌ **FeatureRegistry 完全未使用**
```python
# 實際執行結果
>>> FeatureRegistry.list_features()
{}  # 空字典！完全沒有模組註冊
```

#### ❌ **10 個功能模組無一使用**

| 模組 | 完成度 | 實際架構 | 是否使用 base/ |
|------|--------|---------|---------------|
| function_sqli | 95% | CommandHandler | ❌ 否 |
| function_xss | 90% | CommandHandler | ❌ 否 |
| function_crypto | 95% | 純 Rust CLI | ❌ 否 |
| function_ssrf | 85% | Detector | ❌ 否 |
| function_idor | 80% | CommandHandler | ❌ 否 |
| function_bizlogic | 75% | CLI | ❌ 否 |
| function_info_leak | 100% | Detector | ❌ 否 |
| function_authn_go | 70% | 純 Go CLI | ❌ 否 |
| function_postex | 50% | CommandHandler | ❌ 否 |
| function_web_scanner | 35% | 規劃中 | ❌ 否 |

**統計**: 0/10 模組使用 base 標準接口

#### ❌ **FeatureStepExecutor 無法運作**

```python
# Line 113 永遠失敗
feature_class = FeatureRegistry.get(tool_name)
if feature_class is None:
    raise KeyError(f"功能模組 '{tool_name}' 未註冊")  # ← 永遠執行到這！
```

---

## 🎯 實際執行架構

### **功能模組實際使用 3 種架構**

#### 1️⃣ **CommandHandler 架構** (6 個模組)
```python
class XSSCommandHandler:
    async def handle_command(self, command: AICommand) -> dict:
        # 處理命令，返回字典
```

#### 2️⃣ **純 CLI 架構** (2 個模組)
```bash
# Rust/Go 獨立可執行文件
cargo run -- --analyze-cookies
go run main.go --analyze-sessions
```

#### 3️⃣ **Detector 架構** (2 個模組)
```python
class SSRFEngine:
    async def run(self, *, target_url: str) -> list[SSRFIssue]:
        # 直接執行檢測
```

### **核心模組直接導入檢測器**

```python
# services/core/aiva_core/task_planning/commander/attack_coordinator.py
from services.features.function_xss.traditional_detector import TraditionalXssDetector
from services.features.function_sqli.detector.sqli_detector import SqliDetector

# 直接調用，不通過 FeatureRegistry
detector = TraditionalXssDetector()
results = detector.detect(url)  # ✅ 實際可用
```

---

## 📊 代碼統計

### **搜尋結果**

```bash
# 搜尋 FeatureRegistry 註冊
grep_search: "FeatureRegistry.register("
結果: 1 match - 僅在註釋示例中

# 搜尋 base 導入
grep_search: "from ..base|from services.features.base"  
結果: No matches in function_* modules

# 搜尋 FeatureBase 繼承
grep_search: "class.*\(FeatureBase\)"
結果: No matches
```

### **影響範圍**
- ✅ **已移除引用**: services/features/__init__.py
- ✅ **已歸檔文件**: base/, feature_step_executor.py
- ✅ **無破壞性影響**: 功能模組未使用這些組件

---

## 🔄 如需重新啟用

### **需要完成的工作**

1. **實現標準接口**
   ```python
   # 每個模組需要創建 Feature 類別
   class SqliFeature:
       def run(self, params: dict) -> FeatureResult:
           # 實現檢測邏輯
           return FeatureResult(...)
   ```

2. **註冊所有模組**
   ```python
   # 在模組 __init__.py 中
   from services.features.base import FeatureRegistry
   FeatureRegistry.register("sqli", SqliFeature)
   ```

3. **統一返回格式**
   - 所有模組返回 FeatureResult
   - 使用統一的 Finding 結構

4. **估計工作量**: 10 個模組 × 8-16 小時 = 80-160 小時

---

## 📚 相關文檔

- 分析報告: 見本次對話記錄
- 架構流程圖: 已生成 Mermaid 圖表展示實際 vs 設計架構
- 決策依據: FeatureRegistry 完全為空，FeatureStepExecutor 無法工作

---

## ✅ 歸檔決策

**決策**: 歸檔（非刪除）  
**理由**: 
1. 設計良好但未被採用
2. 保留作為未來標準化的參考
3. 移除以減少代碼混淆和維護負擔
4. 功能模組已有各自穩定的工作架構

**替代方案**: 未來如需統一接口，可參考此設計重新實現
