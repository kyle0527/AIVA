# AIVA 多語言能力分析整合 - 完整實施指南

**版本**: v1.0  
**日期**: 2025-11-16  
**適用對象**: 開發者、系統維護者  
**前置條件**: Python 3.10+, AIVA 專案環境

---

## 📋 目錄

1. [環境準備](#1-環境準備)
2. [現有基礎設施確認](#2-現有基礎設施確認)
3. [實施整合](#3-實施整合)
4. [測試驗證](#4-測試驗證)
5. [使用範例](#5-使用範例)
6. [問題排查](#6-問題排查)
7. [擴展指南](#7-擴展指南)

---

## 1. 環境準備

### 1.1 確認專案結構

```bash
# 進入 AIVA 專案根目錄
cd C:\D\fold7\AIVA-git

# 確認目錄結構
tree /F services\core\aiva_core\internal_exploration
```

**預期輸出**:
```
services\core\aiva_core\internal_exploration\
├── __init__.py
├── README.md
├── capability_analyzer.py
├── language_extractors.py     # ← 關鍵文件
└── module_explorer.py
```

### 1.2 啟動虛擬環境

```powershell
# 如果已有虛擬環境
.\.venv\Scripts\Activate.ps1

# 確認 Python 版本
python --version  # 應顯示 3.10 或更高版本
```

### 1.3 檢查依賴

```bash
# 確認必要的 Python 套件
python -c "import ast, re, pathlib, logging; print('✅ 所有依賴已安裝')"
```

---

## 2. 現有基礎設施確認

### 2.1 驗證 language_extractors.py 存在

```bash
# 檢查文件存在
if (Test-Path "services\core\aiva_core\internal_exploration\language_extractors.py") { 
    echo "✅ language_extractors.py 存在" 
} else { 
    echo "❌ 文件不存在,請先創建" 
}
```

### 2.2 檢查提取器實現

```python
# 驗證腳本: check_extractors.py
from services.core.aiva_core.internal_exploration.language_extractors import (
    get_extractor,
    GoExtractor,
    RustExtractor,
    TypeScriptExtractor
)

# 測試每個提取器
extractors = {
    "go": GoExtractor(),
    "rust": RustExtractor(),
    "typescript": TypeScriptExtractor(),
    "javascript": TypeScriptExtractor()
}

for lang, extractor in extractors.items():
    print(f"✅ {lang.upper()} 提取器已加載: {extractor.__class__.__name__}")

# 測試工廠函數
for lang in ["go", "rust", "typescript", "javascript"]:
    ext = get_extractor(lang)
    if ext:
        print(f"✅ get_extractor('{lang}') 返回: {ext.__class__.__name__}")
    else:
        print(f"❌ get_extractor('{lang}') 返回 None")
```

**執行驗證**:
```bash
python -c "from services.core.aiva_core.internal_exploration.language_extractors import get_extractor; print('✅ language_extractors 可導入')"
```

### 2.3 檢查 module_explorer.py 多語言支援

```bash
# 檢查文件掃描配置
grep -A 10 "file_extensions" services/core/aiva_core/internal_exploration/module_explorer.py
```

**預期輸出**:
```python
self.file_extensions = {
    "python": "*.py",
    "go": "*.go",
    "rust": "*.rs",
    "typescript": "*.ts",
    "javascript": "*.js"
}
```

---

## 3. 實施整合

### 3.1 備份原始文件

```bash
# 創建備份
cp services\core\aiva_core\internal_exploration\capability_analyzer.py services\core\aiva_core\internal_exploration\capability_analyzer.py.backup

# 確認備份成功
if (Test-Path "services\core\aiva_core\internal_exploration\capability_analyzer.py.backup") {
    echo "✅ 備份完成"
}
```

### 3.2 修改 capability_analyzer.py

**步驟 1: 添加導入**

在文件頂部 (約第 10-15 行) 添加:

```python
# 原有導入
import ast
import logging
from pathlib import Path
from typing import Any

# ← 在這裡添加新導入
from .language_extractors import get_extractor

logger = logging.getLogger(__name__)
```

**完整命令**:
```python
# 使用編輯器打開
code services\core\aiva_core\internal_exploration\capability_analyzer.py

# 或使用 sed (PowerShell)
$content = Get-Content services\core\aiva_core\internal_exploration\capability_analyzer.py -Raw
$content = $content -replace "from typing import Any\n\nlogger", "from typing import Any`n`nfrom .language_extractors import get_extractor`n`nlogger"
Set-Content services\core\aiva_core\internal_exploration\capability_analyzer.py $content
```

**步驟 2: 添加語言檢測方法**

在 `_extract_capabilities_from_file` 方法之前添加 (約第 79 行):

```python
    def _detect_language(self, file_path: Path) -> str:
        """檢測文件語言
        
        Args:
            file_path: 文件路徑
            
        Returns:
            語言名稱: python, go, rust, typescript, javascript
        """
        suffix = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".go": "go",
            ".rs": "rust",
            ".ts": "typescript",
            ".js": "javascript"
        }
        return language_map.get(suffix, "unknown")
```

**步驟 3: 重構原有方法**

找到 `async def _extract_capabilities_from_file` 方法,替換為:

```python
    async def _extract_capabilities_from_file(self, file_path: Path, module: str) -> list[dict]:
        """從文件中提取能力 (支援多語言)
        
        Args:
            file_path: 文件路徑 (.py/.go/.rs/.ts/.js)
            module: 所屬模組名稱
            
        Returns:
            能力列表
        """
        # 根據副檔名選擇提取器
        language = self._detect_language(file_path)
        
        if language == "python":
            return self._extract_python_capabilities(file_path, module)
        else:
            # 使用 language_extractors 處理非 Python 語言
            return self._extract_non_python_capabilities(file_path, module, language)
```

**步驟 4: 重命名原有 Python 提取邏輯**

將原有的提取邏輯移到新方法 `_extract_python_capabilities`:

```python
    def _extract_python_capabilities(self, file_path: Path, module: str) -> list[dict]:
        """從 Python 文件提取能力 (使用 AST)
        
        Args:
            file_path: Python 文件路徑
            module: 所屬模組名稱
            
        Returns:
            能力列表
        """
        capabilities = []
        
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if self._has_capability_decorator(node):
                        cap = self._extract_capability_info(node, file_path, module)
                        capabilities.append(cap)
            
            if capabilities:
                logger.debug(f"  Found {len(capabilities)} Python capabilities in {file_path.name}")
            
        except SyntaxError as e:
            logger.warning(f"  Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"  Failed to parse {file_path}: {e}")
        
        return capabilities
```

**步驟 5: 添加非 Python 提取方法**

```python
    def _extract_non_python_capabilities(
        self, 
        file_path: Path, 
        module: str,
        language: str
    ) -> list[dict]:
        """從非 Python 文件提取能力 (使用 language_extractors)
        
        Args:
            file_path: 文件路徑
            module: 所屬模組名稱
            language: 語言名稱
            
        Returns:
            能力列表
        """
        try:
            # 獲取對應語言的提取器
            extractor = get_extractor(language)
            if not extractor:
                logger.warning(f"  No extractor available for language: {language}")
                return []
            
            # 讀取文件內容
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            
            # 使用提取器提取能力
            capabilities = extractor.extract_capabilities(content, str(file_path))
            
            # 添加 module 信息
            for cap in capabilities:
                if "module" not in cap or not cap["module"]:
                    cap["module"] = module
            
            if capabilities:
                logger.debug(f"  Found {len(capabilities)} {language} capabilities in {file_path.name}")
            
            return capabilities
            
        except Exception as e:
            logger.error(f"  Failed to extract from {file_path}: {e}")
            return []
```

### 3.3 驗證語法

```bash
# 檢查 Python 語法錯誤
python -m py_compile services/core/aiva_core/internal_exploration/capability_analyzer.py

# 如果無輸出,表示語法正確
echo $?  # 應輸出 0
```

---

## 4. 測試驗證

### 4.1 創建測試腳本

創建 `test_multi_language_analysis.py`:

```python
"""多語言能力分析整合測試"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from services.core.aiva_core.internal_exploration import ModuleExplorer, CapabilityAnalyzer


async def main():
    print("=" * 80)
    print("🚀 多語言能力分析整合測試")
    print("=" * 80)
    
    # 1. 初始化
    root_path = Path(__file__).parent / "services"
    explorer = ModuleExplorer(root_path=root_path)
    analyzer = CapabilityAnalyzer()
    
    # 2. 掃描模組
    print("\n📂 掃描模組文件...")
    modules_info = await explorer.explore_all_modules()
    
    # 統計文件
    total_files = sum(m["stats"]["total_files"] for m in modules_info.values())
    by_lang = {}
    for module_data in modules_info.values():
        for lang, count in module_data["stats"]["by_language"].items():
            by_lang[lang] = by_lang.get(lang, 0) + count
    
    print(f"\n✅ 掃描完成:")
    print(f"   - 總文件: {total_files}")
    for lang, count in by_lang.items():
        if count > 0:
            print(f"   - {lang}: {count} 個")
    
    # 3. 提取能力
    print("\n🔍 提取能力...")
    capabilities = await analyzer.analyze_capabilities(modules_info)
    
    # 統計能力
    cap_by_lang = {}
    for cap in capabilities:
        lang = cap.get("language", "python")
        cap_by_lang[lang] = cap_by_lang.get(lang, 0) + 1
    
    print(f"\n✅ 提取完成:")
    print(f"   - 總能力: {len(capabilities)}")
    for lang, count in cap_by_lang.items():
        print(f"   - {lang}: {count} 個")
    
    # 4. 顯示範例
    print("\n📝 能力範例:")
    for lang in ["python", "go", "rust", "typescript"]:
        lang_caps = [c for c in capabilities if c.get("language") == lang][:2]
        if lang_caps:
            print(f"\n   {lang.upper()}:")
            for cap in lang_caps:
                print(f"     - {cap['name']}")
    
    # 5. 驗證
    print("\n✅ 驗證結果:")
    checks = {
        "多語言掃描": len(by_lang) >= 3,
        "Python 提取": cap_by_lang.get("python", 0) > 0,
        "Go 提取": cap_by_lang.get("go", 0) > 0,
        "TypeScript 提取": cap_by_lang.get("typescript", 0) > 0,
    }
    
    for check, passed in checks.items():
        print(f"   {'✅' if passed else '❌'} {check}")
    
    print("\n" + "=" * 80)
    print("✅ 測試完成!" if all(checks.values()) else "⚠️ 部分測試未通過")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
```

### 4.2 執行測試

```bash
# 執行測試
python test_multi_language_analysis.py
```

**預期輸出**:
```
================================================================================
🚀 多語言能力分析整合測試
================================================================================

📂 掃描模組文件...
✅ 掃描完成:
   - 總文件: 380
   - python: 320 個
   - go: 27 個
   - rust: 7 個
   - typescript: 18 個
   - javascript: 8 個

🔍 提取能力...
✅ 提取完成:
   - 總能力: 576
   - python: 410 個
   - go: 88 個
   - typescript: 78 個

📝 能力範例:
   PYTHON:
     - analyze_capabilities
     - explore_all_modules
   GO:
     - NewScannerAMQPClient
     - DeclareQueue
   TYPESCRIPT:
     - toStandardFinding
     - analyzeClientSideAuthBypass

✅ 驗證結果:
   ✅ 多語言掃描
   ✅ Python 提取
   ✅ Go 提取
   ✅ TypeScript 提取

================================================================================
✅ 測試完成!
================================================================================
```

### 4.3 單元測試 (可選)

創建 `tests/test_capability_analyzer_multi_lang.py`:

```python
"""capability_analyzer 多語言整合單元測試"""

import pytest
from pathlib import Path
from services.core.aiva_core.internal_exploration.capability_analyzer import CapabilityAnalyzer


@pytest.fixture
def analyzer():
    return CapabilityAnalyzer()


def test_detect_language(analyzer):
    """測試語言檢測"""
    assert analyzer._detect_language(Path("test.py")) == "python"
    assert analyzer._detect_language(Path("test.go")) == "go"
    assert analyzer._detect_language(Path("test.rs")) == "rust"
    assert analyzer._detect_language(Path("test.ts")) == "typescript"
    assert analyzer._detect_language(Path("test.js")) == "javascript"
    assert analyzer._detect_language(Path("test.txt")) == "unknown"


def test_extract_go_capabilities(analyzer, tmp_path):
    """測試 Go 能力提取"""
    go_file = tmp_path / "scanner.go"
    go_file.write_text("""
package scanner

// ScanTarget scans a target URL
func ScanTarget(url string) error {
    return nil
}

// internal function
func internalHelper() {}
""")
    
    caps = analyzer._extract_non_python_capabilities(go_file, "test_module", "go")
    assert len(caps) == 1  # 只有 ScanTarget (大寫開頭)
    assert caps[0]["name"] == "ScanTarget"
    assert caps[0]["language"] == "go"


def test_extract_typescript_capabilities(analyzer, tmp_path):
    """測試 TypeScript 能力提取"""
    ts_file = tmp_path / "scanner.ts"
    ts_file.write_text("""
/**
 * Analyze client auth bypass
 */
export function analyzeAuthBypass(): void {
    // implementation
}

export const helperFunc = () => {};
""")
    
    caps = analyzer._extract_non_python_capabilities(ts_file, "test_module", "typescript")
    assert len(caps) >= 1
    assert any(c["name"] == "analyzeAuthBypass" for c in caps)


def test_extract_python_capabilities(analyzer, tmp_path):
    """測試 Python 能力提取"""
    py_file = tmp_path / "scanner.py"
    py_file.write_text("""
from aiva_core.core_capabilities import register_capability

@register_capability
async def scan_target(url: str) -> dict:
    \"\"\"Scan a target URL\"\"\"
    return {}
""")
    
    caps = analyzer._extract_python_capabilities(py_file, "test_module")
    assert len(caps) == 1
    assert caps[0]["name"] == "scan_target"
    assert caps[0]["is_async"] == True
```

**執行單元測試**:
```bash
pytest tests/test_capability_analyzer_multi_lang.py -v
```

---

## 5. 使用範例

### 5.1 基本使用

```python
import asyncio
from pathlib import Path
from services.core.aiva_core.internal_exploration import ModuleExplorer, CapabilityAnalyzer

async def analyze_system():
    # 初始化
    explorer = ModuleExplorer()
    analyzer = CapabilityAnalyzer()
    
    # 掃描並分析
    modules_info = await explorer.explore_all_modules()
    capabilities = await analyzer.analyze_capabilities(modules_info)
    
    # 按語言分組
    by_language = {}
    for cap in capabilities:
        lang = cap.get("language", "python")
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append(cap)
    
    # 輸出結果
    for lang, caps in by_language.items():
        print(f"\n{lang.upper()}: {len(caps)} capabilities")
        for cap in caps[:5]:  # 顯示前 5 個
            print(f"  - {cap['name']}")
    
    return capabilities

# 執行
capabilities = asyncio.run(analyze_system())
```

### 5.2 過濾特定語言

```python
# 只分析 Go 語言能力
go_capabilities = [
    cap for cap in capabilities 
    if cap.get("language") == "go"
]

print(f"Go 能力: {len(go_capabilities)} 個")
for cap in go_capabilities:
    params = cap.get("parameters", [])
    param_str = ", ".join(p["name"] for p in params)
    print(f"  - {cap['name']}({param_str})")
```

### 5.3 生成報告

```python
def generate_capability_report(capabilities: list) -> str:
    """生成能力分析報告"""
    lines = ["# AIVA 系統能力報告\n"]
    
    # 按語言分組
    by_lang = {}
    for cap in capabilities:
        lang = cap.get("language", "python")
        by_lang.setdefault(lang, []).append(cap)
    
    # 生成每種語言的章節
    for lang, caps in sorted(by_lang.items()):
        lines.append(f"\n## {lang.upper()} ({len(caps)} 個能力)\n")
        
        for cap in caps:
            lines.append(f"### {cap['name']}\n")
            lines.append(f"- **模組**: {cap['module']}\n")
            lines.append(f"- **文件**: {cap['file_path']}\n")
            
            if cap.get("description"):
                lines.append(f"- **說明**: {cap['description']}\n")
            
            if cap.get("parameters"):
                params = ", ".join(p["name"] for p in cap["parameters"])
                lines.append(f"- **參數**: {params}\n")
            
            lines.append("\n")
    
    return "".join(lines)

# 生成並保存報告
report = generate_capability_report(capabilities)
with open("CAPABILITY_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)
```

### 5.4 整合到內部閉環

```python
from services.core.aiva_core.cognitive_core.internal_loop import InternalLoopConnector

async def update_rag_with_capabilities():
    """更新 RAG 系統的能力知識"""
    # 1. 分析能力
    explorer = ModuleExplorer()
    analyzer = CapabilityAnalyzer()
    
    modules_info = await explorer.explore_all_modules()
    capabilities = await analyzer.analyze_capabilities(modules_info)
    
    # 2. 轉換為 RAG 文檔
    documents = []
    for cap in capabilities:
        doc = {
            "content": f"{cap['name']}: {cap.get('description', '')}",
            "metadata": {
                "type": "capability",
                "language": cap.get("language", "python"),
                "module": cap["module"],
                "file_path": cap["file_path"]
            }
        }
        documents.append(doc)
    
    # 3. 更新 RAG
    internal_loop = InternalLoopConnector()
    await internal_loop.update_capability_knowledge(documents)
    
    print(f"✅ 已更新 {len(documents)} 個能力到 RAG 系統")

# 執行更新
asyncio.run(update_rag_with_capabilities())
```

---

## 6. 問題排查

### 6.1 導入錯誤

**問題**: `ImportError: cannot import name 'get_extractor'`

**解決**:
```bash
# 檢查文件是否存在
ls services/core/aiva_core/internal_exploration/language_extractors.py

# 檢查 __init__.py 是否導出
cat services/core/aiva_core/internal_exploration/__init__.py

# 如果沒有,添加導出
echo "from .language_extractors import get_extractor" >> services/core/aiva_core/internal_exploration/__init__.py
```

### 6.2 無法提取能力

**問題**: 某種語言提取 0 個能力

**診斷步驟**:

1. **檢查文件是否被掃描**:
```python
modules_info = await explorer.explore_all_modules()
for module, data in modules_info.items():
    go_files = [f for f in data["files"] if f["type"] == "go"]
    print(f"{module}: {len(go_files)} Go 文件")
```

2. **手動測試提取器**:
```python
from services.core.aiva_core.internal_exploration.language_extractors import GoExtractor

extractor = GoExtractor()
with open("services/scan/some_file.go") as f:
    content = f.read()

caps = extractor.extract_capabilities(content, "test.go")
print(f"提取到 {len(caps)} 個能力")
for cap in caps:
    print(f"  - {cap['name']}")
```

3. **檢查正則模式**:
```python
import re

# Go 函數模式
pattern = re.compile(
    r'func\s+(?:\([^)]*\)\s+)?([A-Z][a-zA-Z0-9_]*)\s*\(',
    re.MULTILINE
)

test_code = """
func ScanTarget(url string) error {
    return nil
}
"""

matches = pattern.findall(test_code)
print(f"匹配到: {matches}")  # 應顯示 ['ScanTarget']
```

### 6.3 提取結果不正確

**問題**: 提取到不應該的函數,或遺漏了某些函數

**Go 語言規則**:
- ✅ 只提取**大寫開頭**的函數 (導出函數)
- ❌ 小寫開頭的函數不會被提取 (內部函數)

```go
// ✅ 會被提取
func PublicFunction() {}

// ❌ 不會被提取
func privateFunction() {}
```

**Rust 語言規則**:
- ✅ 只提取 `pub fn` (公開函數)
- ❌ `impl` 中的方法目前**不支援** (已知限制)

```rust
// ✅ 會被提取
pub fn public_function() {}

// ❌ 不會被提取 (無 pub)
fn private_function() {}

// ❌ 不會被提取 (impl 方法)
impl MyStruct {
    pub fn method(&self) {}
}
```

**TypeScript 語言規則**:
- ✅ 提取 `export function`
- ✅ 提取 `export const x = () =>`
- ❌ 非 export 函數不會被提取

```typescript
// ✅ 會被提取
export function publicFunc() {}

// ❌ 不會被提取
function privateFunc() {}
```

### 6.4 性能問題

**問題**: 掃描大型專案時很慢

**優化方法**:

1. **限制掃描範圍**:
```python
explorer = ModuleExplorer()
explorer.target_modules = ["core/aiva_core"]  # 只掃描核心模組
```

2. **跳過測試文件** (已內建):
```python
# module_explorer.py 已自動跳過
if file_path.name.startswith("test_"):
    continue
```

3. **使用快取**:
```python
analyzer = CapabilityAnalyzer()
# 快取已分析的結果
analyzer.capabilities_cache = {}  # 已內建
```

---

## 7. 擴展指南

### 7.1 新增語言支援

**範例: 添加 Java 支援**

**步驟 1**: 在 `language_extractors.py` 添加提取器

```python
class JavaExtractor(LanguageExtractor):
    """Java 語言函數提取器"""
    
    FUNCTION_PATTERN = re.compile(
        r'public\s+(?:static\s+)?'  # public [static]
        r'(?:\w+)\s+'  # 返回類型
        r'([A-Z][a-zA-Z0-9_]*)\s*'  # 方法名 (大寫開頭)
        r'\(([^)]*)\)',  # 參數列表
        re.MULTILINE
    )
    
    def extract_capabilities(self, content: str, file_path: str) -> list[dict]:
        capabilities = []
        
        for match in self.FUNCTION_PATTERN.finditer(content):
            method_name = match.group(1)
            params = match.group(2)
            
            # 提取 Javadoc
            doc_comments = self._extract_javadoc(content, match.start())
            
            capability = {
                "name": method_name,
                "language": "java",
                "file_path": file_path,
                "parameters": self._parse_java_params(params),
                "description": doc_comments or f"Java method: {method_name}",
                "line_number": content[:match.start()].count('\n') + 1
            }
            
            capabilities.append(capability)
        
        return capabilities
    
    def _parse_java_params(self, params_str: str) -> list[dict]:
        """解析 Java 參數"""
        if not params_str.strip():
            return []
        
        params = []
        for param in params_str.split(','):
            parts = param.strip().split()
            if len(parts) >= 2:
                params.append({
                    "type": parts[0],
                    "name": parts[1]
                })
        return params
    
    def _extract_javadoc(self, content: str, start_pos: int) -> str:
        """提取 Javadoc 註釋"""
        lines = content[:start_pos].split('\n')
        javadoc = []
        in_javadoc = False
        
        for line in reversed(lines):
            stripped = line.strip()
            if stripped == '*/':
                in_javadoc = True
                continue
            elif stripped.startswith('/**'):
                break
            elif in_javadoc:
                cleaned = re.sub(r'^\s*\*\s?', '', stripped)
                javadoc.insert(0, cleaned)
        
        return ' '.join(javadoc)
```

**步驟 2**: 註冊到工廠函數

```python
def get_extractor(language: str) -> LanguageExtractor | None:
    extractors = {
        "go": GoExtractor(),
        "rust": RustExtractor(),
        "typescript": TypeScriptExtractor(),
        "javascript": TypeScriptExtractor(),
        "java": JavaExtractor(),  # ← 添加這行
    }
    return extractors.get(language.lower())
```

**步驟 3**: 更新 module_explorer.py

```python
self.file_extensions = {
    "python": "*.py",
    "go": "*.go",
    "rust": "*.rs",
    "typescript": "*.ts",
    "javascript": "*.js",
    "java": "*.java",  # ← 添加這行
}
```

**步驟 4**: 更新 capability_analyzer.py

```python
def _detect_language(self, file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    language_map = {
        ".py": "python",
        ".go": "go",
        ".rs": "rust",
        ".ts": "typescript",
        ".js": "javascript",
        ".java": "java",  # ← 添加這行
    }
    return language_map.get(suffix, "unknown")
```

**步驟 5**: 測試

```python
# 創建測試文件
test_java = """
public class Scanner {
    /**
     * Scan a target URL
     * @param url The target URL
     * @return Scan results
     */
    public ScanResult ScanTarget(String url) {
        return new ScanResult();
    }
}
"""

from services.core.aiva_core.internal_exploration.language_extractors import JavaExtractor

extractor = JavaExtractor()
caps = extractor.extract_capabilities(test_java, "Scanner.java")

print(f"提取到 {len(caps)} 個 Java 能力")
for cap in caps:
    print(f"  - {cap['name']}: {cap['description']}")
```

### 7.2 自定義能力過濾

**範例: 只提取安全相關能力**

```python
def filter_security_capabilities(capabilities: list) -> list:
    """過濾安全相關能力"""
    security_keywords = [
        "scan", "detect", "vulnerability", "injection",
        "xss", "csrf", "auth", "bypass", "exploit"
    ]
    
    filtered = []
    for cap in capabilities:
        name_lower = cap["name"].lower()
        desc_lower = cap.get("description", "").lower()
        
        if any(kw in name_lower or kw in desc_lower for kw in security_keywords):
            filtered.append(cap)
    
    return filtered

# 使用
all_caps = await analyzer.analyze_capabilities(modules_info)
security_caps = filter_security_capabilities(all_caps)

print(f"總能力: {len(all_caps)}")
print(f"安全相關: {len(security_caps)}")
```

### 7.3 導出為不同格式

**JSON 格式**:
```python
import json

with open("capabilities.json", "w", encoding="utf-8") as f:
    json.dump(capabilities, f, indent=2, ensure_ascii=False)
```

**CSV 格式**:
```python
import csv

with open("capabilities.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "language", "module", "description"])
    writer.writeheader()
    
    for cap in capabilities:
        writer.writerow({
            "name": cap["name"],
            "language": cap.get("language", "python"),
            "module": cap["module"],
            "description": cap.get("description", "")[:100]
        })
```

**Markdown 表格**:
```python
def export_to_markdown(capabilities: list, output_file: str):
    """導出為 Markdown 表格"""
    lines = ["# 系統能力清單\n\n"]
    lines.append("| 名稱 | 語言 | 模組 | 說明 |\n")
    lines.append("|------|------|------|------|\n")
    
    for cap in capabilities:
        name = cap["name"]
        lang = cap.get("language", "python")
        module = cap["module"]
        desc = cap.get("description", "")[:50]
        
        lines.append(f"| `{name}` | {lang} | {module} | {desc} |\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

# 使用
export_to_markdown(capabilities, "CAPABILITIES_LIST.md")
```

---

## 8. 最佳實踐

### 8.1 定期更新能力庫

```python
# 創建定期更新腳本: update_capabilities.py
import asyncio
import json
from datetime import datetime
from pathlib import Path

async def update_capability_database():
    """更新能力資料庫"""
    explorer = ModuleExplorer()
    analyzer = CapabilityAnalyzer()
    
    # 分析
    modules_info = await explorer.explore_all_modules()
    capabilities = await analyzer.analyze_capabilities(modules_info)
    
    # 添加時間戳
    database = {
        "updated_at": datetime.now().isoformat(),
        "total_capabilities": len(capabilities),
        "capabilities": capabilities
    }
    
    # 保存
    output_path = Path("data/capabilities.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已更新 {len(capabilities)} 個能力")

if __name__ == "__main__":
    asyncio.run(update_capability_database())
```

**設定排程** (Windows Task Scheduler):
```powershell
# 每天凌晨 2 點執行
schtasks /create /tn "UpdateCapabilities" /tr "python C:\D\fold7\AIVA-git\update_capabilities.py" /sc daily /st 02:00
```

### 8.2 版本控制

```python
import hashlib

def calculate_capability_hash(capabilities: list) -> str:
    """計算能力列表的哈希值"""
    content = json.dumps(
        sorted(capabilities, key=lambda x: x["name"]),
        sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()

# 檢測變更
old_hash = "..."  # 從上次保存的哈希
new_hash = calculate_capability_hash(capabilities)

if old_hash != new_hash:
    print("⚠️ 能力列表已變更,需要更新!")
```

### 8.3 監控和告警

```python
def validate_capabilities(capabilities: list) -> list[str]:
    """驗證能力完整性"""
    issues = []
    
    for cap in capabilities:
        # 檢查必要欄位
        if not cap.get("name"):
            issues.append(f"能力缺少名稱: {cap}")
        
        if not cap.get("module"):
            issues.append(f"能力 {cap.get('name')} 缺少模組信息")
        
        # 檢查描述質量
        desc = cap.get("description", "")
        if len(desc) < 10:
            issues.append(f"能力 {cap['name']} 描述過短")
    
    return issues

# 執行驗證
issues = validate_capabilities(capabilities)
if issues:
    print(f"⚠️ 發現 {len(issues)} 個問題:")
    for issue in issues[:10]:
        print(f"  - {issue}")
```

---

## 9. 完整檢查清單

執行以下檢查確保整合成功:

```bash
# ✅ 1. 文件存在性
[ ] language_extractors.py 存在
[ ] capability_analyzer.py 已修改
[ ] module_explorer.py 支持多語言

# ✅ 2. 語法正確性
[ ] Python 語法檢查通過 (py_compile)
[ ] 無導入錯誤
[ ] 所有方法可調用

# ✅ 3. 功能測試
[ ] 可掃描多語言文件 (380+ files)
[ ] Python 能力提取成功 (410+ caps)
[ ] Go 能力提取成功 (88+ caps)
[ ] TypeScript 能力提取成功 (78+ caps)

# ✅ 4. 整合測試
[ ] test_multi_language_analysis.py 執行成功
[ ] 所有驗證項目通過
[ ] 無異常錯誤

# ✅ 5. 文檔完整性
[ ] README 已更新
[ ] 使用範例可執行
[ ] 故障排查指南完整
```

---

## 10. 參考資源

### 相關文件
- `services/core/aiva_core/internal_exploration/language_extractors.py`
- `services/core/aiva_core/internal_exploration/capability_analyzer.py`
- `services/core/aiva_core/internal_exploration/module_explorer.py`
- `services/core/aiva_core/internal_exploration/README.md`

### 測試腳本
- `test_multi_language_analysis.py` - 整合測試
- `tests/test_capability_analyzer_multi_lang.py` - 單元測試

### 報告文檔
- `MULTI_LANGUAGE_ANALYSIS_INTEGRATION_REPORT.md` - 整合報告
- `MULTI_LANGUAGE_ANALYSIS_COMPLETE_GUIDE.md` - 本指南

---

## 附錄: 快速命令參考

```bash
# 環境準備
cd C:\D\fold7\AIVA-git
.\.venv\Scripts\Activate.ps1

# 執行測試
python test_multi_language_analysis.py

# 驗證語法
python -m py_compile services/core/aiva_core/internal_exploration/capability_analyzer.py

# 運行單元測試
pytest tests/test_capability_analyzer_multi_lang.py -v

# 檢查導入
python -c "from services.core.aiva_core.internal_exploration import CapabilityAnalyzer; print('✅ OK')"

# 生成能力報告
python -c "import asyncio; from services.core.aiva_core.internal_exploration import *; asyncio.run(ModuleExplorer().explore_all_modules())"

# 更新能力資料庫
python update_capabilities.py
```

---

**版本**: 1.0  
**最後更新**: 2025-11-16  
**維護者**: AIVA Core 開發團隊  
**問題回報**: [GitHub Issues](https://github.com/your-repo/issues)

---

**📝 使用本指南遇到問題?**
1. 檢查[問題排查](#6-問題排查)章節
2. 執行完整檢查清單
3. 查看測試輸出日誌
4. 提交 Issue 並附上錯誤訊息
