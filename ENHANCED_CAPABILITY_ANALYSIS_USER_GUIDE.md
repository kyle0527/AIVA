# 增強版多語言能力分析使用指南

**版本**: v2.0 Enhanced  
**更新日期**: 2025-11-16

---

## 🚀 快速開始

### 基本使用

```python
import asyncio
from services.core.aiva_core.internal_exploration import ModuleExplorer, CapabilityAnalyzer

async def analyze_capabilities():
    # 1. 初始化探索器和分析器
    explorer = ModuleExplorer()
    analyzer = CapabilityAnalyzer()
    
    # 2. 探索所有模組
    modules = await explorer.explore_all_modules()
    print(f"📚 Found {len(modules)} modules")
    
    # 3. 分析能力
    capabilities = await analyzer.analyze_capabilities(modules)
    print(f"✅ Extracted {len(capabilities)} capabilities")
    
    # 4. 查看語言分布
    from collections import Counter
    lang_counts = Counter(cap["language"] for cap in capabilities)
    
    print("\n📊 Language Distribution:")
    for lang, count in lang_counts.most_common():
        percentage = (count / len(capabilities)) * 100
        print(f"  {lang:12} : {count:4} ({percentage:5.1f}%)")
    
    # 5. 查看錯誤報告
    analyzer.print_extraction_report()
    
    return capabilities

# 運行
capabilities = asyncio.run(analyze_capabilities())
```

**輸出範例**:
```
📚 Found 4 modules
🔍 Starting capability analysis for 4 modules...
✅ Extracted 692 capabilities

📊 Language Distribution:
  python       :  411 ( 59.4%)
  rust         :  115 ( 16.6%)
  go           :   88 ( 12.7%)
  typescript   :   78 ( 11.3%)

📊 Capability Extraction Report
==============================================================
📁 Files Processed:
  Total:      382
  ✅ Success:  382
  ❌ Failed:   0
  ⚠️  Skipped:  0
  Success Rate: 100.0%
==============================================================
```

---

## 📝 能力數據結構

### Python 能力
```python
{
    "name": "sql_injection_scan",
    "language": "python",
    "module": "core/aiva_core",
    "description": "執行 SQL 注入漏洞掃描",
    "parameters": [
        {"name": "target_url", "annotation": "str"},
        {"name": "payload_type", "annotation": "str | None"}
    ],
    "file_path": "C:/D/fold7/AIVA-git/services/core/...",
    "return_type": "ScanResult",
    "is_async": True,
    "decorators": ["@register_capability"],
    "docstring": "執行 SQL 注入漏洞掃描...",
    "line_number": 45
}
```

### Rust 能力 (方法)
```python
{
    "name": "SensitiveInfoScanner::scan",
    "language": "rust",
    "module": "scan",
    "struct": "SensitiveInfoScanner",  # ✨ 新增
    "method": "scan",                   # ✨ 新增
    "description": "Scan content for sensitive information",
    "parameters": [
        {"name": "content", "type": "&str"},
        {"name": "source_url", "type": "&str"}
    ],
    "file_path": "C:/D/fold7/AIVA-git/services/scan/...",
    "return_type": "Vec<Finding>",
    "is_async": False,
    "is_method": True,                  # ✨ 新增
    "line_number": 123
}
```

### Go 能力
```python
{
    "name": "DetectSSRF",
    "language": "go",
    "module": "scanner",
    "description": "Go function: DetectSSRF",
    "parameters": [
        {"name": "target", "type": "string"},
        {"name": "options", "type": "*DetectorOptions"}
    ],
    "file_path": "C:/D/fold7/AIVA-git/services/scan/...",
    "return_type": "(*Finding, error)",
    "is_exported": True,
    "line_number": 67
}
```

### TypeScript 能力
```python
{
    "name": "scanWebApplication",
    "language": "typescript",
    "module": "scan",
    "description": "Scan web application for vulnerabilities",
    "parameters": [
        {"name": "url", "type": "string", "description": "Target URL"},
        {"name": "options", "type": "ScanOptions", "description": "Scan options"}
    ],
    "file_path": "C:/D/fold7/AIVA-git/services/scan/...",
    "return_type": "Promise<ScanResult>",
    "is_async": True,
    "is_exported": True,
    "line_number": 89
}
```

---

## 🔍 進階使用

### 1. 過濾特定語言能力

```python
# 只查看 Rust 方法
rust_methods = [
    cap for cap in capabilities 
    if cap["language"] == "rust" and cap.get("is_method")
]

print(f"🦀 Found {len(rust_methods)} Rust methods")
for cap in rust_methods[:10]:
    print(f"  - {cap['name']}")
```

### 2. 按模組分組

```python
analyzer = CapabilityAnalyzer()
grouped = analyzer.get_capabilities_by_module(capabilities)

for module, caps in grouped.items():
    print(f"\n📦 Module: {module}")
    print(f"   Capabilities: {len(caps)}")
    
    # 統計語言分布
    langs = Counter(cap["language"] for cap in caps)
    for lang, count in langs.items():
        print(f"     - {lang}: {count}")
```

### 3. 查找異步能力

```python
async_capabilities = [
    cap for cap in capabilities 
    if cap.get("is_async")
]

print(f"⚡ Found {len(async_capabilities)} async capabilities")

# 按語言分組
by_lang = {}
for cap in async_capabilities:
    lang = cap["language"]
    by_lang.setdefault(lang, []).append(cap)

for lang, caps in by_lang.items():
    print(f"  {lang}: {len(caps)} async capabilities")
```

### 4. 提取錯誤報告

```python
analyzer = CapabilityAnalyzer()
# ... 執行分析 ...

# 獲取詳細報告
report = analyzer.get_extraction_report()

print(f"📊 Statistics:")
print(f"  Total Files:    {report['statistics']['total_files']}")
print(f"  Success Rate:   {report['success_rate']:.1f}%")
print(f"  Total Errors:   {report['total_errors']}")

if report['total_errors'] > 0:
    print(f"\n⚠️  Errors by Type:")
    for err_type, count in report['errors_by_type'].items():
        print(f"    {err_type}: {count}")
    
    print(f"\n📋 Recent Errors:")
    for err in report['recent_errors']:
        print(f"    - {err['file']}")
        print(f"      Type: {err['type']}")
        print(f"      Message: {err['message']}")
```

---

## 🧪 測試和驗證

### 運行完整測試

```bash
# 在專案根目錄
cd C:\D\fold7\AIVA-git

# 運行測試腳本
python -m services.core.aiva_core.internal_exploration.test_enhanced_extraction
```

### 運行特定測試

```python
import asyncio
from services.core.aiva_core.internal_exploration.test_enhanced_extraction import (
    test_rust_extraction,
    test_error_handling,
    test_full_analysis
)

# 只測試 Rust 提取
asyncio.run(test_rust_extraction())

# 只測試錯誤處理
asyncio.run(test_error_handling())

# 完整分析
asyncio.run(test_full_analysis())
```

---

## 🔧 故障排除

### 問題 1: ModuleNotFoundError

**症狀**: `ModuleNotFoundError: No module named 'services'`

**解決方案**:
```bash
# 確保在專案根目錄
cd C:\D\fold7\AIVA-git

# 使用 -m 標誌運行
python -m services.core.aiva_core.internal_exploration.test_enhanced_extraction
```

### 問題 2: 沒有提取到 Rust 能力

**檢查清單**:
1. ✅ 確認 Rust 文件存在
   ```bash
   Get-ChildItem -Path "services" -Recurse -Filter "*.rs" | Measure-Object
   ```

2. ✅ 檢查 Rust 代碼格式
   ```rust
   // ✅ 正確: impl 區塊 + pub fn
   impl Scanner {
       pub fn scan(&self) -> Result<()> { }
   }
   
   // ❌ 錯誤: 私有方法不會被提取
   impl Scanner {
       fn internal_method(&self) { }
   }
   ```

3. ✅ 查看日誌輸出
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### 問題 3: 成功率低於 100%

**診斷步驟**:
```python
analyzer = CapabilityAnalyzer()
# ... 執行分析 ...

# 查看錯誤詳情
report = analyzer.get_extraction_report()

print("❌ Failed Files:")
for err in report['recent_errors']:
    print(f"  File: {err['file']}")
    print(f"  Type: {err['type']}")
    print(f"  Message: {err['message']}\n")
```

---

## 📊 性能優化建議

### 1. 跳過不必要的目錄

```python
explorer = ModuleExplorer()

# 自定義排除模式
explorer.exclude_patterns.extend([
    "**/node_modules/**",
    "**/target/**",        # Rust 編譯輸出
    "**/__pycache__/**",
    "**/venv/**"
])
```

### 2. 並行處理 (未來版本)

```python
# 當前版本: 同步處理
capabilities = await analyzer.analyze_capabilities(modules)

# 未來版本 (P2): 並行處理
analyzer = CapabilityAnalyzer(max_workers=4)
capabilities = await analyzer.analyze_capabilities_parallel(modules)
```

### 3. 使用快取 (未來版本)

```python
# 未來版本 (P2): 智能快取
analyzer = CapabilityAnalyzer(enable_cache=True)

# 首次運行: 完整掃描
capabilities1 = await analyzer.analyze_capabilities(modules)

# 二次運行: 使用快取 (僅掃描變更文件)
capabilities2 = await analyzer.analyze_capabilities(modules)
```

---

## 🎯 最佳實踐

### 1. 定期執行完整掃描

```python
# 在 CI/CD 流程中
async def ci_capability_check():
    explorer = ModuleExplorer()
    analyzer = CapabilityAnalyzer()
    
    modules = await explorer.explore_all_modules()
    capabilities = await analyzer.analyze_capabilities(modules)
    
    # 驗證最小能力數
    MIN_CAPABILITIES = 650
    if len(capabilities) < MIN_CAPABILITIES:
        raise ValueError(f"Capability count dropped: {len(capabilities)} < {MIN_CAPABILITIES}")
    
    # 驗證成功率
    report = analyzer.get_extraction_report()
    if report['success_rate'] < 95.0:
        raise ValueError(f"Success rate too low: {report['success_rate']:.1f}%")
    
    print(f"✅ CI Check Passed: {len(capabilities)} capabilities")
```

### 2. 監控能力變化

```python
# 保存基線
import json

baseline_path = "capabilities_baseline.json"

# 首次運行: 保存基線
with open(baseline_path, 'w') as f:
    json.dump(capabilities, f, indent=2)

# 後續運行: 比較變化
with open(baseline_path) as f:
    baseline = json.load(f)

# 比較差異
new_caps = {cap['name'] for cap in capabilities}
old_caps = {cap['name'] for cap in baseline}

added = new_caps - old_caps
removed = old_caps - new_caps

if added:
    print(f"✅ Added capabilities: {len(added)}")
    for name in list(added)[:10]:
        print(f"  + {name}")

if removed:
    print(f"⚠️  Removed capabilities: {len(removed)}")
    for name in list(removed)[:10]:
        print(f"  - {name}")
```

### 3. 生成能力文檔

```python
def generate_capability_docs(capabilities: list[dict], output_path: str):
    """生成能力清單 Markdown 文檔"""
    
    lines = [
        "# AIVA 能力清單",
        f"\n**生成時間**: {datetime.now().isoformat()}",
        f"**總計**: {len(capabilities)} 個能力\n",
    ]
    
    # 按語言分組
    by_lang = {}
    for cap in capabilities:
        lang = cap["language"]
        by_lang.setdefault(lang, []).append(cap)
    
    for lang, caps in sorted(by_lang.items()):
        lines.append(f"\n## {lang.title()} ({len(caps)} 個能力)\n")
        
        # 按模組分組
        by_module = {}
        for cap in caps:
            module = cap.get("module", "unknown")
            by_module.setdefault(module, []).append(cap)
        
        for module, module_caps in sorted(by_module.items()):
            lines.append(f"\n### 模組: {module}\n")
            
            for cap in sorted(module_caps, key=lambda x: x['name']):
                lines.append(f"- **`{cap['name']}`**")
                if cap.get("description"):
                    lines.append(f"  - {cap['description']}")
                if cap.get("parameters"):
                    params = ", ".join(p["name"] for p in cap["parameters"])
                    lines.append(f"  - 參數: `{params}`")
                lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Documentation generated: {output_path}")

# 使用
generate_capability_docs(capabilities, "CAPABILITIES.md")
```

---

## 📚 相關資源

### 內部文檔
- [MULTI_LANGUAGE_INTEGRATION_IMPROVEMENT_PLAN.md](./MULTI_LANGUAGE_INTEGRATION_IMPROVEMENT_PLAN.md) - 改善計劃
- [P0_IMPLEMENTATION_COMPLETION_REPORT.md](./P0_IMPLEMENTATION_COMPLETION_REPORT.md) - 完成報告
- [MULTI_LANGUAGE_ANALYSIS_INTEGRATION_REPORT.md](./MULTI_LANGUAGE_ANALYSIS_INTEGRATION_REPORT.md) - 原始分析

### API 文檔
- `ModuleExplorer` - 模組探索器
- `CapabilityAnalyzer` - 能力分析器
- `LanguageExtractor` - 語言提取器基類
  - `GoExtractor`
  - `RustExtractor` ✨ 增強版
  - `TypeScriptExtractor`

### 測試文件
- `test_enhanced_extraction.py` - 增強版測試腳本

---

## 🤝 貢獻指南

### 添加新語言支援

1. 在 `language_extractors.py` 創建新的提取器類
   ```python
   class KotlinExtractor(LanguageExtractor):
       def extract_capabilities(self, content: str, file_path: str) -> list[dict[str, Any]]:
           # 實現提取邏輯
           pass
   ```

2. 在 `get_extractor()` 註冊語言
   ```python
   extractors = {
       "go": GoExtractor(),
       "rust": RustExtractor(),
       "typescript": TypeScriptExtractor(),
       "javascript": TypeScriptExtractor(),
       "kotlin": KotlinExtractor(),  # 新增
   }
   ```

3. 在 `capability_analyzer.py` 添加語言檢測
   ```python
   language_map = {
       ".py": "python",
       ".go": "go",
       ".rs": "rust",
       ".ts": "typescript",
       ".js": "javascript",
       ".kt": "kotlin",  # 新增
   }
   ```

4. 添加測試用例
   ```python
   def test_kotlin_extraction():
       # 測試邏輯
       pass
   ```

---

**指南版本**: v2.0  
**最後更新**: 2025-11-16  
**維護者**: AIVA 架構團隊
