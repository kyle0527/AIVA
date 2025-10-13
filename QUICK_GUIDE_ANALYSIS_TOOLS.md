# 程式碼分析工具快速指南

## 快速開始

### 1. 分析整個程式碼庫

```bash
cd /path/to/AIVA
python tools/analyze_codebase.py
```

**輸出位置**: `_out/analysis/analysis_report_*.txt` 和 `.json`

### 2. 生成流程圖

```bash
# 為整個 services 目錄生成流程圖
python tools/py2mermaid.py

# 為單一檔案生成流程圖
python tools/py2mermaid.py -i services/core/aiva_core/ai_engine/tools.py
```

**輸出位置**: `docs/diagrams/*.mmd`

### 3. 在程式碼中使用

```python
from core.aiva_core.ai_engine.tools import CodeAnalyzer

# 建立分析器
analyzer = CodeAnalyzer("/path/to/codebase")

# 詳細分析
result = analyzer.execute(path="relative/path/to/file.py", detailed=True)

# 查看結果
print(f"複雜度: {result['cyclomatic_complexity']}")
print(f"類型提示: {result['has_type_hints']}")
print(f"文檔字串: {result['has_docstrings']}")
```

## 常用命令

### 分析特定模組

```bash
# 只分析 core 模組的流程圖
python tools/py2mermaid.py -i services/core -o docs/diagrams/core -m 30

# 只分析 scan 模組的流程圖  
python tools/py2mermaid.py -i services/scan -o docs/diagrams/scan -m 30
```

### 不同流程圖方向

```bash
# 從左到右（適合寬螢幕）
python tools/py2mermaid.py -d LR

# 從上到下（預設）
python tools/py2mermaid.py -d TB

# 從下到上
python tools/py2mermaid.py -d BT

# 從右到左
python tools/py2mermaid.py -d RL
```

## 報告解讀

### 程式碼品質指標

| 指標 | 良好 | 需改進 | 差 |
|------|------|--------|-----|
| 類型提示覆蓋率 | > 70% | 50-70% | < 50% |
| 文檔字串覆蓋率 | > 80% | 60-80% | < 60% |
| 平均複雜度 | < 10 | 10-20 | > 20 |
| 單檔案複雜度 | < 20 | 20-50 | > 50 |

### AIVA 當前狀態

✅ **類型提示覆蓋率**: 74.8% (良好)
✅ **文檔字串覆蓋率**: 81.9% (良好)
⚠️ **平均複雜度**: 11.94 (需改進)

**需要重構的檔案**:
1. `aiva_common/utils/network/ratelimit.py` (複雜度: 91)
2. `scan/aiva_scan/dynamic_engine/dynamic_content_extractor.py` (複雜度: 54)
3. `function/function_ssrf/aiva_func_ssrf/worker.py` (複雜度: 49)

## 測試工具

```bash
# 執行測試套件
python tools/test_tools.py
```

預期輸出:
```
測試結果: 4 通過, 0 失敗
```

## 整合到工作流程

### Git Pre-commit Hook

在 `.git/hooks/pre-commit` 中加入:

```bash
#!/bin/bash
echo "執行程式碼分析..."
python tools/analyze_codebase.py
```

### CI/CD Pipeline

在 GitHub Actions / GitLab CI 中加入:

```yaml
- name: Code Analysis
  run: |
    python tools/analyze_codebase.py
    python tools/test_tools.py
```

## 疑難排解

### 問題: ImportError

確保從專案根目錄執行：

```bash
cd /path/to/AIVA
python tools/analyze_codebase.py  # ✓
```

不要從 tools 目錄執行：

```bash
cd /path/to/AIVA/tools
python analyze_codebase.py  # ✗
```

### 問題: UnicodeDecodeError

工具會自動嘗試 UTF-8 和 CP950 編碼。如果仍有問題，檢查檔案編碼：

```bash
file -i your_file.py
```

### 問題: 記憶體不足

限制分析的檔案數：

```python
# 在 analyze_codebase.py 中修改
stats = analyze_directory(
    root_path=project_root / "services",
    output_dir=output_dir,
    max_files=100,  # 減少數量
)
```

## 進階用法

### 自訂忽略模式

```python
from tools.analyze_codebase import analyze_directory

stats = analyze_directory(
    root_path=Path("services"),
    output_dir=Path("_out/analysis"),
    ignore_patterns=[
        "__pycache__",
        "test_*",  # 忽略測試檔案
        "backup_*",  # 忽略備份
    ],
    max_files=500,
)
```

### 整合到 Python 腳本

```python
import json
from pathlib import Path
from tools.analyze_codebase import analyze_directory

# 執行分析
stats = analyze_directory(
    root_path=Path("services"),
    output_dir=Path("_out/analysis"),
)

# 檢查品質標準
type_hint_coverage = (stats['files_with_type_hints'] / stats['total_files']) * 100
if type_hint_coverage < 70:
    print(f"警告: 類型提示覆蓋率過低 ({type_hint_coverage:.1f}%)")
    exit(1)

print("✓ 程式碼品質檢查通過")
```

## 相關資源

- **工具文檔**: `tools/README.md`
- **升級報告**: `CODE_ANALYSIS_UPGRADE_REPORT.md`
- **測試套件**: `tools/test_tools.py`
- **範例報告**: `_out/analysis/analysis_report_*.txt`

## 聯絡支援

如有問題或建議，請：
1. 查看本指南的疑難排解部分
2. 查看 `tools/README.md` 詳細文檔
3. 提交 GitHub Issue
