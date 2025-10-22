# AIVA Import Path 最佳實踐指南

## 📋 概述

本指南提供 AIVA 項目中 import 路徑的最佳實踐，防止常見的 import 錯誤並維護代碼品質。

## 🚨 常見問題與解決方案

### ❌ 錯誤的 Import 模式

```python
# 錯誤 - 舊的直接導入模式
from aiva_core.ai_engine import AIModelManager
from aiva_common.schemas import Task
from aiva_scan.scanner import Scanner

# 錯誤 - 缺少 services 前綴
import aiva_core
import aiva_common
```

### ✅ 正確的 Import 模式

```python
# 正確 - 新的絕對路徑模式
from services.core.aiva_core.ai_engine import AIModelManager
from services.aiva_common.schemas import Task
from services.scan.aiva_scan.scanner import Scanner

# 正確 - 包含完整路徑
import services.core.aiva_core
import services.aiva_common
```

## 📁 項目結構對應

### 當前的目錄結構
```
AIVA-git/
├── services/
│   ├── core/
│   │   └── aiva_core/          # 核心 AI 引擎
│   ├── aiva_common/            # 共用 schemas 和 enums
│   ├── features/               # 功能模組
│   ├── scan/                   # 掃描相關
│   └── integration/            # 整合模組
├── examples/                   # 示例程序
└── tools/                      # 開發工具
```

### Import 路徑映射表

| 舊路徑 | 新路徑 | 用途 |
|--------|--------|------|
| `aiva_core.*` | `services.core.aiva_core.*` | AI 核心組件 |
| `aiva_common.*` | `services.aiva_common.*` | 共用 schemas/enums |
| `aiva_scan.*` | `services.scan.aiva_scan.*` | 掃描功能 |
| `aiva_integration.*` | `services.integration.aiva_integration.*` | 整合功能 |

## 🔧 具體修復範例

### 範例 1: AI 引擎組件

```python
# ❌ 錯誤
from aiva_core.ai_engine import AIModelManager, BioNeuronRAGAgent
from aiva_core.learning import ModelTrainer

# ✅ 正確
from services.core.aiva_core.ai_engine import AIModelManager, BioNeuronRAGAgent
from services.core.aiva_core.learning import ModelTrainer
```

### 範例 2: 共用 Schemas 和 Enums

```python
# ❌ 錯誤
from aiva_common.schemas import Task, Target, ScanStrategy
from aiva_common.enums import TaskType, TaskStatus

# ✅ 正確
from services.aiva_common.schemas import Task, Target, ScanStrategy
from services.aiva_common.enums import TaskType, TaskStatus
```

### 範例 3: 工具類別初始化

```python
# ❌ 錯誤 - 缺少必要參數
cmd_executor = CommandExecutor()
code_reader = CodeReader()

# ✅ 正確 - 提供必要的 codebase_path 參數
cmd_executor = CommandExecutor(codebase_path=".")
code_reader = CodeReader(codebase_path=".")
```

### 範例 4: 作用域內導入

```python
# ❌ 錯誤 - 在函數外部未導入
async def check_system():
    manager = AIModelManager()  # 錯誤：未定義

# ✅ 正確 - 在正確作用域內導入
async def check_system():
    from services.core.aiva_core.ai_engine import AIModelManager
    manager = AIModelManager()
```

## 🛠️ 自動化工具

### Import Path Checker 工具

我們提供了自動化工具來檢測和修復 import 問題：

```bash
# 僅檢查問題
python tools/import_path_checker.py --check

# 自動修復問題
python tools/import_path_checker.py --fix

# 生成詳細報告
python tools/import_path_checker.py --report
```

### 工具功能
- ✅ 自動檢測錯誤的 import 模式
- ✅ 批量修復 import 路徑
- ✅ 生成詳細問題報告
- ✅ 自動備份原始檔案
- ✅ 支援整個項目掃描

## 🚀 預防措施

### 1. Pre-commit Hook

在 `.pre-commit-config.yaml` 中加入：

```yaml
repos:
  - repo: local
    hooks:
      - id: import-path-check
        name: AIVA Import Path Check
        entry: python tools/import_path_checker.py --check
        language: system
        files: \.py$
        pass_filenames: false
```

### 2. CI/CD Pipeline

在 GitHub Actions 中加入：

```yaml
- name: Check Import Paths
  run: |
    python tools/import_path_checker.py --check
    if [ $? -ne 0 ]; then
      echo "發現 import 路徑問題，請運行 'python tools/import_path_checker.py --fix' 修復"
      exit 1
    fi
```

### 3. IDE 配置

#### VS Code 設定
在 `.vscode/settings.json` 中：

```json
{
    "python.analysis.extraPaths": [
        "./services"
    ],
    "python.defaultInterpreterPath": "./.venv/bin/python"
}
```

## 📊 錯誤分類與優先級

### 🔴 高優先級（阻塞性錯誤）
- Import 路徑無法解析
- 模組未找到錯誤
- 類別構造函數參數錯誤

### 🟡 中優先級（警告）
- Pylance 符號解析警告
- 動態 import 限制

### 🟢 低優先級（建議）
- 變數可能未繫結警告（在正確的異常處理中）

## 📝 開發檢查清單

在每次 commit 前檢查：

- [ ] 所有 import 語句使用正確的 `services.*` 路徑
- [ ] 類別構造函數提供必要參數
- [ ] 在正確的作用域內進行導入
- [ ] 運行 `python tools/import_path_checker.py --check`
- [ ] 解決所有高優先級錯誤

## 🔍 故障排除

### 常見錯誤訊息

1. **"無法解析匯入"**
   - 檢查是否使用了正確的 `services.*` 路徑
   - 確認模組檔案存在

2. **"參數遺漏引數"**
   - 檢查類別構造函數的必要參數
   - 常見於 `CommandExecutor`, `CodeReader`, `CodeWriter`

3. **"可能未繫結"**
   - 確認在正確的作用域內導入
   - 檢查 try-catch 異常處理邏輯

### 快速修復步驟

1. 運行診斷工具：
   ```bash
   python tools/import_path_checker.py --report
   ```

2. 查看生成的報告：`reports/import_path_check_report.md`

3. 自動修復：
   ```bash
   python tools/import_path_checker.py --fix
   ```

4. 驗證修復：
   ```bash
   python tools/import_path_checker.py --check
   ```

## 📚 延伸閱讀

- [Python Import 系統官方文檔](https://docs.python.org/3/reference/import.html)
- [PEP 328 - Absolute and Relative Imports](https://pep8.org/)
- [AIVA 架構文檔](./ARCHITECTURE/)

---

**注意**: 此指南隨項目演進持續更新。如發現新的問題模式，請更新此文檔並相應修改自動化工具。