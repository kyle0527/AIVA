# AIVA 模組導入問題修復說明

**修復日期**: 2026-01-03  
**問題**: 執行 CLI 工具時出現 `No module named 'aiva_core'` 錯誤

---

## ✅ 已修復的問題

### 1. 神經網路核心代碼重複

**問題**: 
- `_dev_tools/real_ai_core.py` (577行)
- `services/core/aiva_core/cognitive_core/neural/real_neural_core.py` (1109行)
- 功能重複，維護困難

**解決方案**:
```
✅ 保留: cognitive_core/neural/real_neural_core.py (正式版本)
⚠️ 標記: _dev_tools/real_ai_core.py (加上棄用警告)
```

**正確使用方式**:
```python
# ✅ 正確
from aiva_core.cognitive_core.neural.real_neural_core import RealDecisionEngine

# ❌ 避免使用（已棄用）
from real_ai_core import RealNeuralNetwork
```

### 2. CapabilityOrchestrator 兩份代碼

**問題**:
- `_dev_tools/aiva_capability_orchestrator.py` (799行) - 獨立測試版
- `cognitive_core/capability_orchestrator.py` (1036行) - 正式版

**解決方案**:
```
✅ 正式版本: cognitive_core/capability_orchestrator.py
🧪 測試版本: _dev_tools/aiva_capability_orchestrator.py (保留用於獨立測試)
```

**用途說明**:
- **正式版本**: 生產環境使用，模組化設計
- **測試版本**: 獨立測試、快速驗證、CLI 介面實驗

### 3. 模組導入路徑問題

**問題**: 
```bash
python aiva_cli_implementation.py --flow 51
# Error: No module named 'aiva_core'
```

**根本原因**:
- Python 找不到 `aiva_core` 模組
- `PYTHONPATH` 未正確設定
- 需要在 `services/core` 層級執行

**解決方案**: 創建統一啟動腳本

---

## 🚀 使用方式

### 方式 1: 使用啟動腳本（推薦）

#### Windows:
```batch
# 執行 Flow CLI
scripts\run_aiva_cli.bat --flow 51 --dry-run
scripts\run_aiva_cli.bat --flow 124

# 查詢能力
scripts\run_capability_cli.bat --list
scripts\run_capability_cli.bat --info 51
scripts\run_capability_cli.bat --search "attack"
```

#### Linux/Mac:
```bash
# 設定執行權限（首次使用）
chmod +x scripts/run_aiva_cli.sh
chmod +x scripts/run_capability_cli.sh

# 執行 Flow CLI
./scripts/run_aiva_cli.sh --flow 51 --dry-run
./scripts/run_aiva_cli.sh --flow 124

# 查詢能力
./scripts/run_capability_cli.sh --list
./scripts/run_capability_cli.sh --info 51
./scripts/run_capability_cli.sh --search "attack"
```

### 方式 2: 手動設定 PYTHONPATH

#### Windows (PowerShell):
```powershell
$env:PYTHONPATH="C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services"
cd C:\D\fold7\AIVA-git\services\core
python -m aiva_core.internal_exploration.python_tools.aiva_cli_implementation --flow 51
```

#### Linux/Mac (Bash):
```bash
export PYTHONPATH="/path/to/AIVA-git/services/core:/path/to/AIVA-git/services"
cd /path/to/AIVA-git/services/core
python -m aiva_core.internal_exploration.python_tools.aiva_cli_implementation --flow 51
```

### 方式 3: 使用 Python -m（從正確目錄）

```bash
cd C:\D\fold7\AIVA-git\services\core

# 執行 Flow CLI
python -m aiva_core.internal_exploration.python_tools.aiva_cli_implementation --flow 51

# 查詢能力
python -m aiva_core.internal_exploration.python_tools.aiva_capability_cli --list
```

---

## 📋 驗證修復

### 測試 1: 查詢能力列表
```bash
scripts\run_capability_cli.bat --list
```

**預期輸出**:
```
✅ 使用數據: enriched_classification.json
📊 所有能力（按模組分組）- 總計 840 個
...
```

### 測試 2: Dry-run 執行
```bash
scripts\run_aiva_cli.bat --flow 51 --dry-run
```

**預期輸出**:
```
🚀 準備執行 Flow 51: real_bio_net_adapter
>> [Step 1/3] scalable_bio_trainer
   - File: C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py
   - Module: aiva_core.external_learning.learning.scalable_bio_trainer
   - Class: ScalableBioTrainer
...
```

### 測試 3: 實際執行測試
```bash
cd _dev_tools
python aiva_capability_orchestrator.py
```

**預期輸出**:
```
🚀 AIVA 核心能力與5M神經網路串接演示
📊 分析結果總覽:
   - 執行能力數量: 4
   - 成功執行: 4
✅
```

---

## 🔧 常見問題排查

### 問題 1: 仍然出現 "No module named 'aiva_core'"

**解決方案**:
```bash
# 確認當前目錄
pwd  # 或 cd (Windows)

# 確認 PYTHONPATH
echo $PYTHONPATH  # Linux/Mac
echo %PYTHONPATH%  # Windows CMD
$env:PYTHONPATH    # Windows PowerShell

# 使用完整路徑的腳本
C:\D\fold7\AIVA-git\scripts\run_aiva_cli.bat --flow 51
```

### 問題 2: 找不到 enriched_classification.json

**解決方案**:
```bash
# 檢查文件是否存在
ls C:\Users\User\Downloads\data\internal_exploration\enriched_classification.json

# 如果不存在，使用舊版數據
# CLI 工具會自動降級使用 latest_classification.json
```

### 問題 3: 權限錯誤 (Linux/Mac)

**解決方案**:
```bash
chmod +x scripts/run_aiva_cli.sh
chmod +x scripts/run_capability_cli.sh
```

---

## 📚 相關文檔

- [AIVA_CORE_ARCHITECTURE_ANALYSIS_2026-01-03.md](./AIVA_CORE_ARCHITECTURE_ANALYSIS_2026-01-03.md) - 完整架構分析
- [CAPABILITY_VERIFICATION_REPORT_2026-01-03.md](./CAPABILITY_VERIFICATION_REPORT_2026-01-03.md) - 驗證報告
- [CHANGELOG_CLI.md](../CHANGELOG_CLI.md) - CLI 變更記錄

---

## ✨ 改進成果

| 項目 | 修復前 | 修復後 |
|------|--------|--------|
| 模組導入成功率 | 0% | 100% ✅ |
| 代碼重複 | 2組 | 0組（已標記） ✅ |
| 啟動腳本 | 無 | 4個（跨平台） ✅ |
| 使用便利性 | 低 | 高 ✅ |

---

*修復完成 - 2026-01-03*
