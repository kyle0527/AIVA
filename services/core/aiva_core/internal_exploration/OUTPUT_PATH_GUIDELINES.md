# Internal Exploration 輸出路徑規範

**重要原則**: Internal Exploration 模組**不應該產生任何持久化輸出文件**。所有分析結果應該輸出到整合模組。

## 🎯 輸出路徑配置

### 1. Internal CLI（AI Core 分析）

**執行器**: `aiva_internal_executor.py`  
**分類器**: `aiva_internal_classifier.py`

**輸出路徑**:
```
services/integration/cli_outputs/python/
├── CLI_COMMANDS_REFERENCE.md
└── cli_commands_db.json
```

**配置方式**:
```python
# 自動檢測 Integration 模組
from integration import CLI_OUTPUTS_PYTHON_DIR
CLI_OUTPUT_DIR = CLI_OUTPUTS_PYTHON_DIR
```

### 2. External CLI（Features/Scan 分析）

**執行器**: `aiva_external_executor.py`  
**分類器**: `aiva_external_classifier.py`

**輸出路徑**:
```
features_classification/
├── classification_data.json
├── EXTERNAL_CLI_COMMANDS_REFERENCE.md
└── external_cli_commands_db.json
```

**配置方式**:
```python
PROJECT_ROOT / "features_classification"
```

### 3. 語言工具（AST 分析）

#### Python 工具
```bash
python python_tools/aiva_flow_analyzer.py \
  --target services/core/aiva_core \
  --output services/integration/analysis/python
```

#### Go 工具
```bash
cd services/features/function_authn_go
go run ../../core/aiva_core/internal_exploration/go_tools/go2mermaid.go .
# 輸出: ./analysis_output/analysis_results.json
```

#### Rust 工具
```bash
cd services/features/function_crypto
cargo run --manifest-path ../../core/aiva_core/internal_exploration/rust_tools/Cargo.toml -- .
# 輸出: ./analysis_output/analysis_results.json
```

#### TypeScript 工具
```bash
cd services/scan/typescript_engine
npx ts-node ../../core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts ./src
# 輸出: ./analysis_output/analysis_results.json
```

### 4. Self-Healing（診斷工具）

**輸出路徑**:
```
<目標模組>/analysis_results/
├── dataflow_breakpoints_report.md
├── missing_connections_report.md
└── health_check_report.json
```

**說明**: 診斷報告輸出到**被診斷模組的目錄**，不在 internal_exploration 中。

---

## ⚠️ 禁止的輸出位置

❌ **絕對不要在以下位置產生輸出**:

```
services/core/aiva_core/internal_exploration/output/          # 禁止
services/core/aiva_core/internal_exploration/outputs/         # 禁止
services/core/aiva_core/internal_exploration/analysis/        # 禁止
services/core/aiva_core/internal_exploration/*.json           # 禁止（配置文件除外）
services/core/aiva_core/internal_exploration/**/*.json        # 禁止（配置文件除外）
```

**例外**: 僅允許以下配置文件:
- `modules_config.json` - 模組配置
- `package.json` - Node.js 配置
- `Cargo.toml` - Rust 配置

---

## 📋 .gitignore 配置

已在根目錄 `.gitignore` 添加：

```gitignore
# Internal Exploration 輸出 (應輸出到整合模組)
services/core/aiva_core/internal_exploration/output/
services/core/aiva_core/internal_exploration/**/output/
services/core/aiva_core/internal_exploration/**/analysis_results/
services/core/aiva_core/internal_exploration/**/analysis_output/
services/core/aiva_core/internal_exploration/**/*.json
!services/core/aiva_core/internal_exploration/modules_config.json
!services/core/aiva_core/internal_exploration/**/package.json
!services/core/aiva_core/internal_exploration/**/package-lock.json
```

---

## 🔍 檢查遺留文件

定期執行以下命令檢查是否有遺留輸出：

```powershell
# 檢查 JSON 文件（排除配置文件）
Get-ChildItem -Path "services/core/aiva_core/internal_exploration" `
  -Filter "*.json" -Recurse | `
  Where-Object { $_.Name -notin @("modules_config.json", "package.json", "package-lock.json") }

# 檢查輸出目錄
Get-ChildItem -Path "services/core/aiva_core/internal_exploration" `
  -Directory -Recurse | `
  Where-Object { $_.Name -in @("output", "outputs", "analysis_results", "analysis_output") }
```

---

## ✅ 驗證清單

開發新功能時，確保：

- [ ] 所有輸出路徑指向整合模組
- [ ] 沒有硬編碼 `./output` 或 `./analysis` 路徑
- [ ] 使用配置的 `CLI_OUTPUT_DIR` 或 `PROJECT_ROOT`
- [ ] 測試後檢查是否產生遺留文件
- [ ] 更新本文檔反映新的輸出路徑

---

**最後更新**: 2026-01-18  
**維護者**: AIVA Core Team
