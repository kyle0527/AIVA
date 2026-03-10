# AIVA 統一輸出路徑配置說明

## 概述

為了更好地組織專案輸出數據，AIVA 已實施統一的輸出路徑管理系統。所有分析工具（Python、TypeScript、Go、Rust）的輸出將集中存儲在 `services/integration/data/` 目錄下。

## 目錄結構

```
services/
├── integration/
│   └── data/
│       ├── internal_exploration/         # Internal Exploration 模組數據
│       │   ├── analysis_results/        # 各語言工具的分析結果
│       │   │   ├── python/             # Python 工具輸出
│       │   │   ├── typescript/         # TypeScript 工具輸出
│       │   │   ├── go/                 # Go 工具輸出
│       │   │   └── rust/               # Rust 工具輸出
│       │   ├── analysis_history/       # 分析歷史版本 (v1, v2, v3...)
│       │   └── self_healing/           # Self-Healing 診斷報告
│       ├── attack_paths/               # 攻擊路徑數據
│       ├── experiences/                # 經驗學習數據
│       └── training/                   # 訓練數據
├── core/
│   └── aiva_core/
│       └── internal_exploration/       # 工具源碼位置
│           ├── python_tools/
│           ├── typescript_tools/
│           ├── go_tools/
│           └── rust_tools/
└── aiva_common/
    └── config/
        └── paths.py                    # 統一路徑配置文件
```

## 配置文件

### Python: `services/aiva_common/config/paths.py`

統一的路徑配置模組，提供：
- 所有輸出目錄的路徑定義
- 自動創建目錄的輔助函數
- 多語言配置導出功能

```python
from services.aiva_common.config.paths import (
    ANALYSIS_RESULTS_DIR,
    get_analysis_output_dir,
    ensure_directories
)

# 使用示例
ensure_directories()
output_dir = get_analysis_output_dir("python")
```

### TypeScript: `typescript_tools/paths.config.ts`

```typescript
import { getDefaultOutputDir, ensureDirectories } from './paths.config';

ensureDirectories();
const outputDir = getDefaultOutputDir();
```

### Go: `go_tools/paths_config.go`

```go
pathsConfig := GetPathsConfig()
outputDir := pathsConfig.GetDefaultOutputDir()
```

### Rust: `rust_tools/src/paths_config.rs`

```rust
use paths_config::PathsConfig;

let paths_config = PathsConfig::new();
let output_dir = paths_config.get_default_output_dir();
```

## 環境變量控制

### AIVA_USE_INTEGRATED_PATHS

控制是否使用新的集中式路徑（默認：true）

```bash
# 使用新路徑（默認）
export AIVA_USE_INTEGRATED_PATHS=true

# 使用舊路徑（向後兼容）
export AIVA_USE_INTEGRATED_PATHS=false
```

設置為 `false` 時，工具將使用原來的 `./analysis_output` 目錄。

## 已修改的工具

### Python 工具
1. ✅ `aiva_flow_analyzer.py` - AST 分析器
2. ✅ `aiva_exploration_pipeline.py` - 三階段管道
3. ✅ `core_analyzer.py` - Self-Healing 診斷
4. ✅ `run_analysis.py` - Self-Healing CLI

### TypeScript 工具
1. ✅ `ts2mermaid.ts` - TypeScript AST 分析器

### Go 工具
1. ✅ `go2mermaid.go` - Go AST 分析器

### Rust 工具
1. ✅ `src/main.rs` - Rust AST 分析器

## 使用方式

### 1. Python 工具

```bash
# 使用新路徑（默認）
python aiva_flow_analyzer.py --target path/to/code

# 使用舊路徑
AIVA_USE_INTEGRATED_PATHS=false python aiva_flow_analyzer.py --target path/to/code

# 自定義輸出路徑
python aiva_flow_analyzer.py --target path/to/code --output /custom/path
```

### 2. TypeScript 工具

```bash
# 使用新路徑（默認）
npm run analyze

# 使用舊路徑
AIVA_USE_INTEGRATED_PATHS=false npm run analyze

# 自定義輸出路徑
npm run analyze -- --output=/custom/path
```

### 3. Go 工具

```bash
# 使用新路徑（默認）
go run go2mermaid.go --input=./path/to/code

# 使用舊路徑
AIVA_USE_INTEGRATED_PATHS=false go run go2mermaid.go --input=./path/to/code

# 自定義輸出路徑
go run go2mermaid.go --input=./path/to/code --output=/custom/path
```

### 4. Rust 工具

```bash
# 使用新路徑（默認）
cargo run -- --input=./path/to/code

# 使用舊路徑
AIVA_USE_INTEGRATED_PATHS=false cargo run -- --input=./path/to/code

# 自定義輸出路徑
cargo run -- --input=./path/to/code --output=/custom/path
```

## 向後兼容性

所有修改都保持向後兼容：

1. **環境變量控制**：可通過 `AIVA_USE_INTEGRATED_PATHS=false` 使用舊路徑
2. **自定義路徑優先**：命令行 `--output` 參數優先於默認配置
3. **try/except 保護**：Python 工具在無法導入配置時使用舊路徑
4. **漸進式遷移**：現有數據不受影響，新輸出使用新路徑

## 輸出示例

### 分析結果輸出

```
services/integration/data/internal_exploration/analysis_results/
├── python/
│   ├── flow_graphs/
│   │   ├── module1_flow.mmd
│   │   └── module2_flow.mmd
│   ├── statistics.json
│   └── summary.md
├── typescript/
│   ├── flow_graphs/
│   ├── statistics.json
│   └── summary.md
├── go/
│   └── ...
└── rust/
    └── ...
```

### 歷史版本輸出

```
services/integration/data/internal_exploration/analysis_history/
├── v1/
│   ├── timestamp_2024-01-01/
│   └── ...
├── v2/
│   └── ...
└── v3/
    └── ...
```

### Self-Healing 輸出

```
services/integration/data/internal_exploration/self_healing/
├── core_analysis_20240101_120000.json
├── breakpoint_analysis_20240101_120000.json
└── practical_suggestions_20240101_120000.json
```

## 優點

1. **統一管理**：所有輸出數據集中在 integration 模組
2. **多語言支持**：Python/TypeScript/Go/Rust 工具使用相同的路徑結構
3. **易於備份**：一個目錄包含所有分析數據
4. **清晰組織**：按工具類型和功能分類存儲
5. **向後兼容**：不影響現有工作流程
6. **環境隔離**：通過環境變量控制行為

## 遷移檢查清單

- [x] 創建統一路徑配置文件
- [x] 修改 Python 工具
- [x] 修改 TypeScript 工具
- [x] 修改 Go 工具
- [x] 修改 Rust 工具
- [x] 更新文檔
- [ ] 測試所有工具
- [ ] 通知團隊成員

## 測試建議

1. **基本功能測試**
   ```bash
   # 測試 Python 工具
   python aiva_flow_analyzer.py --target services/core
   
   # 測試 TypeScript 工具
   npm run analyze -- --input=src
   
   # 測試 Go 工具
   go run go2mermaid.go --input=.
   
   # 測試 Rust 工具
   cargo run -- --input=src
   ```

2. **向後兼容性測試**
   ```bash
   AIVA_USE_INTEGRATED_PATHS=false python aiva_flow_analyzer.py --target services/core
   ```

3. **自定義路徑測試**
   ```bash
   python aiva_flow_analyzer.py --target services/core --output /tmp/test_output
   ```

## 問題排查

### 找不到配置文件
- 確保 `services/aiva_common/config/paths.py` 存在
- 檢查 Python 路徑是否包含 services 目錄
- 使用 `AIVA_USE_INTEGRATED_PATHS=false` 回退到舊路徑

### 權限問題
- 確保對 `services/integration/data/` 目錄有寫入權限
- 檢查目錄所有權和權限設置

### 路徑不正確
- 檢查 `PROJECT_ROOT` 環境變量
- 驗證相對路徑計算是否正確
- 使用 `--output` 參數指定絕對路徑

## 聯繫方式

如有問題或建議，請聯繫 AIVA 開發團隊。
