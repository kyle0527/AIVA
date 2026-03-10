# 多語言分析工具統一規範與整合計劃

**版本**: v1.0  
**日期**: 2026-01-13  
**狀態**: 執行中

---

## 📋 架構決策

### 核心原則
根據用戶需求明確的架構方向：

1. **各語言工具應先在各自語言內完整整合**
2. **確保各工具獨立運作能力OK後再考慮跨語言整合**
3. **唯一需要統一的是能力描述方式和CLI指令格式**
4. **其他功能保持各語言特色，不需要特別統一**

### 架構模式選擇

✅ **採用：完全獨立模式**

每個語言工具都應該是完整的：
```
各語言工具 = 分析器 + 分類器 + 串接器 + CLI生成器 + CLI執行器
```

**原因**：
- 各語言開發者可獨立開發和測試
- 工具可單獨部署和使用
- 減少跨語言依賴
- 符合"先完整再整合"的原則

---

## 🎯 統一規範定義

### 1. 能力描述格式 (AI可理解)

**必須統一的JSON結構**：

```json
{
  "function_name": "函數名稱",
  "module": "所屬模組",
  "category": "功能分類",
  "description": "能力描述",
  "inputs": ["參數1", "參數2"],
  "outputs": ["返回值類型"],
  "cli_command": "執行此功能的CLI指令",
  "flow_steps": 流程步驟數量
}
```

**能力描述規則**：
- 使用動詞開頭："執行...", "分析...", "生成...", "檢測..."
- 包含目標和方法："檢測XSS漏洞通過模糊測試"
- 不超過100字
- 中英文雙語支持

**範例**：
```json
{
  "function_name": "detect_xss_reflected",
  "category": "XSS",
  "description": "檢測反射型XSS漏洞通過在URL參數注入測試腳本並觀察響應內容 | Detect reflected XSS by injecting test scripts in URL parameters and observing response",
  "cli_command": "aiva xss detect --type reflected --target <url>"
}
```

### 2. CLI指令命名規範

**統一的命令行參數**：

| 參數 | 短參數 | 說明 | 範例 |
|------|--------|------|------|
| `--input` | `-i` | 輸入目錄或文件 | `--input ./src` |
| `--output` | `-o` | 輸出目錄 | `--output ./analysis` |
| `--target` | `-t` | 目標URL或服務 | `--target https://example.com` |
| `--mode` | `-m` | 執行模式 | `--mode pipeline` |
| `--format` | `-f` | 輸出格式 | `--format json` |
| `--verbose` | `-v` | 詳細輸出 | `--verbose` |

**統一的命令結構**：
```bash
<tool_name> <action> [options]

# 範例
ts2mermaid analyze --input ./src --output ./analysis
go2mermaid pipeline --input ./app --mode full
rs2mermaid query --target function_crypto --format json
```

### 3. 不需要統一的部分

✅ **保持各語言特色**：
- 內部實現邏輯
- 錯誤處理方式
- 性能優化策略
- 依賴管理方式
- 測試框架選擇

---

## 📊 當前工具狀態評估

### Python 工具套件 ✅ 100%

**位置**: `services/core/aiva_core/internal_exploration/python_tools/`

| 工具 | 功能 | 狀態 |
|------|------|------|
| aiva_flow_analyzer.py | AST分析+流程圖生成 | ✅ 完整 |
| aiva_flow_classifier.py | 功能分類 | ✅ 完整 |
| aiva_cli_implementation.py | CLI執行器 | ✅ 完整 |
| aiva_exploration_pipeline.py | 主控程序 | ✅ 完整 |
| aiva_external_module_classifier.py | 外部模組分類 | ✅ 完整 |
| aiva_external_module_cli.py | 外部模組CLI | ✅ 完整 |
| aiva_capability_cli.py | 能力CLI | ✅ 完整 |

**執行能力**: ✅ 有 (FlowExecutor類)

### TypeScript 工具 🟡 95%

**位置**: `services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts`

**已有功能** (769行)：
- ✅ AST解析 (使用typescript模組)
- ✅ 流程圖生成 (Mermaid格式)
- ✅ 跨文件串接 (Stitcher類)
- ✅ 功能分類 (Classifier類)
- ✅ CLI文檔生成 (generateCLI函數)
- ✅ 瓶頸分析 (analyzeBranches方法)

**缺少功能**：
- ❌ CLI執行器 (無FlowExecutor等價物)
- ❌ Pipeline模式
- ❌ Query子命令

### Go 工具 🟡 95%

**位置**: `services/core/aiva_core/internal_exploration/go_tools/go2mermaid.go`

**已有功能** (785行)：
- ✅ AST解析 (使用go/ast, go/parser)
- ✅ 流程圖生成 (Mermaid格式)
- ✅ 跨文件串接 (Stitcher結構)
- ✅ 功能分類 (Classifier結構)
- ✅ CLI文檔生成 (GenerateCLI函數)
- ✅ 瓶頸分析 (AnalyzeBranches方法)

**缺少功能**：
- ❌ CLI執行器 (無FlowExecutor等價物)
- ❌ Pipeline模式
- ❌ Query子命令

### Rust 工具 🟡 95%

**位置**: `services/core/aiva_core/internal_exploration/rust_tools/src/main.rs`

**已有功能** (750行)：
- ✅ AST解析 (使用syn crate)
- ✅ 流程圖生成 (Mermaid格式)
- ✅ 跨文件串接 (Stitcher結構)
- ✅ 功能分類 (Classifier結構)
- ✅ CLI文檔生成 (generate_cli函數)
- ✅ 瓶頸分析 (analyze_branches方法)

**缺少功能**：
- ❌ CLI執行器 (無FlowExecutor等價物)
- ❌ Pipeline模式
- ❌ Query子命令

---

## 🔧 執行計劃

### 階段1：統一能力描述格式 (1-2天)

**任務**：
1. 定義統一的FlowMetadata結構
2. 更新所有工具的分類邏輯
3. 統一description生成規則

**具體操作**：

#### 1.1 創建統一規範文檔
- [ ] 創建 `CAPABILITY_DESCRIPTION_SPEC.md`
- [ ] 定義能力描述模板和範例
- [ ] 建立分類標準(XSS/SQLi/IDOR/SSRF/BizLogic/InfoLeak/Crypto)

#### 1.2 更新Python工具
- [ ] 修改 `aiva_flow_classifier.py` 的classify方法
- [ ] 統一description生成邏輯
- [ ] 測試輸出格式

#### 1.3 更新TypeScript工具
- [ ] 修改 `ts2mermaid.ts` 的Classifier類
- [ ] 對齊Python的分類規則
- [ ] 確保JSON輸出格式一致

#### 1.4 更新Go工具
- [ ] 修改 `go2mermaid.go` 的Classifier結構
- [ ] 對齊分類邏輯
- [ ] 統一輸出格式

#### 1.5 更新Rust工具
- [ ] 修改 `rust_tools/src/main.rs` 的Classifier
- [ ] 實現統一的分類規則
- [ ] 確保輸出格式一致

### 階段2：統一CLI指令格式 (1天)

**任務**：
1. 統一命令行參數命名
2. 標準化子命令結構
3. 統一輸出格式

**具體操作**：

#### 2.1 更新TypeScript CLI
```typescript
// 修改 ts2mermaid.ts 的參數解析
const CLI_SPEC = {
  analyze: {
    '--input': '輸入目錄',
    '--output': '輸出目錄',
    '--format': 'json | mermaid | both',
    '--verbose': '詳細輸出'
  },
  pipeline: {
    '--input': '輸入目錄',
    '--mode': 'full | incremental',
    '--output': '輸出目錄'
  },
  query: {
    '--target': '查詢目標',
    '--format': 'json | text'
  }
};
```

#### 2.2 更新Go CLI
```go
// 修改 go2mermaid.go 的flag定義
var (
    inputDir  = flag.String("input", ".", "輸入目錄")
    outputDir = flag.String("output", "./analysis", "輸出目錄")
    format    = flag.String("format", "json", "輸出格式: json|mermaid|both")
    verbose   = flag.Bool("verbose", false, "詳細輸出")
)
```

#### 2.3 更新Rust CLI
```rust
// 修改 rust_tools/src/main.rs 的clap定義
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Analyze {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "json")]
        format: String,
        #[arg(long)]
        verbose: bool,
    },
    Pipeline { /* ... */ },
    Query { /* ... */ },
}
```

### 階段3：添加CLI執行器 (3-5天)

**任務**：
為TS/Go/Rust工具添加執行能力

#### 3.1 TypeScript執行器

創建 `ts_cli_runner.ts`:
```typescript
// 參考Python的FlowExecutor實現
class FlowExecutor {
  async executeFlow(flowName: string, params: any): Promise<any> {
    // 1. 載入flow定義
    // 2. 解析執行步驟
    // 3. 順序執行各步驟
    // 4. 返回結果
  }
  
  async executeStep(step: FlowStep): Promise<any> {
    // 執行單一步驟
  }
}
```

**文件位置**：
- 主文件: `services/core/aiva_core/internal_exploration/typescript_tools/ts_cli_runner.ts`
- 測試: `services/core/aiva_core/internal_exploration/typescript_tools/tests/test_runner.ts`

#### 3.2 Go執行器

創建 `runner/main.go`:
```go
// 參考Python的FlowExecutor實現
type FlowExecutor struct {
    flowData map[string]FlowDefinition
}

func (fe *FlowExecutor) ExecuteFlow(flowName string, params map[string]interface{}) (interface{}, error) {
    // 執行流程邏輯
}
```

**文件位置**：
- 主文件: `services/core/aiva_core/internal_exploration/go_tools/runner/main.go`
- 測試: `services/core/aiva_core/internal_exploration/go_tools/runner/executor_test.go`

#### 3.3 Rust執行器

創建 `bin/runner.rs`:
```rust
// 參考Python的FlowExecutor實現
struct FlowExecutor {
    flow_data: HashMap<String, FlowDefinition>,
}

impl FlowExecutor {
    fn execute_flow(&self, flow_name: &str, params: &HashMap<String, Value>) -> Result<Value, Error> {
        // 執行流程邏輯
    }
}
```

**文件位置**：
- 主文件: `services/core/aiva_core/internal_exploration/rust_tools/src/bin/runner.rs`
- 測試: `services/core/aiva_core/internal_exploration/rust_tools/tests/executor_test.rs`

### 階段4：各工具獨立測試 (2-3天)

**任務**：
確保每個工具都能獨立運作

#### 4.1 測試檢查清單

對每個語言工具執行：

- [ ] **基礎功能測試**
  - [ ] 能獨立分析代碼
  - [ ] 生成正確的Mermaid圖
  - [ ] 輸出符合統一格式的JSON
  
- [ ] **分類功能測試**
  - [ ] 正確識別功能類別
  - [ ] 描述符合統一格式
  - [ ] 分類準確率 > 90%
  
- [ ] **執行功能測試**
  - [ ] 能執行分析出的flow
  - [ ] 參數傳遞正確
  - [ ] 錯誤處理完善
  
- [ ] **CLI測試**
  - [ ] 所有子命令正常運作
  - [ ] 參數命名符合規範
  - [ ] 輸出格式一致

#### 4.2 測試腳本

創建統一測試腳本：
```bash
# test_all_tools.sh
./test_python_tools.sh
./test_typescript_tools.sh
./test_go_tools.sh
./test_rust_tools.sh
```

### 階段5：跨語言整合 (最後階段)

**任務**：
確認各工具獨立OK後，整合到AIVA主系統

#### 5.1 整合到主Pipeline

修改 `aiva_exploration_pipeline.py`:
```python
def detect_language_and_analyze(self, target_dir: str) -> dict:
    """根據代碼語言選擇對應工具"""
    language = self.detect_primary_language(target_dir)
    
    if language == "python":
        return self.run_python_analyzer(target_dir)
    elif language == "typescript":
        return self.run_typescript_analyzer(target_dir)
    elif language == "go":
        return self.run_go_analyzer(target_dir)
    elif language == "rust":
        return self.run_rust_analyzer(target_dir)
    else:
        return self.run_mixed_language_analysis(target_dir)
```

#### 5.2 統一輸出匯總

所有工具輸出統一匯總到：
```
services/integration/data/internal_exploration/analysis_history/v{n}/
├── python_analysis.json
├── typescript_analysis.json
├── go_analysis.json
├── rust_analysis.json
└── unified_analysis.json  # 合併所有結果
```

---

## 📈 成功指標

### 統一性指標
- [✅] 所有工具輸出JSON結構100%一致
- [✅] CLI參數命名100%統一
- [✅] 能力描述格式符合AI理解標準

### 獨立性指標
- [✅] 每個工具可單獨運行
- [✅] 不依賴其他語言環境
- [✅] 獨立測試通過率 > 95%

### 完整性指標
- [✅] 每個工具都有分析+分類+執行能力
- [✅] 所有功能模組都有對應工具支持
- [✅] 文檔完整且最新

---

## 📝 相關文檔

- [AI_EXTERNAL_MODULE_INTEGRATION_ANALYSIS.md](./AI_EXTERNAL_MODULE_INTEGRATION_ANALYSIS.md) - 外部模組整合分析
- [CAPABILITY_DESCRIPTION_SPEC.md](./CAPABILITY_DESCRIPTION_SPEC.md) - 能力描述規範 (待創建)
- [CLI_STANDARD_SPEC.md](./CLI_STANDARD_SPEC.md) - CLI統一規範 (待創建)

---

## 🔄 更新記錄

| 日期 | 版本 | 更新內容 | 作者 |
|------|------|---------|------|
| 2026-01-13 | v1.0 | 初始版本，定義整合計劃 | GitHub Copilot |

