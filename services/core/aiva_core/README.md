# 🤖 AIVA Core - 程式化核心服務

> **版本**: v3.0-dev | **狀態**: ⚠️ 架構完整但關鍵功能需優化 | **更新**: 2025-12-02

**導航**: [← 返回 Services](../../services/README.md) | [關鍵缺陷報告](../../../AI核心關鍵缺陷報告.md)

---

## ⚠️ 重要聲明 (2025-12-02)

### 🚨 核心系統需優化點

經深度代碼審查,發現以下**優先改進問題**:

#### 1. 程式決策核心需強化 (HIGH PRIORITY)

**現狀**: `BioNeuronDecisionController` 只有 NLU (指令解析),決策邏輯需強化

```python
# ⚠️ 現狀: 只做指令解析
async def _parse_ui_command(self, text: str):
    # 簡單的關鍵字匹配,需強化決策邏輯
    if "掲描" in text:
        return "start_scan", {}

# ⚠️ 需強化: 無法生成 SystemCommand
# ⚠️ 需強化: 無法查詢 RAG 能力
# ⚠️ 需強化: 無法決定引擎組合策略
```

**影響**: 
- 無法實現 13 步驟程式化流程
- 無法指揮其他模組
- 內閭環數據未充分利用

#### 2. 內閉環數據未使用 (CRITICAL)

**現狀**: `InternalLoopConnector.query_capabilities()` 存在但從未被調用

```python
# ✅ 方法存在
async def query_capabilities(self, query: str) -> RAGQueryResult:
    # 可以查詢 RAG 中的能力數據
    pass

# ❌ 問題: 沒有任何地方調用這個方法
# grep -r "query_capabilities" 只找到定義,沒有使用
```

**影響**:
- AI 不知道自己有什麼能力
- 無法動態適應新模組
- 雙閉環斷裂

invoker = get_global_invoker()
invoker.register_feature(ModuleName.XSS_SCANNER, xss_feature)
response = await invoker.invoke(FeatureRequest(...))

commander = AICommanderV2()
command = await commander.process_command("掃描 example.com")
```

---

### 7. Persistence - 持久化
**路徑**: `persistence/`

數據持久化層，提供任務管理和存儲接口。

**核心組件**：
- **task_manager.py** - 任務生命週期管理
- **storage.py** - 統一存儲接口

---

### 8. Reporting - 報告生成
**路徑**: `reporting/`

報告生成系統，支援多種格式輸出。

**核心組件**：
- **report_generator.py** - 報告生成器（Markdown, HTML, PDF）

---

### 9. System - 系統管理
**路徑**: `system/`

系統級管理功能，資源監控和健康檢查。

**核心組件**：
- **resource_watchdog.py** - 資源監控和自動調整
- **health_checker.py** - 健康檢查

---

## 🚀 快速開始

### 基本使用示例

```python
from aiva_core.cognitive_core import RealNeuralCore, RAGEngine
from aiva_core.task_planning import EnhancedPlanner, TaskExecutor
from aiva_core.integration import get_global_invoker

# 1. 初始化認知核心
neural_core = RealNeuralCore(use_5m_model=True)
neural_core.load_weights()
rag = RAGEngine(vector_store_type="postgresql")

# 2. 創建任務計劃
planner = EnhancedPlanner(neural_core)
plan = await planner.create_plan(
    goal="Web安全評估",
    target="https://example.com"
)

# 3. 執行任務
executor = TaskExecutor(get_global_invoker())
results = await executor.start_execution(plan)

# 4. 生成報告
from aiva_core.reporting import ReportGenerator
generator = ReportGenerator()
report = generator.generate_markdown_report(results)
```

---

## 📂 目錄結構

```
aiva_core/
├── cognitive_core/           # 認知核心
│   ├── neural/              # 神經網路（6個文件，2000+行）
│   ├── decision/            # 決策系統（2個文件，700+行）
│   ├── rag/                 # RAG系統（4個文件，1450+行）
│   └── anti_hallucination/  # 反幻覺（1個文件，350+行）
├── task_planning/           # 任務規劃
│   ├── planner/             # 規劃器（2個文件，800+行）
│   └── executor/            # 執行器（3個文件，1250+行）
├── core_capabilities/       # 核心能力
│   ├── analysis/            # 分析能力
│   ├── attack/              # 攻擊能力
│   ├── dialog/              # 對話管理
│   ├── ingestion/           # 數據攝取
│   ├── processing/          # 數據處理
│   ├── output/              # 輸出格式化
│   └── plugins/             # 插件系統
├── service_backbone/        # 服務骨幹
│   ├── api/                 # API層
│   ├── adapters/            # 適配器
│   ├── messaging/           # 消息系統
│   ├── state/               # 狀態管理
│   ├── storage/             # 存儲層
│   └── ...                  # 其他基礎設施
├── external_learning/       # 對外學習
│   ├── ai_model/            # AI模型整合
│   ├── analysis/            # 外部分析
│   ├── learning/            # 持續學習
│   └── ...
├── integration/             # 整合層
│   ├── features_invoker.py  # Features調用
│   ├── feedback_processor.py # 反饋處理
│   └── ai_commander_v2.py   # AI指揮官
├── persistence/             # 持久化
│   ├── task_manager.py      # 任務管理
│   └── storage.py           # 存儲接口
├── reporting/               # 報告生成
│   └── report_generator.py
├── system/                  # 系統管理
│   └── resource_watchdog.py
├── plugin_system/           # 插件系統（已廢棄）
├── plugins/                 # 插件目錄（已廢棄）
├── internal_exploration/    # 內部探索（整合中）
└── ui_panel/                # UI面板（整合中）
```

**統計**：
- **總模組數**: 9 個主要模組
- **總文件數**: 96 個 Python 文件（不含 __init__.py）
- **總代碼量**: ~25,000+ 行

---

## 🔗 相關服務

- [AIVA Common](../aiva_common/README.md) - 公共數據結構和工具
- [Features](../features/README.md) - 功能模組實現
- [Scan](../scan/README.md) - 掃描引擎和協調器
- [Integration](../integration/README.md) - 外部系統整合

---

**最後更新**: 2025-12-01 | **維護者**: AIVA Team
