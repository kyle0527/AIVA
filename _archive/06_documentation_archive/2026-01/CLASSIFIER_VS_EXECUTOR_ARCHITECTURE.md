# Classifier 阶段 vs Executor 阶段架构说明

## 🔄 双阶段架构流程

```
┌─────────────────────────────────────────────────────────────┐
│                    阶段 1: Classifier                        │
│                   （生成时分类 - 静态分析）                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ classification_data  │
                 │      .json           │
                 └──────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    阶段 2: Executor                          │
│                  （执行时使用 - 动态执行）                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 阶段 1: Classifier（生成时分类）

### 职责
**静态分析源代码，生成元数据**

### 文件
- `aiva_internal_classifier.py` - 分析内部模块（Python）
- `aiva_external_classifier.py` - 分析外部模块（Python/Rust/Go/TypeScript）

### 工作内容

#### 1. **扫描源代码**
```python
# aiva_internal_classifier.py
- 扫描 services/core/aiva_core/ 下的所有 Python 文件
- 使用 AST 分析函数调用关系
- 识别数据流（哪个函数调用哪个函数）
```

#### 2. **构建 Flow 定义**
```json
{
  "id": 20,
  "path": ["aiva_exploration_pipeline", "aiva_flow_analyzer"],
  "full_path": ["C:\\...\\aiva_exploration_pipeline.py", "C:\\...\\aiva_flow_analyzer.py"],
  "length": 2,
  "modules": ["internal_exploration", "internal_exploration"],
  "component_types": ["程式組件", "程式組件"]
}
```

#### 3. **分类 Flow 类型**（应该在这里添加）
```python
# 在 classify_flows() 中添加 Loop 类型判断
flow['loop_type'] = self._classify_loop_type(flow)  # 新增
flow['ai_notification'] = {
    'required': flow['loop_type'] in ['internal_loop', 'external_loop'],
    'connector': 'InternalLoopConnector' if flow['loop_type'] == 'internal_loop' else 'ExternalLoopConnector'
}
```

#### 4. **输出 classification_data.json**
```json
{
  "metadata": {
    "generated_at": "2026-01-20T10:00:00",
    "total_flows": 286
  },
  "flows": [
    {
      "id": 1,
      "path": [...],
      "loop_type": "internal_loop",  // ← 应该在这里添加
      "ai_notification": {...}
    }
  ]
}
```

### 执行时机
- **开发时/部署前** 运行一次
- 代码变更后重新生成
- 通常不在运行时执行

### 命令示例
```bash
# 生成内部模块分类
python -m aiva_core.internal_exploration.aiva_internal_classifier

# 生成外部模块分类
python -m aiva_core.internal_exploration.aiva_external_classifier
```

---

## ⚡ 阶段 2: Executor（执行时使用）

### 职责
**读取分类结果，动态执行 Flow**

### 文件
- `aiva_internal_executor.py` - 执行内部模块 Flow
- `aiva_external_executor.py` - 执行外部模块 Flow

### 工作内容

#### 1. **加载分类数据**
```python
# aiva_internal_executor.py
class FlowExecutor:
    def __init__(self):
        # 读取 Classifier 生成的数据
        self.data = self._load_data()  # 加载 classification_data.json
```

#### 2. **动态执行 Flow**
```python
def execute_flow(self, flow_id):
    flow = self.get_flow_by_id(flow_id)
    
    # 遍历 path，动态导入并执行
    for step in flow['path']:
        module = importlib.import_module(module_path)
        instance = module.SomeClass()
        result = instance.execute()
```

#### 3. **使用分类信息**（应该在这里使用）
```python
def execute_flow(self, flow_id):
    # ... 执行 Flow ...
    
    # 执行完成后，读取 Classifier 生成的分类
    loop_type = flow.get('loop_type', 'uncertain')
    
    # 根据分类通知对应的 Connector
    if loop_type == 'internal_loop':
        self.internal_connector.record_exploration(result)
    elif loop_type == 'external_loop':
        self.external_connector.record_feedback(result)
```

#### 4. **运行时通知 AI**
```python
# 基于 Classifier 的分类结果决定通知哪个 Connector
notification_config = flow.get('ai_notification', {})
if notification_config.get('required'):
    connector = notification_config.get('connector')
    self._notify_connector(connector, result)
```

### 执行时机
- **运行时** 每次用户调用时执行
- 频繁执行（每个 CLI 命令）
- 性能要求高

### 命令示例
```bash
# 执行内部 Flow
python -m aiva_core.internal_exploration.aiva_internal_executor --flow 20

# 执行外部 Flow
python -m aiva_core.internal_exploration.aiva_external_executor --lang python --flow 1
```

---

## 🔑 关键差异对比

| 维度 | Classifier（分类器） | Executor（执行器） |
|------|---------------------|-------------------|
| **时机** | 开发时/部署前 | 运行时 |
| **频率** | 偶尔执行（代码变更后） | 频繁执行（每次调用） |
| **输入** | 源代码文件 | classification_data.json |
| **输出** | classification_data.json | 执行结果 + AI 通知 |
| **操作** | 静态分析（AST） | 动态执行（importlib） |
| **性能** | 可以慢（复杂分析） | 必须快（用户等待） |
| **职责** | 分析+分类 | 执行+通知 |

---

## 💡 正确的实施方式

### ❌ 错误（我之前的做法）
```python
# 在 Executor 中动态分类（运行时计算）
class FlowExecutor:
    def execute_flow(self, flow_id):
        # ... 执行 ...
        
        # ❌ 每次执行都重新分类（浪费性能）
        category = self.classify_flow(flow)  
        self.notify_connectors(flow, result, category)
```

**问题**：
- 每次执行都要重新计算分类
- 分类逻辑重复
- 性能开销大

### ✅ 正确（应该这样做）

#### 1. **在 Classifier 中添加分类逻辑**
```python
# aiva_internal_classifier.py
class AIVAFlowClassifier:
    
    LOOP_CLASSIFICATION_RULES = {
        "internal_loop": {
            "keywords": ["rag", "knowledge", "vector", "capability", "registry"],
        },
        "external_loop": {
            "keywords": ["scan", "attack", "learning", "feedback"],
        },
        # ...
    }
    
    def _classify_loop_type(self, flow):
        """分类 Flow 的 Loop 类型"""
        combined_text = ' '.join(flow['path'] + flow['modules']).lower()
        
        for loop_type, config in self.LOOP_CLASSIFICATION_RULES.items():
            if any(kw in combined_text for kw in config['keywords']):
                return loop_type
        
        return 'uncertain'
    
    def classify_flows(self):
        for flow in self.flows:
            # 现有分类...
            flow['modules'] = [...]
            flow['component_types'] = [...]
            
            # ✅ 新增：Loop 类型分类
            flow['loop_type'] = self._classify_loop_type(flow)
            flow['ai_notification'] = self._get_notification_config(flow['loop_type'])
```

#### 2. **在 Executor 中直接使用**
```python
# aiva_internal_executor.py
class FlowExecutor:
    def execute_flow(self, flow_id):
        flow = self.get_flow_by_id(flow_id)
        
        # 执行 Flow
        result = self._do_execute(flow)
        
        # ✅ 直接读取 Classifier 生成的分类
        loop_type = flow.get('loop_type', 'uncertain')
        notification_config = flow.get('ai_notification', {})
        
        # 根据预先分类的结果通知
        if notification_config.get('required'):
            self._notify_ai_system(notification_config, result)
```

---

## 📐 数据流向

```
源代码
  │
  ▼
[Classifier] ──────────> classification_data.json
  │                             │
  │ 分析                         │ 包含:
  │ - 函数调用关系                │ - flow 定义
  │ - 模块分类                    │ - loop_type
  │ - Loop 类型 ✅               │ - ai_notification
  │                             │
  │                             ▼
  │                      [Executor]
  │                             │
  │                             ▼ 读取分类
  │                       动态执行 Flow
  │                             │
  │                             ▼
  │                      根据 loop_type
  │                      通知 Connector
  │                             │
  │                             ▼
  │                    AI 学习/知识更新
```

---

## 🎯 实施建议

### 优先级 1: 修改 Classifier
1. 在 `aiva_internal_classifier.py` 添加 `_classify_loop_type()` 方法
2. 在 `classify_flows()` 中调用，将结果保存到 flow 对象
3. 重新运行 Classifier 生成新的 classification_data.json

### 优先级 2: 修改 Executor
1. 在 `aiva_internal_executor.py` 读取 `loop_type` 字段
2. 在 `execute_flow()` 完成后根据 `loop_type` 通知对应 Connector
3. 添加容错机制（如果没有 loop_type 则使用默认值）

### 优先级 3: 同步外部模块
1. 在 `aiva_external_classifier.py` 添加类似的分类逻辑
2. 在 `aiva_external_executor.py` 添加通知机制

---

## ✅ 总结

**正确的方式**：
- **Classifier**: 负责分析和分类（生成 loop_type）
- **Executor**: 负责执行和通知（使用 loop_type）

**优点**：
- ✅ 分离关注点
- ✅ 性能优化（分类只做一次）
- ✅ 易于维护（修改分类规则只需重新运行 Classifier）
- ✅ 可追溯（classification_data.json 记录了所有分类决策）

**我之前的错误**：
- ❌ 在 Executor 中动态分类
- ❌ 修改了错误的文件（aiva_cli_implementation.py 已被取代）

**下一步正确做法**：
1. 修改 `aiva_internal_classifier.py` 添加 Loop 分类
2. 重新生成 `classification_data.json`
3. 修改 `aiva_internal_executor.py` 使用分类结果
