# AI 学习数据流规范

## 📊 数据流架构（三层 + 双路分离）

**架构设计（可插拔式 AI + 安全防护）**：
```
外部 HTTP 请求
   ↓
main.py (整个程序对外入口) ← 第一道安全防线
   ↓ 安全检测（防木马、恶意请求、注入攻击等）
   ↓ 通过检测后转发
   ↓
app.py (程序与 AI 的沟通接口) ← 可插拔设计，实现双路分离
   ↓
   ├─────────────┬─────────────┐
   ↓             ↓
第一路：      第二路：
整合模块      任务规划 AI
(实时存储)    (分析决策)
   ↓             ↓
数据库        EnhancedDecisionAgent
持久化        (认知核心)
                ↓
           【AI 决策并下令】
                ↓
        ┌───────┴───────┐
        ↓               ↓
   下令给扫描模块   下令给功能模块
   (Phase0/1)      (XSS/SQLi/SSRF)
        ↓               ↓
   独立执行模块     独立执行模块
```

**学习系统（异步独立）**：
```
任务结束后
   ↓
从三个数据源读取：
├─ 1. 整合模块（实际执行记录，JSONL）
│     services/integration/data/experiences/*.jsonl
│     - 本次任务的完整记录
│     - 按能力分类：xss, sqli, ssrf, phase0 等
│     - 包含时间戳、请求、响应、结果
│
├─ 2. 历史数据（之前的执行记录，JSONL）
│     同样从整合模块读取，但是历史时间段
│     - 作为"预期响应"的参考
│     - 相同场景的不同结果
│
└─ 3. 能力知识库（各能力的分析报告，Markdown）
      services/core/aiva_core/cognitive_core/learning_system/knowledge/
      - XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
      - SQLI_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
      - SSRF_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
      - 描述攻击手法、绕过技巧、预期响应模式
   ↓
三路数据比对和评估：
├─ 本次 vs 历史：发现差异和变化趋势
├─ 本次 vs 知识库：匹配已知模式
└─ 历史 vs 知识库：验证知识库准确性
   ↓
判断：是否为已知情况？
├─ ✅ 已知情况 → 直接生成优化方案
│
└─ ❌ 未知情况（不在任何数据中）
    ↓
    触发 RAG 搜索
    ├─ 搜索范围：技术文档、CVE、安全研究、外部资源
    └─ 返回：相似案例、解决方案、技术资料
    ↓
学习优化目标：
├─ 参数优化（timeout, depth, threads, 检测阈值）
└─ 方法优化（绕过技巧, Payload 变形, 攻击手法）
   ↓
生成新权重/策略
   ↓
验证新权重效果（测试环境/模拟场景）
   ↓
   ├─ 效果好 → ✅ 保存新权重，更新模型和知识库
   └─ 效果差 → ❌ 丢弃新权重，保留原有权重
```

**统一架构原则**（适用于所有13个步骤）：
- ✅ **三层入口**：外部 → main.py（安全检测）→ app.py（双路分离）→ AI 内部
- ✅ **main.py 职责**：第一道安全防线
  - 检测恶意请求（木马、注入攻击、异常参数）
  - 速率限制和访问控制
  - 通过检测后才转发给 app.py
- ✅ **app.py 职责**：实现双路分离（整合模块存储 + 任务规划 AI）
- ✅ **分层防护意义**：main.py 挡住攻击，保护 AI 系统不受污染
- ✅ **AI 只负责规划和下令**：不直接接触外部，不负责实际执行
- ✅ **执行由独立模块完成**：扫描模块和功能模块独立执行
- ✅ **学习系统异步运行**：任务结束后从三个数据源读取并学习
  - 整合模块：实际执行记录（本次+历史）
  - 能力知识库：各能力的分析报告和预期模式
- ✅ **安全设计**：学习系统不在任务执行期间介入（避免修改运行中的代码）

---

## 1️⃣ 发送端：外部模块发布的数据

### 数据来源（已完成 CLI 分析）
- **总模块**: 8个
- **总 Flow**: 210个
- **语言**: Python (203), Go (4), TypeScript (3)
- **类型**: XSS, SQLi, SSRF, IDOR, Authentication, Business Logic

### 发送格式（MQ 消息）

```python
# services/features/function_xss/result_publisher.py
# services/features/function_sqli/result_binder_publisher.py
# 等等...

await broker.publish(
    topic=Topic.LOG_RESULTS_ALL,  # "log.results.all"
    message=AivaMessage(
        header=MessageHeader(
            message_id="msg_xxx",
            trace_id="trace_xxx",
            correlation_id="scan_xxx",
            source_module=ModuleName.FUNC_XSS  # 或 FUNC_SQLI, FUNC_SSRF...
        ),
        topic=Topic.LOG_RESULTS_ALL,
        payload=FindingPayload(...)  # 或 TaskUpdatePayload
    )
)
```

### Payload 结构

#### A. FindingPayload（漏洞发现）

```json
{
  "finding_id": "finding_20260120_001",
  "scan_id": "scan_abc123",
  "task_id": "task_xyz789",
  "vulnerability": {
    "name": "xss",  // VulnerabilityType enum
    "severity": "high",  // RiskLevel enum: CRITICAL, HIGH, MEDIUM, LOW
    "confidence": "high",  // ConfidenceLevel enum
    "description": "发现存储型 XSS 漏洞"
  },
  "location": {
    "url": "https://target.com/profile",
    "parameter": "bio",
    "method": "POST"
  },
  "evidence": {
    "request": {
      "url": "https://target.com/profile",
      "method": "POST",
      "headers": {...},
      "body": "{\"bio\": \"<script>alert(1)</script>\"}"
    },
    "response": {
      "status_code": 200,
      "body": "Profile updated: <script>alert(1)</script>"
    },
    "payload": "<script>alert(1)</script>",
    "matched_pattern": "<script.*>.*</script>"
  },
  "remediation": "对用户输入进行 HTML 转义",
  "references": [
    "https://owasp.org/www-community/attacks/xss/"
  ],
  "metadata": {
    "module": "function_xss",
    "language": "Python",
    "flow_id": 15,
    "execution_time": 2.5
  },
  "discovered_at": "2026-01-20T10:30:00Z"
}
```

#### B. TaskUpdatePayload（任务状态）

```json
{
  "task_id": "task_xyz789",
  "scan_id": "scan_abc123",
  "status": "COMPLETED",  // or IN_PROGRESS, FAILED
  "worker_id": "xss-worker-12345",
  "details": {
    "findings_count": 3,
    "execution_time": 5.2,
    "urls_tested": 25
  },
  "timestamp": "2026-01-20T10:30:00Z"
}
```

---

## 2️⃣ 接收端：AI 需要的学习数据

### AI 学习目标
根据训练数据集（`distillation_train.json`），AI 需要学习：
1. **漏洞识别**：从 HTTP 请求/响应中识别漏洞类型
2. **严重性评估**：0.0-1.0 的连续值
3. **置信度判断**：0.0-1.0 的连续值
4. **推理能力**：解释为什么判定为某类漏洞

### AI 学习数据格式

```python
# 转换后的格式（用于 AI 训练）
{
    "scenario_text": "在 HTTP 响应中发现 反射型 XSS 相关的 <script> 模式",
    "raw_context": "GET /profile?bio=<script>alert(1)</script> HTTP/1.1\nHost: target.com",
    "vulnerability_type": "xss",  # 漏洞类型
    "severity": 0.8,  # 严重性（0.0-1.0）
    "confidence": 0.9,  # 置信度（0.0-1.0）
    "reasoning": "基于 反射型 XSS 特征分析，发现 <script> 标签未转义，判定为 xss，严重性评估为 0.80",
    "source_module": "function_xss",
    "language": "Python",
    "flow_id": 15,
    "scenario_id": "xss_high_20260120_001",
    "difficulty_level": "medium"  # easy, medium, hard
}
```

---

## 3️⃣ 数据转换逻辑

### ExternalLearningListener 的职责

```python
# services/core/aiva_core/cognitive_core/learning_system/event_listener.py

class ExternalLearningListener:
    async def _on_task_completed(self, message: dict):
        """接收 MQ 消息并转换为 AI 学习数据"""
        
        # 1. 解析 MQ 消息
        payload = message.get("payload", {})
        finding = FindingPayload(**payload)
        
        # 2. 提取关键信息
        scenario_text = self._build_scenario_text(finding)
        raw_context = self._build_raw_context(finding.evidence)
        
        # 3. 转换为 AI 学习格式
        learning_sample = {
            "scenario_text": scenario_text,
            "raw_context": raw_context,
            "vulnerability_type": finding.vulnerability.name.value,
            "severity": self._map_severity(finding.vulnerability.severity),
            "confidence": self._map_confidence(finding.vulnerability.confidence),
            "reasoning": self._generate_reasoning(finding),
            "source_module": message.get("header", {}).get("source_module"),
            "metadata": finding.metadata
        }
        
        # 4. 传递给学习系统
        await self.connector.process_execution_result(learning_sample)
```

### 数据转换映射表

#### 严重性映射
```python
SEVERITY_MAPPING = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.5,
    "LOW": 0.3,
    "INFO": 0.1
}
```

#### 置信度映射
```python
CONFIDENCE_MAPPING = {
    "CONFIRMED": 1.0,
    "high": 0.85,
    "medium": 0.6,
    "low": 0.4
}
```

---

## 4️⃣ 学习系统集成

### ExternalLoopConnector 的职责

```python
# services/core/aiva_core/cognitive_core/external_loop_connector.py

class ExternalLoopConnector:
    async def process_execution_result(self, learning_sample: dict):
        """处理学习样本"""
        
        # 1. 偏差分析（如果有预测值）
        if learning_sample.get("predicted_type"):
            deviation = self._analyze_deviation(learning_sample)
        
        # 2. 构建训练样本
        training_sample = TrainingDataSample(
            scenario_text=learning_sample["scenario_text"],
            raw_context=learning_sample["raw_context"],
            teacher_vulnerability_type=learning_sample["vulnerability_type"],
            teacher_severity=learning_sample["severity"],
            teacher_confidence=learning_sample["confidence"],
            teacher_reasoning=learning_sample["reasoning"],
            source_doc="实战数据",
            scenario_id=learning_sample.get("scenario_id"),
            difficulty_level=self._assess_difficulty(learning_sample)
        )
        
        # 3. 添加到训练队列
        await self.trainer.add_training_sample(training_sample)
        
        # 4. 触发增量训练（如果样本数达到阈值）
        if self.trainer.sample_count >= TRAINING_THRESHOLD:
            await self.trainer.incremental_training()
```

---

## 5️⃣ 词汇表和训练数据

### 已有的安全词汇表
- **路径**: `training/data/security_vocabulary/security_vocabulary.json`
- **总术语**: 63个
- **Top 术语**: JWT (76), GraphQL (68), RCE (59), XSS (56), WebSocket (45)

### 已有的训练数据集
- **路径**: `training/data/distillation_dataset/distillation_train.json`
- **样本数**: 589个训练样本 + 148个验证样本
- **漏洞类型**: XSS, SQLi, SSRF, IDOR, RCE, Authentication, Business Logic

### 实战数据增强
```python
# 实战数据会自动添加到训练集
training_data = [
    # 原有的蒸馏数据（来自 Teacher Model）
    *load_distillation_dataset(),
    
    # 新增的实战数据（来自真实执行）
    *load_production_samples()
]
```

---

## 6️⃣ 数据质量保证

### 数据验证
```python
def validate_learning_sample(sample: dict) -> bool:
    """验证学习样本的完整性和有效性"""
    
    required_fields = [
        "scenario_text",
        "raw_context",
        "vulnerability_type",
        "severity",
        "confidence"
    ]
    
    # 1. 检查必填字段
    if not all(field in sample for field in required_fields):
        return False
    
    # 2. 验证数值范围
    if not (0.0 <= sample["severity"] <= 1.0):
        return False
    if not (0.0 <= sample["confidence"] <= 1.0):
        return False
    
    # 3. 验证漏洞类型
    valid_types = {"xss", "sqli", "ssrf", "idor", "rce", "authentication", "business_logic"}
    if sample["vulnerability_type"] not in valid_types:
        return False
    
    return True
```

### 数据去重
```python
def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """去除重复的学习样本"""
    
    seen = set()
    unique_samples = []
    
    for sample in samples:
        # 基于关键特征生成指纹
        fingerprint = (
            sample["vulnerability_type"],
            sample["scenario_text"][:50],  # 前50字符
            sample["severity"],
            sample["confidence"]
        )
        
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_samples.append(sample)
    
    return unique_samples
```

---

## 7️⃣ 实施步骤

### Step 1: 确认外部模块数据格式 ✅
- [x] XSS 模块发送 FindingPayload
- [x] SQLi 模块发送 FindingPayload
- [x] SSRF 模块发送 FindingPayload
- [x] IDOR 模块发送 FindingPayload
- [x] 所有模块统一发送到 `LOG_RESULTS_ALL`

### Step 2: 实现数据转换器
```python
# services/core/aiva_core/cognitive_core/learning_system/data_transformer.py

class LearningDataTransformer:
    """将 MQ 消息转换为 AI 学习格式"""
    
    def transform_finding_to_learning_sample(
        self, 
        finding: FindingPayload
    ) -> dict:
        """转换漏洞发现为学习样本"""
        pass
    
    def build_scenario_text(self, finding: FindingPayload) -> str:
        """构建场景描述文本"""
        pass
    
    def build_raw_context(self, evidence: dict) -> str:
        """构建原始上下文（HTTP 请求/响应）"""
        pass
    
    def generate_reasoning(self, finding: FindingPayload) -> str:
        """生成推理文本"""
        pass
```

### Step 3: 修改 ExternalLearningListener
```python
# 添加数据转换逻辑
from .data_transformer import LearningDataTransformer

class ExternalLearningListener:
    def __init__(self):
        self.transformer = LearningDataTransformer()
    
    async def _on_task_completed(self, message: dict):
        # 转换数据
        learning_sample = self.transformer.transform_finding_to_learning_sample(
            FindingPayload(**message["payload"])
        )
        
        # 验证数据
        if validate_learning_sample(learning_sample):
            await self.connector.process_execution_result(learning_sample)
```

### Step 4: 实现增量学习
```python
# services/core/aiva_core/cognitive_core/learning_system/learning/incremental_trainer.py

class IncrementalTrainer:
    """增量训练器"""
    
    async def add_training_sample(self, sample: dict):
        """添加训练样本到缓冲区"""
        self.sample_buffer.append(sample)
        
        # 达到阈值时触发训练
        if len(self.sample_buffer) >= self.training_threshold:
            await self.train()
    
    async def train(self):
        """执行增量训练"""
        # 1. 加载现有模型
        # 2. 使用新样本微调
        # 3. 保存更新后的模型
        pass
```

---

## 8️⃣ 监控和评估

### 学习统计
```python
learning_stats = {
    "total_samples_collected": 1523,
    "samples_by_type": {
        "xss": 450,
        "sqli": 320,
        "ssrf": 280,
        "idor": 200,
        "rce": 150,
        "others": 123
    },
    "avg_confidence": 0.82,
    "last_training": "2026-01-20T12:00:00Z",
    "model_version": "v1.2.5"
}
```

### 学习效果评估
```python
evaluation_metrics = {
    "accuracy": 0.89,  # 准确率
    "precision": 0.87,  # 精确率
    "recall": 0.85,  # 召回率
    "f1_score": 0.86,  # F1 分数
    "improvement_rate": 0.05  # 相比上个版本的改进率
}
```

---

## 📝 总结

### 核心要点
1. **统一入口模式**（适用于所有13个步骤）：
   - 所有对外请求 → app.py (唯一入口)
   - 接收方式完全相同：双路处理（储存 + AI 决策）
   
2. **双路处理架构**（每次都一样）：
   - **第一路**：整合模块储存（数据库持久化）
   - **第二路**：AI 分析处理（EnhancedDecisionAgent）→ AI 决策并下令
   
3. **差别在下令目标**（根据 AI 决策）：
   - 下令给扫描模块（Phase0/Phase1）
   - 下令给功能模块（XSS/SQLi/SSRF/IDOR 等）
   - 下令给其他能力模块
   
4. **训练数据独立管理**：所有训练相关文件在 training/ 目录
   - `training/data/distillation_train.json` - 训练数据
   - `training/scripts/train_student_model.py` - 训练脚本
   - `training/scripts/generate_distillation_dataset.py` - 数据生成

### 数据流总结（统一模式）
```
外部请求 (任何步骤)
  ↓
app.py (唯一入口) ← 统一接收所有请求
  ↓
  ├──────────────────┬──────────────────┐
  ↓                  ↓
第一路：          第二路：
整合模块储存      AI 分析处理
  ↓                  ↓
数据库持久化    EnhancedDecisionAgent (认知核心)
                     ↓
                【AI 决策分析】
                     ↓
                下令给目标模块
                     ↓
            ┌────────┴────────┐
            ↓                 ↓
       扫描模块           功能模块
    (Phase0/Phase1)    (XSS/SQLi/SSRF/IDOR...)
            ↓                 ↓
       扫描结果 ──────→ MQ ←─── 执行结果
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
    ScanResultProcessor    ExternalLoopConnector
    (结果处理)             (学习更新)


训练阶段（独立）：
training/
  ↓
distillation_train.json
  ↓
train_student_model.py
  ↓
AI Model (.pth)
```
