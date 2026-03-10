# AIVA 双 CLI 架构设计（纯粹版）

> **⚠️ 文檔狀態**: 📚 **參考用** - 核心設計理念已整合至 [SIMPLIFIED_DUAL_CLI_DESIGN.md](docs/05_implementation_guides/SIMPLIFIED_DUAL_CLI_DESIGN.md)  
> **兼容性**: ✅ 與「AI 排序器方案」98% 兼容  
> **建議**: 本文件的 CLI 通訊機制（subprocess + JSON）仍然有效，請參考最新的 AI 排序器方案進行實施

**文档日期**: 2026年1月11日  
**核心理念**: AI 内部模块一套，对外功能模块一套，各自独立

---

## 🎯 核心设计原则

### 双层 CLI 架构

```
┌─────────────────────────────────────────────────────────┐
│              AI 核心系统（内部）                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ cognitive_core + task_planning +                  │  │
│  │ internal_exploration + service_backbone           │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↕                               │
│                    内部 CLI                             │
│        (AI 模块之间的通讯，可能更复杂)                   │
└─────────────────────────────────────────────────────────┘
                         ↓
                    外部 CLI
            (subprocess + JSON，极简)
                         ↓
┌─────────────────────────────────────────────────────────┐
│              外部功能模块                                │
│  ┌────────────────┐  ┌────────────────┐                │
│  │   Features     │  │     Scan       │                │
│  │  (XSS/SQLi等)  │  │  (扫描引擎)     │                │
│  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

**关键点**：
- ✅ **内部 CLI** - AI 模块之间通讯（可以复杂、紧密）
- ✅ **外部 CLI** - AI 调用功能模块（必须简单、标准）
- ❌ 外部调用**无** Dispatcher/Coordinator 中间层
- ✅ 外部调用直接 subprocess + json.loads()
- ✅ 内部通讯方式由 AI 自己决定

---

## 🏠 内部 CLI（AI 模块内部通讯）

### 范围
AI 核心系统的内部模块：
- `cognitive_core` - 认知核心、决策引擎
- `task_planning` - 任务规划
- `internal_exploration` - 内部探索
- `service_backbone` - 服务骨干

### 特点
- 🔗 **紧密耦合** - 模块间可能需要复杂交互
- 🧠 **自我管理** - AI 自己决定通讯方式
- 📚 **知识同步** - RAG、向量库、经验管理
- ⚡ **灵活实现** - 可以用任何方式（CLI、函数调用、消息队列等）

### 通讯方式（AI 自己决定）

**方式 1：直接函数调用**（推荐，最快）

```python
# cognitive_core 调用 internal_exploration
from aiva_core.internal_exploration import InternalLoopConnector

connector = InternalLoopConnector()
capabilities = await connector.discover_capabilities()
```

**方式 2：内部 CLI**（如果需要隔离）

```bash
# 通过 CLI 调用内部模块
python -m aiva_core.internal_exploration.capability_discovery \
  --scan-path services/features \
  --output json
```

**方式 3：消息队列**（如果需要异步）

```python
# 通过 RabbitMQ 发送消息
await dispatcher.broadcast("capability_updated", data)
```

**关键**：内部通讯方式**不限制**，AI 自己决定最合适的方式！

### 内部 CLI 示例

```bash
# 1. 能力发现
python -m aiva_core.internal_exploration.capability_discovery \
  --scan-path services/features/function_xss \
  --output json

# 输出 (简单 JSON，AI 直接处理)
{
  "module": "function_xss",
  "capabilities": [
    {"name": "reflected_xss_scan", "confidence": 0.95},
    {"name": "stored_xss_scan", "confidence": 0.90}
  ],
  "entry_points": ["__main__.py", "scanner.py"]
}

# 2. 代码分析
python -m aiva_core.internal_exploration.code_analyzer \
  --target services/scan/engines \
  --depth 2

# 输出
{
  "analyzed_files": 15,
  "functions_found": 48,
  "patterns": ["async_scan", "multi_engine", "result_merge"]
}

# 3. RAG 同步
python -m aiva_core.internal_exploration.rag_sync \
  --source classification_data.json \
  --vector-db update

# 输出
{
  "status": "success",
  "documents_updated": 276,
  "embeddings_created": 1024
}
```

### AI 处理方式

```python
# AI 执行内循环 CLI
cmd = ["python", "-m", "aiva_core.internal_exploration.capability_discovery",
       "--scan-path", "services/features/function_xss",
       "--output", "json"]

result = await subprocess_run(cmd)
data = json.loads(result.stdout)  # ← 直接处理，无中间层

# AI 自己决定如何使用这些数据
if data["capabilities"]:
    await self.update_knowledge_base(data)
    logger.info(f"发现 {len(data['capabilities'])} 个新能力")
```

---

## 🌐 外部 CLI（调用功能模块）

### 范围
AI 外部的独立功能模块：
- `features/` - 漏洞检测功能（XSS、SQLi、SSRF、IDOR 等）
- `scan/` - 扫描引擎（Python、Rust、Go 多引擎）

### 特点
- 🔌 **松耦合** - 独立进程，通过 CLI 调用
- 📦 **标准接口** - 必须遵循简单的 JSON 输出
- 🚀 **跨语言** - 支持 Python、Rust、Go、TypeScript
- ⚡ **高性能** - subprocess 调用，无额外开销

### 强制要求

**唯一硬性规则**：
1. ✅ 必须能通过命令行调用
2. ✅ 必须输出 JSON 到 stdout
3. ✅ 错误输出到 stderr
4. ✅ 用退出码表示成功/失败

**就这些！** 格式、字段、结构完全自由。

### 外部 CLI 示例

```bash
# 1. XSS 扫描
python -m function_xss --url https://target.com --type reflected

# 输出 (格式自由，只要是 JSON)
{
  "target": "https://target.com",
  "type": "reflected",
  "vulnerable": true,
  "findings": [
    {
      "payload": "<script>alert(1)</script>",
      "param": "search",
      "status": 200,
      "evidence": "...<script>alert(1)</script>..."
    }
  ]
}

# 2. SQL 注入
python -m function_sqli --url https://target.com/api/user?id=1

# 输出 (可以完全不同格式)
{
  "url": "https://target.com/api/user",
  "injection_point": "id",
  "sqli_type": "error-based",
  "database": "MySQL 5.7",
  "extracted": ["admin", "user", "guest"]
}

# 3. 多引擎扫描
cargo run --manifest-path services/scan/rust_scanner/Cargo.toml \
  -- --target https://target.com --fast

# 输出 (Rust CLI 的 JSON)
{
  "engine": "rust_fast_scanner",
  "scan_time_ms": 1234,
  "endpoints_found": 25,
  "vulnerabilities": [...]
}
```

### AI 处理方式

```python
# AI 执行外循环 CLI
cmd = ["python", "-m", "function_xss",
       "--url", target_url,
       "--type", "reflected"]

result = await subprocess_run(cmd)
data = json.loads(result.stdout)  # ← 直接处理

# AI 自己分析结果
if data.get("vulnerable"):
    await self.record_success(data)
    await self.learn_from_success(data["findings"])
else:
    await self.adjust_strategy()
```

---

## 🔄 内部与外部交互

### AI 如何协调两套 CLI

```python
class CapabilityOrchestrator:
    """AI 主控，协调内部模块和外部功能"""
    
    async def execute_task(self, target: str):
        # ========== 内部通讯（方式自由） ==========
        
        # 方式 1：直接调用（推荐）
        from aiva_core.internal_exploration import InternalLoopConnector
        connector = InternalLoopConnector()
        capabilities = await connector.query_capabilities("xss_scan")
        
        # 或方式 2：内部 CLI（如果需要）
        # internal_cmd = ["python", "-m", "aiva_core.internal_exploration.capability_query"]
        # internal_result = await subprocess_run(internal_cmd)
        # capabilities = json.loads(internal_result.stdout)
        
        # AI 决策：选择最佳工具
        best_tool = self._select_best_tool(capabilities)
        
        # ========== 外部调用（必须 subprocess + JSON） ==========
        
        # 调用外部功能模块
        external_cmd = [
            "python", "-m", best_tool["module"],  # features.function_xss
            "--url", target,
            "--type", "reflected"
        ]
        
        # subprocess 执行
        external_result = await subprocess_run(external_cmd)
        
        # 解析 JSON（外部模块的唯一要求）
        attack_result = json.loads(external_result.stdout)
        
        # ========== 内部处理结果 ==========
        
        # 方式 1：直接调用
        await self.experience_manager.record_success(attack_result)
        
        # 或方式 2：内部消息
        # await self.dispatcher.broadcast("task_completed", attack_result)
        
        return attack_result
```

### 关键差异

| 项目 | 内部 CLI | 外部 CLI |
|------|---------|---------|
| **范围** | AI 核心模块 | 功能模块 |
| **调用方式** | 灵活（函数/CLI/MQ） | **必须 subprocess** |
| **输出格式** | 可以任意 | **必须 JSON** |
| **中间层** | 可以有（如需要） | **绝对不能有** |
| **耦合度** | 可以紧密 | **必须松散** |
| **语言** | 主要 Python | Python/Rust/Go/TS |

---

## 📋 CLI 格式要求（最小约束）

### 唯一硬性要求

```bash
# 必须：输出 JSON 到 stdout
python -m your_module --args ...
# stdout: {"key": "value", ...}

# 错误：输出到 stderr
# 退出码：非 0
```

### 推荐但非强制

```json
{
  "status": "success|failed",  // 推荐有状态
  "target": "...",              // 推荐有目标
  "findings": [...],            // 结果列表（名字随意）
  "metadata": {...}             // 其他信息（可选）
}
```

### 格式可以完全不同

```python
# XSS CLI 输出（4 字段）
{"target": "...", "type": "...", "vulnerable": true, "findings": [...]}

# SQLi CLI 输出（5 字段）
{"url": "...", "injection_point": "...", "sqli_type": "...", 
 "database": "...", "extracted": [...]}

# Scan CLI 输出（3 字段）
{"engine": "...", "scan_time_ms": 1234, "vulnerabilities": [...]}
```

**AI 怎么处理？** 自己适配！

```python
# AI 的灵活处理
data = json.loads(stdout)

# 根据执行的命令知道格式
if "function_xss" in cmd:
    vulnerable = data.get("vulnerable", False)
    findings = data.get("findings", [])
elif "function_sqli" in cmd:
    vulnerable = len(data.get("extracted", [])) > 0
    findings = data.get("extracted", [])
# ...
```

---

## 🛠️ 实现示例

### 内部模块示例（方式灵活）

**方式 1：作为 Python 模块（推荐）**

```python
# services/core/aiva_core/internal_exploration/internal_loop_connector.py

class InternalLoopConnector:
    """内部探索连接器（可以直接导入使用）"""
    
    async def discover_capabilities(self, scan_path: str):
        """发现能力"""
        return {
            "module": scan_path.split("/")[-1],
            "capabilities": [
                {"name": "xss_scan", "confidence": 0.95}
            ]
        }
    
    async def sync_to_rag(self, data: dict):
        """同步到 RAG"""
        # 内部实现...
        pass

# AI 直接调用（不需要 CLI）
connector = InternalLoopConnector()
capabilities = await connector.discover_capabilities("services/features")
```

**方式 2：也可以提供 CLI（可选）**

```python
# services/core/aiva_core/internal_exploration/__main__.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["discover", "sync"])
    args = parser.parse_args()
    
    connector = InternalLoopConnector()
    
    if args.action == "discover":
        result = asyncio.run(connector.discover_capabilities(...))
        print(json.dumps(result))
```

### 外部功能模块示例（必须 CLI）

```python
# services/features/features_ready/function_xss/__main__.py

import argparse
import json
import sys

async def scan_xss(url: str, xss_type: str):
    """执行 XSS 扫描"""
    # 实际扫描逻辑...
    return {
        "target": url,
        "type": xss_type,
        "vulnerable": True,
        "findings": [
            {
                "payload": "<script>alert(1)</script>",
                "param": "search",
                "status": 200
            }
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--type", choices=["reflected", "stored"])
    args = parser.parse_args()
    
    result = asyncio.run(scan_xss(args.url, args.type))
    print(json.dumps(result, ensure_ascii=False))  # ← stdout JSON
```

---

## 🎯 优势总结

### 为什么这么简单就够了？

1. **AI 足够聪明**
   - 知道执行什么命令
   - 知道期待什么格式
   - 能够灵活适配不同输出

2. **subprocess 足够可靠**
   - 标准的进程调用
   - stdout/stderr 自然分离
   - 退出码清晰表示成功/失败

3. **JSON 足够通用**
   - 所有语言都支持
   - 人类可读，机器可解析
   - 灵活的数据结构

4. **无需中间层**
   - ❌ 不需要 Dispatcher 路由消息
   - ❌ 不需要 Coordinator 协调结果
   - ❌ 不需要复杂的数据模型（20+ 字段）
   - ✅ AI 直接决策、直接调用、直接处理

---

## 📊 对比传统架构

### 传统架构（复杂，未使用）

```
AI → Dispatcher → Coordinator → Module CLI → JSON
              ↓           ↓
          路由层      分析层
          (未用)     (未用)
```

**问题**：
- 过度设计
- 中间层从未被调用
- 增加复杂度和延迟

### 当前架构（简单，实际运行）

```部模块（方式自由）

```python
# 1. 创建新模块
services/core/aiva_core/cognitive_core/new_component.py

class NewComponent:
    """新的 AI 内部组件"""
    
    async def process(self, data):
        # 业务逻辑
        return result

# 2. AI 直接导入使用（推荐）
from aiva_core.cognitive_core import NewComponent
component = NewComponent()
result = await component.process(data)

# 或者提供 CLI（可选）
# python -m aiva_core.cognitive_core.new_component --action process
```

### 添加新的外部功能模块（必须 CLI）
```bash
# 1. 创建新模块
services/core/aiva_core/internal_exploration/new_analyzer.py

# 2. 实现 __main__
if __name__ == "__main__":
    # argparse + 业务逻辑
    result = analyze()
    print(json.dumps(result))

# 3. AI 直接调用
cmd = ["python", "-m", "aiva_core.internal_exploration.new_analyzer", ...]
```

### 添加新的外循环 CLI

```bash
# 1. 创建新功能模块
services/features/features_ready/function_new/

# 2. 实现 __main__.py
if __name__ == "__main__":
    result = execute_attack()
    print(j外部功能模块？

通过**内部模块发现**（直接调用，不需要 subprocess）：

```python
# AI 内部调用内部模块（方式 1：推荐）
from aiva_core.internal_exploration import InternalLoopConnector

connector = InternalLoopConnector()
capabilities = await connector.discover_capabilities(
    scan_path="services/features"
)

# AI 更新自己的知识库
await self.rag.update(capabilities)

# 或者通过内部 CLI（方式 2：可选）
# cmd = ["python", "-m", "aiva_core.internal_exploration", "--action", "discover"]
# result = await subprocess_run(cmd)
# capabilities = json.loads(result.stdoutatures"]

result = await subprocess_run(cmd)
capabilities = json.loads(result.stdout)

# AI 更新自己的知识库
await self.rag.update(capabilities)
```"""AI 内部有分析能力"""
    
    async def execute_with_analysis(self, external_cmd):
        # 调用外部功能模块
        result = await subprocess_run(external_cmd)
        raw = json.loads(result.stdout)  # 外部模块的简单输出
        
        # AI 内部分析（不需要外部 Coordinator）
        analyzed = self._analyze_result(raw)
        quality = self._evaluate_quality(raw)
        recommendations = self._generate_tips(raw)
        
        return {
            "raw": raw,
            "analysis": analyzed,
            "quality": quality,
            "recommendations": recommendations
        }
```

**方案 B：内部模块提供分析**（可选）

```python
# AI 调用内部分析模块（直接函数调用）
from aiva_core.cognitive_core.result_analyzers import XSSAnalyzer

# 外部功能返回简单结果
external_result = await self.call_external_cli("function_xss", ...)
raw = json.loads(external_result.stdout)

# 内部模块分析（不通过 CLI）
analyzer = XSSAnalyzer()
enhanced = await analyzer.analyze(raw)
```

**关键差异**：
- ❌ 外部功能模块：只输出简单 JSON
- ✅ 内部 AI 模块：负责复杂分析
- ✅ 分析在 AI 内部完成，不需要外部 Coordinator

**方案 B：可选的后处理层**（仅当需要时）

```python
# 仅在需要时调用
if needs_deep_analysis:
    from cognitive_core.result_analyzers import XSSAnalyzer
    analyzer = XSSAnalyzer()
    enhanced = await analyzer.analyze(raw_result)
```

**关键**：后处理是**可选的**，不是**必需的**！

---

## 📝 设计哲学

### KISS 原则

**Keep It Simple, Stupid**

- AI 能做的，不要委托给其他组件
- 进程能做的，不要用复杂的消息队列
- JSON 能解决的，不要设计复杂协议

### YAGNI 原则

**You Aren't Gonna Need It**

- Dispatcher？实际验证：不需要
- Coordinator？实际验证：不需要
-层 CLI 架构 = AI 内部模块（灵活） + 外部功能模块（标准 CLI）
```

### 关键要素

**AI 内部模块**（cognitive_core, task_planning, internal_exploration）
1. ✅ 通讯方式**自由** - 函数调用、CLI、消息队列都可以
2. ✅ 可以**紧密耦合** - 内部模块可以复杂交互
3. ✅ AI **自己决定** - 用什么方式最合适

**外部功能模块**（features, scan）
1. ✅ **必须** subprocess CLI
2. ✅ **必须**输出 JSON 到 stdout
3. ✅ **必须**简单、标准、独立
4. ❌ **绝不**需要 Dispatcher/Coordinator 中间层

### 成功标准

**内部模块**：
- AI 能调用到（方式不限）
- 能完成功能

**外部模块**：
## ✅ 总结

### 核心架构

```
双 CLI 架构 = 内循环 CLI + 外循环 CLI + AI 直接协调
```

### 关键要素

1. ✅ **内循环 CLI** - 自我探索，输出 JSON
2. ✅ **外循环 CLI** - 实战攻击，输出 JSON
3. ✅ **AI 主控** - subprocess 调用，json.loads() 处理
4. ❌ **无中间层** - 无 Dispatcher，无 Coordinator

### 成功标准

- 能执行：`python -m module_name --args`
- 能输出：JSON 到 stdout
- 能解析：AI 能 `json.loads()`
- 能工作：实现功能目标

**就这么简单！** 🎉
