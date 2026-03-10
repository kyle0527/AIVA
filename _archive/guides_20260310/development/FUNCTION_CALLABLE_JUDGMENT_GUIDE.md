# 如何判断函数能否直接调用？- 标准判断指南

## 🎯 快速判断法（3 秒规则）

看到一个函数，问自己 3 个问题：

1. **它在类里面吗？** → 是 = ❌ 不能直接调用
2. **它有 `_` 或 `__` 开头吗？** → 是 = ❌ 不能直接调用  
3. **它在 `__main__.py` 文件里吗？** → 否 = ⚠️ 可能不能调用

---

## 📖 Python 编程基础规则（业界标准）

### 规则 1: 类方法 vs 顶层函数

#### ✅ 顶层函数（Top-level Function）- 可以直接调用

```python
# 文件: some_module.py

def run_test(url, param):  # ← 顶层函数
    """这是独立函数，可以直接调用"""
    return test(url, param)

# 调用方式：
from some_module import run_test
result = run_test("http://example.com", "q")  # ✅ 直接调用
```

**特征**：
- 不在任何 `class` 块里面
- 直接定义在文件顶层
- 可以直接 `import` 使用

#### ❌ 类方法（Class Method）- 不能直接调用

```python
# 文件: some_module.py

class TestDetector:  # ← 这是一个类
    def __init__(self, config):
        self.config = config
    
    def execute(self, url):  # ← 这是类方法
        """必须先创建实例才能调用"""
        return self._process(url)
    
    def _process(self, url):  # ← 私有类方法
        pass

# 调用方式：
from some_module import TestDetector

# ❌ 错误 - 不能直接调用
TestDetector.execute("http://example.com")  # TypeError!

# ✅ 正确 - 必须先创建实例
detector = TestDetector(config)  # 先实例化
result = detector.execute("http://example.com")  # 再调用
```

**特征**：
- 在 `class ClassName:` 块里面
- 第一个参数是 `self`
- 需要先创建对象实例

---

### 规则 2: 命名约定（PEP 8 标准）

Python 社区遵循 **PEP 8** 命名规范：

#### ✅ 公开函数（Public）

```python
def run_test():        # ✅ 公开函数
    pass

def calculate_score(): # ✅ 公开函数
    pass

class MyClass:
    def execute(self): # ✅ 公开方法
        pass
```

**特征**：
- 普通命名，没有特殊前缀
- 表示可以被外部使用

#### ⚠️ 单下划线开头（Protected）

```python
def _internal_helper():     # ⚠️ 内部辅助函数
    pass

class MyClass:
    def _build_config(self): # ⚠️ 受保护方法
        pass
```

**含义**：
- **约定俗成**：这是内部实现
- **不是强制**：技术上可以调用
- **不应使用**：可能随时改变，不保证兼容性

#### ❌ 双下划线开头（Private）

```python
def __secret_function():      # ❌ 私有函数（不常用）
    pass

class MyClass:
    def __init__(self):        # ✅ 特殊方法（构造函数）
        pass
    
    def __private_method(self): # ❌ 私有方法
        pass
    
    def __str__(self):          # ✅ 特殊方法（魔术方法）
        pass
```

**双下划线有两种用途**：

1. **魔术方法**（Magic Methods）：`__init__`, `__str__`, `__call__`
   - Python 内置，有特殊含义
   - 不直接调用，由 Python 自动调用

2. **名称修饰**（Name Mangling）：普通方法名
   - Python 会改变名称（`_ClassName__method`）
   - 强制私有化

---

### 规则 3: 特殊方法（Magic Methods）

```python
class MyClass:
    def __init__(self):      # ❌ 不直接调用
        """构造函数"""
        pass
    
    def __str__(self):       # ❌ 不直接调用
        """字符串表示"""
        return "MyClass"
    
    def __call__(self):      # ❌ 不直接调用
        """使对象可调用"""
        pass
    
    def __len__(self):       # ❌ 不直接调用
        """返回长度"""
        return 0

# 使用方式：
obj = MyClass()          # 自动调用 __init__
print(obj)               # 自动调用 __str__
obj()                    # 自动调用 __call__
len(obj)                 # 自动调用 __len__
```

**规则**：双下划线包围的方法 = 魔术方法 = Python 自动调用

---

### 规则 4: 模块结构约定

#### ✅ `__main__.py` - CLI 入口

```python
# services/features/function_xss/__main__.py

async def run_reflected_test(args):  # ✅ CLI 可调用
    """反射型 XSS 测试入口"""
    pass

async def run_dom_test(args):        # ✅ CLI 可调用
    """DOM XSS 测试入口"""
    pass

if __name__ == "__main__":
    # 这个文件设计为命令行入口
    asyncio.run(main())
```

**约定**：
- `__main__.py` 表示这是**可执行模块**
- 里面的函数通常是**用户接口**
- 可以通过 `python -m module_name` 运行

#### ❌ 内部模块 - 实现细节

```python
# services/features/function_xss/detector.py

class XssDetector:                   # ❌ 内部实现
    def execute(self):
        pass

# services/features/function_xss/worker.py

async def process_task(task):        # ❌ 内部实现
    pass
```

**约定**：
- 非 `__main__.py` 的文件 = 内部实现
- 不设计为直接调用

---

## 🔍 实际判断步骤

### 步骤 1: 看文件路径

```
services/features/function_xss/
├── __main__.py          ✅ CLI 入口（看这里找可用函数）
├── __init__.py          ⚠️ 模块接口（通常是类）
├── detector.py          ❌ 内部实现
├── worker.py            ❌ 内部实现
├── payload_generator.py ❌ 内部实现
└── utils.py             ❌ 内部实现
```

**判断**：
- 在 `__main__.py` 中？ → ✅ 很可能可以调用
- 在其他文件中？ → ❌ 很可能是内部实现

---

### 步骤 2: 看函数定义

#### 示例 1: 顶层函数

```python
# ✅ 可以调用
async def run_reflected_test(args):
    ^^^^^^^^^^^^^^^^^^^^^^^^
    └─ 没有缩进 = 顶层函数
```

#### 示例 2: 类方法

```python
# ❌ 不能直接调用
class TraditionalXssDetector:
    def execute(self, payloads):
        ^^^^
        └─ 有缩进 = 类方法
```

**判断**：
- 函数定义没有缩进？ → ✅ 顶层函数
- 函数定义有缩进？ → ❌ 可能是类方法

---

### 步骤 3: 看函数签名

#### 示例 1: 简单参数

```python
# ✅ CLI 友好
def run_test(url: str, param: str, timeout: int):
    pass
```

#### 示例 2: 复杂参数

```python
# ❌ CLI 不友好
def process_task(
    task: FunctionTaskPayload,      # ← 复杂对象
    detector: XssDetector,           # ← 对象实例
    config: Dict[str, Any]           # ← 字典
):
    pass
```

**判断**：
- 参数是基本类型（str, int, bool）？ → ✅ 可能可以调用
- 参数是对象/类实例？ → ❌ 不能通过 CLI 调用

---

### 步骤 4: 看函数名

```python
# ✅ 公开接口
run_reflected_test()
run_dom_test()
execute_scan()

# ⚠️ 内部实现
_build_request()
_inject_payload()
_verify_result()

# ❌ 魔术方法
__init__()
__str__()
__call__()
```

**判断**：
- 普通命名？ → ✅ 可能是公开接口
- `_` 开头？ → ⚠️ 内部实现
- `__` 开头结尾？ → ❌ 魔术方法

---

## 📊 完整判断流程图

```
看到一个函数
    ↓
┌─────────────────────────┐
│ 1. 它在 class 里面吗？   │
└───┬─────────────────┬───┘
    │ 是              │ 否
    ↓                 ↓
  ❌ 不能直接调用     继续判断
    (需要实例化)       ↓
                 ┌──────────────────────┐
                 │ 2. 函数名有 _ 开头吗？│
                 └───┬──────────────┬───┘
                     │ 是           │ 否
                     ↓              ↓
                   ⚠️ 内部实现     继续判断
                   (不建议调用)     ↓
                              ┌─────────────────────┐
                              │ 3. 在 __main__.py？  │
                              └───┬─────────────┬───┘
                                  │ 是          │ 否
                                  ↓             ↓
                              ✅ 可能可用    ⚠️ 可能内部
                                             (看文档)
```

---

## 📚 实际案例对比

### 案例 1: XSS 模块

#### ✅ 可以调用（在 `__main__.py`）

```python
# services/features/function_xss/__main__.py

async def run_reflected_test(args):     # ✅ 顶层 + 公开 + CLI友好
    """反射型 XSS 测试"""
    pass

async def run_dom_test(args):           # ✅ 顶层 + 公开 + CLI友好
    """DOM XSS 测试"""
    pass

async def run_stored_test(args):        # ✅ 顶层 + 公开 + CLI友好
    """存储型 XSS 测试"""
    pass
```

**判断依据**：
1. ✅ 在 `__main__.py` 中
2. ✅ 顶层函数（没有缩进）
3. ✅ 公开命名（没有 `_`）
4. ✅ 简单参数（argparse.Namespace）

#### ❌ 不能调用（在其他文件）

```python
# services/features/function_xss/traditional_detector.py

class TraditionalXssDetector:           # ❌ 这是一个类
    def __init__(self, task, timeout):  # ❌ 魔术方法
        self._task = task
    
    async def execute(self, payloads):  # ❌ 类方法 + 复杂参数
        pass
    
    def _build_request_parts(self):     # ❌ 类方法 + 私有
        pass
```

**判断依据**：
1. ❌ 不在 `__main__.py` 中
2. ❌ 类方法（在 class 里面）
3. ❌ 有私有方法（`_` 开头）
4. ❌ 复杂参数（对象实例）

---

## 🎓 业界标准参考

### PEP 8 - Python 代码风格指南

**官方规范**：https://peps.python.org/pep-0008/

关键约定：
- **单下划线前缀** (`_name`)：内部使用
- **双下划线前缀** (`__name`)：私有（名称修饰）
- **双下划线包围** (`__name__`)：魔术方法
- **全大写** (`CONSTANT`)：常量

### Google Python Style Guide

**Google 规范**：https://google.github.io/styleguide/pyguide.html

额外建议：
- 模块的公开 API 应该在 `__all__` 中列出
- 私有函数和类应该以 `_` 开头
- 不要直接调用魔术方法

### 常见框架的约定

#### Django
```python
# views.py - 可以直接调用的视图函数
def my_view(request):  # ✅ 公开接口
    pass

# utils.py - 内部辅助函数
def _helper_function():  # ⚠️ 内部使用
    pass
```

#### Flask
```python
@app.route('/api/test')
def api_test():  # ✅ 路由处理函数
    pass

def _validate_input():  # ⚠️ 内部验证
    pass
```

---

## 🛠️ 实用工具

### 使用 Python 内置函数检查

```python
import inspect

# 检查是否为函数
inspect.isfunction(run_reflected_test)  # True

# 检查是否为方法
inspect.ismethod(detector.execute)  # True

# 检查是否为类
inspect.isclass(TraditionalXssDetector)  # True

# 查看函数签名
sig = inspect.signature(run_reflected_test)
print(sig)  # (args)
```

### 查看模块的公开 API

```python
# 查看模块导出的内容
import function_xss
print(function_xss.__all__)  
# ['XSSCommandHandler', 'XssWorkerService', 'TraditionalXssDetector', ...]

# 查看所有公开属性
print([name for name in dir(function_xss) if not name.startswith('_')])
```

---

## 📋 快速检查清单

在调用一个函数之前，检查：

- [ ] **文件位置**：在 `__main__.py` 中？
- [ ] **函数类型**：是顶层函数（不在 class 里）？
- [ ] **命名规范**：没有 `_` 前缀？
- [ ] **参数类型**：都是简单类型（str, int, bool）？
- [ ] **文档说明**：有使用示例吗？
- [ ] **返回类型**：返回简单数据结构？

**6 个都是 ✅** → 很可能可以直接调用  
**有 ❌** → 可能是内部实现

---

## 💡 实战技巧

### 技巧 1: 看 `__all__`

```python
# services/features/function_xss/__init__.py

__all__ = [
    "XSSCommandHandler",      # 导出的是类，不是函数
    "XssWorkerService",       # 导出的是类，不是函数
    "TraditionalXssDetector", # 导出的是类，不是函数
]
```

**判断**：导出的都是类 → 需要实例化使用

### 技巧 2: 看 CLI 参数解析

```python
# 如果看到这样的代码：
parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--param", default="q")

# 说明这个模块设计为 CLI 使用
# 里面的函数很可能可以直接调用
```

### 技巧 3: 看导入方式

```python
# ✅ 顶层函数的导入
from function_xss.__main__ import run_reflected_test

# ❌ 类的导入
from function_xss.traditional_detector import TraditionalXssDetector
detector = TraditionalXssDetector(...)  # 需要实例化
```

---

## 🎯 总结：判断标准

### 三大黄金规则

1. **位置规则**：在 `__main__.py` → 可能可用
2. **命名规则**：没有 `_` 前缀 → 可能可用
3. **结构规则**：是顶层函数 → 可能可用

### 三大红旗信号

1. **在 class 里面** → ❌ 需要实例化
2. **有 `_` 前缀** → ⚠️ 内部实现
3. **复杂参数** → ❌ CLI 不友好

### 记住一句话

> **如果一个函数设计为让你直接调用，它会表现得很明显：**
> - 在 `__main__.py` 中
> - 名字清晰（如 `run_xxx`, `execute_xxx`）
> - 参数简单（字符串、数字）
> - 有文档说明

如果你需要"猜"它能不能用，答案通常是**不能**。

---

## 📖 延伸阅读

- **PEP 8**: https://peps.python.org/pep-0008/
- **Python Tutorial - Classes**: https://docs.python.org/3/tutorial/classes.html
- **Python Data Model**: https://docs.python.org/3/reference/datamodel.html
- **Google Python Style Guide**: https://google.github.io/styleguide/pyguide.html

---

**关键概念**：Python 的命名和结构约定不是技术限制，而是**社区共识**。遵循这些约定能让代码更易理解和维护！
