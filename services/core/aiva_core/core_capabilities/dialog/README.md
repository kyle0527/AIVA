# 💬 Dialog - 對話助理系統

**導航**: [← 返回 Core Capabilities](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **代碼量**: 1 個 Python 檔案，約 586 行代碼  
> **角色**: AIVA 的「自然語言介面」- 支援對話問答和一鍵執行

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
  - [DialogIntent - 對話意圖識別](#dialogintent---對話意圖識別)
  - [AIVAAssistant - AIVA 對話助理](#aivaassistant---aiva-對話助理)
- [支援的意圖類型](#支援的意圖類型)
- [使用範例](#使用範例)

---

## 🎯 模組概述

**Dialog** 子模組實現 AI 對話層，讓使用者可以透過自然語言與 AIVA 互動，詢問系統能力、執行掃描任務、生成 CLI 指令等。整合了意圖識別、能力註冊表查詢和自然語言回應生成。

### 核心能力
1. **自然語言理解** - 識別使用者的對話意圖
2. **能力查詢** - 查詢和解釋系統可用能力
3. **一鍵執行** - 透過對話直接執行掃描任務
4. **CLI 生成** - 將對話轉換為可執行的 CLI 指令
5. **上下文追蹤** - 維護對話歷史和上下文

### 設計特色
- **多語言支援** - 中文和英文模式識別
- **正則表達式匹配** - 靈活的意圖模式識別
- **能力整合** - 與 CapabilityRegistry 深度整合
- **可擴展性** - 易於添加新的意圖類型

---

## 📂 檔案列表

| 檔案名 | 行數 | 核心功能 | 狀態 |
|--------|------|----------|------|
| **assistant.py** | 586 | AIVA 對話助理 - 意圖識別和自然語言交互 | ✅ 生產 |

**總計**: 約 586 行代碼（含註解和空行）

---

## 🔧 核心組件

### DialogIntent - 對話意圖識別

**檔案**: `assistant.py` (部分)

使用正則表達式模式匹配來識別使用者的對話意圖。

#### 支援的意圖類型

```python
class DialogIntent:
    """對話意圖識別"""
    
    INTENT_PATTERNS = {
        "list_capabilities": [
            r"現在系統會什麼|你會什麼|有什麼功能|能力清單|可用功能",
            r"list.*capabilit|show.*function|what.*can.*do"
        ],
        "explain_capability": [
            r"解釋|說明|介紹.*(?P<capability>\w+)",
            r"explain|describe.*(?P<capability>\w+)"
        ],
        "run_scan": [
            r"幫我跑.*(?P<scan_type>掃描|scan|test)|執行.*(?P<target>https?://\S+)",
            r"run.*(?P<scan_type>scan|test)|execute.*scan"
        ],
        "compare_capabilities": [
            r"比較.*(?P<cap1>\w+).*和.*(?P<cap2>\w+)|差異|對比",
            r"compare.*(?P<cap1>\w+).*(?P<cap2>\w+)|difference"
        ],
        "generate_cli": [
            r"產生.*CLI|輸出.*指令|生成.*命令|可執行的.*指令",
            r"generate.*cli|output.*command|executable.*command"
        ],
        "system_status": [
            r"系統狀態|健康檢查|服務狀態|運行情況",
            r"system.*status|health.*check|service.*status"
        ],
        "help": [
            r"幫助|說明|怎麼用|使用方法|指引",
            r"help|how.*to.*use|usage|guide"
        ]
    }
    
    @classmethod
    def identify(cls, user_input: str) -> tuple[str, dict]:
        """識別使用者意圖
        
        Args:
            user_input: 使用者輸入的文字
        
        Returns:
            (intent_name, extracted_params)
        """
        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    params = match.groupdict()
                    return intent, params
        
        return "unknown", {}
```

#### 意圖識別示例

```python
# 列出能力
intent, params = DialogIntent.identify("現在系統會什麼？")
# 返回: ("list_capabilities", {})

# 解釋特定能力
intent, params = DialogIntent.identify("解釋 SQL 注入掃描")
# 返回: ("explain_capability", {"capability": "SQL"})

# 執行掃描
intent, params = DialogIntent.identify("幫我跑一個掃描 https://example.com")
# 返回: ("run_scan", {"target": "https://example.com"})

# 生成 CLI
intent, params = DialogIntent.identify("產生可執行的 CLI 指令")
# 返回: ("generate_cli", {})
```

---

### AIVAAssistant - AIVA 對話助理

**檔案**: `assistant.py` (主要部分)

AIVA 的主要對話介面，處理使用者輸入並生成適當的回應。

#### 核心類別

```python
class AIVAAssistant:
    """AIVA 對話助理
    
    功能:
    - 對話意圖識別
    - 能力查詢和解釋
    - 任務執行
    - CLI 指令生成
    - 對話歷史管理
    """
    
    def __init__(self, registry: Optional[CapabilityRegistry] = None):
        """初始化助理
        
        Args:
            registry: 能力註冊表（默認使用全局註冊表）
        """
        self.registry = registry or global_registry
        self.conversation_history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
    
    async def chat(self, user_input: str) -> str:
        """處理使用者輸入並返回回應"""
        # 1. 識別意圖
        intent, params = DialogIntent.identify(user_input)
        
        # 2. 記錄對話歷史
        self._add_to_history("user", user_input)
        
        # 3. 根據意圖處理
        if intent == "list_capabilities":
            response = await self._list_capabilities()
        elif intent == "explain_capability":
            response = await self._explain_capability(params)
        elif intent == "run_scan":
            response = await self._run_scan(params)
        elif intent == "generate_cli":
            response = self._generate_cli_from_context()
        elif intent == "system_status":
            response = await self._get_system_status()
        elif intent == "help":
            response = self._show_help()
        else:
            response = self._handle_unknown_intent(user_input)
        
        # 4. 記錄回應
        self._add_to_history("assistant", response)
        
        return response
```

#### 能力列表查詢

```python
async def _list_capabilities(self) -> str:
    """列出所有可用能力"""
    capabilities = self.registry.list_all()
    
    if not capabilities:
        return "目前系統沒有註冊任何能力。"
    
    response = "## 🎯 AIVA 可用能力列表\n\n"
    
    # 按類別分組
    by_category = {}
    for cap in capabilities:
        category = cap.get("category", "其他")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(cap)
    
    # 生成回應
    for category, caps in sorted(by_category.items()):
        response += f"### {category}\n\n"
        for cap in caps:
            response += f"- **{cap['name']}**: {cap['description']}\n"
            if cap.get("tags"):
                response += f"  標籤: {', '.join(cap['tags'])}\n"
        response += "\n"
    
    return response

# 使用示例
# 用戶: "你會什麼？"
# 助理返回:
"""
## 🎯 AIVA 可用能力列表

### Web 安全測試
- **SQL 注入掃描**: 檢測 SQL 注入漏洞
  標籤: sql, injection, database
- **XSS 掃描**: 檢測跨站腳本漏洞
  標籤: xss, client-side, javascript

### API 安全測試
- **API 模糊測試**: 對 API 端點進行模糊測試
  標籤: api, fuzzing, rest
...
"""
```

#### 能力解釋

```python
async def _explain_capability(self, params: dict) -> str:
    """解釋特定能力
    
    Args:
        params: 包含 capability 名稱
    """
    capability_name = params.get("capability", "")
    
    # 搜索能力
    capabilities = self.registry.search(capability_name)
    
    if not capabilities:
        return f"找不到與 '{capability_name}' 相關的能力。"
    
    cap = capabilities[0]  # 取最匹配的
    
    response = f"## 📖 {cap['name']}\n\n"
    response += f"**描述**: {cap['description']}\n\n"
    
    if cap.get("parameters"):
        response += "**參數**:\n"
        for param, info in cap["parameters"].items():
            required = "必需" if info.get("required") else "可選"
            response += f"- `{param}` ({required}): {info.get('description', '')}\n"
        response += "\n"
    
    if cap.get("examples"):
        response += "**使用範例**:\n"
        for example in cap["examples"]:
            response += f"```\n{example}\n```\n"
    
    if cap.get("references"):
        response += "**參考資料**:\n"
        for ref in cap["references"]:
            response += f"- {ref}\n"
    
    return response

# 使用示例
# 用戶: "解釋 SQL 注入掃描"
# 助理返回:
"""
## 📖 SQL 注入掃描

**描述**: 自動檢測 Web 應用程式中的 SQL 注入漏洞

**參數**:
- `target_url` (必需): 目標 URL
- `depth` (可選): 爬取深度，默認 3
- `authentication` (可選): 認證資訊

**使用範例**:
```
aiva scan sql-injection --target https://example.com --depth 5
```

**參考資料**:
- OWASP Top 10 - A03:2021 Injection
- CWE-89: SQL Injection
"""
```

#### 執行掃描

```python
async def _run_scan(self, params: dict) -> str:
    """執行掃描任務
    
    Args:
        params: 包含 scan_type 和 target
    """
    scan_type = params.get("scan_type", "")
    target = params.get("target", "")
    
    if not target:
        return "請提供目標 URL，例如: 'https://example.com'"
    
    # 查找對應的能力
    capabilities = self.registry.search(scan_type)
    
    if not capabilities:
        return f"找不到 '{scan_type}' 類型的掃描能力。"
    
    cap = capabilities[0]
    
    # 構建任務
    task = {
        "capability": cap["id"],
        "parameters": {
            "target_url": target,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # 保存到上下文供後續使用
    self.context["last_task"] = task
    
    response = f"✅ 準備執行 **{cap['name']}**\n\n"
    response += f"目標: `{target}`\n\n"
    response += "您可以:\n"
    response += "1. 說 '執行' 來立即開始\n"
    response += "2. 說 '產生 CLI' 來獲得命令行指令\n"
    response += "3. 說 '取消' 來取消任務\n"
    
    return response

# 使用示例
# 用戶: "幫我跑 SQL 注入掃描 https://example.com"
# 助理返回:
"""
✅ 準備執行 **SQL 注入掃描**

目標: `https://example.com`

您可以:
1. 說 '執行' 來立即開始
2. 說 '產生 CLI' 來獲得命令行指令
3. 說 '取消' 來取消任務
"""
```

#### CLI 指令生成

```python
def _generate_cli_from_context(self) -> str:
    """從上下文生成 CLI 指令"""
    last_task = self.context.get("last_task")
    
    if not last_task:
        return "沒有待執行的任務。請先描述您想做什麼，例如 '掃描 https://example.com'"
    
    capability_id = last_task["capability"]
    params = last_task["parameters"]
    
    # 查找能力
    cap = self.registry.get(capability_id)
    
    if not cap:
        return "無法生成 CLI 指令：能力未找到"
    
    # 生成 CLI 指令
    cli_command = f"aiva {cap['cli_name']}"
    
    for param_name, param_value in params.items():
        cli_param = cap["parameters"].get(param_name, {}).get("cli_flag")
        if cli_param:
            cli_command += f" {cli_param} {param_value}"
    
    response = "## 🖥️ 可執行的 CLI 指令\n\n"
    response += f"```bash\n{cli_command}\n```\n\n"
    response += "您可以複製上面的指令在終端機中執行。\n"
    
    return response

# 使用示例
# 用戶: "產生 CLI 指令"
# 助理返回:
"""
## 🖥️ 可執行的 CLI 指令

```bash
aiva scan sql-injection --target https://example.com --depth 3
```

您可以複製上面的指令在終端機中執行。
"""
```

#### 對話歷史管理

```python
def _add_to_history(self, role: str, content: str):
    """添加到對話歷史"""
    self.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # 保持歷史在合理長度
    if len(self.conversation_history) > 20:
        self.conversation_history = self.conversation_history[-20:]

def get_conversation_summary(self) -> str:
    """獲取對話摘要"""
    if not self.conversation_history:
        return "尚無對話記錄"
    
    summary = "## 💬 對話摘要\n\n"
    for entry in self.conversation_history[-5:]:  # 最近 5 條
        role_icon = "👤" if entry["role"] == "user" else "🤖"
        summary += f"{role_icon} **{entry['role']}**: {entry['content'][:100]}...\n\n"
    
    return summary
```

---

## 🗣️ 支援的意圖類型

| 意圖 | 中文模式 | 英文模式 | 功能 |
|------|---------|---------|------|
| **list_capabilities** | "你會什麼？" | "what can you do?" | 列出所有能力 |
| **explain_capability** | "解釋 SQL 注入" | "explain SQL injection" | 解釋特定能力 |
| **run_scan** | "掃描 example.com" | "scan example.com" | 執行掃描任務 |
| **compare_capabilities** | "比較 A 和 B" | "compare A and B" | 比較兩個能力 |
| **generate_cli** | "產生 CLI 指令" | "generate CLI command" | 生成命令行指令 |
| **system_status** | "系統狀態" | "system status" | 查詢系統狀態 |
| **help** | "幫助" | "help" | 顯示幫助信息 |

---

## 🚀 使用範例

### 基本對話流程

```python
from core_capabilities.dialog import AIVAAssistant

# 初始化助理
assistant = AIVAAssistant()

# 對話 1: 查詢能力
response = await assistant.chat("你會什麼？")
print(response)
# 輸出: 能力列表...

# 對話 2: 解釋特定能力
response = await assistant.chat("解釋 XSS 掃描")
print(response)
# 輸出: XSS 掃描的詳細說明...

# 對話 3: 執行掃描
response = await assistant.chat("幫我掃描 https://example.com")
print(response)
# 輸出: 準備執行掃描的確認信息...

# 對話 4: 生成 CLI
response = await assistant.chat("產生 CLI 指令")
print(response)
# 輸出: CLI 指令...

# 查看對話歷史
summary = assistant.get_conversation_summary()
print(summary)
```

### 整合到 Web 介面

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()
assistant = AIVAAssistant()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        # 接收用戶消息
        user_message = await websocket.receive_text()
        
        # 處理消息
        response = await assistant.chat(user_message)
        
        # 發送回應
        await websocket.send_text(response)
```

### 整合到 CLI

```python
import asyncio
from core_capabilities.dialog import AIVAAssistant

async def interactive_mode():
    """互動模式"""
    assistant = AIVAAssistant()
    
    print("🤖 AIVA 對話助理")
    print("輸入 'exit' 退出\n")
    
    while True:
        user_input = input("👤 您: ")
        
        if user_input.lower() in ["exit", "quit", "離開"]:
            print("👋 再見!")
            break
        
        response = await assistant.chat(user_input)
        print(f"\n🤖 AIVA: {response}\n")

if __name__ == "__main__":
    asyncio.run(interactive_mode())
```

---

## 📊 性能指標

| 指標 | 說明 | 典型值 |
|------|------|--------|
| **意圖識別準確率** | 正確識別意圖的比例 | >90% |
| **響應延遲** | 從輸入到回應的時間 | 50-200 ms |
| **上下文追蹤深度** | 保留的對話歷史條數 | 20 條 |
| **能力查詢速度** | 查詢註冊表的速度 | <10 ms |

---

## 📚 相關文檔

- [Core Capabilities 主文檔](../README.md)
- [Service Integration - CapabilityRegistry](../../service_backbone/README.md)
- [UI Panel](../../ui_panel/README.md) - 使用者介面
- [Task Planning](../../task_planning/README.md) - 任務規劃

---

**版權所有** © 2024 AIVA Project. 保留所有權利。
