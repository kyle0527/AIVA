# 🤖 AIVA 專用AI核心設計

> **設計理念**: 基於官方四大模組架構，專注於程式操作和用戶溝通的輕量級AI  
> **核心原則**: 簡單、可靠、高效  
> **目標**: 完美理解用戶指令，精確操作程式

---

## 🏗️ 官方四大模組架構整合

### AIVA 標準化四大模組
基於官方架構定義，AIVA採用以下標準模組分工：

#### 1. 🧩 **aiva_common** - 通用基礎模組
- **官方定義**: 所有模組共享的基礎設施和官方標準實現
- **專用AI角色**: 提供程式操作AI的核心通信協議
- **核心組件**: 
  - `MessageHeader`, `AivaMessage` (統一通信協議)
  - `CVSSv3Metrics`, `CVEReference`, `CWEReference` (官方安全標準)
  - `ModuleName`, `Severity` 等基礎枚舉

#### 2. 🧠 **core** - 核心業務模組  
- **官方定義**: AI核心引擎、任務編排、決策邏輯、風險評估
- **專用AI角色**: 程式理解與操作決策中心
- **核心組件**: 
  - AI訓練與經驗管理 (`AITrainingStartPayload`, `ExperienceSample`)
  - 任務執行控制 (`TaskExecution`, `PlanExecutionResult`)
  - 風險決策 (`RiskAssessment`, `AttackPathAnalysis`)

#### 3. 🔍 **scan** - 掃描發現模組
- **官方定義**: 目標發現、指紋識別、漏洞掃描、資產管理
- **專用AI角色**: 程式結構分析與問題發現
- **核心組件**: 
  - 資產掃描 (`Asset`, `AssetInventoryItem`, `ScanScope`)
  - 技術指紋 (`Fingerprints`, `TechnicalFingerprint`)
  - 漏洞發現 (`Vulnerability`, `VulnerabilityDiscovery`)

#### 4. ⚙️ **function** - 功能檢測模組
- **官方定義**: 專業化檢測功能（XSS/SQLi/SSRF/IDOR等）
- **專用AI角色**: 程式功能驗證與測試執行
- **核心組件**: 
  - 功能測試 (`FunctionTaskPayload`, `TestResult`, `TestExecution`)
  - 漏洞利用 (`ExploitResult`, `ExploitConfiguration`)
  - 專項測試 (`AuthZTest`, `PostExTest`, `SensitiveDataTest`)

#### 5. 🔗 **integration** - 整合服務模組
- **官方定義**: 外部服務整合、API閘道、報告系統、威脅情報
- **專用AI角色**: 程式環境整合與外部工具對接
- **核心組件**: 
  - 威脅情報 (`ThreatIntelPayload`, `IOCRecord`)
  - SIEM整合 (`SIEMEvent`, `SIEMIntegration`)
  - 通知系統 (`NotificationPayload`, `WebhookPayload`)

## 🎯 專用AI核心架構

### 核心模組設計
```
specialized_ai_core/
├── command_processor/          # 命令處理器
│   ├── intent_parser.py       # 意圖解析 (理解用戶要做什麼)
│   ├── command_mapper.py      # 命令映射 (指令轉換為程式操作)
│   └── safety_checker.py      # 安全檢查 (防止危險操作)
├── program_controller/         # 程式控制器  
│   ├── system_executor.py     # 系統命令執行
│   ├── file_manager.py        # 文件操作管理
│   └── service_manager.py     # 服務管理
├── communication/              # 溝通模組
│   ├── response_generator.py  # 回應生成器
│   ├── status_reporter.py     # 狀態報告器
│   └── clarification_handler.py # 澄清處理器
├── context_manager/            # 上下文管理
│   ├── conversation_state.py  # 對話狀態
│   ├── operation_history.py   # 操作歷史
│   └── user_preferences.py    # 用戶偏好
└── simple_ai_core.py          # 主核心 (< 200 行)
```

---

## 💡 核心功能實現

### 1. 意圖解析器 (理解您說什麼)
```python
class IntentParser:
    """輕量級意圖解析器 - 理解用戶指令"""
    
    def __init__(self):
        # 簡單的關鍵詞映射，不需要複雜的NLP
        self.intent_patterns = {
            'file_operation': ['創建', '刪除', '移動', '複製', '編輯'],
            'system_control': ['啟動', '停止', '重啟', '檢查狀態'],
            'code_analysis': ['分析', '檢查', '掃描', '報告'],
            'communication': ['說明', '解釋', '為什麼', '怎麼做'],
        }
    
    def parse_intent(self, user_input: str) -> dict:
        """解析用戶意圖"""
        intent = self._match_patterns(user_input)
        params = self._extract_parameters(user_input)
        
        return {
            'intent': intent,
            'parameters': params,
            'confidence': self._calculate_confidence(user_input, intent)
        }
    
    def _match_patterns(self, text: str) -> str:
        """匹配意圖模式"""
        for intent, keywords in self.intent_patterns.items():
            if any(keyword in text for keyword in keywords):
                return intent
        return 'unknown'
```

### 2. 程式執行器 (執行實際操作)
```python
class SystemExecutor:
    """系統命令執行器 - 安全地執行程式操作"""
    
    def __init__(self):
        self.safe_commands = {
            'list_files': 'ls -la',
            'check_status': 'systemctl status',
            'disk_usage': 'df -h',
            'memory_info': 'free -h',
        }
        self.dangerous_commands = ['rm -rf', 'format', 'dd if=']
    
    async def execute_command(self, command: str, params: dict) -> dict:
        """安全執行命令"""
        # 安全檢查
        if self._is_dangerous(command):
            return {
                'success': False,
                'message': '這個操作可能有危險，需要您確認',
                'requires_confirmation': True
            }
        
        # 執行命令
        try:
            result = await self._run_command(command, params)
            return {
                'success': True,
                'output': result,
                'message': '操作完成'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'執行失敗: {e}'
            }
    
    def _is_dangerous(self, command: str) -> bool:
        """檢查命令是否危險"""
        return any(danger in command.lower() for danger in self.dangerous_commands)
```

### 3. 溝通回應器 (和您對話)
```python
class ResponseGenerator:
    """回應生成器 - 和用戶自然溝通"""
    
    def __init__(self):
        self.response_templates = {
            'success': [
                "✅ 完成了！{details}",
                "✅ 好的，已經執行完畢。{details}",
                "✅ 操作成功：{details}"
            ],
            'error': [
                "❌ 抱歉，執行時遇到問題：{error}",
                "❌ 操作失敗：{error}",
                "❌ 出現錯誤：{error}"
            ],
            'clarification': [
                "🤔 我需要確認一下：{question}",
                "🤔 請問您是要：{options}",
                "🤔 這個操作有點不確定，{question}"
            ],
            'dangerous': [
                "⚠️ 這個操作可能有風險：{risk}，確定要繼續嗎？",
                "⚠️ 注意：{risk}，需要您確認後才執行",
            ]
        }
    
    def generate_response(self, result: dict, context: dict = None) -> str:
        """生成自然的回應"""
        if result.get('success'):
            template = random.choice(self.response_templates['success'])
            return template.format(details=result.get('output', ''))
        
        elif result.get('requires_confirmation'):
            template = random.choice(self.response_templates['dangerous'])
            return template.format(risk=result.get('message', ''))
        
        elif result.get('error'):
            template = random.choice(self.response_templates['error'])
            return template.format(error=result['error'])
        
        else:
            return "🤔 我不太確定該怎麼回應這個情況。"
```

### 4. 專用AI主核心 (整合所有功能)
```python
class SpecializedAICore:
    """專用AI核心 - 輕量級程式操作AI"""
    
    def __init__(self):
        self.intent_parser = IntentParser()
        self.system_executor = SystemExecutor()
        self.response_generator = ResponseGenerator()
        self.conversation_state = ConversationState()
        
        # 簡單的狀態管理
        self.is_running = True
        self.waiting_for_confirmation = False
        self.pending_operation = None
    
    async def process_user_input(self, user_input: str) -> str:
        """處理用戶輸入的主要方法"""
        
        # 1. 解析用戶意圖
        intent_result = self.intent_parser.parse_intent(user_input)
        
        # 2. 處理確認回應
        if self.waiting_for_confirmation:
            return await self._handle_confirmation(user_input)
        
        # 3. 根據意圖執行操作
        if intent_result['intent'] == 'file_operation':
            return await self._handle_file_operation(intent_result)
        elif intent_result['intent'] == 'system_control':
            return await self._handle_system_control(intent_result)
        elif intent_result['intent'] == 'communication':
            return await self._handle_communication(intent_result)
        else:
            return "🤔 我不太明白您想要做什麼，能再詳細說明一下嗎？"
    
    async def _handle_file_operation(self, intent: dict) -> str:
        """處理文件操作"""
        # 根據參數構建命令
        command = self._build_file_command(intent['parameters'])
        
        # 執行命令
        result = await self.system_executor.execute_command(command, intent['parameters'])
        
        # 如果需要確認
        if result.get('requires_confirmation'):
            self.waiting_for_confirmation = True
            self.pending_operation = (command, intent['parameters'])
        
        # 生成回應
        return self.response_generator.generate_response(result)
    
    async def _handle_confirmation(self, user_input: str) -> str:
        """處理用戶確認"""
        if '是' in user_input or 'yes' in user_input.lower() or '確定' in user_input:
            # 執行待確認的操作
            command, params = self.pending_operation
            result = await self.system_executor.execute_command(command, params, force=True)
            
            self.waiting_for_confirmation = False
            self.pending_operation = None
            
            return self.response_generator.generate_response(result)
        else:
            self.waiting_for_confirmation = False
            self.pending_operation = None
            return "好的，我取消了該操作。"
```

---

## 🚀 實施步驟

### Step 1: 創建精簡核心 (今天)
```bash
cd services/core/aiva_core/
mkdir specialized_ai_core
cd specialized_ai_core

# 創建核心文件
touch simple_ai_core.py
touch command_processor/__init__.py
touch program_controller/__init__.py
touch communication/__init__.py
```

### Step 2: 移除複雜模組 (明天)
```bash
# 歸檔複雜的AI模組
mv ai_engine/ _archive/
mv analysis/ _archive/
mv training/ _archive/
mv rag/ _archive/
```

### Step 3: 測試基本功能 (後天)
```python
# 簡單測試
ai = SpecializedAICore()
response = await ai.process_user_input("檢查系統狀態")
print(response)  # ✅ 系統狀態檢查完成：CPU 15%, 記憶體 60%
```

---

## 💬 使用示例

### 程式操作對話
```
您: "幫我檢查一下服務狀態"
AI: "✅ 系統服務狀態檢查完成：
    - Docker: 運行中
    - PostgreSQL: 運行中  
    - Redis: 運行中
    - RabbitMQ: 運行中"

您: "停止 Docker 服務"
AI: "⚠️ 停止 Docker 服務會影響所有容器運行，確定要繼續嗎？"

您: "確定"
AI: "✅ Docker 服務已停止。"
```

### 文件操作對話
```
您: "創建一個新的配置文件"
AI: "🤔 請問您要創建什麼類型的配置文件？在哪個目錄？"

您: "在當前目錄創建 config.json"
AI: "✅ 已創建 config.json 文件，是否需要添加一些基本配置？"
```

---

## 🔗 官方模組依賴關係

### 標準依賴鏈
基於AIVA官方架構，模組間依賴關係如下：

```
scan → aiva_common
function → aiva_common  
integration → aiva_common
core → aiva_common + (scan/function/integration 的部分模式)
```

### 專用AI通信協議
專用AI核心將使用官方通信標準：

```python
# 使用官方MessageHeader和AivaMessage
from services.aiva_common import MessageHeader, AivaMessage, ModuleName

class SpecializedAICore:
    async def send_command(self, target_module: ModuleName, payload: dict):
        """發送標準化命令到其他模組"""
        header = MessageHeader(
            message_id=generate_uuid(),
            source_module=ModuleName.CORE,
            target_module=target_module,
            timestamp=datetime.utcnow()
        )
        
        message = AivaMessage(
            header=header,
            topic=f"commands.{target_module.lower()}.execute",
            payload=payload
        )
        
        return await self.message_bus.send(message)
```

### 模組整合策略
1. **保持向後兼容**: 使用 `schemas_compat.py` 的重新導出機制
2. **標準化通信**: 統一使用 `AivaMessage` 協議
3. **模組化設計**: 專用AI作為 `core` 模組的簡化版本
4. **官方枚舉**: 使用標準 `ModuleName`, `Severity` 等枚舉

## 🏆 預期成果

這樣的專用AI核心：
- **輕量**: 總代碼量 < 1000 行
- **專注**: 只做程式操作和溝通
- **安全**: 內建安全檢查機制
- **自然**: 像和真人對話一樣
- **可靠**: 簡單邏輯，不易出錯
- **架構兼容**: 完全符合AIVA官方四大模組架構

---

**📝 備註**: 這是基於AIVA官方四大模組架構（aiva_common、core、scan、function、integration），從複雜通用AI轉向專用程式操作AI的實用設計方案。重點在於**簡化**而非**功能豐富**，同時保持與現有系統的完全兼容！