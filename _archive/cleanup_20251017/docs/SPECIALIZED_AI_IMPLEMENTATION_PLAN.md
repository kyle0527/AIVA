# 🛠️ AIVA 專業化AI實施計劃

> **基於官方四大模組架構的簡化AI實施策略**

---

## 📋 實施概覽

### 目標
在AIVA官方四大模組架構（`aiva_common`, `core`, `scan`, `function`, `integration`）基礎上，將複雜通用AI簡化為專業化程式操作AI。

### 策略
- ✅ **保持官方架構**：完全符合現有四大模組設計
- ✅ **使用官方合約**：採用 `MessageHeader`、`AivaMessage` 通信協議
- ✅ **向後兼容**：透過 `schemas_compat.py` 保持兼容性
- ✅ **模組化簡化**：在 `core` 模組內創建輕量化AI核心

---

## 🚀 階段一：基礎架構準備 (今天)

### 1.1 創建專業化AI目錄結構
```bash
cd c:\F\AIVA\services\core\aiva_core\
mkdir specialized_ai/
cd specialized_ai/

# 創建核心模組
mkdir command_processor/
mkdir program_controller/ 
mkdir communication/
mkdir context_manager/

# 創建主文件
touch __init__.py
touch specialized_ai_core.py
```

### 1.2 確認官方依賴關係
```python
# 從 aiva_common 導入官方標準
from services.aiva_common import (
    MessageHeader,
    AivaMessage, 
    ModuleName,
    Topic,
    Severity
)

# 從 core.ai_models 導入現有AI組件
from services.core.ai_models import (
    AIVARequest,
    AIVAResponse,
    SessionState
)
```

### 1.3 建立通信接口
```python
class SpecializedAICore:
    """專業化AI核心 - 符合AIVA官方架構"""
    
    def __init__(self):
        self.module_name = ModuleName.CORE
        self.session_state = SessionState()
    
    async def process_message(self, message: AivaMessage) -> AivaMessage:
        """處理來自其他模組的標準消息"""
        # 實現官方消息處理邏輯
        pass
```

---

## 📦 階段二：模組整合 (明天)

### 2.1 與Scan模組整合
```python
# 發送掃描請求到scan模組
async def request_scan(self, target: str):
    header = MessageHeader(
        message_id=generate_uuid(),
        source_module=ModuleName.CORE,
        timestamp=datetime.utcnow()
    )
    
    message = AivaMessage(
        header=header,
        topic=Topic.TASK_SCAN_START,
        payload={"target": target, "scope": "basic"}
    )
    
    return await self.message_bus.send_to_module(ModuleName.SCAN, message)
```

### 2.2 與Function模組整合  
```python
# 發送功能測試請求
async def request_function_test(self, test_type: str, target: str):
    topic_mapping = {
        "xss": Topic.TASK_FUNCTION_XSS,
        "sqli": Topic.TASK_FUNCTION_SQLI, 
        "ssrf": Topic.TASK_FUNCTION_SSRF,
        "idor": Topic.FUNCTION_IDOR_TASK
    }
    
    message = AivaMessage(
        header=MessageHeader(
            message_id=generate_uuid(),
            source_module=ModuleName.CORE
        ),
        topic=topic_mapping[test_type],
        payload={"target": target}
    )
    
    return await self.message_bus.send_to_module(ModuleName.FUNCTION, message)
```

### 2.3 與Integration模組整合
```python
# 發送威脅情報查詢
async def query_threat_intel(self, ioc: str):
    message = AivaMessage(
        header=MessageHeader(
            message_id=generate_uuid(), 
            source_module=ModuleName.CORE
        ),
        topic=Topic.TASK_THREAT_INTEL_LOOKUP,
        payload={"ioc": ioc, "sources": ["all"]}
    )
    
    return await self.message_bus.send_to_module(ModuleName.INTEGRATION, message)
```

---

## 🎯 階段三：核心功能實現 (後天)

### 3.1 意圖解析器
```python
class IntentParser:
    """輕量級意圖解析器 - 理解用戶指令"""
    
    def __init__(self):
        self.intent_patterns = {
            'scan_request': ['掃描', '檢查', '分析目標'],
            'function_test': ['測試', 'XSS', 'SQL注入', 'SSRF', 'IDOR'],  
            'threat_lookup': ['威脅情報', '查詢', 'IOC'],
            'system_operation': ['啟動', '停止', '重啟', '狀態'],
            'file_operation': ['創建', '刪除', '編輯', '移動']
        }
    
    def parse_intent(self, user_input: str) -> dict:
        """解析用戶意圖並映射到官方模組操作"""
        # 實現簡單關鍵詞匹配邏輯
        pass
```

### 3.2 程式控制器  
```python
class ProgramController:
    """程式操作控制器 - 安全執行系統命令"""
    
    def __init__(self):
        self.allowed_operations = [
            'service_status', 'file_create', 'file_edit', 
            'directory_list', 'process_check'
        ]
    
    async def execute_operation(self, operation: str, params: dict) -> dict:
        """執行授權的程式操作"""
        if operation not in self.allowed_operations:
            raise ValueError(f"未授權的操作: {operation}")
        
        # 實現安全的程式操作邏輯
        pass
```

### 3.3 通信處理器
```python  
class CommunicationHandler:
    """自然語言通信處理器"""
    
    def __init__(self):
        self.response_templates = {
            'success': "✅ {operation} 完成：{details}",
            'error': "❌ {operation} 失敗：{reason}", 
            'confirmation': "⚠️ 即將執行 {operation}，確定要繼續嗎？",
            'clarification': "🤔 請澄清：{question}"
        }
    
    def generate_response(self, result: dict, context: dict) -> str:
        """生成自然的回應文字"""
        # 實現簡單模板響應邏輯
        pass
```

---

## 🔧 階段四：整合測試 (第4天)

### 4.1 單模組測試
```python
# 測試與scan模組通信
async def test_scan_integration():
    ai = SpecializedAICore()
    result = await ai.process_user_input("掃描 example.com")
    assert result.success == True

# 測試與function模組通信  
async def test_function_integration():
    ai = SpecializedAICore()
    result = await ai.process_user_input("測試 XSS 漏洞")
    assert result.success == True
```

### 4.2 跨模組工作流測試
```python
async def test_full_workflow():
    """測試完整的跨模組工作流"""
    ai = SpecializedAICore()
    
    # 1. 用戶請求掃描
    scan_result = await ai.process_user_input("掃描並分析 target.com")
    
    # 2. AI自動調用scan模組
    # 3. 根據掃描結果調用function模組進行深度測試
    # 4. 調用integration模組查詢威脅情報
    # 5. 生成綜合報告並回應用戶
    
    assert scan_result.modules_called == ["scan", "function", "integration"]
```

---

## 📊 階段五：性能優化 (第5天)

### 5.1 簡化現有複雜模組
```bash
# 歸檔複雜AI組件
cd c:\F\AIVA\services\core\aiva_core\
mkdir _archive_complex_ai/

# 移動複雜模組
mv ai_engine/ _archive_complex_ai/
mv analysis/ _archive_complex_ai/ 
mv training/ _archive_complex_ai/
mv rag/ _archive_complex_ai/
```

### 5.2 更新模組導入
```python
# 更新 core/__init__.py
from .specialized_ai.specialized_ai_core import SpecializedAICore

# 保持向後兼容
from .specialized_ai.specialized_ai_core import SpecializedAICore as AICore
```

### 5.3 性能基準測試
```python
# 測試啟動時間和響應速度
import time

def test_performance():
    start = time.time()
    ai = SpecializedAICore()
    init_time = time.time() - start
    
    start = time.time()  
    response = ai.process_user_input("檢查系統狀態")
    response_time = time.time() - start
    
    assert init_time < 2.0  # 啟動時間 < 2秒
    assert response_time < 1.0  # 響應時間 < 1秒
```

---

## 🎯 成功標準

### 功能標準
- ✅ 完全符合AIVA官方四大模組架構
- ✅ 使用官方 `MessageHeader`、`AivaMessage` 通信協議
- ✅ 保持向後兼容性
- ✅ 能夠操作所有四大模組（scan、function、integration、core）
- ✅ 自然語言理解與回應

### 性能標準  
- ✅ 啟動時間 < 2秒
- ✅ 響應時間 < 1秒
- ✅ 記憶體使用量 < 100MB
- ✅ 核心代碼 < 1000行

### 可靠性標準
- ✅ 安全操作檢查
- ✅ 錯誤處理和復原
- ✅ 操作確認機制
- ✅ 完整的日志記錄

---

## 📝 實施檢查清單

### Day 1 - 基礎搭建
- [ ] 創建 `specialized_ai/` 目錄結構
- [ ] 實現基礎 `SpecializedAICore` 類
- [ ] 建立官方消息通信接口
- [ ] 測試與 `aiva_common` 的導入

### Day 2 - 模組整合  
- [ ] 實現與 `scan` 模組通信
- [ ] 實現與 `function` 模組通信
- [ ] 實現與 `integration` 模組通信
- [ ] 測試跨模組消息傳遞

### Day 3 - 核心功能
- [ ] 完成意圖解析器
- [ ] 完成程式控制器
- [ ] 完成通信處理器  
- [ ] 集成用戶交互邏輯

### Day 4 - 測試驗證
- [ ] 單模組測試全部通過
- [ ] 跨模組工作流測試通過
- [ ] 用戶交互測試通過
- [ ] 性能基準測試通過

### Day 5 - 部署優化
- [ ] 歸檔複雜AI組件
- [ ] 更新模組導入配置
- [ ] 性能優化完成
- [ ] 文檔更新完成

---

**🎉 完成後，AIVA將擁有一個完全符合官方架構的專業化程式操作AI，既保持系統完整性，又大幅簡化了AI複雜度！**