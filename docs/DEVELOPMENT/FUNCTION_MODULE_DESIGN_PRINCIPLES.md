# 功能模組設計原則

> **制定日期**: 2025-10-16  
> **適用範圍**: AIVA 功能模組 (services/function/)  
> **設計哲學**: 功能性優先，語言特性最大化

---

## 🎯 核心設計原則

### 1. **功能性優先原則**
- ✅ **以檢測效果為核心指標** - 模組的價值由其檢測能力決定
- ✅ **實用性勝過架構一致性** - 優先確保功能正常運作
- ✅ **快速迭代和部署** - 支持獨立開發和部署週期

### 2. **語言特性最大化原則**
- ✅ **充分利用語言優勢** - Python的靈活性、Go的並發性、Rust的安全性
- ✅ **遵循語言最佳實踐** - 符合各語言的慣用法和規範
- ✅ **不強制統一架構** - 允許不同語言採用不同的設計模式

### 3. **模組間通信標準**
- ✅ **統一消息格式** - 必須支持 `AivaMessage` + `MessageHeader` 協議
- ✅ **標準主題命名** - 使用 `Topic` 枚舉中定義的標準主題
- ✅ **錯誤處理一致性** - 統一的錯誤回報格式

---

## 📋 各語言模組實現指南

### 🐍 **Python 模組**
**優勢**: 快速開發、豐富庫生態、AI/ML 整合  
**適用場景**: 複雜邏輯檢測、機器學習驅動檢測、快速原型

**實現要求**:
```python
# 必須實現的接口
from services.aiva_common.schemas.messaging import AivaMessage
from services.aiva_common.schemas.base import MessageHeader
from services.aiva_common.enums.modules import ModuleName, Topic

class PythonFunctionWorker:
    """Python 功能模組基礎類別"""
    
    async def process_message(self, message: AivaMessage) -> AivaMessage:
        """處理標準 AIVA 消息"""
        # 實現檢測邏輯
        pass
    
    def get_module_name(self) -> ModuleName:
        """返回模組標識"""
        pass
```

**推薦架構**:
- 使用 asyncio 進行並發處理
- 採用 Pydantic 進行資料驗證
- 利用 Python 生態進行複雜分析

### 🔷 **Go 模組**
**優勢**: 高性能並發、快速編譯、記憶體安全  
**適用場景**: 高吞吐量檢測、系統級掃描、網路相關檢測

**實現要求**:
```go
// 必須實現的接口
type FunctionWorker interface {
    ProcessMessage(ctx context.Context, msg *AivaMessage) (*AivaMessage, error)
    GetModuleName() string
    Shutdown(ctx context.Context) error
}

// 標準消息結構
type AivaMessage struct {
    Header  MessageHeader      `json:"header"`
    Topic   string            `json:"topic"`
    Payload map[string]interface{} `json:"payload"`
}
```

**推薦架構**:
- 使用 goroutines 和 channels 進行並發
- 採用 context 進行超時和取消控制
- 利用 Go 的網路和系統程式設計能力

### 🦀 **Rust 模組**
**優勢**: 記憶體安全、零成本抽象、極致性能  
**適用場景**: 安全關鍵檢測、底層分析、高性能處理

**實現要求**:
```rust
// 必須實現的 trait
pub trait FunctionWorker {
    async fn process_message(&self, message: AivaMessage) -> Result<AivaMessage, Error>;
    fn get_module_name(&self) -> &str;
    async fn shutdown(&self) -> Result<(), Error>;
}

// 標準消息結構
#[derive(Serialize, Deserialize)]
pub struct AivaMessage {
    pub header: MessageHeader,
    pub topic: String,
    pub payload: serde_json::Value,
}
```

**推薦架構**:
- 使用 tokio 進行異步處理
- 採用 serde 進行序列化/反序列化
- 利用 Rust 的安全性進行關鍵檢測

### 📘 **TypeScript 模組**
**優勢**: 前端整合、動態分析、瀏覽器自動化  
**適用場景**: DOM 分析、前端安全檢測、瀏覽器行為模擬

**實現要求**:
```typescript
// 必須實現的接口
interface FunctionWorker {
    processMessage(message: AivaMessage): Promise<AivaMessage>;
    getModuleName(): string;
    shutdown(): Promise<void>;
}

// 標準消息結構
interface AivaMessage {
    header: MessageHeader;
    topic: string;
    payload: Record<string, any>;
}
```

**推薦架構**:
- 使用 async/await 進行異步處理
- 採用 Playwright/Puppeteer 進行瀏覽器自動化
- 利用 Node.js 生態進行前端分析

---

## 🔄 模組間協作機制

### **消息佇列通信**
```yaml
# 標準通信流程
Core Module -> Function Module:
  Topic: "tasks.function.{type}"
  Payload: FunctionTaskPayload

Function Module -> Core Module:
  Topic: "results.function.completed"
  Payload: FindingPayload
```

### **配置系統整合**
```python
# 統一配置接口 (可選實現)
from services.function.common.detection_config import BaseDetectionConfig

# 各模組可自定義配置，但建議繼承基礎配置
class CustomModuleConfig(BaseDetectionConfig):
    custom_option: bool = True
```

### **錯誤處理標準**
```python
# 統一錯誤格式
{
    "error_id": "error_uuid",
    "error_type": "detection_error|network_error|config_error",
    "message": "詳細錯誤描述", 
    "module": "模組名稱",
    "timestamp": "ISO格式時間戳"
}
```

---

## 🚀 開發和部署指南

### **獨立開發**
- ✅ 每個模組可獨立開發和測試
- ✅ 支持不同的開發週期和版本控制
- ✅ 允許不同的依賴管理策略

### **獨立部署**
- ✅ 支持容器化部署 (Docker)
- ✅ 支持微服務架構
- ✅ 支持水平擴展

### **測試策略**
- ✅ 單元測試：各語言使用原生測試框架
- ✅ 整合測試：通過消息佇列進行端到端測試
- ✅ 性能測試：各模組獨立進行性能基準測試

---

## 📊 質量標準

### **功能性指標**
- 🎯 **檢測準確率** > 95%
- 🎯 **誤報率** < 5%
- 🎯 **覆蓋率** > 90%

### **性能指標**
- ⚡ **響應時間** < 30秒 (標準檢測)
- ⚡ **吞吐量** > 100 requests/minute
- ⚡ **資源使用** < 512MB 記憶體

### **可靠性指標**
- 🛡️ **可用性** > 99.5%
- 🛡️ **錯誤恢復** < 60秒
- 🛡️ **資料一致性** 100%

---

## 🔮 未來擴展

### **新語言支援**
- 考慮支援 Java (Spring Boot) - 企業級整合
- 考慮支援 C# (.NET) - Windows 環境優化
- 考慮支援 Swift - macOS/iOS 安全檢測

### **新檢測技術**
- AI/ML 驅動的異常檢測
- 區塊鏈安全檢測
- IoT 設備安全評估
- 雲原生安全檢測

---

**設計原則總結**: 
**功能為王，語言為器，通信為橋，質量為本**

---

**文檔版本**: v1.0  
**最後更新**: 2025-10-16  
**維護者**: AIVA 開發團隊