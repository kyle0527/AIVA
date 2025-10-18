# AIVA 功能模組 | Function Module

> **目錄**: `services/function/`  
> **設計原則**: [功能模組設計原則](../../docs/DEVELOPMENT/FUNCTION_MODULE_DESIGN_PRINCIPLES.md)  
> **設計哲學**: **功能為王，語言為器，通信為橋，質量為本**

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

## 📦 模組清單 | Module List

### 🐍 Python 模組 (5個)

| 模組 | 路徑 | 功能 | 狀態 |
|------|------|------|------|
| **SQLi** | `function_sqli/` | SQL 注入檢測（5引擎） | ✅ 穩定 |
| **XSS** | `function_xss/` | XSS 檢測（Reflected/Stored/DOM） | ✅ 穩定 |
| **IDOR** | `function_idor/` | IDOR 檢測（Horizontal/Vertical） | ✅ 強化中 |
| **SSRF** | `function_ssrf/` | SSRF 檢測（OAST整合） | ✅ 強化中 |
| **PostEx** | `function_postex/` | 後滲透測試 | ⚠️ 開發中 |

#### Python 模組特點
- **優勢**: 快速開發、豐富庫生態、AI/ML 整合
- **適用場景**: 複雜邏輯檢測、機器學習驅動檢測、快速原型
- **技術棧**: asyncio, Pydantic, aiohttp, httpx

### 🔷 Go 模組 (4個)

| 模組 | 路徑 | 功能 | 狀態 |
|------|------|------|------|
| **AuthN** | `function_authn_go/` | 身份認證漏洞檢測 | ✅ 穩定 |
| **CSPM** | `function_cspm_go/` | 雲端安全態勢管理 | ✅ 穩定 |
| **SCA** | `function_sca_go/` | 軟體成分分析 | ✅ 穩定 |
| **SSRF (Go)** | `function_ssrf_go/` | SSRF 檢測（高並發版） | ✅ 穩定 |

#### Go 模組特點
- **優勢**: 高性能並發、快速編譯、記憶體安全
- **適用場景**: 高吞吐量檢測、系統級掃描、網路相關檢測
- **技術棧**: goroutines, channels, context, go-swagger

### 🦀 Rust 模組 (1個)

| 模組 | 路徑 | 功能 | 狀態 |
|------|------|------|------|
| **SAST** | `function_sast_rust/` | 靜態應用程式安全測試 | ✅ 穩定 |

#### Rust 模組特點
- **優勢**: 記憶體安全、零成本抽象、極致性能
- **適用場景**: 安全關鍵檢測、底層分析、高性能處理
- **技術棧**: tokio, serde, tree-sitter

---

## 🔄 統一通信協議

### 標準消息格式

所有功能模組必須支持 AIVA 標準消息協議：

```python
from aiva_common.schemas import AivaMessage, MessageHeader, FunctionTaskPayload

# 接收任務
async def process_message(message: AivaMessage) -> AivaMessage:
    """
    處理標準 AIVA 消息
    
    Args:
        message: 包含 header, topic, payload 的標準消息
        
    Returns:
        包含檢測結果的標準消息
    """
    task = FunctionTaskPayload(**message.payload)
    
    # 執行檢測邏輯
    findings = await detect_vulnerabilities(task)
    
    # 返回標準格式
    return AivaMessage(
        header=MessageHeader(
            message_id=generate_id(),
            source=ModuleName.FUNCTION_SQLI,
            destination=ModuleName.CORE,
            timestamp=datetime.now(UTC)
        ),
        topic=Topic.RESULTS_FUNCTION_COMPLETED,
        payload=FindingPayload(findings=findings)
    )
```

### 標準主題命名

```yaml
# 接收任務
tasks.function.sqli
tasks.function.xss
tasks.function.idor
tasks.function.ssrf

# 返回結果
results.function.completed
results.function.error

# 狀態更新
events.function.progress
events.function.heartbeat
```

---

## 📊 質量標準

### 功能性指標
- 🎯 **檢測準確率** > 95%
- 🎯 **誤報率** < 5%
- 🎯 **覆蓋率** > 90%

### 性能指標
- ⚡ **響應時間** < 30秒 (標準檢測)
- ⚡ **吞吐量** > 100 requests/minute
- ⚡ **資源使用** < 512MB 記憶體

### 可靠性指標
- 🛡️ **可用性** > 99.5%
- 🛡️ **錯誤恢復** < 60秒
- 🛡️ **資料一致性** 100%

---

## 🚀 開發指南

### 新增模組流程

1. **選擇語言** - 根據檢測需求選擇最合適的語言
   - Python: 複雜邏輯、ML驅動
   - Go: 高並發、系統級
   - Rust: 安全關鍵、高性能

2. **實現標準接口**
   ```python
   # Python 模組必須實現
   class FunctionWorker:
       async def process_message(self, msg: AivaMessage) -> AivaMessage
       def get_module_name(self) -> ModuleName
   ```

3. **遵循目錄結構**
   ```
   function_{name}/
   ├── aiva_func_{name}/
   │   ├── worker.py          # 核心檢測邏輯
   │   ├── engines/           # 檢測引擎（如適用）
   │   └── config.py          # 配置管理
   ├── tests/                 # 單元測試
   ├── README.md              # 模組文檔
   └── requirements.txt       # 依賴（Python）
   ```

4. **編寫測試** - 使用語言原生測試框架
   - Python: pytest
   - Go: go test
   - Rust: cargo test

5. **更新文檔** - 在模組 README 中記錄設計原則引用

### 測試策略

```bash
# Python 模組
cd function_sqli
pytest tests/ -v --cov

# Go 模組
cd function_sca_go
go test ./... -v -cover

# Rust 模組
cd function_sast_rust
cargo test --verbose
```

---

## 🔗 相關文檔

- **設計原則**: [FUNCTION_MODULE_DESIGN_PRINCIPLES.md](../../docs/DEVELOPMENT/FUNCTION_MODULE_DESIGN_PRINCIPLES.md)
- **架構圖**: [05_function_module.mmd](../../_out/architecture_diagrams/05_function_module.mmd)
- **Schema 指南**: [SCHEMA_GUIDE.md](../../docs/DEVELOPMENT/SCHEMA_GUIDE.md)
- **多語言架構**: [ARCHITECTURE_MULTILANG.md](../../docs/ARCHITECTURE_MULTILANG.md)

---

## 📈 統計資訊

- **總模組數**: 10 個
- **Python 模組**: 5 個
- **Go 模組**: 4 個
- **Rust 模組**: 1 個
- **總程式碼行數**: ~15,000+ lines
- **測試覆蓋率**: 平均 85%

---

## 🎯 最佳實踐

### ✅ DO - 應該做的

1. **充分利用語言特性**
   - Python: 使用 asyncio, type hints, dataclasses
   - Go: 使用 goroutines, channels, defer
   - Rust: 使用 ownership, traits, async/await

2. **遵循通信協議**
   - 使用標準 `AivaMessage` 格式
   - 遵循 `Topic` 枚舉命名
   - 統一錯誤處理格式

3. **獨立測試和部署**
   - 每個模組可獨立開發
   - 支持容器化部署
   - 支持水平擴展

### ❌ DON'T - 不應該做的

1. **不要強制架構統一**
   - 不要求所有模組使用相同設計模式
   - 不要求統一依賴注入框架
   - 允許語言特定的最佳實踐

2. **不要破壞通信協議**
   - 不要自定義消息格式
   - 不要繞過標準主題命名
   - 不要忽略錯誤處理標準

3. **不要犧牲功能性**
   - 不要為了架構美觀而降低檢測能力
   - 不要過度設計
   - 實用性優先於完美主義

---

**維護團隊**: AIVA 開發團隊  
**最後更新**: 2025-10-16  
**版本**: v1.0.0
