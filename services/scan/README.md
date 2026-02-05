# 🎯 AIVA Scan - 多語言掃描引擎調度器

> **版本**: v3.1 | **狀態**: ✅ Production Ready | **更新**: 2025-12-26

**導航**: [← 返回 Services](../README.md) | [📋 發展方向報告](./DEVELOPMENT_ROADMAP.md)

---

## 📑 目錄

- [🎯 模組定位](#-模組定位)
- [🏗️ 當前架構](#️-當前架構)
- [📊 引擎狀態](#-引擎狀態)
- [🔗 相關模組](#-相關模組)
- [📊 使用方式](#-使用方式)
- [🔧 開發指南](#-開發指南)

---

## 🎯 模組定位

**Scan 模組**: 多語言掃描引擎調度器

**當前職責**:
- ✅ 調度 Go/Rust/TypeScript/Python 原生掃描引擎
- ✅ 處理 CLI 命令分發  
- ✅ 提供多語言引擎統一接口

**架構分工**:
```
Go Engine (Fast)      → 參數模糊測試、基礎 SSRF (廉價並發)
Rust Engine (Deep)    → HTTP 走私、認證爆破 (極致性能)
TypeScript (Dynamic)  → DOM XSS、SPA 爬蟲 (瀏覽器環境)
Python (Intelligence) → XXE、反序列化、被動分析 (複雜邏輯)
```

---

## 🏗️ 當前架構（無協調器設計）

```
services/scan/
├── go_engine/              # Go 掃描引擎 (SSRF, SCA, CSPM)
├── rust_engine/            # Rust 掃描引擎 (端口掃描, 信息收集)
├── typescript_engine/      # TypeScript 掃描引擎 (DOM XSS, SPA)
├── python_engine/          # Python 智能引擎 (XXE, 反序列化, 被動分析)
│   ├── xxe_detector.py
│   ├── deserialization_detector_v2.py
│   └── passive_analyzer.py
└── README.md

✅ **無協調器** - 每個引擎完全獨立，AI 直接調用
✅ **無中間層** - 最大化性能，最小化複雜度
✅ **純 CLI 模式** - subprocess 直接調用二進制
```

---

## 📊 引擎狀態與加強方向

| 引擎 | 語言 | 狀態 | 當前功能 | 🎯 加強方向 |
|------|------|------|----------|------------|
| **Go Engine** | Go 1.23.1 | ⚠️ 需編譯 | SSRF、SCA、CSPM | ✅ **完善** - 已實現基線比對、OOB回調、雲端metadata掃描 |
| **Rust Engine** | Rust 2021 | ✅ **已加強** | 端口掃描、HTTP Smuggling、認證爆破 | ✅ **完成** - 已添加 CL.TE/TE.CL/TE.TE 檢測 + 智能速率控制 |
| **TypeScript Engine** | Node 20+ | ✅ **已加強** | DOM XSS、SPA路由、WebSocket安全 | ✅ **完成** - Source-to-Sink追蹤、PostMessage檢測、DOM Clobbering |
| **Python Engine** | Python 3.11+ | ✅ **95% 完成** | XXE、反序列化、被動分析 | ✅ **優秀** - 無需加強，可作為參考標準 |

### 🔧 詳細加強建議

#### 1. **Go Engine - SSRF 掃描器** ✅ 已優化
**當前實現** (基於 OWASP WSTG-INPV-19):
- ✅ 基線響應比對（MD5 hash + body length）
- ✅ OOB 回調支持（Burp Collaborator 集成）
- ✅ 雲端 Metadata 掃描（AWS/Azure/GCP）
- ✅ 常見繞過技巧（IP進制轉換、URL語義攻擊）
- ✅ DNS Rebinding 檢測
- ✅ 內部服務探測

**無需加強** - 已達到 PortSwigger + OWASP 標準

#### 2. **Rust Engine** ✅ 已完成加強
**當前狀態**: 高性能掃描 + HTTP Smuggling 檢測 + 智能認證爆破

**新增功能**:
```rust
// ✅ HTTP Request Smuggling 檢測器 v2.0 (smuggling_detector_v2.rs)
pub struct SmugglingDetector {
    // CL.TE (Content-Length vs Transfer-Encoding)
    async fn detect_cl_te() -> Option<SmugglingFinding>
    
    // TE.CL (Transfer-Encoding vs Content-Length)
    async fn detect_te_cl() -> Option<SmugglingFinding>
    
    // TE.TE (雙重 Transfer-Encoding 混淆)
    async fn detect_te_te() -> Option<SmugglingFinding>
    
    // Chunk 編碼混淆檢測
    async fn detect_chunk_obfuscation() -> Option<SmugglingFinding>
    
    // 時間差異檢測（與基線比對）
    async fn measure_baseline() -> Duration
}

// ✅ 增強認證爆破器 v2.0 (auth_brute_v2.rs)
pub struct AuthBruteForcer {
    // 智能速率控制策略
    rate_limit_strategy: RateLimitStrategy // Fixed, ExponentialBackoff, Adaptive
    
    // 多協議支持
    protocol: AuthProtocol // HTTP, HTTPBasic, HTTPDigest, FormBased, JWT, OAuth2
    
    // 自適應速率調整（根據響應時間）
    fn adjust_adaptive_rate() -> Duration
    
    // 統計信息收集
    fn get_statistics() -> HashMap<String, String>
}
```

**OWASP 映射**:
- OWASP A05:2021 - Security Misconfiguration (HTTP Smuggling)
- OWASP A07:2021 - Identification and Authentication Failures (Brute Force)

**參考資源**:
- [HTTP Smuggling Research by Albinowax](https://portswigger.net/research/http-desync-attacks-request-smuggling-reborn)
- OWASP: Testing for HTTP Splitting/Smuggling (WSTG-INPV-15)

#### 3. **TypeScript Engine** ✅ 已完成加強
**當前狀態**: Playwright 動態掃描 + DOM 安全分析 + SPA 路由檢測

**新增功能**:
```typescript
// ✅ DOM 安全分析器 v2.0 (dom-security-analyzer.ts)
export class DOMSecurityAnalyzer {
    // Source-to-Sink 數據流追蹤
    async analyzeSourceToSink(): Promise<void>
    
    // PostMessage 安全檢測（Origin 驗證）
    async analyzePostMessage(): Promise<void>
    
    // DOM Clobbering 檢測
    async analyzeDOMClobbering(): Promise<void>
    
    // WebSocket 安全分析
    async analyzeWebSocket(): Promise<void>
    
    // SPA 路由安全分析
    async analyzeSPARoutes(): Promise<void>
}

// ✅ WebSocket 安全分析器 (websocket-security-analyzer.ts)
export class WebSocketSecurityAnalyzer {
    // WebSocket 連接監控
    async injectWebSocketMonitor(): Promise<void>
    
    // 敏感數據洩露檢測
    containsSensitiveData(data: string): boolean
    
    // Origin 驗證檢查
    analyzeConnection(conn: any): void
}

// ✅ SPA 路由安全分析器 (spa-route-analyzer.ts)
export class SPARouteSecurityAnalyzer {
    // 自動發現 SPA 框架 (React/Vue/Angular/Next.js)
    async detectSPAFramework(): Promise<string | null>
    
    // 路由遍歷攻擊測試（../admin, /%2e%2e/admin）
    async testRouteTraversal(): Promise<void>
    
    // 客戶端繞過檢測
    async testClientSideBypass(): Promise<void>
    
    // 未授權訪問測試
    async testUnauthorizedAccess(): Promise<void>
}
```

**OWASP 映射**:
- OWASP A03:2021 - Injection (DOM XSS)
- OWASP A01:2021 - Broken Access Control (SPA Route)
- OWASP A05:2021 - Security Misconfiguration (WebSocket)

**參考資源**:
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)
- OWASP: Testing for DOM XSS (WSTG-INPV-01)

#### 4. **Python Engine** ✅ 參考標準
**當前實現**:
- ✅ XXE 檢測器: 7種攻擊類型，完整證據收集
- ✅ 反序列化檢測器: Java/Python/PHP/.NET，15+ Gadget Chains
- ✅ 被動分析器: 8類敏感數據，完整 OWASP 映射

**優秀實踐**（其他引擎可參考）:
1. 完整的 dataclass 結構
2. 詳細的 OWASP 標籤
3. 多層次證據收集
4. 完善的錯誤處理
5. 時間基礎檢測

---

## 🔗 相關模組

### 功能模組 (實際檢測邏輯)
- **[XSS 檢測](../features/function_xss/README.md)** - XSStrike/Dalfox
- **[SQLI 檢測](../features/function_sqli/README.md)** - 6 種引擎並行
- **[IDOR 檢測](../features/function_idor/README.md)** - 水平/垂直權限測試
- **[SSRF 檢測](../features/function_ssrf/README.md)** - 內網探測+OAST
- **[信息洩露檢測](../features/function_info_leak/README.md)** - 敏感信息檢測

### 核心模組
- **[AI Core](../core/aiva_core/README.md)** - AI 命令中心
- **[Integration](../integration/README.md)** - 結果聚合與分析

---

## 📊 使用方式

### 1. Go Engine (CLI 模式)

```powershell
# 編譯
cd services/scan/go_engine
go build -o bin/sca-scanner ./cmd/sca-scanner
go build -o bin/ssrf-scanner ./cmd/ssrf-scanner

# Python 調用
import subprocess
result = subprocess.run([
    "./services/scan/go_engine/bin/sca-scanner"
], input=json.dumps({"url": "https://target.com"}), 
   capture_output=True, text=True)
```

### 2. Rust Engine (CLI 模式)

```powershell
# 編譯
cd services/scan/rust_engine
cargo build --release

# 快速掃描
./target/release/aiva-info-gatherer scan \
    --url "https://target.com" \
    --mode fast \
    --format json

# Python 調用
result = subprocess.run([
    "./services/scan/rust_engine/target/release/aiva-info-gatherer",
    "scan", "--url", target, "--mode", "deep"
], capture_output=True, text=True)
```

### 3. TypeScript Engine (RabbitMQ 模式)

```powershell
# 安裝並啟動
cd services/scan/typescript_engine
npm install
npm run build
npm run install:browsers
npm start  # 監聽 RabbitMQ

# Python 發送任務
import pika
channel.basic_publish(
    exchange='',
    routing_key='task.scan.dynamic',
    body=json.dumps({
        "scan_id": "001",
        "target_url": "https://target.com"
    })
)
```

### 4. Python Engine (獨立模組)

```powershell
# 安裝依賴
cd services/scan/python_engine
pip install -r requirements.txt

# 運行測試
python test_detectors.py

# Python 直接調用
from xxe_detector import XXEDetector
from deserialization_detector_v2 import DeserializationDetector
from passive_analyzer import PassiveAnalyzer

# XXE 檢測
detector = XXEDetector(callback_server="http://your-callback.com")
findings = detector.test_xxe("http://target.com/api/xml", "xml_param", "POST")

# 反序列化檢測
detector = DeserializationDetector()
findings = detector.test_deserialization(
    url="http://target.com/api",
    param="data",
    language="java"
)

# 被動分析
analyzer = PassiveAnalyzer()
findings = analyzer.analyze_har('traffic.har')
```

---

## 🔧 開發指南

### 添加新引擎

1. 在對應語言目錄創建引擎代碼
2. 實現統一接口：`scan(target, options) -> result`
3. 在 `command_handler.py` 註冊引擎
4. 更新本 README

### 引擎接口規範

```python
class EngineInterface:
    """所有引擎必須實現此接口"""
    
    async def scan(self, target: str, options: dict) -> dict:
        """
        Args:
            target: 掃描目標 URL/IP
            options: 掃描選項
        
        Returns:
            {
                "engine": "go|rust|typescript|python",
                "findings": [...],
                "stats": {...}
            }
        """
        pass
```

---

## 📚 引擎文檔

| 引擎 | 文檔 | 狀態 | 完成度 |
|------|------|------|--------|
| Go Engine | [README](./go_engine/README.md) | ⚠️ 需編譯 | 代碼完整 |
| Rust Engine | [README](./rust_engine/README.md) | ⚠️ 需編譯 | CLI 實現 |
| TypeScript Engine | [README](./typescript_engine/README.md) | ⚠️ 需安裝瀏覽器 | RabbitMQ 完成 |
| Python Engine | [README](./python_engine/README.md) | ✅ Production Ready | 95% |

---

## 🎯 總結

Scan 模組提供統一的多語言掃描引擎調度接口：

| 引擎 | 定位 | 狀態 |
|------|------|------|
| Go | The Active Fuzzer (主動模糊測試器) | ⚠️ 需編譯 |
| Rust | The Fast Filter (極速過濾器) | ⚠️ 需編譯 |
| TypeScript | The Browser Emulator (瀏覽器模擬器) | ⚠️ 需安裝 |
| Python | The Intelligence Engine (智能分析引擎) | ✅ Ready |

**掃描流程**:
```
AI Decision → 選擇引擎 → 執行掃描 → 返回結果 → 聚合分析
```

---

**最後更新**: 2025-12-26 | **版本**: v3.1 | **狀態**: ✅ Production Ready
