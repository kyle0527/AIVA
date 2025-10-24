# AIVA Features - 安全功能架構詳解 🛡️

> **定位**: AIVA 平台的安全檢測和防護核心  
> **規模**: 2111 個安全組件 (78.4% 系統重心)  
> **主力語言**: Rust (SAST 引擎) + Python (漏洞檢測)

---

## 🎯 **安全功能總覽**

### **🔥 安全引擎分佈**

```
🛡️ 安全功能層 (2,111 組件)
├── 🔍 靜態分析引擎 (SAST)
│   └── 🦀 Rust: 1,798 組件 (85.2%) ⭐ 絕對主力
├── 🚨 漏洞檢測引擎
│   ├── 🐍 XSS Detection: 63 組件
│   ├── 🐍 SQL Injection: 59 組件  
│   └── 🐍 SSRF Detection: 58 組件
└── 🔐 其他安全模組
    └── 各種專業安全工具
```

### **⚡ 核心安全能力**

| 安全領域 | 主要引擎 | 語言 | 組件數 | 功能描述 |
|----------|----------|------|--------|----------|
| **靜態分析** | SAST Engine | 🦀 Rust | 1,798 | 程式碼安全掃描、AST 分析、漏洞識別 |
| **注入攻擊** | SQL Injection | 🐍 Python | 59 | 5引擎檢測、盲注識別、時間延遲檢測 |
| **跨站腳本** | XSS Detection | 🐍 Python | 63 | Reflected/Stored/DOM XSS 檢測 |
| **請求偽造** | SSRF Detection | 🐍 Python | 58 | 內網掃描、協議濫用、OAST 整合 |

---

## 🦀 **Rust SAST 引擎 (主力系統)**

### **🔥 SAST 引擎架構**

SAST 引擎是整個 AIVA 平台的安全分析核心，使用 Rust 實現極致的效能和記憶體安全：

```rust
// 核心 SAST 架構示例
pub struct SastEngine {
    parser: CodeParser,
    analyzer: VulnerabilityAnalyzer,  
    reporter: SecurityReporter,
}

impl SastEngine {
    pub async fn scan_codebase(&self, target: &Path) -> SastResult {
        let ast = self.parser.parse_files(target).await?;
        let vulnerabilities = self.analyzer.analyze(&ast).await?;
        self.reporter.generate_report(vulnerabilities).await
    }
}
```

### **📊 SAST 引擎能力**
- **掃描速度**: ~500 files/sec
- **記憶體使用**: <50MB baseline  
- **支援語言**: 20+ 程式語言
- **漏洞類型**: 100+ OWASP 漏洞模式
- **準確率**: >95% (極低誤報率)

### **🛠️ SAST 開發指南**
```bash
# Rust SAST 環境設定
cd services/features/function_sast_rust/
cargo build --release
cargo test

# 效能測試
cargo bench

# 新增漏洞規則
cargo run --example add_rule -- --rule-file new_rule.toml
```

---

## 🐍 **Python 漏洞檢測引擎群**

### **💉 SQL Injection 檢測引擎 (59組件)**

**核心能力:**
- **5大檢測引擎**: Boolean-based, Time-based, Error-based, Union-based, Stacked queries  
- **智能 Payload**: 自適應 payload 生成和優化
- **盲注檢測**: 布林盲注和時間盲注的精確識別
- **WAF 繞過**: 多種編碼和混淆技術

**使用範例:**
```python
from aiva.features.sqli import SQLiDetector

# 初始化檢測器
detector = SQLiDetector(
    engines=['boolean', 'time', 'error', 'union'],
    timeout=30,
    payloads='aggressive'
)

# 執行檢測
result = await detector.scan_parameter(
    url="https://target.com/search", 
    param="q",
    method="GET"
)

if result.vulnerable:
    print(f"發現 SQL 注入: {result.injection_type}")
    print(f"Payload: {result.successful_payload}")
```

### **🔗 XSS 檢測引擎 (63組件)**

**檢測類型:**
- **Reflected XSS**: 反射型跨站腳本
- **Stored XSS**: 儲存型跨站腳本  
- **DOM XSS**: DOM 型跨站腳本
- **Universal XSS**: 通用跨站腳本

**智能特性:**
- **Context 分析**: HTML/JS/CSS/URL 上下文識別
- **編碼繞過**: 自動嘗試各種編碼方式
- **WAF 識別**: 自動識別和繞過 Web 防火牆
- **Polyglot Payload**: 多上下文通用 payload

### **🌐 SSRF 檢測引擎 (58組件)**

**檢測能力:**
- **內網掃描**: 自動探測內部服務
- **協議支援**: HTTP/HTTPS/FTP/File/Gopher 等
- **OAST 整合**: Out-of-band 應用安全測試
- **盲 SSRF**: 無回顯 SSRF 的檢測

**高級功能:**
```python
from aiva.features.ssrf import SSRFDetector

# OAST 整合的 SSRF 檢測
detector = SSRFDetector(
    oast_server="burpcollaborator.net",
    internal_ranges=["10.0.0.0/8", "192.168.0.0/16"],
    protocols=["http", "https", "ftp", "file"]
)

result = await detector.test_ssrf(
    url="https://target.com/fetch",
    param="url"
)
```

---

## 🔄 **跨引擎協作模式**

### **🤝 Rust ↔ Python 資料交換**

```python
# Python 調用 Rust SAST 引擎
import sast_engine  # Rust FFI binding

class SecurityScanner:
    def __init__(self):
        self.sast = sast_engine.SastEngine()
        self.sqli_detector = SQLiDetector()
        self.xss_detector = XSSDetector()
    
    async def comprehensive_scan(self, target):
        # 1. Rust SAST 靜態分析
        sast_results = await self.sast.scan_codebase(target.code_path)
        
        # 2. Python 動態檢測
        sqli_results = await self.sqli_detector.scan_endpoints(target.endpoints)
        xss_results = await self.xss_detector.scan_forms(target.forms)
        
        # 3. 結果整合和關聯分析
        return self.correlate_results(sast_results, sqli_results, xss_results)
```

### **📊 統一報告格式**

```json
{
  "scan_id": "uuid-here",
  "target": "https://target.com",
  "engines": ["sast", "sqli", "xss", "ssrf"],
  "vulnerabilities": [
    {
      "id": "SAST-001",
      "engine": "rust_sast",
      "type": "sql_injection",
      "severity": "high",
      "confidence": 0.95,
      "location": {
        "file": "src/login.rs",
        "line": 42
      },
      "description": "Potential SQL injection in user input handling"
    }
  ],
  "statistics": {
    "total_vulnerabilities": 15,
    "high_severity": 3,
    "medium_severity": 8,
    "low_severity": 4
  }
}
```

---

## 🧪 **安全功能測試指南**

### **🔍 單元測試**
```bash
# Rust SAST 測試
cd function_sast_rust/
cargo test --lib
cargo test --integration

# Python 漏洞檢測測試  
cd function_sqli/
python -m pytest tests/ -v --coverage

cd function_xss/
python -m pytest tests/ -v --coverage

cd function_ssrf/  
python -m pytest tests/ -v --coverage
```

### **🎯 整合測試**
```bash
# 跨引擎整合測試
python -m pytest tests/integration/ -v
python -m pytest tests/security_pipeline/ -v

# 效能基準測試
python scripts/security_benchmarks.py
```

---

## 🚀 **效能指標**

### **⚡ 各引擎效能基準**

| 引擎 | 掃描速度 | 記憶體使用 | 準確率 | 誤報率 |
|------|----------|------------|--------|--------|
| **🦀 SAST** | 500 files/sec | <50MB | >95% | <3% |
| **🐍 SQLi** | 100 requests/sec | <100MB | >92% | <5% |
| **🐍 XSS** | 150 requests/sec | <80MB | >90% | <7% |  
| **🐍 SSRF** | 80 requests/sec | <60MB | >88% | <8% |

### **📊 整體安全掃描效能**
- **綜合掃描速度**: ~300 files+requests/sec
- **總記憶體使用**: <400MB
- **掃描準確率**: >93% (加權平均)
- **完整掃描時間**: <5min (中型應用)

---

## ⚠️ **安全開發最佳實踐**

### **🔒 Rust SAST 開發**
```rust
// ✅ 良好實踐
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct SafeAnalyzer {
    rules: Arc<Vec<SecurityRule>>,
    cache: Arc<Mutex<AnalysisCache>>,
}

// ❌ 避免
// 不要使用不安全的記憶體操作
// 不要忽略錯誤處理
```

### **🐍 Python 檢測開發**
```python
# ✅ 良好實踐
import asyncio
import aiohttp
from typing import Optional, List

class VulnDetector:
    async def scan_with_timeout(self, target: str, timeout: int = 30) -> Optional[Result]:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
                return await self._perform_scan(session, target)
        except asyncio.TimeoutError:
            logger.warning(f"Scan timeout for {target}")
            return None

# ❌ 避免  
# 不要使用同步 HTTP 請求
# 不要忽略超時處理
# 不要硬編碼 payload
```

---

## 🔧 **故障排除指南**

### **常見問題**

**Q1: Rust SAST 編譯失敗**
```bash
# 檢查 Rust 版本
rustc --version  # 需要 1.70+

# 清理並重新編譯
cargo clean && cargo build --release
```

**Q2: Python 檢測器記憶體洩漏**
```python
# 確保正確關閉 HTTP 連接
async with aiohttp.ClientSession() as session:
    # 使用 session...
    pass  # 自動清理
```

**Q3: 跨引擎通信失敗**
```bash
# 檢查 FFI 綁定
python -c "import sast_engine; print('Rust binding OK')"

# 檢查資料格式相容性  
python scripts/test_data_format.py
```

---

**📝 版本**: v2.0 - Security-Focused Documentation  
**🔄 最後更新**: 2025-10-24  
**🛡️ 安全等級**: 最高機密 - 內部使用  
**👥 維護團隊**: AIVA Security Architecture Team

*本文件專門針對 AIVA Features 模組的安全功能進行深度解析。包含了所有安全引擎的架構、使用方法和最佳實踐。*
