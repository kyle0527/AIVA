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

## 🏗️ 當前架構

```
services/scan/
├── go_engine/              # Go 掃描引擎 (參數模糊測試)
├── rust_engine/            # Rust 掃描引擎 (HTTP Smuggling)
├── typescript_engine/      # TypeScript 掃描引擎 (DOM XSS, SPA)
├── python_engine/          # Python 智能引擎 (XXE, 反序列化, 被動分析)
├── command_handler.py      # CLI 命令處理
└── README.md              # 本文件
```

---

## 📊 引擎狀態

| 引擎 | 語言 | 狀態 | 架構模式 | 用途 |
|------|------|------|----------|------|
| **Go Engine** | Go 1.23.1 | ⚠️ 需編譯 | CLI | 參數模糊測試、SSRF、SCA、CSPM |
| **Rust Engine** | Rust 2021 | ⚠️ 需編譯 | CLI | HTTP Smuggling、認證爆破、端點發現 |
| **TypeScript Engine** | Node 20+ | ⚠️ 需安裝瀏覽器 | RabbitMQ | DOM XSS、SPA 爬蟲、動態掃描 |
| **Python Engine** | Python 3.11+ | ✅ **95% 完成** | 獨立模組 | XXE、反序列化、被動分析 |

### Python Engine (最新更新 2025-12-26)

**實現狀態**:
- ✅ **XXE 檢測器** (95% 完成) - 7 種攻擊類型，12 種檢測模式，306 行代碼，6 測試
- ✅ **反序列化檢測器 v2** (90% 完成) - 4 語言支持，15+ Java Gadget Chains，675 行代碼，10 測試
- ✅ **被動流量分析器** (100% 完成) - 8 類敏感數據，6 個安全頭部，456 行代碼，15 測試
- ✅ **測試套件** (85% 覆蓋) - 35+ 測試用例，580+ 行代碼

**OWASP 覆蓋**:
- A02:2021 - 敏感數據洩露、錯誤信息洩露
- A05:2021 - XXE、安全頭部缺失、Cookie 安全
- A08:2021 - 不安全反序列化

**詳細文檔**: [Python Engine README](./python_engine/README.md) | [完整文檔 (7000+ 行)](./python_engine/README_v2.md)

---

## 🔗 相關模組

### 功能模組 (實際檢測邏輯)
- **[XSS 檢測](../features/features_ready/function_xss/README.md)** - XSStrike/Dalfox
- **[SQLI 檢測](../features/features_ready/function_sqli/README.md)** - 6 種引擎並行
- **[IDOR 檢測](../features/features_ready/function_idor/README.md)** - 水平/垂直權限測試
- **[SSRF 檢測](../features/features_ready/function_ssrf/README.md)** - 內網探測+OAST
- **[信息洩露檢測](../features/features_ready/function_info_leak/README.md)** - 敏感信息檢測

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
