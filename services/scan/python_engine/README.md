# 🐍 AIVA Python Engine - 智能分析引擎

> **版本**: v3.0 | **狀態**: ✅ Production Ready | **更新**: 2026-01-23

---

## 📋 概述

**Python Engine** 是 AIVA 的智能分析引擎，專注於需要複雜邏輯和深度分析的安全檢測任務。

### 🎯 核心能力

- ✅ **XXE 檢測** - XML 外部實體注入檢測
- ✅ **反序列化漏洞檢測** - Java/PHP/Python 反序列化
- ✅ **被動分析** - 流量模式分析與異常檢測
- ✅ **複雜邏輯檢測** - 業務邏輯漏洞識別

---

## 🏗️ 架構設計

```
python_engine/
├── xxe_detector.py                  # XXE 檢測器
├── deserialization_detector_v2.py   # 反序列化檢測器 v2
├── passive_analyzer.py              # 被動分析器
├── __init__.py                      # 模組初始化
└── README.md                        # 本文件
```

---

## 🚀 快速開始

### 1️⃣ 安裝依賴

```bash
cd services/scan/python_engine
pip install -r requirements.txt  # 如果有 requirements.txt
```

### 2️⃣ 直接使用

```python
# XXE 檢測
from services.scan.python_engine.xxe_detector import XXEDetector

detector = XXEDetector()
result = detector.detect(xml_content, target_url)

# 反序列化檢測
from services.scan.python_engine.deserialization_detector_v2 import DeserializationDetector

detector = DeserializationDetector()
findings = detector.scan(response_data, content_type)

# 被動分析
from services.scan.python_engine.passive_analyzer import PassiveAnalyzer

analyzer = PassiveAnalyzer()
insights = analyzer.analyze_traffic(http_flows)
```

---

## 🔧 主要模組

### 1. XXE 檢測器

**文件**: `xxe_detector.py`

**檢測方法**:
- **OOB (Out-of-Band)** - DNS/HTTP 外帶數據
- **Error-based** - 錯誤信息洩露
- **Blind XXE** - 盲注檢測

**Payload 類型**:
- 本地文件讀取（`file:///etc/passwd`）
- 內網探測（`http://169.254.169.254/`）
- DoS 攻擊（Billion Laughs）

**符合標準**:
- ✅ OWASP WSTG-INPV-07
- ✅ CWE-611

### 2. 反序列化檢測器 v2

**文件**: `deserialization_detector_v2.py`

**支援語言**:
- **Java** - ObjectInputStream 反序列化
- **PHP** - unserialize() 漏洞
- **Python** - pickle 反序列化
- **.NET** - BinaryFormatter

**檢測方法**:
- Magic Bytes 識別（`rO0`, `a:`, `\x80\x03`）
- Gadget Chain 檢測
- 已知漏洞特徵匹配

**防禦建議**:
- 使用白名單反序列化
- 數據簽名驗證
- 沙箱執行環境

### 3. 被動分析器

**文件**: `passive_analyzer.py`

**分析能力**:
- **流量模式分析** - 識別異常請求模式
- **參數特徵提取** - 發現潛在注入點
- **響應分析** - 敏感信息洩露檢測
- **時序分析** - 競態條件識別

**輸出**:
- 風險評分
- 異常流量報告
- 建議檢測點

---

## 📊 使用場景

| 模組 | 適用場景 | 檢測深度 |
|------|---------|---------|
| XXE 檢測器 | XML 處理接口 | 深度（OOB + Error） |
| 反序列化檢測器 | 二進制協議、Cookie、Session | 深度（多語言支持） |
| 被動分析器 | 大規模流量分析 | 廣度（模式識別） |

---

## 🔗 相關文檔

- [主掃描模組 README](../README.md)
- [XXE 檢測指南](https://owasp.org/www-community/vulnerabilities/XML_External_Entity_(XXE)_Processing)
- [反序列化漏洞參考](https://owasp.org/www-community/vulnerabilities/Deserialization_of_untrusted_data)

---

## 📝 許可證

MIT License - 詳見主專案 LICENSE 文件
