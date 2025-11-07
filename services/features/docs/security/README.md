# 🛡️ 安全功能檢測模組

**導航**: [← 返回文檔中心](../README.md) | [← 返回主模組](../../README.md)

---

## 📑 目錄

- [安全模組概覽](#安全模組概覽)
- [完整實現模組](#完整實現模組)
- [技術架構](#技術架構)
- [使用指南](#使用指南)
- [開發規範](#開發規範)

---

## 🔍 安全模組概覽

AIVA Features的安全功能模組是核心競爭力，提供全面的Web應用安全檢測能力，涵蓋OWASP Top 10中的關鍵漏洞類型。

### 📊 **安全模組統計**
- **總模組數**: 7個模組 (5個完整實現, 2個進行中)
- **代碼規模**: 91個檔案 (75個Python, 11個Go, 5個其他)
- **檢測範圍**: 覆蓋OWASP Top 10主要漏洞類型
- **實現程度**: 平均85%完成度

---

## ✅ 完整實現模組

### [🗃️ SQL注入檢測模組](../../function_sqli/README.md)
**最完整的檢測模組** - 95%實現度

- **檔案數量**: 20個Python檔案
- **主要引擎**: 4個專業檢測引擎
  - 布爾盲注檢測 (`boolean_detection.py`)
  - 時間盲注檢測 (`time_detection.py`) 
  - 聯合查詢注入 (`union_detection.py`)
  - 錯誤注入檢測 (`error_detection.py`)
- **特色功能**:
  - 後端資料庫指紋識別
  - 智能負載編碼器
  - 進階配置管理
  - 結果綁定發布器

**核心能力**:
```
✅ MySQL, PostgreSQL, MSSQL, Oracle支援
✅ WAF繞過技術
✅ 智能檢測策略
✅ 並行檢測優化
```

### [🔍 XSS檢測模組](../../function_xss/README.md)
**跨站腳本檢測** - 90%實現度

- **檔案數量**: 10個Python檔案
- **檢測類型**:
  - Reflected XSS (`traditional_detector.py`)
  - Stored XSS (`stored_detector.py`)
  - DOM XSS (`dom_xss_detector.py`)
  - Blind XSS (`blind_xss_listener_validator.py`)
- **特色功能**:
  - 智能payload生成器
  - 多引擎檢測系統
  - 盲注XSS監聽驗證

**核心能力**:
```
✅ 三種XSS類型全覆蓋
✅ 上下文感知檢測
✅ 先進的payload變異
✅ 實時監聽驗證
```

### [🌐 SSRF檢測模組](../../function_ssrf/README.md)
**服務端請求偽造檢測** - 90%實現度

- **檔案數量**: 12個Python檔案
- **核心組件**:
  - 智能SSRF檢測器 (`smart_ssrf_detector.py`)
  - 內網地址檢測器 (`internal_address_detector.py`)
  - 參數語義分析器 (`param_semantics_analyzer.py`)
  - OAST調度器 (`oast_dispatcher.py`)
- **特色功能**:
  - Out-of-Band檢測
  - 內網掃描防護
  - 語義級參數分析

**核心能力**:
```
✅ 內網地址檢測
✅ 雲端元數據保護
✅ OAST技術應用
✅ 協議多樣性支援
```

### [🔐 IDOR檢測模組](../../function_idor/README.md)
**不安全直接對象引用檢測** - 85%實現度

- **檔案數量**: 12個Python檔案
- **檢測方向**:
  - 垂直權限提升 (`vertical_escalation_tester.py`)
  - 水平權限繞過 (`cross_user_tester.py`)
  - 資源ID提取 (`resource_id_extractor.py`)
  - 智能檢測器 (`smart_idor_detector.py`)
- **特色功能**:
  - 增強版Worker
  - 跨用戶測試機制
  - 智能ID模式識別

**核心能力**:
```
✅ 垂直/水平越權檢測
✅ 多用戶會話管理  
✅ 資源ID智能提取
✅ 權限邊界測試
```

### [🔑 認證檢測模組](../../function_authn_go/README.md)
**Go語言高效能認證安全檢測** - 100%實現度

- **檔案數量**: 8個檔案 (Go + Python)
- **檢測類型**:
  - 弱認證機制檢測
  - 認證繞過檢測 
  - 會話管理問題檢測
  - 多重認證繞過檢測
- **特色功能**:
  - Go高併發檢測引擎
  - Python無縫整合
  - SARIF標準輸出
  - 智能模式識別

**核心能力**:
```
✅ 高效能Go實現
✅ 弱密碼檢測
✅ 會話安全分析
✅ MFA繞過檢測
```

### [🔒 密碼學檢測模組](../../function_crypto/README.md)
**密碼學安全弱點檢測** - 40%實現度

- **檔案數量**: 8個Python檔案
- **檢測類型**:
  - 加密算法弱點
  - 金鑰管理問題
  - 隨機性檢測
  - 數位簽章驗證
- **特色功能**:
  - NIST統計檢測
  - 證書鏈分析
  - 合規性檢查

**核心能力**:
```
🔹 密碼強度分析
🔹 證書安全檢測
🔹 隨機性統計檢測  
🔹 加密實現審計
```

### [⚡ 後滲透檢測模組](../../function_postex/README.md)
**滲透後活動檢測** - 30%實現度

- **檔案數量**: 9個Python檔案
- **檢測類型**:
  - 權限提升檢測
  - 橫向移動偵測
  - 持久化機制檢測
  - C&C通訊分析
- **特色功能**:
  - MITRE ATT&CK框架整合
  - 行為模式分析
  - 威脅情報整合

**核心能力**:
```
🔹 攻擊鏈重構
🔹 行為異常分析
🔹 持久化偵測
🔹 橫向移動追蹤
```

---

## 🏗️ 技術架構

### **統一架構模式**
所有安全模組遵循統一的架構設計：

```
function_[type]/
├── worker.py              # 主要Worker入口
├── detector/              # 檢測器目錄
│   ├── smart_detector.py  # 智能檢測器
│   └── specific_*.py      # 特定類型檢測器
├── engine/               # 檢測引擎目錄
│   ├── detection_*.py    # 各種檢測引擎
│   └── result_*.py       # 結果處理
├── config/              # 配置目錄
│   ├── payloads/        # 負載配置
│   └── rules/          # 檢測規則
└── README.md           # 模組說明文檔
```

### **共享組件**
- **aiva_common**: 統一的枚舉和Schema
- **智能檢測管理器**: 跨模組檢測策略
- **結果標準化**: 符合SARIF 2.1.0格式

---

## 📖 使用指南

### **快速開始**
1. **選擇檢測模組**: 根據目標漏洞類型選擇對應模組
2. **配置檢測參數**: 設置目標URL、認證信息等
3. **執行檢測**: 透過Worker接口啟動檢測
4. **分析結果**: 獲取SARIF格式的檢測報告

### **模組選擇建議**

| 目標場景 | 推薦模組 | 檢測重點 |
|----------|----------|----------|
| **Web應用掃描** | SQLI + XSS + SSRF + IDOR | 全面漏洞檢測 |
| **API安全測試** | SQLI + IDOR | 資料庫和權限檢測 |
| **內網安全評估** | SSRF | 內網滲透風險 |
| **前端安全** | XSS | 腳本注入風險 |

### **整合使用範例**
```python
# 使用智能檢測管理器進行綜合檢測
from smart_detection_manager import SmartDetectionManager

manager = SmartDetectionManager()
results = await manager.comprehensive_scan(
    target="https://example.com",
    modules=["sqli", "xss", "ssrf", "idor"]
)
```

---

## 🔧 開發規範

### **新增安全模組指南**
1. **繼承基礎架構**: 使用統一的目錄結構
2. **實現必要接口**: Worker、Detector、Engine
3. **配置標準化**: 使用aiva_common的標準枚舉
4. **測試覆蓋**: 提供完整的單元測試
5. **文檔完善**: 建立完整的README文檔

### **代碼品質要求**
- **型別提示**: 使用Python型別提示
- **錯誤處理**: 完善的異常處理機制
- **日誌記錄**: 統一的日誌格式
- **效能優化**: 支援並行和異步執行

### **安全考慮**
- **輸入驗證**: 嚴格的參數驗證
- **輸出過濾**: 防止檢測結果洩漏敏感信息
- **權限控制**: 最小權限原則
- **審計跟蹤**: 完整的操作日誌

---

## 🔗 相關連結

### **📚 開發規範與指南**
- [🏗️ **AIVA Common 規範**](../../../../services/aiva_common/README.md) - 共享庫標準與開發規範
- [🛠️ **開發快速指南**](../../../../guides/development/DEVELOPMENT_QUICK_START_GUIDE.md) - 環境設置與部署
- [🌐 **多語言環境標準**](../../../../guides/development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md) - 開發環境配置
- [🔒 **安全框架規範**](../../../../services/aiva_common/SECURITY_FRAMEWORK_COMPLETED.md) - 安全開發標準
- [📦 **依賴管理指南**](../../../../guides/development/DEPENDENCY_MANAGEMENT_GUIDE.md) - 依賴問題解決

### **模組文檔**
- [🗃️ SQL注入檢測詳細文檔](../../function_sqli/README.md)
- [🔍 XSS檢測詳細文檔](../../function_xss/README.md)
- [🌐 SSRF檢測詳細文檔](../../function_ssrf/README.md)
- [🔐 IDOR檢測詳細文檔](../../function_idor/README.md)
- [🔑 認證檢測詳細文檔](../../function_authn_go/README.md)
- [🔒 密碼學檢測詳細文檔](../../function_crypto/README.md)
- [⚡ 後滲透檢測詳細文檔](../../function_postex/README.md)

### **開發資源**
- [🔧 開發中功能](../development/README.md) - 待完成的安全模組
- [🐍 Python模組指南](../python/README.md) - Python開發規範
- [📖 文檔中心](../README.md) - 完整文檔導航

---

*最後更新: 2025年11月7日*  
*維護團隊: AIVA Security Team*