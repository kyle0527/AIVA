# AIVA 通用模組重構計劃 - 分類重組方案

## 🎯 目標
將過大的 schemas.py (1789 行) 按功能域重新分類，實現：
- 單一職責原則 (SRP)
- 單一事實來源 (Single Source of Truth)
- 模組化組織
- 可維護性提升

## 📋 當前分析

### 當前文件大小
- `schemas.py`: 1789 行 ❌ 過大
- `ai_schemas.py`: 318 行 ✅ 適中
- `enums.py`: 329 行 ✅ 適中

### 發現的功能域 (基於現有註釋)
1. **基礎通信協議** (35 行)
   - MessageHeader, AivaMessage, Authentication

2. **掃描相關** (約 200 行)
   - ScanScope, ScanStartPayload, ScanCompletedPayload
   - Asset, Summary, Fingerprints

3. **功能測試** (約 150 行)
   - FunctionTaskTarget, FunctionTaskPayload
   - TestResult, ExploitResult

4. **漏洞管理** (約 300 行)
   - Vulnerability, VulnerabilityFinding
   - FindingEvidence, FindingImpact

5. **威脅情報** (約 100 行)
   - ThreatIntel Payloads, IOCRecord, ThreatIndicator

6. **風險評估** (約 200 行)
   - RiskAssessment, AttackPathAnalysis

7. **官方標準** (約 400 行)
   - CVSS v3.1, SARIF v2.1.0, CVE/CWE/CAPEC

8. **AI 和機器學習** (約 300 行)
   - 強化學習, AI 驅動驗證

9. **整合服務** (約 200 行)
   - SIEM, EASM, API 安全

10. **系統管理** (約 100 行)
    - ModuleStatus, SystemStatus

## 🎨 重構方案

### 新文件結構
```
services/aiva_common/
├── __init__.py                 # 統一導出入口
├── types/                      # 類型定義模組
│   ├── __init__.py
│   ├── base.py                # 基礎類型和 BaseModel
│   ├── messaging.py           # 消息和通信協議
│   ├── security.py           # 安全相關基礎類型
│   └── system.py             # 系統狀態和配置
├── schemas/                   # 業務模式定義
│   ├── __init__.py
│   ├── scanning.py           # 掃描相關模式
│   ├── vulnerability.py      # 漏洞管理模式
│   ├── testing.py           # 功能測試模式
│   ├── threat_intel.py      # 威脅情報模式
│   ├── risk_assessment.py   # 風險評估模式
│   └── integration.py       # 整合服務模式
├── standards/                # 官方標準實現
│   ├── __init__.py
│   ├── cvss.py              # CVSS v3.1 標準
│   ├── sarif.py             # SARIF v2.1.0 標準
│   ├── mitre.py             # MITRE ATT&CK 標準
│   └── cve_cwe.py           # CVE/CWE/CAPEC 標準
├── ai/                      # AI 和機器學習
│   ├── __init__.py
│   ├── schemas.py           # AI 相關模式 (重命名)
│   ├── learning.py          # 強化學習模式
│   └── analysis.py          # AI 分析模式
└── enums.py                # 保持現狀
```

## 🔄 遷移步驟

### 階段 1: 創建新結構 (不破壞現有)
1. 創建新目錄結構
2. 按功能域分割內容
3. 建立完整的導入鏈

### 階段 2: 更新導入 (向後兼容)
1. 更新 `__init__.py` 以支持兩種導入方式
2. 添加棄用警告
3. 確保所有模組正常工作

### 階段 3: 清理 (移除舊文件)
1. 移除或重命名大型 schemas.py
2. 更新所有引用
3. 清理過時導入

## 📊 預期收益

### 可維護性
- 文件大小從 1789 行分散到 10+ 個 100-300 行的文件
- 每個文件職責單一且清晰
- 更容易進行代碼審查

### 開發體驗
- 更快的 IDE 載入和分析
- 更精確的自動完成
- 更容易定位相關定義

### 架構清晰度
- 明確的功能邊界
- 標準化的分層結構
- 更好的依賴管理

## 🛡️ 風險緩解

### 向後兼容性
```python
# 在 __init__.py 中保持舊的導入路徑
from .schemas.vulnerability import VulnerabilityFinding
from .schemas.scanning import ScanRequest
# 等等...

# 同時支持新的導入方式
__all__ = [...] # 包含所有導出
```

### 漸進遷移
- 新代碼使用新結構
- 舊代碼繼續工作
- 逐步遷移現有引用

### 測試覆蓋
- 為每個新模組添加測試
- 確保導入兼容性
- 驗證功能完整性

## 📋 實施檢查清單

- [ ] 創建新目錄結構
- [ ] 分割 schemas.py 內容
- [ ] 建立標準模組 (CVSS, SARIF, MITRE)
- [ ] 重組 AI 相關定義
- [ ] 更新所有 __init__.py 文件
- [ ] 測試導入兼容性
- [ ] 更新文檔
- [ ] 漸進遷移現有代碼

這個重構將使 AIVA 的架構更加清晰和可維護！
