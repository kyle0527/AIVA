# 📘 AIVA 使用者手冊 - 第 2 冊：AI 決策流程

> **版本**: v3.0  
> **最後更新**: 2026-02-01  
> **適用對象**: 安全研究員、Bug Hunter、滲透測試人員  
> **閱讀時間**: 約 20 分鐘

---

## 📑 目錄

### 第一部分：用戶視角（必讀）
1. [快速開始：一分鐘發起掃描](#1-快速開始一分鐘發起掃描)
2. [CLI 命令使用指南](#2-cli-命令使用指南)
3. [掃描結果解讀](#3-掃描結果解讀)

### 第二部分：內部運作（可選閱讀）
4. [AI 決策流程概述](#4-ai-決策流程概述)
5. [三階段決策詳解](#5-三階段決策詳解)
6. [Bug Bounty 優化邏輯](#6-bug-bounty-優化邏輯)

---

# 第一部分：用戶視角

## 1. 快速開始：一分鐘發起掃描

### 方式一：互動式選單（推薦新手）

```bash
# 雙擊執行
啟動能力選單.bat

# 按照提示選擇掃描類型
```

### 方式二：命令行（推薦熟練用戶）

```bash
# 基本掃描
aiva scan https://example.com

# 高強度掃描
aiva scan https://example.com -i 0.8

# 指定參數
aiva scan https://example.com --param max_depth=5
```

### 方式三：HTTP API（推薦自動化）

```bash
curl -X POST http://localhost:9000/scan \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://example.com",
    "scan_type": "full",
    "max_depth": 3
  }'
```

---

## 2. CLI 命令使用指南

### 2.1 基本命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `aiva scan <target>` | 啟動掃描 | `aiva scan https://example.com` |
| `aiva status <scan_id>` | 查詢掃描狀態 | `aiva status scan_abc123` |
| `aiva health` | 系統健康檢查 | `aiva health` |
| `aiva list-flows` | 列出可用功能 | `aiva list-flows --stats` |

### 2.2 進階參數

```bash
# 調整 AI 強度（0.0-1.0）
aiva scan https://example.com -i 0.9

# 傳遞額外參數
aiva scan https://example.com \
  --param stealth_level=high \
  --param rate_limit=500

# 預覽模式（不實際執行）
aiva scan https://example.com --dry-run
```

### 2.3 常用 Flow 命令

```bash
# Flow 0: 內部查詢
aiva query "find SQL injection vulnerabilities"

# Flow 8: 攻擊面掃描
aiva flow8 --target https://example.com -i 0.8

# Flow 列表
aiva list-flows --by-endpoint
```

---

## 3. 掃描結果解讀

### 3.1 掃描狀態

```json
{
  "scan_id": "scan_abc123",
  "status": "running",
  "progress": 45,
  "current_phase": "phase1_deep_scan",
  "vulnerabilities_found": 3,
  "estimated_remaining_time": 120
}
```

**狀態說明**：
- `queued` - 已排隊等待執行
- `phase0_started` - Phase 0 快速偵察中
- `phase1_deep_scan` - Phase 1 深度掃描中
- `phase2_attack` - Phase 2 攻擊測試中
- `completed` - 掃描完成
- `failed` - 掃描失敗

### 3.2 漏洞報告

```json
{
  "vulnerabilities": [
    {
      "type": "SQL_INJECTION",
      "severity": "HIGH",
      "confidence": 0.95,
      "location": "/api/user?id=",
      "cvss_score": 7.5,
      "estimated_bounty": 5000,
      "poc_ready": true
    }
  ]
}
```

**嚴重性等級**：
- `CRITICAL` - 嚴重（預估獎金 $10k+）
- `HIGH` - 高危（預估獎金 $3k-$10k）
- `MEDIUM` - 中危（預估獎金 $500-$3k）
- `LOW` - 低危（預估獎金 $100-$500）

---

# 第二部分：內部運作（可選閱讀）

> **提示**: 這部分內容是給想深入了解 AI 決策邏輯的用戶。如果您只想使用系統，可以跳過這部分。

---

## 4. AI 決策流程概述

### 4.1 核心理念

當您執行 `aiva scan https://example.com` 時，系統內部會自動執行一個複雜的三階段決策流程：

```
用戶命令
  ↓
Phase 0: 快速偵察（資產發現）
  ↓
AI 決策 1: 是否需要深度掃描？掃描什麼？
  ↓
Phase 1: 深度掃描（漏洞檢測）
  ↓
AI 決策 2: 攻擊哪些目標？優先級如何？
  ↓
Phase 2: 攻擊測試（漏洞利用）
  ↓
AI 決策 3: 提交報告還是繼續深挖？
  ↓
返回結果
```

### 4.2 為什麼需要三階段決策？

**傳統掃描器的問題**：
- ❌ 盲目掃描所有端口和路徑
- ❌ 浪費時間在低價值目標上
- ❌ 容易觸發 WAF 被封禁
- ❌ 無法根據發現動態調整策略

**AIVA 的 AI 決策優勢**：
- ✅ 根據偵察結果智能決定掃描範圍
- ✅ 優先測試高價值漏洞（ROI 導向）
- ✅ 動態調整策略避免 WAF 封禁
- ✅ 評估結果決定是否繼續深挖

---

## 5. 三階段決策詳解

### 5.1 Phase 0: 快速偵察（Step 0-5）

**目標**: 快速了解目標的基本情況

```
Step 0: 接收用戶輸入
  ↓
Step 1-2: 解析目標，創建任務計劃
  ↓
Step 3-4: 執行快速偵察
  - 發現開放端口
  - 識別 Web 技術棧
  - 檢測 WAF 存在
  - 發現子域名和 API 端點
  ↓
Step 5: 整合偵察結果
```

**輸出範例**：
```json
{
  "urls_found": 127,
  "apis_found": 15,
  "forms_found": 8,
  "technologies": ["PHP", "Laravel", "MySQL"],
  "waf_detected": true,
  "waf_vendor": "Cloudflare"
}
```

---

### 5.2 AI 決策 1: Phase 0 → Phase 1 策略（Step 6）

**決策方法**: `decide_phase1_strategy()`

**AI 考量因素**：

1. **高價值目標識別**
   - 發現 API 端點 → 可能有 IDOR/Auth 問題
   - 發現文件上傳 → 可能有 RCE 風險
   - 發現支付流程 → 高獎金目標
   - 發現 Admin 面板 → 權限提升機會

2. **技術棧風險評估**
   - PHP/WordPress → 歷史漏洞多，風險高
   - Spring/Struts → 已知 CVE 風險
   - Node.js → 原型鏈污染風險

3. **WAF 和限制**
   - 檢測到 WAF → 切換隱匿模式
   - Rate Limiting → 降低掃描速度
   - Program Scope → 遵守範圍限制

4. **ROI 計算**
   - 預估獎金價值 vs 掃描成本
   - 時間投資回報率

**決策輸出範例**：
```json
{
  "needs_deep_scan": true,
  "focus_areas": ["api_endpoints", "file_upload", "auth_flows"],
  "scan_strategy": "api_focused",
  "stealth_level": "high",
  "estimated_value": 8500,
  "reasoning": "發現 15 個 API 端點和文件上傳功能，檢測到 Cloudflare WAF，建議使用隱匿模式深度掃描"
}
```

---

### 5.3 Phase 1: 深度掃描（Step 7-8）

**目標**: 根據 AI 決策進行針對性深度掃描

```
Step 7: 執行深度掃描
  - SQL 注入檢測
  - XSS 漏洞檢測
  - SSRF 檢測
  - IDOR 檢測
  - 文件上傳漏洞檢測
  ↓
Step 8: 整合掃描結果
```

**輸出範例**：
```json
{
  "vulnerabilities": [
    {
      "type": "SQL_INJECTION",
      "location": "/api/user?id=",
      "severity": "HIGH",
      "confidence": 0.95
    },
    {
      "type": "XSS_STORED",
      "location": "/comment",
      "severity": "MEDIUM",
      "confidence": 0.80
    }
  ]
}
```

---

### 5.4 AI 決策 2: Phase 1 → Phase 2 目標選擇（Step 9）

**決策方法**: `decide_phase2_targets()`

**AI 優先級排序邏輯**（基於 HackerOne/Bugcrowd 實戰經驗）：

**Tier 1: Critical 獎金 $10k+**
- Account Takeover (ATO) 鏈
- RCE/SSRF 到內部服務
- 支付/金融繞過
- PII 大規模洩露

**Tier 2: High 獎金 $3k-$10k**
- SQL Injection（有數據影響）
- IDOR 到敏感資源
- Auth Bypass
- Privilege Escalation

**Tier 3: Medium 獎金 $500-$3k**
- XSS（Stored > DOM > Reflected）
- CSRF（敏感操作）
- 信息洩露（API keys, credentials）

**決策輸出範例**：
```json
{
  "targets": [
    {
      "vuln_type": "sql_injection",
      "tier": 2,
      "priority": 1,
      "cvss_estimate": 7.5,
      "estimated_bounty": 5000,
      "attack_vector": "time_based_blind",
      "reasoning": "高信心 SQLi，位於用戶 API，可能洩露敏感數據"
    },
    {
      "vuln_type": "xss_stored",
      "tier": 3,
      "priority": 2,
      "cvss_estimate": 5.5,
      "estimated_bounty": 800,
      "attack_vector": "dom_based",
      "reasoning": "Stored XSS 在評論區，可能升級為 ATO"
    }
  ]
}
```

---

### 5.5 Phase 2: 攻擊測試（Step 10）

**目標**: 驗證漏洞可利用性，準備 POC

```
Step 10: 執行攻擊測試
  - 按優先級測試目標
  - 生成 POC
  - 驗證可複現性
  - 評估實際影響
```

---

### 5.6 AI 決策 3: Phase 2 結果評估（Step 11）

**決策方法**: `evaluate_phase2_results()`

**AI 決策選項**：

1. **SUBMIT_REPORT** - 提交報告
   - 條件: 高信心漏洞 + POC 已準備 + CVSS > 7.0
   - 行動: 生成報告，提交到 Bug Bounty 平台

2. **CONTINUE_DEEP_DIVE** - 繼續深挖
   - 條件: 發現潛在攻擊鏈 + 時間允許
   - 行動: 探索漏洞組合，提升嚴重性

3. **CHAIN_VULNERABILITIES** - 串聯漏洞
   - 條件: 多個低/中危漏洞可組合
   - 行動: 嘗試組合成高危攻擊鏈（例: XSS + CSRF = ATO）

4. **SWITCH_STRATEGY** - 切換策略
   - 條件: 當前攻擊向量無效 + WAF 持續封鎖
   - 行動: 嘗試其他攻擊方法

5. **ABANDON_TARGET** - 放棄目標
   - 條件: ROI 過低 + 疑似 honeypot + 重複風險高
   - 行動: 終止掃描，節省時間

**決策輸出範例**：
```json
{
  "action": "SUBMIT_REPORT",
  "priority": "HIGH",
  "reasoning": "發現高信心 SQL 注入，POC 已準備，預估獎金 $5000，建議立即提交報告",
  "estimated_bounty": 5000,
  "cvss_score": 7.5,
  "report_ready": true
}
```

---

## 6. Bug Bounty 優化邏輯

### 6.1 Program Scope 合規

AI 會自動檢查：
- ✅ 目標是否在 Program 範圍內
- ✅ 是否遵守測試限制（例如: 禁止自動化工具）
- ✅ 是否避開禁止測試的區域

### 6.2 WAF 規避策略

當檢測到 WAF 時，AI 會：
- 切換到隱匿模式（`stealth_level=high`）
- 降低請求速率（`rate_limit=500`）
- 使用 WAF 規避技術
- 必要時切換攻擊向量

### 6.3 重複風險評估

AI 會查詢歷史數據：
- 該漏洞類型是否已被大量報告
- 該 Program 的重複率
- 調整優先級避免浪費時間

### 6.4 時間投資回報率（ROI）

AI 會計算：
```
ROI = (預估獎金 × 成功率) / 預估時間成本

範例:
- SQLi: ($5000 × 0.9) / 2小時 = $2250/小時 → 高優先級
- XSS: ($500 × 0.7) / 1小時 = $350/小時 → 中優先級
```

---

## 📚 系列手冊導覽

| 冊別 | 主題 | 連結 |
|------|------|------|
| 第1冊 | 系統入門與架構 | [前往](使用者手冊_第1冊_系統入門與架構.md) |
| **第2冊** | AI 決策流程（本冊） | - |
| 第3冊 | 執行與適應 | [前往](使用者手冊_第3冊_執行與適應.md) |
| 第4冊 | 功能模組操作 | [前往](使用者手冊_第4冊_功能模組操作.md) |
| 第5冊 | 進階開發 | [前往](使用者手冊_第5冊_進階開發.md) |

---

## 📝 附錄

### A. 常見問題

**Q1: AI 決策會不會出錯？**

A: AI 決策基於 5M 參數神經網路和 HackerOne/Bugcrowd 實戰經驗，但仍可能出錯。系統設計了多層驗證機制，並會在不確定時降低信心度。

**Q2: 我可以覆蓋 AI 決策嗎？**

A: 可以。使用 `--param` 參數手動指定策略，例如:
```bash
aiva scan https://example.com --param scan_strategy=aggressive
```

**Q3: AI 決策需要多長時間？**

A: 每個決策點通常在 1-3 秒內完成。整個三階段流程取決於目標複雜度，通常 10-30 分鐘。

---

**更新日期**: 2026-02-01  
**維護者**: AIVA Development Team
