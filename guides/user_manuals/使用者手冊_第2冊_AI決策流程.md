# 📘 AIVA 使用者手冊 - 第 2 冊：AI 決策流程

> **版本**: v3.2  
> **最後更新**: 2026-02-05  
> **適用對象**: 安全研究員、Bug Hunter、滲透測試人員  
> **閱讀時間**: 約 20 分鐘

---

## ⚠️ 重要更新提示（v2.0+）

**如果您使用 AIVA v2.0 或更新版本，策略系統已進行重大更新**：

- ❌ **已移除**: `AIVA_ENV`、`scan_type`、`mode` 等舊參數
- ✅ **新增**: `target_sensitivity` (0.0-1.0) 和 4 維度策略模型

**請先閱讀**: [第 2-1 冊：策略系統更新指南](使用者手冊_第2-1冊_策略系統更新指南.md)

> 💡 **提示**: 本冊部分範例代碼使用舊參數（如 `scan_type: "aggressive"`），僅供理解決策邏輯。實際使用請參考第 2-1 冊的新參數。

---

## 📑 目錄

### 第一部分：用戶視角（必讀）
1. [快速開始：一分鐘發起掃描](#1-快速開始一分鐘發起掃描)
2. [API 與 Flow 命令使用指南](#2-api-與-flow-命令使用指南)
3. [掃描結果解讀](#3-掃描結果解讀)

### 第二部分：內部運作（可選閱讀）
4. [AI 決策流程概述](#4-ai-決策流程概述)
5. [三階段決策詳解](#5-三階段決策詳解)
6. [Bug Bounty 優化邏輯](#6-bug-bounty-優化邏輯)

---

# 第一部分：用戶視角

## 1. 快速開始：一分鐘發起掃描

### 方式一：互動式選單（推薦新手）

```powershell
# 進入專案目錄
cd C:\D\fold7\AIVA-git\services\core

# 啟動互動式選單
python -m aiva_core.internal_exploration.aiva_internal_executor --menu
```

**選單功能**：
- 列出所有可用的 Flow（數據流）
- 選擇並執行特定 Flow
- 查看 Flow 詳細資訊

### 方式二：Flow 執行器（推薦熟練用戶）

```powershell
# 1. 進入專案目錄
cd C:\D\fold7\AIVA-git\services\core

# 2. 執行特定 Flow（以 Flow 8 為例）
python -m aiva_core.internal_exploration.aiva_internal_executor --flow 8

# 3. Dry Run 模式（預覽不執行）

# 4. 列出所有 Flow
python -m aiva_core.internal_exploration.aiva_internal_executor --list

# 5. 啟動互動式選單（可用於瀏覽和搜尋能力）
python -m aiva_core.internal_exploration.aiva_internal_executor --menu
```

### 方式三：HTTP API（推薦自動化）

> ⚠️ 需先啟動 API 服務（參見第1冊）

```powershell
# HTTP API 模式
curl -X POST http://localhost:8000/scan `
  -H "Content-Type: application/json" `
  -d '{
    "target": "https://example.com",
    "scan_type": "full",
    "max_depth": 3
  }'

# CLI 直接掃描模式（最簡單）
python services\core\aiva_core\service_backbone\api\app.py --target https://example.com
```

---

## 2. API 與 Flow 命令使用指南

### 2.1 HTTP API 端點

| 端點 | 方法 | 說明 | 範例 |
|------|------|------|------|
| `/health` | GET | 系統健康檢查 | `curl http://localhost:8000/health` |
| `/status/{scan_id}` | GET | 查詢掃描狀態 | `curl http://localhost:8000/status/scan_abc123` |
| `/scan` | POST | 啟動掃描 | 見上方範例 |

### 2.2 Flow 執行器參數

```powershell
# 基本執行
python -m aiva_core.internal_exploration.aiva_internal_executor --flow <ID>

# 可用參數
--flow <ID>           # 執行指定 Flow
--list                # 列出前 20 個可用的 Flow
--menu                # 啟動互動式選單（按模組和能力瀏覽）
--generate-doc {md,json}  # 生成 CLI 參考文件
--data <路徑>         # 指定分類數據檔案路徑
```

### 2.3 常用 Flow 範例

```powershell
# 進入工作目錄
cd C:\D\fold7\AIVA-git\services\core

# 列出所有可用 Flow
python -m aiva_core.internal_exploration.aiva_internal_executor --list

# 啟動互動式選單（推薦）
python -m aiva_core.internal_exploration.aiva_internal_executor --menu

# 執行特定 Flow（如 Flow 8）
python -m aiva_core.internal_exploration.aiva_internal_executor --flow 8

# Dry Run 模式預覽

# 生成 CLI 參考文件（Markdown 格式）
python -m aiva_core.internal_exploration.aiva_internal_executor --generate-doc md
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

### 4.0 核心模組檔案一覽

AI 決策流程由以下核心檔案實現：

| 檔案 | 位置 | 職責 |
|------|------|------|
| `ai_decision_core.py` | `cognitive_core/` | 決策核心邏輯，負責三階段決策判斷 |
| `enhanced_decision_agent.py` | `cognitive_core/decision/` | 增強決策代理，整合多種決策方法 |
| `adaptive_weight_manager.py` | `cognitive_core/decision/` | 動態權重管理，根據實戰結果調整決策權重 |
| `anti_hallucination_module.py` | `cognitive_core/anti_hallucination/` | 防幻覺模組，確保 AI 決策基於真實數據 |

#### 核心模組調用關係

```
用戶請求
    ↓
ai_decision_core.py          ← 決策入口
    ├── enhanced_decision_agent.py   ← 增強決策
    │       └── adaptive_weight_manager.py  ← 權重調整
    └── anti_hallucination_module.py ← 結果驗證
    ↓
決策輸出 (JSON)
```

#### 關鍵方法說明

| 模組 | 方法 | 說明 |
|------|------|------|
| `ai_decision_core` | `decide_phase1_scan()` | Phase 0→1 決策 |
| `ai_decision_core` | `decide_phase2_targets()` | Phase 1→2 決策 |
| `ai_decision_core` | `decide_submit_or_continue()` | Phase 2 後決策 |
| `enhanced_decision_agent` | `make_decision()` | 整合決策入口 |
| `adaptive_weight_manager` | `update_weights()` | 根據結果更新權重 |
| `anti_hallucination_module` | `validate_decision()` | 驗證決策可靠性 |

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

**黑盒測試策略**: 因為不知道目標情況，所以規劃多路並行探查

```
Step 0: 接收用戶輸入
  ↓
Step 1-2: 解析目標，創建任務計劃
  ↓
Step 3: 規劃多路並行探查任務（黑盒策略）
  - 規劃 HTTP 基礎探測
  - 規劃端口掃描任務
  - 規劃 WAF 檢測任務
  - 規劃技術棧指紋識別
  - 規劃目錄發現任務
  （5+ 個並行任務）
  ↓
Step 4: 執行並行探查（Rust 引擎高性能執行）
  - 所有任務同時執行
  - 收集多路探查結果
  ↓
Step 5: 整合多路探查結果
  - 合併端口信息
  - 合併技術棧信息
  - 統一情報視圖
```

**多路探查輸出範例**：
```json
{
  "open_ports": [80, 443, 3306],
  "technologies": ["PHP", "Laravel", "MySQL", "Express.js"],
  "waf_detected": true,
  "waf_vendor": "Cloudflare",
  "directories_found": ["/admin", "/api", "/upload"],
  "http_info": {
    "status_code": 200,
    "headers": {"server": "nginx", "x-powered-by": "Express"}
  },
  "tasks_executed": 5,
  "successful_probes": 4
}
```

**關鍵點**:
- ✅ **並行執行**: 5 個探查任務同時進行，節省時間
- ✅ **互補性**: HTTP 探測 + 端口掃描 + WAF 檢測，多角度了解目標
- ✅ **容錯性**: 即使某個探查失敗，其他探查仍可提供情報

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
| 第2-2冊 | 13 步驟黑盒測試架構詳解 | [前往](使用者手冊_第2-2冊_13步驟黑盒測試架構詳解.md) |
| 第3冊 | 執行與適應 | [前往](使用者手冊_第3冊_執行與適應.md) |
| 第4冊 | 功能模組操作 | [前往](使用者手冊_第4冊_功能模組操作.md) |
| 第5冊 | 數據流分析與執行器 | [前往](使用者手冊_第5冊_數據流分析與執行器.md) |
| 第6冊 | 進階開發 | [前往](使用者手冊_第6冊_進階開發.md) |

---

## 📝 附錄

### A. 常見問題

**Q1: AI 決策會不會出錯？**

A: AI 決策基於 5M 參數神經網路和 HackerOne/Bugcrowd 實戰經驗，但仍可能出錯。系統設計了多層驗證機制，並會在不確定時降低信心度。

**Q2: 我可以覆蓋 AI 決策嗎？**

A: 可以。透過 HTTP API 傳入自定義參數：
```powershell
curl -X POST http://localhost:8000/scan `
  -H "Content-Type: application/json" `
  -d '{
    "target": "https://example.com",
    "scan_type": "aggressive",
    "ai_intensity": 0.9
  }'
```

或在 Flow 執行時使用 context_data 傳入自定義策略。

**Q3: AI 決策需要多長時間？**

A: 每個決策點通常在 1-3 秒內完成。整個三階段流程取決於目標複雜度，通常 10-30 分鐘。

---

**更新日期**: 2026-02-02  
**維護者**: AIVA Development Team
