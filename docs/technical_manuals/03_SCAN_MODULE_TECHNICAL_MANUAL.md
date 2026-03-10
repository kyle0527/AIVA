# AIVA Scan 模組技術手冊

**版本**: v3.1 | **狀態**: ✅ Production Ready | **路徑**: `services/scan/`

---

## 目錄

1. [模組概述](#1-模組概述)
2. [四大引擎架構](#2-四大引擎架構)
3. [各引擎技術細節](#3-各引擎技術細節)
   - 3.1 [Go Engine — 高速掃描](#31-go-engine--高速掃描)
   - 3.2 [Rust Engine — 高精度攻擊檢測](#32-rust-engine--高精度攻擊檢測)
   - 3.3 [TypeScript Engine — 現代 Web 應用](#33-typescript-engine--現代-web-應用)
   - 3.4 [Python Engine — 參考標準實作](#34-python-engine--參考標準實作)
4. [智能速率控制](#4-智能速率控制)
5. [引擎選擇邏輯](#5-引擎選擇邏輯)
6. [輸出格式](#6-輸出格式)
7. [完成狀態](#7-完成狀態)
   - 7.1 [已完成功能](#71-已完成功能-)
   - 7.2 [待完成 / 目標功能](#72-待完成--目標功能-)
8. [與其他模組的整合](#8-與其他模組的整合)
9. [搭配閱讀](#9-搭配閱讀)

---

## 1. 模組概述

Scan 模組是 AIVA 的多語言掃描引擎調度層，負責管理 Go、Rust、TypeScript、Python 四個獨立掃描引擎。

**架構原則**：無中央協調器——各引擎獨立運作，由 Core 模組根據目標特性選擇引擎組合。

**OWASP WSTG 合規**：所有引擎遵循 OWASP Web Security Testing Guide 標準。

---

## 2. 四大引擎架構

```
┌──────────────────────────────────────────────────────┐
│                    Scan Module                       │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │
│  │ Go Engine  │  │ Rust Engine│  │  TS Engine     │  │
│  │ Fast SSRF  │  │ HTTP Smug  │  │  DOM XSS       │  │
│  │ SCA / CSPM │  │ Auth Brute │  │  SPA Routing   │  │
│  └────────────┘  └────────────┘  │  WebSocket     │  │
│  ⚠️ 需編譯      ✅ 生產就緒      └────────────────┘  │
│                                  ✅ 生產就緒          │
│  ┌──────────────────────────────────────────────┐    │
│  │          Python Engine（Reference Standard）  │    │
│  │   XXE / Deserialization / Passive Analysis   │    │
│  │                    95% 完成                   │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## 3. 各引擎技術細節

### 3.1 Go Engine — 高速掃描

**狀態**：⚠️ 功能完整，需編譯後使用

**適用場景**：大規模快速掃描

| 功能 | 說明 |
|---|---|
| SSRF 掃描 | 伺服器端請求偽造快速探測 |
| SCA（軟體成分分析）| 第三方依賴漏洞識別 |
| CSPM（雲端安全態勢）| 雲端配置錯誤檢測 |

**技術優勢**：Go 並發模型，掃描速度遠優於 Python

```bash
# 需先編譯
cd services/scan/go_engine && go build -o scan_go .
./scan_go --target https://target.com --mode ssrf --concurrency 100
```

### 3.2 Rust Engine — 高精度攻擊檢測

**狀態**：✅ 生產就緒（已增強）

**適用場景**：精確漏洞驗證，需要高效能記憶體安全

| 功能 | 說明 |
|---|---|
| HTTP Smuggling | CL.TE, TE.CL, TE.TE, Chunk encoding 四種變體 |
| Auth Brute Force | 增強版認證爆破，智能速率控制 |

**HTTP Smuggling 技術細節**：
```
CL.TE：Content-Length + Transfer-Encoding 衝突
TE.CL：Transfer-Encoding + Content-Length 衝突
TE.TE：雙 Transfer-Encoding header 混淆
Chunk：Chunked 編碼注入
```

### 3.3 TypeScript Engine — 現代 Web 應用

**狀態**：✅ 生產就緒（已增強）

**適用場景**：SPA、動態渲染、WebSocket 應用

**依賴工具**：Playwright（headless Chrome/Firefox）

| 功能 | 說明 |
|---|---|
| DOM XSS | Playwright 驅動，真實瀏覽器執行 |
| SPA Route 遍歷 | 客戶端路由安全測試 |
| WebSocket 安全 | 雙向通信安全分析 |
| PostMessage 偵測 | 跨源通信安全檢測 |
| 客戶端繞過 | 前端驗證邏輯繞過測試 |

### 3.4 Python Engine — 參考標準實作

**狀態**：✅ 95% 完成（作為其他引擎的參考標準）

| 功能 | 說明 |
|---|---|
| XXE | XML External Entity 注入 |
| Deserialization | 反序列化漏洞（Java, Python, PHP） |
| Passive Analysis | 被動流量分析，無侵入性掃描 |

---

## 4. 智能速率控制

Scan 模組實作自適應速率算法，避免觸發 WAF 或被封鎖：

```
初始速率 → 監測回應時間/狀態碼 → 動態調整
  │
  ├── 429/503 → 降速 50%，等待 backoff
  ├── 正常回應 → 逐步提升速率
  └── 超時增加 → 降速 30%
```

---

## 5. 引擎選擇邏輯

由 Core 模組的 `decide_scan_strategy()` 根據目標特性自動選擇：

```python
# 決策邏輯示意
if target.is_spa:
    engines = ["typescript", "python"]
if target.has_api:
    engines += ["go", "rust"]
if target.has_websocket:
    engines += ["typescript"]
if target.is_large_scope:
    engines += ["go"]  # 高並發掃描
```

---

## 6. 輸出格式

各引擎統一輸出 JSON 格式，符合 aiva_common schemas：

```json
{
  "engine": "rust",
  "scan_type": "http_smuggling",
  "target": "https://target.com",
  "vulnerability_found": true,
  "technique": "CL.TE",
  "severity": "HIGH",
  "cvss_score": 7.5,
  "evidence": { ... }
}
```

---

## 7. 完成狀態

### 7.1 已完成功能 ✅

| 功能 | 引擎 | 說明 |
|---|---|---|
| HTTP Smuggling（CL.TE, TE.CL, TE.TE）| Rust | 增強版，已測試 |
| 自適應認證爆破 | Rust | 智能速率控制 |
| DOM XSS 偵測 | TypeScript | Playwright 真實瀏覽器 |
| SPA 路由遍歷 | TypeScript | 客戶端安全測試 |
| WebSocket 安全分析 | TypeScript | PostMessage 偵測 |
| XXE 注入偵測 | Python | 95% 完成 |
| 反序列化漏洞 | Python | Java/Python/PHP 覆蓋 |
| SSRF 快速掃描 | Go | 功能完整，需編譯 |
| SCA 軟體成分分析 | Go | 功能完整，需編譯 |
| OWASP WSTG 合規 | 全引擎 | 測試標準對齊 |

### 7.2 待完成 / 目標功能 🎯

| 功能 | 優先級 | 說明 |
|---|---|---|
| Go Engine CI/CD 自動編譯 | P1 | 整合到建置流程，無需手動編譯 |
| Rust Engine CI/CD 整合 | P1 | 同上 |
| Browser 依賴自動安裝 | P1 | TypeScript Engine Playwright 瀏覽器自動設定 |
| Python Engine 剩餘 5% | P1 | 補齊被動分析邊緣案例 |
| CORS 安全測試引擎 | P2 | 新增跨域資源共享配置錯誤掃描 |
| GraphQL 安全掃描 | P2 | GraphQL introspection、注入、越權 |
| gRPC 協定掃描 | P2 | gRPC 服務的安全測試 |
| Cloud API 掃描擴展 | P2 | AWS/GCP/Azure API 特化掃描 |
| 分散式掃描協調 | P3 | 多節點並行掃描，超大規模目標 |
| 掃描結果去重算法 | P3 | 多引擎同時掃描時避免重複漏洞報告 |
| 自動速率校準 | P3 | 根據目標歷史行為自動設定初始速率 |

---

## 8. 與其他模組的整合

| 模組 | 關係 |
|---|---|
| `core/` | 接收指令，由 Core 調度 |
| `features/` | 平行執行（非上下游關係） |
| `integration/` | 掃描結果傳送 |
| `aiva_common/` | 共用 schemas 和 enums |

---

## 9. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第5冊_數據流分析與執行器.md`
- **技術手冊**：`docs/technical_manuals/01_CORE_MODULE_TECHNICAL_MANUAL.md`（調度方）
