# AIVA 掃描工作流程與資料流

## 📑 目錄

- [📋 完整掃描流程](#完整掃描流程)
  - [Phase 0: Rust 快速偵察 (必執行)](#phase-0-rust-快速偵察-必執行)
  - [AI 決策: 核心模組分析](#ai-決策-核心模組分析)
  - [Phase 1: 多引擎深度掃描](#phase-1-多引擎深度掃描)
    - [Python 引擎 (爬蟲 + 表單提取)](#python-引擎-爬蟲-表單提取)
    - [TypeScript 引擎 (動態渲染)](#typescript-引擎-動態渲染)
  - [AI 再次決策: 是否繼續](#ai-再次決策-是否繼續)
  - [Phase 2: Go 引擎 (SSRF 專項測試) - 條件性執行](#phase-2-go-引擎-ssrf-專項測試-條件性執行)
- [🔄 資料流總結](#資料流總結)
- [📊 各引擎提供的資訊對比](#各引擎提供的資訊對比)
- [🎯 關鍵發現與修正](#關鍵發現與修正)
  - [問題: Go 引擎無法獨立工作](#問題-go-引擎無法獨立工作)
  - [當前狀態](#當前狀態)
- [💡 最佳實踐](#最佳實踐)

---

## 📋 完整掃描流程

### Phase 0: Rust 快速偵察 (必執行)
**目的**: 快速探測攻擊面,為後續決策提供基礎資訊

**輸入**:
```json
{
  "url": "http://target.com",
  "mode": "fast",
  "timeout": 10
}
```

**輸出** (Rust Engine):
```json
{
  "mode": "FastDiscovery",
  "targets": [{
    "url": "http://target.com",
    "success": true,
    "endpoints": [
      {
        "path": "/api/users",
        "method": "GET",
        "status_code": 401,
        "risk_level": "high"
      }
    ],
    "js_findings": [
      "ApiEndpoint: /api/Products",
      "ApiEndpoint: /api/SecurityQuestions"
    ],
    "technologies": ["Express.js", "Node.js"],
    "sensitive_info": []
  }],
  "summary": {
    "total_endpoints": 40,
    "total_sensitive_info": 0
  }
}
```

**Phase 0 提供的資訊**:
- ✅ **端點列表**: 40+ 個常見路徑 (如 `/api/users`, `/admin`, `/graphql`)
- ✅ **技術棧**: Express.js, Node.js, Angular 等
- ✅ **風險等級**: high/medium/low/info
- ✅ **JS Findings**: JS 文件中發現的 API 端點
- ❌ **無參數詳情**: 只有路徑,沒有參數名稱和類型
- ❌ **無表單結構**: 沒有表單字段信息

---

### AI 決策: 核心模組分析

**輸入**: Phase 0 結果 + 歷史數據 + RAG 搜索

**核心模組 (AI) 分析**:
1. **技術棧判斷**: Express.js + Node.js → 選擇 Node.js 相關 Payload
2. **端點風險評估**: 9 個 high-risk 端點 → 優先測試
3. **引擎選擇策略**:
   - 發現 `/api/*` 端點多 → 建議使用 **Python 引擎**爬取完整資訊
   - 發現 SPA 技術 (Angular) → 建議使用 **TypeScript 引擎**動態渲染
   - 發現高風險 API → 暫不使用 Go 引擎 (需要完整參數)

**整合模組協助**:
- 查詢資料庫: 類似目標的歷史掃描記錄
- 對比數據: 該技術棧常見的漏洞類型
- 補充建議: 如果遇到未知情況,AI 使用 RAG 搜索網路

**AI 決策輸出**:
```json
{
  "selected_engines": ["python", "typescript"],
  "recommended_strategy": "deep",
  "priority_endpoints": ["/api/users", "/api/config"],
  "stop_condition": "found_confirmed_vulnerability",
  "max_depth": 3
}
```

---

### Phase 1: 多引擎深度掃描

#### Python 引擎 (爬蟲 + 表單提取)

**輸入** (來自協調器):
```python
{
  "scan_id": "scan_001",
  "strategy": "deep",      # AI 決定
  "max_depth": 3,          # AI 決定
  "max_pages": 100,        # AI 決定
  "timeout": 10
}
```

**Python 引擎執行**:
1. 爬取 `http://target.com`
2. 提取所有鏈接和表單
3. 分析每個表單的字段和參數

**Python 引擎輸出** (Asset 格式):
```python
[
  {
    "asset_id": "asset_001",
    "type": "url",
    "value": "http://target.com/api/users",
    "parameters": ["id", "name", "email"],  # ✅ 參數列表
    "has_form": False
  },
  {
    "asset_id": "asset_002",
    "type": "url",
    "value": "http://target.com/login",
    "parameters": ["username", "password"],
    "has_form": True  # ✅ 有表單
  }
]
```

**Python 引擎提供的關鍵資訊**:
- ✅ **完整 URL**: 包含路徑和可能的參數
- ✅ **參數名稱**: `["id", "name", "email"]`
- ✅ **表單結構**: `has_form=True` 表示有表單
- ✅ **表單字段**: 可以提取 `<input name="username">` 等

---

#### TypeScript 引擎 (動態渲染)

**適用場景**: SPA (React/Vue/Angular)

**輸入**:
```json
{
  "scan_id": "scan_001",
  "max_depth": 3,
  "timeout": 10
}
```

**TypeScript 引擎執行**:
1. 使用 Playwright 渲染頁面
2. 執行 JavaScript,獲取動態生成的內容
3. 提取 AJAX 請求和動態路由

**TypeScript 引擎輸出**:
```python
[
  {
    "asset_id": "asset_003",
    "type": "api",
    "value": "http://target.com/api/Products",
    "parameters": [],
    "has_form": False
  }
]
```

---

### AI 再次決策: 是否繼續

**條件判斷**:
1. **已確認漏洞**: 如果 Python 或 TypeScript 發現明確漏洞 → **停止**
2. **深層探測**: 如果需要測試 SSRF/CSRF → 繼續使用 **Go 引擎**
3. **資訊不足**: 如果參數不明確 → 使用 **RAG 搜索**建議

**AI 第 2 次決策: 是否進入 Phase 2**

**決策邏輯**:
```python
# 評估資產質量與攻擊價值
assessment = self._assess_assets(phase1_result.assets)

if assessment.has_attack_surface:
    # ✅ 有可攻擊資產 → 進入 Phase 2
    return Phase2Decision(
        continue_to_phase2=True,
        selected_modules=["function_ssrf", "function_sqli", "function_xss"],
        reason="發現 50 個表單資產 + 80 個 API 端點"
    )
else:
    # ⚠️ 無有效攻擊面 → 跳過 Phase 2，直接進入 Integration
    return Phase2Decision(
        continue_to_phase2=False,
        selected_modules=[],
        reason="僅發現靜態頁面，無表單或 API 參數"
    )

# ⚠️ 重要: 無論是否進入 Phase 2，最終都會進入 Integration 產生報告
# Integration 會進行資料庫比對、歷史分析，並產出完整報告
```

---

### Phase 2: Go 引擎 (SSRF 專項測試) - 條件性執行

**前置條件** (由 Python 引擎提供):
```python
{
  "asset_id": "asset_001",
  "type": "url",
  "value": "http://target.com/api/fetch",
  "parameters": ["url", "callback"],  # ✅ Python 發現的參數
  "has_form": False
}
```

**Go 引擎輸入** (協調器轉換):
```json
{
  "scan_id": "scan_002",
  "targets": [
    "http://target.com/api/fetch?url=",
    "http://target.com/api/fetch?callback="
  ],
  "concurrency": 5,
  "timeout": 10
}
```

**Go 引擎執行**:
1. 接收**完整的帶參數 URL**
2. 測試 SSRF payload: `?url=file:///etc/passwd`
3. 驗證響應內容,確認是否真的執行了 SSRF

**Go 引擎輸出**:
```json
{
  "assets": [{
    "type": "web_vulnerability",
    "name": "SSRF - File Protocol",
    "severity": "high",
    "confidence": "high",
    "details": {
      "affected_url": "http://target.com/api/fetch?url=file:///etc/passwd",
      "vulnerable_param": "url",
      "response_preview": "root:x:0:0:root:/root:/bin/bash..."
    }
  }]
}
```

---

## 🔄 資料流總結

```
Phase 0 (Rust)
   ↓
  [端點列表, 技術棧, JS Findings]
   ↓
AI 核心模組 (分析)
   ↓
  [選擇引擎: python, typescript]
   ↓
Phase 1 (Python + TypeScript)
   ↓
  [完整 URL, 參數名稱, 表單結構]
   ↓
AI 再次決策
   ↓
  判斷: 是否需要深層測試?
   ├─ YES → Phase 2 (Go)
   │   ↓
   │  [SSRF 漏洞確認]
   │   ↓
   └─ NO → 整合模組產生報告
       ↓
      [最終報告]
```

---

## 📊 各引擎提供的資訊對比

| 引擎 | 端點路徑 | 參數名稱 | 表單結構 | 漏洞驗證 | 執行時間 |
|------|---------|---------|---------|---------|---------|
| **Rust** | ✅ 40+ | ❌ | ❌ | ❌ | ~200ms |
| **Python** | ✅ 完整 | ✅ 提取 | ✅ 分析 | ⚠️ 部分 | ~30s |
| **TypeScript** | ✅ 動態 | ⚠️ 部分 | ⚠️ 部分 | ❌ | ~45s |
| **Go** | ✅ 接收 | ✅ 需要 | ❌ | ✅ SSRF | ~5s |

---

## 🎯 關鍵發現與修正

### 問題: Go 引擎無法獨立工作

**原因**:
- Go 引擎設計用於**專項 SSRF 測試**
- 需要**完整的帶參數 URL**: `http://target.com/api?url=xxx`
- Rust Phase 0 只提供**路徑**: `/api/users`

**修正方案**:
1. ✅ **Rust Phase 0**: 提供端點列表 (已實現)
2. ✅ **Python Engine**: 爬取並提取參數 (已實現)
3. ✅ **AI 決策**: 根據資訊決定是否使用 Go (需實現)
4. ✅ **協調器轉換**: 將 Python 資產轉換為 Go 輸入 (需實現)

### 當前狀態

**協調器修復**:
- ✅ 修正參數轉換邏輯
- ✅ 為每個引擎準備正確格式
- ✅ Go 引擎只收到 4 個字段 (符合使用指南)

**Go 引擎測試**:
- ✅ 成功發送 18 個請求到 Juice Shop
- ❌ 但都是盲目測試 (無效參數)
- ⚠️ 需要 Python 先提取真實參數

**下一步**:
1. 測試 Python 引擎輸出,確認能否提取參數
2. 實現協調器的資產轉換邏輯
3. 完整測試: Rust → Python → Go 流程
4. 實現 AI 決策邏輯 (停止條件)

---

## 💡 最佳實踐

1. **總是先執行 Rust Phase 0**: 快速了解攻擊面
2. **Python 引擎用於資訊收集**: 提取參數和表單
3. **Go 引擎用於專項測試**: 只在有明確參數時使用
4. **AI 決策停止條件**: 發現確認漏洞即停止
5. **整合模組產生報告**: 所有階段完成後統一輸出
