# AIVA AI 智能安全平台使用者指南

> **📖 閱讀對象**: 滲透測試人員、安全研究員、Bug Bounty獵人、企業安全團隊  
> **🎯 使用場景**: 基於統一Schema的智能化安全測試、AI驅動漏洞發現、標準化報告生成  
> **⏱️ 預計閱讀時間**: 15 分鐘  
> **🚀 版本**: v5.0 跨語言統一版本 (2025.10.28)  
> **� 核心突破**: 100% Schema標準化 + AI智能協同 + 企業級數據治理

---

## 📚 使用者指南目錄

1. [🚀 3分鐘快速上手](#-3分鐘快速上手)
2. [⚡ CLI智能指令系統](#-cli智能指令系統)
3. [🎯 六大掃描策略詳解](#-六大掃描策略詳解)
4. [📊 實戰案例與結果解讀](#-實戰案例與結果解讀)
5. [🔧 進階使用技巧](#-進階使用技巧)
6. [⚠️ 使用注意事項與限制](#️-使用注意事項與限制)
7. [🆘 常見問題與解決方案](#-常見問題與解決方案)
8. [� 最佳實踐指南](#-最佳實踐指南)

---

## 🚀 3分鐘快速上手

### 🎯 超簡單！三步驟開始使用AIVA

#### **Step 1: 環境準備** (1分鐘)

```bash
# 1. 確保您已在AIVA專案目錄中
cd /path/to/AIVA-git

# 2. 檢查環境是否準備就緒 (可選)
python health_check.py

# 3. 啟動靶場環境 (OWASP Juice Shop)
# 注意：靶場由您自行啟動，AIVA不會自動啟動
docker run -d -p 3000:3000 bkimminich/juice-shop
```

#### **Step 2: 選擇您的第一個指令** (30秒)

**新手推薦指令**:
```bash
# 快速掃描 - 最適合第一次使用
python core_scan_integration_cli.py quick-scan http://localhost:3000

# 如果您想看詳細的JSON結果
python core_scan_integration_cli.py quick-scan http://localhost:3000 --output json
```

**經驗用戶指令**:
```bash
# 深度綜合掃描
python core_scan_integration_cli.py deep-scan https://target.com --comprehensive

# 隱匿情報收集
python core_scan_integration_cli.py intel https://target.com --stealth
```

#### **Step 3: 查看 AI 執行結果**

```python
# 結果包含以下關鍵資訊:
print(result)
# {
#     'status': 'success',
#     'mode': 'hybrid',
#     'decision': {
#         'attack_vector': 'reflected_xss',
#         'confidence': 0.87,
#         'requires_approval': True  # 混合模式下高風險操作需確認
#     },
#     'plan': {
#         'steps': [...],  # 攻擊計畫步驟
#         'estimated_time': '5-10 minutes',
#         'risk_level': 'medium'
#     },
#     'results': [...],  # 執行結果
#     'ai_summary': '檢測到 3 個 XSS 注入點...'
# }
```

#### **Step 3: 執行您的第一次掃描** (1分鐘)

**基本使用流程**:

1. **設定目標** (必填):
   ```bash
   python core_scan_integration_cli.py quick-scan [您的目標URL]
   ```

2. **可選參數設定**:
   ```bash
   # 指定輸出格式
   --output [text|json|markdown|xml]
   
   # 增加詳細度
   --verbose
   
   # 全面掃描模式
   --comprehensive
   
   # 隱匿模式
   --stealth
   ```

3. **實際執行範例**:
   ```bash
   # 新手範例
   python core_scan_integration_cli.py quick-scan http://testfire.net
   
   # 進階範例
   python core_scan_integration_cli.py deep-scan https://target.com --output json --verbose
   ```

**執行時間預估**:
- quick-scan: ~1.7秒
- deep-scan: ~3.7秒
- intel/discovery: ~2.5秒

---

## 🧠 AI 核心能力概覽

### AIVA 的 AI 大腦如何工作？

```
┌──────────────────────────────────────────────────┐
│          🧠 BioNeuron AI 決策引擎                 │
│  - 500萬參數生物神經網路                          │
│  - RAG 知識增強 (7種知識類型)                     │
│  - 反幻覺保護機制                                 │
└─────────────┬────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌──▼───┐ ┌──▼────┐
│知識檢索│ │決策樹│ │執行器 │
│RAG引擎│ │推理  │ │調度   │
└───────┘ └──────┘ └───────┘
```

### **三層智能決策系統**

#### **Layer 1: BioNeuronMasterController** (主控制器)
- **職責**: 接收使用者請求,選擇操作模式,協調 AI 組件
- **使用者感知**: 這是您直接互動的入口
- **典型操作**: 模式切換、任務路由、風險評估

#### **Layer 2: BioNeuronRAGAgent** (核心 AI 大腦)
- **職責**: AI 決策推理、知識增強、策略生成
- **使用者感知**: 背後的智能決策引擎
- **核心能力**:
  - 📚 RAG 知識檢索 (自動搜尋相關漏洞知識庫)
  - 🧮 反幻覺驗證 (避免 AI 產生錯誤決策)
  - 🎯 攻擊計畫生成 (基於 500萬參數神經網路)

#### **Layer 3: AICommander** (多 AI 協調器)
- **職責**: 管理多語言 AI 組件 (Python/Go/Rust/TypeScript)
- **使用者感知**: 自動選擇最適合的工具執行任務
- **典型場景**:
  - 🐍 Python AI: 業務邏輯分析、漏洞推理
  - 🚀 Go AI: 高性能網路掃描
  - 🦀 Rust AI: 安全分析、漏洞驗證
  - 📘 TypeScript AI: Web 前端漏洞檢測

### **AI 核心能力矩陣**

| 能力 | 說明 | 實戰應用 |
|------|------|---------|
| 🎯 **智能攻擊編排** | 根據目標特徵自動生成攻擊策略 | SQL注入、XSS、IDOR 等漏洞自動化測試 |
| 📚 **知識增強檢索 (RAG)** | 從 7種知識庫檢索相關案例 | 查找類似漏洞的歷史利用方式 |
| 🛡️ **反幻覺保護** | 驗證 AI 決策的可靠性 | 避免執行危險或無效的攻擊 |
| 🧠 **持續學習** | 從每次測試中學習優化 | 成功的攻擊策略會被記錄並複用 |
| 🌐 **多語言協調** | 整合 Python/Go/Rust/TS 工具 | 自動選擇最佳工具執行任務 |
| 📊 **風險評估** | 評估攻擊風險並要求確認 | 防止誤操作生產環境 |
| 💬 **自然語言理解** | 理解口語化的測試指令 | 說「找 XSS」而不是寫複雜指令 |

---

## 🎯 六種指令完整使用指南

### **CLI指令系統架構**

```
CLI Command System v5.0
├── quick-scan    │ 快速掃描 (1.7s)
├── deep-scan     │ 深度掃描 (3.7s) 
├── intel         │ 情報收集 (2.5s)
├── discovery     │ 服務發現 (2.8s)
├── vuln          │ 漏洞掃描 (3.2s)
└── audit         │ 全面稽核 (3.6s)
```

### **指令分類與使用場景**

#### 🚀 **基礎掃描指令**

##### 1. `quick-scan` - 快速掃描
```bash
# 基本語法
python core_scan_integration_cli.py quick-scan [目標URL]

# 範例
python core_scan_integration_cli.py quick-scan http://localhost:3000
python core_scan_integration_cli.py quick-scan https://example.com --output json
```

**適用場景:**
- ✅ 第一次掃描目標
- ✅ Bug Bounty 初步偵察
- ✅ 快速檢查網站狀態
- ✅ 預算有限的時間內進行基本檢測

**執行時間:** ~1.65秒

##### 2. `deep-scan` - 深度掃描
```bash
# 基本語法  
python core_scan_integration_cli.py deep-scan [目標URL] [選項]

# 範例
python core_scan_integration_cli.py deep-scan https://target.com --comprehensive
python core_scan_integration_cli.py deep-scan https://example.com --stealth --output markdown
```

**適用場景:**
- ✅ 全面漏洞評估
- ✅ 紅隊演練
- ✅ 企業安全稽核
- ✅ 高價值目標深度分析

**執行時間:** ~3.69秒

#### 🕵️ **情報收集指令**

##### 3. `intel` - 隱匿情報收集
```bash
# 基本語法
python core_scan_integration_cli.py intel [目標URL] [選項]

# 範例  
python core_scan_integration_cli.py intel https://target.com --stealth
python core_scan_integration_cli.py intel https://example.com --output json --verbose
```

**適用場景:**
- ✅ 被動信息收集
- ✅ OSINT 調查
- ✅ 隱匿模式偵察
- ✅ 法規合規的初步調查

**執行時間:** ~2.53秒

##### 4. `discovery` - 服務發現
```bash
# 基本語法
python core_scan_integration_cli.py discovery [目標URL] [選項]

# 範例
python core_scan_integration_cli.py discovery https://target.com
python core_scan_integration_cli.py discovery https://example.com --comprehensive --output xml
```

**適用場景:**
- ✅ 網路拓撲分析
- ✅ 服務埠掃描
- ✅ 技術棧識別
- ✅ 攻擊面分析

**執行時間:** ~2.87秒

#### 🔍 **專業評估指令**

##### 5. `vuln` - 漏洞掃描
```bash
# 基本語法
python core_scan_integration_cli.py vuln [目標URL] [選項]

# 範例
python core_scan_integration_cli.py vuln https://target.com --comprehensive
python core_scan_integration_cli.py vuln https://example.com --stealth --output json
```

**適用場景:**
- ✅ 專業漏洞評估
- ✅ CVE 檢測
- ✅ 配置錯誤發現
- ✅ 合規性檢查

**執行時間:** ~3.21秒

##### 6. `audit` - 全面稽核
```bash
# 基本語法
python core_scan_integration_cli.py audit [目標URL] [選項]

# 範例
python core_scan_integration_cli.py audit https://target.com --comprehensive --verbose
python core_scan_integration_cli.py audit https://example.com --output markdown
```

**適用場景:**
- ✅ 企業安全審計
- ✅ 合規性報告
- ✅ 全方位安全評估
- ✅ 最終安全檢查

**執行時間:** ~3.65秒

---

## 💡 傳統模式參考 (已整合至CLI系統)

### 🎨 **CLI 輸出格式選項**

AIVA v5.0 支援多種輸出格式，滿足不同使用場景:

```bash
# 文本格式 (預設) - 適合終端閱讀
python core_scan_integration_cli.py quick-scan https://example.com

# JSON 格式 - 適合程式處理
python core_scan_integration_cli.py quick-scan https://example.com --output json

# Markdown 格式 - 適合文檔整合
python core_scan_integration_cli.py quick-scan https://example.com --output markdown

# XML 格式 - 適合企業報告
python core_scan_integration_cli.py quick-scan https://example.com --output xml
```

### 🔧 **進階選項參數**

```bash
# 詳細模式 - 顯示更多診斷信息
--verbose

# 全面模式 - 啟動深度分析
--comprehensive  

# 隱匿模式 - 降低檢測機率
--stealth

# 組合使用範例
python core_scan_integration_cli.py deep-scan https://target.com \
  --comprehensive --stealth --output json --verbose
```

---

## 🧠 AI功能模組專業檢測系統

### **AIVA Features AI-Driven CLI 系統**

除了基礎掃描指令外，AIVA v5.0 還提供專業的AI驅動功能模組檢測系統，專門針對特定漏洞類型進行深度分析。

#### **🎯 功能模組檢測指令**

```bash
# 功能模組檢測語法
python features_ai_cli.py [檢測類型] [目標URL] [AI選項]
```

##### **核心功能檢測類型**

| 檢測指令 | 功能模組 | 檢測重點 | Bug Bounty價值 |
|----------|---------|---------|---------------|
| `sqli-detect` | SQL注入檢測 | 資料庫注入漏洞 | High |
| `xss-detect` | XSS檢測 | 跨站腳本攻擊 | Medium-High |
| `ssrf-detect` | SSRF檢測 | 服務端請求偽造 | High |
| `idor-detect` | IDOR檢測 | 直接對象引用 | Medium |
| `authn-test` | 認證測試 | 身份認證繞過 | High |
| `authz-test` | 授權測試 | 授權檢查繞過 | High |
| `jwt-bypass` | JWT攻擊 | JWT混淆攻擊 | Medium-High |
| `oauth-confuse` | OAuth混淆 | OAuth流程攻擊 | High |
| `payment-bypass` | 支付繞過 | 支付邏輯漏洞 | Critical |
| `high-value-scan` | 高價值掃描 | Bug Bounty熱門 | Critical |
| `comp-features` | 全功能檢測 | 全面功能分析 | 綜合 |

##### **AI分析模式**

| AI模式 | 適用場景 | 執行策略 | 平均時間 |
|--------|---------|---------|---------|
| `intelligent` | 日常測試 | 智能學習 | 3.22s |
| `rapid` | 快速檢測 | 高速掃描 | 2.47s |
| `expert` | 深度分析 | 專家策略 | 6.16s |
| `guided` | 學習模式 | 引導探索 | 4.5s |

##### **實戰使用範例**

```bash
# SQL注入智能檢測
python features_ai_cli.py sqli-detect https://example.com --ai-mode intelligent

# 高價值漏洞專家分析
python features_ai_cli.py high-value-scan https://target.com --ai-mode expert --output json

# 全功能快速掃描
python features_ai_cli.py comp-features https://webapp.test --ai-mode rapid --output markdown

# 認證繞過隱匿測試
python features_ai_cli.py authn-test https://app.com --stealth --comprehensive
```

---

## 💡 效能與執行統計

### **基礎掃描指令執行時間**

基於 AIVA v5.0 實際測試結果:

| 指令 | 平均執行時間 | 適用場景 | 輸出項目數 |
|------|-------------|---------|-----------|
| `quick-scan` | 1.65秒 | 快速檢測 | 5-8項 |
| `deep-scan` | 3.69秒 | 全面分析 | 12-15項 |
| `intel` | 2.53秒 | 情報收集 | 8-10項 |
| `discovery` | 2.87秒 | 服務發現 | 10-12項 |
| `vuln` | 3.21秒 | 漏洞評估 | 10-14項 |
| `audit` | 3.65秒 | 全面稽核 | 15-18項 |

### **AI功能模組檢測效能**

| AI模式 | 平均執行時間 | 功能模組數 | AI信心度 | 記憶體使用 |
|--------|-------------|-----------|---------|----------|
| `rapid` | 2.47秒 | 6個 | 86.73% | ~80MB |
| `intelligent` | 3.22秒 | 3個 | 85.71% | ~95MB |
| `expert` | 6.16秒 | 4個 | 88.58% | ~120MB |
| `guided` | ~4.5秒 | 5個 | ~87.2% | ~105MB |

### **系統資源使用**

- **CPU使用率**: 10-25% (單核心)
- **記憶體消耗**: 50-120MB (基礎) / 80-120MB (AI功能)
- **網路頻寬**: 最小化 (隱匿模式 <100KB/s)
- **同時連線數**: 可調整 (預設 5個連線)
- **AI並行處理**: 支援異步執行，最多6個功能模組並行

---

### 🎯 **AI功能模組檢測深度指南**

#### **Step 1: 選擇合適的檢測類型**

根據您的目標和需求選擇最適合的功能模組：

**🔴 高價值Bug Bounty漏洞:**
```bash
# 專門針對高價值漏洞
python features_ai_cli.py high-value-scan https://target.com --ai-mode expert

# 支付邏輯漏洞 (Critical級)
python features_ai_cli.py payment-bypass https://shop.example.com --comprehensive
```

**🟡 常見Web應用漏洞:**
```bash
# SQL注入深度檢測
python features_ai_cli.py sqli-detect https://webapp.com --ai-mode intelligent

# XSS全面掃描
python features_ai_cli.py xss-detect https://site.com --output json
```

**� 認證授權漏洞:**
```bash
# 認證繞過測試
python features_ai_cli.py authn-test https://api.example.com --stealth

# JWT攻擊檢測
python features_ai_cli.py jwt-bypass https://app.com --ai-mode expert
```

#### **Step 2: AI模式選擇指南**

| 使用場景 | 推薦AI模式 | 理由 |
|---------|----------|------|
| **Bug Bounty獵人** | `expert` | 深度分析，高信心度結果 |
| **企業安全團隊** | `intelligent` | 平衡效率與準確性 |
| **CI/CD整合** | `rapid` | 快速檢測，適合自動化 |
| **學習研究** | `guided` | 提供詳細分析過程 |

#### **Step 3: 結果解讀與行動**

**AI信心度解讀:**
- **90%+**: 高可信度，建議立即處理
- **80-90%**: 較高可信度，建議手動驗證
- **70-80%**: 中等可信度，需要進一步測試
- **<70%**: 低可信度，可能為誤報

**Bug Bounty價值評估:**
- **Critical**: $5000-$15000+ (支付邏輯、業務邏輯)
- **High**: $1000-$5000 (SQL注入、SSRF、認證繞過)
- **Medium**: $300-$1000 (XSS、IDOR)

### �🖥️ **傳統 UI 模式** (適合新手學習)
- 🛡️ **最安全**: 每步都可控
- 📚 **適合學習**: 看到 AI 的決策過程
- 🎓 **逐步指導**: 理解每個攻擊步驟的意義

**缺點:**
- ⏱️ **效率較低**: 需要頻繁確認
- 👨‍💻 **需人工參與**: 無法批量自動化

**實戰案例:**
```python
# 案例: 測試銀行網站的登入頁面
controller = BioNeuronMasterController(default_mode="ui")

result = await controller.process_request(
    request="""
    測試銀行登入頁面的以下漏洞:
    1. SQL 注入 (登入繞過)
    2. 暴力破解保護
    3. 會話固定攻擊
    目標: https://bank.example.com/login
    """,
    context={
        "risk_level": "high",
        "require_approval_for": ["sql_injection", "brute_force"]
    }
)

# AI 會在執行 SQL 注入和暴力破解前暫停等待確認
# 您可以檢視生成的 Payload 後決定是否繼續
```

---

### 🤖 **模式 2: AI 模式** (完全自主,適合批量測試)

**什麼時候用?**
- ✅ 測試自己的應用程式
- ✅ 批量掃描多個目標
- ✅ 已知場景的重複測試
- ✅ 信任 AI 決策能力

**操作流程:**
```python
controller = BioNeuronMasterController(default_mode="ai")

# AI 完全自主決策和執行
result = await controller.process_request(
    request="掃描 targets.txt 中的所有目標,尋找 IDOR 漏洞",
    context={
        "targets_file": "targets.txt",
        "auto_exploit": True,  # 自動嘗試利用
        "auto_report": True    # 自動生成報告
    }
)

# AI 會自動完成以下步驟 (無需確認):
# 1. ✅ 讀取目標列表
# 2. ✅ 逐個掃描 IDOR
# 3. ✅ 發現漏洞後自動驗證
# 4. ✅ 生成 HackerOne 報告草稿
```

**優點:**
- ⚡ **效率最高**: 無需人工干預
- 🔄 **批量處理**: 可同時處理多個目標
- 🌙 **後台運行**: 設定後可離開

**缺點:**
- ⚠️ **風險較高**: AI 可能誤判或過度攻擊
- 🚫 **不適合生產**: 可能造成服務中斷
- 📉 **學習機會少**: 看不到決策過程

**實戰案例:**
```python
# 案例: 批量測試 Bug Bounty 程式中的 100 個子域名
controller = BioNeuronMasterController(default_mode="ai")

result = await controller.process_request(
    request="""
    對以下目標執行完整的漏洞掃描:
    - 目標清單: subdomains.txt (100個子域名)
    - 掃描類型: XSS, SQLi, IDOR, SSRF, Open Redirect
    - 深度: Medium (避免 DoS)
    - 輸出: HackerOne 報告格式
    """,
    context={
        "parallel_workers": 10,  # 10個並發任務
        "timeout_per_target": 600,  # 每個目標最多10分鐘
        "confidence_threshold": 0.8  # 僅報告高置信度漏洞
    }
)

# 預計執行時間: 100 targets × 10 mins / 10 workers = ~100 mins
# AI 會自動生成報告並標記最有價值的漏洞
```

---

### 💬 **模式 3: Chat 模式** (對話式探索)

**什麼時候用?**
- ✅ 探索新的攻擊思路
- ✅ 學習漏洞利用技術
- ✅ 需要 AI 協助分析複雜場景
- ✅ 與 AI 協作解決問題

**操作流程:**
```python
controller = BioNeuronMasterController(default_mode="chat")

# 自然語言對話
await controller.process_request("我發現一個登入頁面,如何測試?")
# AI: "建議從以下幾個方向測試: 1. SQL注入登入繞過 2. 暴力破解..."

await controller.process_request("SQL注入要怎麼做?")
# AI: "常見的SQL注入Payload包括: ' OR '1'='1, ..."

await controller.process_request("好,幫我測試 admin' OR '1'='1")
# AI: [執行測試並回報結果]
```

**優點:**
- 🎓 **最佳學習模式**: AI 會解釋每個步驟
- 🧠 **靈活探索**: 可隨時調整策略
- 💡 **獲得建議**: AI 提供專業建議

**缺點:**
- ⏱️ **效率中等**: 需要對話往返
- 📝 **需描述清楚**: 對話品質影響效果

**實戰案例:**
```python
# 案例: 與 AI 協作分析複雜的認證繞過場景
controller = BioNeuronMasterController(default_mode="chat")

# 第一輪對話: 描述情況
await controller.process_request("""
我在測試一個 API,發現以下行為:
1. POST /api/login 需要 username + password
2. 回傳 JWT token
3. 但我發現 GET /api/admin 不檢查 token 就能訪問
這是 IDOR 還是認證繞過?
""")
# AI: "這更像是「未授權訪問」漏洞 (Broken Access Control)..."

# 第二輪對話: 請求協助
await controller.process_request("如何證明這個漏洞的嚴重性?")
# AI: "建議測試以下端點: /api/users, /api/settings, ..."

# 第三輪對話: 執行測試
await controller.process_request("幫我測試所有 /api/* 端點")
# AI: [自動掃描並生成報告]
```

---

### 🔄 **模式 4: 混合模式** (推薦用於實際 Bug Bounty)

**什麼時候用?**
- ✅ **日常 Bug Bounty 工作** ← 最常用!
- ✅ 需要平衡效率與安全
- ✅ 信任 AI 處理常規任務
- ✅ 僅在關鍵決策時確認

**智能規則:**
- 🟢 **低風險操作** → AI 自動執行 (如: 資訊收集、端口掃描)
- 🟡 **中風險操作** → AI 執行但記錄詳情 (如: XSS 測試、開放重定向)
- 🔴 **高風險操作** → 暫停等待確認 (如: SQL注入、RCE 嘗試、DoS)

**操作流程:**
```python
controller = BioNeuronMasterController(default_mode="hybrid")

result = await controller.process_request(
    request="完整測試 https://target.com 的所有漏洞",
    context={
        "auto_approve_risk_levels": ["low", "medium"],  # 自動執行低中風險
        "require_approval_for": ["sql_injection", "rce"],  # 高危需確認
        "max_auto_requests": 1000  # 自動請求上限
    }
)

# 執行流程示例:
# 1. ✅ 子域名枚舉 (自動)
# 2. ✅ 端口掃描 (自動)
# 3. ✅ 目錄爆破 (自動)
# 4. ✅ XSS 測試 (自動)
# 5. ⏸️ 發現 SQL 注入可能性 → 暫停等待確認
# 6. [您確認後] ✅ 執行 SQL 注入測試
# 7. ✅ 生成報告 (自動)
```

**優點:**
- ⚖️ **最佳平衡**: 效率與安全兼顧
- 🎯 **實用性強**: 符合實際工作流程
- 🛡️ **風險可控**: 危險操作仍需確認

**缺點:**
- ⚙️ **需配置規則**: 初次使用需設定風險閾值

**實戰案例:**
```python
# 案例: 典型的 Bug Bounty 一天工作
controller = BioNeuronMasterController(default_mode="hybrid")

# 早上: 快速掃描多個新增目標
morning_result = await controller.process_request(
    request="掃描今日新增的 5 個目標,尋找快速勝利 (Quick Wins)",
    context={
        "targets": ["app1.example.com", "app2.example.com", ...],
        "focus_on": ["open_redirect", "cors_misconfiguration", "sensitive_data_exposure"],
        "auto_approve_risk_levels": ["low", "medium"],
        "time_limit": 3600  # 1小時完成
    }
)
# AI 會自動測試低中風險漏洞,找到立即報告

# 下午: 深入分析高價值目標
afternoon_result = await controller.process_request(
    request="深度測試 critical.example.com 的認證和授權機制",
    context={
        "depth": "deep",
        "require_approval_for": ["all"],  # 所有操作都需確認
        "generate_poc": True  # 生成 PoC
    }
)
# AI 會在每個關鍵步驟暫停,讓您決策

# 晚上: 批量驗證舊報告
evening_result = await controller.process_request(
    request="驗證上週提交的 10 個報告是否已修復",
    context={
        "reports_file": "submitted_reports.json",
        "retest_mode": True,
        "auto_update_status": True  # 自動更新報告狀態
    }
)
# AI 自動重測並更新 HackerOne 報告狀態
```

---

## 🎯 實戰場景範例

### 場景 1: XSS 漏洞挖掘

```python
from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController

controller = BioNeuronMasterController(default_mode="hybrid")

result = await controller.process_request(
    request="""
    測試 https://shop.example.com 的所有輸入點,尋找 XSS 漏洞
    重點關注:
    1. 搜尋功能
    2. 用戶評論
    3. 個人資料頁面
    """,
    context={
        "payload_types": ["reflected", "stored", "dom_based"],
        "bypass_waf": True,  # 嘗試繞過 WAF
        "generate_poc": True  # 生成 PoC
    }
)

# 典型輸出:
# {
#     'found_vulnerabilities': [
#         {
#             'type': 'reflected_xss',
#             'location': '/search?q=',
#             'payload': '<script>alert(document.domain)</script>',
#             'severity': 'medium',
#             'poc': 'https://shop.example.com/search?q=<script>...',
#             'waf_bypass': True
#         }
#     ],
#     'total_tested': 47,
#     'ai_confidence': 0.91
# }
```

### 場景 2: IDOR 批量測試

```python
result = await controller.process_request(
    request="""
    測試 API 端點的 IDOR 漏洞:
    - GET /api/users/{id}
    - GET /api/orders/{id}
    - GET /api/invoices/{id}
    使用我的認證 token: eyJhbGc...
    """,
    context={
        "auth_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "test_range": range(1, 1000),  # 測試 ID 1-1000
        "parallel_requests": 50,  # 50個並發
        "detect_horizontal": True,  # 橫向越權
        "detect_vertical": True     # 縱向越權
    }
)

# AI 會自動:
# 1. 測試您的合法 ID (如 ID=123)
# 2. 嘗試訪問其他用戶 ID (如 ID=124, 125, ...)
# 3. 比對回應差異,判斷是否存在 IDOR
# 4. 生成詳細報告
```

### 場景 3: SQL 注入深度測試

```python
result = await controller.process_request(
    request="""
    深度測試登入頁面的 SQL 注入:
    目標: https://app.example.com/login
    參數: username, password
    """,
    context={
        "injection_points": ["username", "password"],
        "techniques": [
            "error_based",
            "boolean_based",
            "time_based",
            "union_based"
        ],
        "dbms_fingerprint": True,  # 識別資料庫類型
        "extract_data": ["users", "passwords"],  # 提取資料
        "require_approval": True  # 高風險,需確認
    }
)

# AI 會暫停並詢問:
# ⚠️ 檢測到可能的 SQL 注入
# 建議 Payload: admin' OR '1'='1'--
# 風險: HIGH (可能影響資料庫)
# 是否繼續? [Y/n]
```

### 場景 4: 完整的 Bug Bounty 工作流

```python
# 完整流程: 從偵察到報告提交
controller = BioNeuronMasterController(default_mode="hybrid")

# Phase 1: 資訊收集 (自動)
recon = await controller.process_request(
    "對 example.com 執行完整偵察",
    context={
        "tasks": ["subdomain_enum", "port_scan", "tech_stack_detection"],
        "passive_only": False  # 包含主動掃描
    }
)

# Phase 2: 漏洞掃描 (半自動)
scan = await controller.process_request(
    f"掃描發現的 {len(recon['subdomains'])} 個子域名",
    context={
        "targets": recon['subdomains'],
        "vulnerability_types": "all",
        "auto_approve_risk_levels": ["low", "medium"]
    }
)

# Phase 3: 漏洞驗證 (手動確認)
for vuln in scan['potential_vulnerabilities']:
    validation = await controller.process_request(
        f"驗證 {vuln['type']} 漏洞: {vuln['location']}",
        context={
            "require_approval": True,  # 每個都需確認
            "generate_poc": True
        }
    )

# Phase 4: 報告生成 (自動)
report = await controller.process_request(
    "生成 HackerOne 報告草稿",
    context={
        "vulnerabilities": [v for v in scan['found'] if v['verified']],
        "format": "hackerone_markdown",
        "include_timeline": True
    }
)

print(f"📊 完成! 發現 {len(report['vulnerabilities'])} 個漏洞")
print(f"💰 預估賞金: ${report['estimated_bounty']}")
```

---

## 🔧 進階配置

### 配置 AI 決策行為

```python
# 自訂 AI 決策參數
controller = BioNeuronMasterController(
    codebase_path="/workspaces/AIVA",
    default_mode="hybrid"
)

# 設定風險閾值
controller.config.update({
    "risk_thresholds": {
        "low": 0.3,      # 風險評分 < 0.3 為低風險
        "medium": 0.6,   # 0.3-0.6 為中風險
        "high": 1.0      # > 0.6 為高風險
    },
    "auto_approve_confidence": 0.85,  # AI 置信度 > 0.85 自動執行
    "max_parallel_tasks": 20,         # 最多 20 個並發任務
    "request_rate_limit": 100         # 每分鐘最多 100 個請求
})
```

### 整合知識庫 (RAG)

```python
# 添加自訂漏洞知識
from services.core.aiva_core.rag import KnowledgeBase

kb = controller.rag_engine.knowledge_base

# 添加您的成功案例
await kb.add_knowledge(
    category="successful_exploits",
    content={
        "vulnerability": "IDOR in /api/profile",
        "payload": "Change user_id parameter",
        "target": "example.com",
        "bounty": 500,
        "notes": "No rate limiting, easy to automate"
    }
)

# AI 會在未來的測試中參考這個案例
```

### 配置多語言 AI 協調

```python
# 啟用特定語言的 AI 模組
from services.core.aiva_core.ai_commander import AICommander

commander = AICommander(codebase_path="/workspaces/AIVA")

# 配置任務分配策略
commander.config = {
    "task_routing": {
        "vulnerability_detection": "python_ai",  # Python AI 處理漏洞檢測
        "code_analysis": "rust_ai",              # Rust AI 處理代碼分析
        "network_scanning": "go_ai",             # Go AI 處理網路掃描
        "web_exploitation": "typescript_ai"      # TS AI 處理 Web 漏洞
    }
}
```

---

## ⚠️ 安全注意事項

### 🚨 重要警告

1. **僅測試授權目標**
   - ❌ 切勿使用 AIVA 攻擊未經授權的系統
   - ✅ 確保目標在 Bug Bounty 範圍內
   - ✅ 保存授權證明 (如 Bug Bounty 計畫頁面截圖)

2. **避免 DoS 攻擊**
   ```python
   # ❌ 錯誤: 可能造成 DoS
   result = await controller.process_request(
       "暴力破解登入",
       context={"rate_limit": None, "parallel": 1000}
   )
   
   # ✅ 正確: 設定合理限制
   result = await controller.process_request(
       "測試暴力破解保護",
       context={
           "max_attempts": 100,
           "delay_between_requests": 1.0,  # 1秒間隔
           "parallel": 5
       }
   )
   ```

3. **數據隱私保護**
   - ✅ 不要提取真實用戶數據
   - ✅ 使用測試帳號進行測試
   - ✅ 及時刪除測試數據

4. **AI 決策審查**
   - ⚠️ AI 可能產生誤判
   - ✅ 高風險操作務必人工確認
   - ✅ 定期檢查 AI 決策日誌

### 🛡️ 風險控制最佳實踐

```python
# 建議的安全配置
safe_config = {
    "mode": "hybrid",  # 使用混合模式
    "auto_approve_risk_levels": ["low"],  # 僅自動執行低風險
    "require_approval_for": [
        "sql_injection",
        "rce",
        "xxe",
        "ssrf_internal",
        "file_upload"
    ],
    "rate_limiting": {
        "max_requests_per_minute": 60,
        "max_requests_per_target": 1000
    },
    "safety_checks": {
        "verify_scope": True,  # 驗證目標在範圍內
        "check_robots_txt": True,
        "respect_rate_limits": True
    }
}

controller = BioNeuronMasterController(default_mode="hybrid")
controller.config.update(safe_config)
```

---

## 📊 效能與限制

### 效能指標

| 指標 | 典型值 | 說明 |
|------|--------|------|
| **AI 決策時間** | 0.5-2 秒 | 簡單任務更快,複雜任務稍慢 |
| **RAG 知識檢索** | < 0.1 秒 | 從知識庫檢索相關資訊 |
| **並發任務數** | 最多 50 | 可同時執行的獨立任務 |
| **請求速率** | 100 req/min | 預設限制,可調整 |

### 已知限制

1. **AI 模型限制**
   - 500萬參數神經網路 (中等規模)
   - 複雜推理能力不如 GPT-4
   - 需依賴 RAG 知識增強

2. **支援的漏洞類型**
   - ✅ 完全支援: XSS, SQLi, IDOR, CSRF, Open Redirect
   - ⚠️ 部分支援: SSRF, XXE, Deserialization
   - ❌ 不支援: 0-day 發現 (需人工分析)

3. **目標限制**
   - ✅ Web 應用程式
   - ✅ REST API
   - ⚠️ GraphQL (基礎支援)
   - ❌ 二進制協議 (如 Protobuf)

---

## 🆘 常見問題

### Q1: AI 做出錯誤決策怎麼辦?

**A:** 使用混合模式或 UI 模式,在高風險操作前人工審查。同時報告錯誤決策幫助 AI 學習:

```python
# 報告錯誤決策
await controller.report_incorrect_decision(
    task_id="task_12345",
    issue="AI 誤判 false positive 為真實漏洞",
    correct_action="應該跳過這個結果"
)
```

### Q2: 如何加快掃描速度?

**A:** 調整並發參數和使用 AI 模式:

```python
result = await controller.process_request(
    "快速掃描 100 個目標",
    context={
        "mode": "ai",  # 完全自動化
        "parallel_workers": 50,  # 增加並發
        "depth": "shallow",  # 淺層掃描
        "skip_verification": False  # 保持驗證以確保準確性
    }
)
```

### Q3: AI 如何避免重複測試?

**A:** AI 會自動記錄已測試的目標和方法:

```python
# 檢查歷史記錄
history = await controller.get_testing_history(
    target="example.com",
    timeframe="last_7_days"
)

# AI 會自動跳過重複測試
result = await controller.process_request(
    "測試 example.com",
    context={"skip_if_tested_recently": True}
)
```

### Q4: 如何導出報告?

**A:** 多種格式支援:

```python
# HackerOne 格式
hackerone_report = await controller.export_report(
    format="hackerone",
    vulnerabilities=result['found'],
    include_poc=True
)

# Markdown 格式
markdown_report = await controller.export_report(
    format="markdown",
    vulnerabilities=result['found']
)

# JSON 格式 (適合自動化處理)
json_report = await controller.export_report(
    format="json",
    vulnerabilities=result['found']
)
```

### Q5: AI 訓練數據從何而來?

**A:** AIVA 從多個來源學習:
- 📚 公開漏洞資料庫 (CVE, CWE)
- 🎓 HackerOne 公開報告
- 💼 您的成功測試經驗 (隱私保護)
- 🔬 安全研究論文

### Q6: 支援哪些程式語言的代碼分析?

**A:** 目前支援:
- ✅ Python, JavaScript, TypeScript
- ✅ PHP, Java
- ⚠️ Go, Rust (基礎支援)
- ❌ C/C++ (計畫中)

---

## 📚 延伸閱讀

- 📖 [AI 引擎技術文件](README_AI_ENGINE.md)
- 🔧 [開發者指南](README_DEVELOPMENT.md)
- 🧪 [測試指南](README_TESTING.md)
- 📊 [架構設計](../README.md)

---

## 🤝 回饋與支援

遇到問題或有改進建議?

1. 📝 查看 [問題排查指南](../TROUBLESHOOTING.md)
2. 💬 加入社群討論
3. 🐛 提交 Bug 報告
4. 💡 提出功能建議

---

**📝 文件版本**: v2.0  
**🔄 最後更新**: 2025-10-27  
**👥 目標讀者**: HackerOne 漏洞獵人、滲透測試人員  
**⏱️ 預估學習時間**: 1-2 小時上手基礎功能,1週精通進階用法

---

> **💡 提示**: 建議從「混合模式」開始使用 AIVA,這是效率與安全的最佳平衡點。隨著經驗累積,可逐步增加自動化程度。

> **🎯 快速開始**: 複製上方的「場景 4: 完整的 Bug Bounty 工作流」代碼,修改目標後執行,立即體驗 AIVA 的完整能力!
