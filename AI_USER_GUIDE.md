# AIVA AI 助手使用指南 🤖

> **📋 使用者指南**：適合一般使用者，用簡單易懂的方式介紹如何使用 AIVA AI  
> **🔧 技術指南**：如需詳細的技術資訊，請參考 `services/core/docs/AI_SERVICES_USER_GUIDE.md`

> **一句話說明**：AIVA AI 助手是您的智能安全測試夥伴，能聽懂您的需求、自動執行掃描、學習優化策略。

## 🚀 立即開始

### 1️⃣ 啟動 AIVA AI
```bash
# 方法一：使用啟動器（推薦）
python aiva_launcher.py

# 方法二：直接啟動對話助手
python -c "from services.core.aiva_core.dialog.assistant import dialog_assistant; 
import asyncio; 
asyncio.run(dialog_assistant.process_user_input('你好，你會什麼？'))"
```

### 2️⃣ 開始對話
啟動後，您可以直接用自然語言與 AIVA 對話：

```
您：你好，你現在會什麼？
AIVA：🚀 AIVA 目前可用功能:
      📊 總能力數: 174+ 個
      🔤 語言分布: Python(120), Go(35), Rust(15)
      💚 健康狀態: 165 個健康
      
      🎯 主要功能模組:
      ✅ XSS 掃描 (Python)
      ✅ SQL 注入檢測 (Python) 
      ✅ SSRF 測試 (Go)
      ...
```

## 💬 五種使用方式

### 🔍 **1. 查詢能力**
**問什麼**：
- `你會什麼？` / `有什麼功能？`
- `系統狀況如何？` / `健康檢查`

**得到什麼**：
- 完整的能力清單
- 每個功能的健康狀態
- 支援的程式語言分布

---

### 📖 **2. 了解功能**
**問什麼**：
- `解釋 XSS 掃描功能`
- `SQL 注入檢測怎麼用？`
- `說明 SSRF 測試`

**得到什麼**：
- 功能詳細說明
- 輸入參數要求
- 輸出結果格式
- 使用建議

---

### ⚡ **3. 一鍵執行**
**問什麼**：
- `幫我跑 XSS 掃描 https://example.com`
- `測試這個網站的 SQL 注入`
- `執行 SSRF 檢測`

**得到什麼**：
- 自動選擇最佳掃描策略
- 即時執行進度回報
- 完整的測試結果
- 發現的漏洞詳情

---

### 🆚 **4. 比較分析**
**問什麼**：
- `比較 Python 和 Go 版本的 SSRF 差異`
- `XSS 掃描和 SQL 注入哪個更準確？`

**得到什麼**：
- 功能對比分析
- 性能差異說明
- 使用場景建議
- 準確率統計

---

### 💻 **5. 生成指令**
**問什麼**：
- `產生可執行的 CLI 指令`
- `給我 XSS 掃描的命令`
- `輸出可直接用的指令`

**得到什麼**：
```bash
# XSS 掃描:
aiva capability execute cap.func_xss.reflected --url https://example.com --timeout 30

# SQL 注入檢測:  
aiva capability execute cap.func_sqli.boolean --url https://example.com --timeout 30

# SSRF 測試:
aiva capability execute cap.func_ssrf.basic --url https://example.com
```

## 🎯 實用場景範例

### 場景 1：新手探索
```
您：我是新手，不知道從哪開始
AIVA：歡迎！建議您先了解我們的基礎功能：
      1. XSS 掃描 - 檢測跨站腳本攻擊
      2. SQL 注入 - 檢測資料庫注入漏洞  
      3. SSRF 測試 - 檢測伺服器端請求偽造
      
      想先試試哪一個？
```

### 場景 2：日常工作
```
您：我有個登入頁面要測試 https://target.com/login
AIVA：好的！正在為您安排測試計劃...
      
      🎯 建議測試順序：
      1. SQL 注入檢測（登入框常見）
      2. XSS 反射型測試  
      3. 暴力破解防護檢查
      
      是否開始執行？
```

### 場景 3：學習提升
```
您：我想學習 SSRF 攻擊技巧
AIVA：SSRF（伺服器端請求偽造）是很重要的攻擊技術！
      
      🧠 核心概念：讓目標服務器發送您構造的請求
      🎯 常見場景：URL 預覽、圖片處理、API 調用
      🛡️ 防護方法：白名單、內網隔離
      
      想看實際的測試案例嗎？
```

## ⚙️ 自動化特色

### 🧠 **智能決策**
- 根據目標特徵自動選擇測試策略
- 學習歷史成功經驗優化測試順序
- 動態調整測試參數提升準確率

### 🔄 **持續學習**
- 記錄每次測試的成果和問題
- 分析成功率趨勢調整權重
- 發現新的攻擊路徑組合

### 🚀 **一鍵部署**
- 支援 Python、Go、Rust 多語言工具
- 自動處理工具間的數據轉換
- 統一的結果格式和錯誤處理

## 🔧 進階使用

### 自定義配置
```python
# 調整對話助手設定
from services.aiva_common.ai import AIVADialogAssistant, DialogConfig

config = DialogConfig(
    session_timeout_minutes=60,    # 會話超時
    max_response_length=3000,      # 回應長度
    intent_confidence_threshold=0.8 # 意圖識別信心值
)

assistant = AIVADialogAssistant(config=config)
```

### 批量操作
```python
# 同時測試多個目標
targets = ["https://site1.com", "https://site2.com", "https://site3.com"]
for target in targets:
    result = await assistant.send_message(
        user_id="batch_user",
        message=f"幫我跑完整掃描 {target}"
    )
    print(f"{target}: {result}")
```

## 📊 當前能力統計

根據最新實施進度報告：

| 項目 | 完成度 | 說明 |
|------|--------|------|
| 🎯 **整體完成度** | **82%** | 核心功能已可正常使用 |
| 🤖 **AI 對話層** | **85%** | 自然語言理解和回應 |
| 🔄 **計劃執行** | **80%** | 自動化測試執行 |
| 🌐 **跨語言支援** | **50%** | Python 完整，Go/Rust 部分支援 |
| 📚 **經驗學習** | **80%** | 成功率追蹤和優化 |
| 🔍 **能力評估** | **85%** | 多維度性能分析 |

## ⚠️ 使用注意事項

### ✅ **適合的場景**
- 🎯 滲透測試和安全評估
- 🔍 漏洞挖掘和 Bug Bounty
- 📚 安全技術學習和研究
- 🚀 自動化安全測試流程

### ⚠️ **限制和風險**
- 🚫 僅限合法授權的測試目標
- ⏱️ 部分跨語言功能仍在完善中
- 🔄 AI 學習需要一定時間累積經驗
- 🛡️ 建議在隔離環境中進行測試

## 🆘 常見問題

**Q: AIVA 回應很慢怎麼辦？**
A: 首次使用需要初始化各種組件，後續會更快。可以檢查系統資源和網路狀況。

**Q: 如何查看測試的詳細日誌？**
A: 問 AIVA `系統狀況如何？` 可以看到健康檢查和統計資訊。

**Q: 能否自定義測試規則？**
A: 目前 AI 會自動選擇最佳策略，未來版本將支援更多自定義選項。

**Q: 支援哪些目標類型？**
A: 主要支援 Web 應用程式，包括 HTTP/HTTPS 網站、API 端點等。

**Q: 如何匯出測試結果？**
A: 測試結果會自動記錄在系統中，可以要求 AIVA 生成報告或 CLI 指令查看。

---

## 🎉 開始您的 AI 安全測試之旅！

現在就試試對 AIVA 說：**「你好，幫我開始第一次安全測試！」**

AIVA AI 助手將成為您最可靠的安全測試夥伴，讓複雜的滲透測試變得像聊天一樣簡單！ 🚀✨