# AIVA OOB 盲注升級計畫與分析 (Interactsh 整合方案)

> **建立日期**: 2026-03-26
> **目標**: 解決真實黑盒漏洞懸賞 (Bug Bounty) 環境中，傳統 Blind SQLi (Time/Boolean/Error) 受到 WAF 阻擋與伺服器非同步處理的問題，導入 100% 準確率的外帶盲注 (Out-of-Band, OOB) 驗證機制。

## 1. 核心概念與原理

在一線的 Bug Bounty 中，由於 WAF 和後端微服務架構的普及，直接透過 HTTP 回應來驗證 SQLi 越來越困難。OOB 盲注的原理是：
1. **不依賴前門 (HTTP)**：放棄從網頁回應文字或延遲時間來判斷。
2. **強迫走後門 (DNS/HTTP)**：強迫後端資料庫 (如 MySQL, MSSQL) 解析一個包含專屬憑證的外部網域，或向外發送 HTTP 請求。
3. **驗證連線 (OAST Server)**：AIVA 透過 API 輪詢專屬的 OAST (Out-of-Band Application Security Testing) 伺服器，若有收到帶有特徵的請求，代表 Payload 已成功在資料庫核心內部執行。

## 2. 工具選擇：Interactsh (ProjectDiscovery)

Interactsh 是一個開源且去中心化的 OAST 解決方案，支援 API 互動。

### 方案 A：使用公開伺服器 (Public API)
*   **作法**：程式直接呼叫 `oast.pro`, `interact.sh` 等官方免費節點。
*   **優點**：
    *   **無成本、免註冊**，立即可用。
    *   快速測試與驗證架構設計。
*   **缺點**：
    *   有時會遇到 API 速率限制 (Rate limiting)。
    *   **隱私風險**：竊取出來的資產內部資訊 (如資料庫版本、甚至密碼 hash) 會流經公開伺服器，**某些嚴格的 Bug Bounty 計畫會判定違反保密協議 (NDA)**。

### 方案 B：自建私有伺服器 (Self-Hosted Private API)
*   **作法**：在 AWS/DigitalOcean 租用 VPS，運行 `docker run projectdiscovery/interactsh-server` 並綁定自定義網域 `oob.your-domain.com`。AIVA 取代預設 URL 改打這個私有 API。
*   **優點**：
    *   **極致安全**：資料 100% 控制在自己手上，絕對符合 NDA 規範。
    *   **沒有存取限制**，穩定度由自己掌控。
    *   IP 不容易被 WAF 當作黑名單攔截 (官方公開的 interact.sh 網域有時會被企業信譽庫判定為潛在威脅)。
*   **缺點**：
    *   每月需負擔少量的 VPS 與網域名稱成本 (約 $5~$10 USD/月)。
    *   需自行維護與設定 DNS Nameserver。

## 3. 在 AIVA 系統的整合計畫
*於 `services/features/function_sqli/engines/oob_detection_engine.py` 中實作：*

1.  **InteractshClient 模組**：
    新增一個非同步的 HTTP Client (基於 `aiohttp`)，用來向 Interactsh API 發送 `/register` (取得 uuid.oast.me) 以及 `/poll` (拿取連線紀錄 JSON)。
2.  **OOB 邏輯重構**：
    原本的假 OOB (僅由網頁回應字眼檢查) 改為：送出 Payload 後 -> 等待 1~5 秒 -> 啟動 `poll()` 去問 Interactsh -> 若有 DNS 紀錄，則標記 `is_vulnerable = True` 並萃取資料。
3.  **Config 擴充**：
    在 `SqliConfig` 加入 `oast_server_url` 和 `oast_auth_token` 等設定項，讓使用者未來能無縫從公開切換至私有伺服器。

---
*備註：此文件已留存，先跳回執行 WAF Bypass Payload Encoder 的改進計畫。*
