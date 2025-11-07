# 02｜CRYPTO 設計規格（成熟版）
**日期**：2025-11-06  

## 1. 目標與範圍
- 偵測弱雜湊（MD5/SHA1）、弱對稱（DES/RC4/ECB）、不安全 TLS（verify=False, TLS1.0/1.1）、硬編碼密鑰、弱隨機等。
- 接收 `FunctionTaskPayload`，輸出 `FindingPayload` 與 SARIF。

## 2. 架構
- **Engine（Rust, PyO3）**：高效規則與啟發式；可引入 openssl / rustls。
- **Detector（Python）**：將 Engine 輸出映射為統一 `Vulnerability`/`Finding*`。
- **Worker（Python, async）**：AMQP 訂閱 `Topic.TASK_FUNCTION_CRYPTO`，回報 `Topic.FINDING_DETECTED`、`Topic.STATUS_TASK_UPDATE`。
- **Config（Pydantic v2）**：規則、白名單、臨界值。

## 3. 通訊與合約
- 僅使用 `services.aiva_common` 之合約：`AivaMessage`、`MessageHeader`、`FunctionTaskPayload`、`FindingPayload` 等。
- 範例 Payload：見《04》文件。

## 4. 安全與可觀測
- 僅檢測/讀檔；**不**主動連外掃描目標。
- 以統一 logger 產生 structured log；hook 指標到 Telemetry。

## 5. 測試與驗收
- 準確率：對照 SSLyze / testssl.sh 樣本 ≥95%。
- 效能：較純 Python 檢測 ≥ +50%。
- 覆蓋 ≥ 100 個測例；CI 強制。

## 6. 目錄
- 參見 `services/features/function_crypto/*` 與《06》部署文件。
